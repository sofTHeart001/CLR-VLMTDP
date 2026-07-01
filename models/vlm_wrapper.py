"""
VLM Wrapper Module
封装视觉-语言模型，用于任务拆解、轨迹生成和状态验证

架构：
  VLMWrapperBase (nn.Module, abstract)
    ├── OpenAIVLMWrapper  ──  默认后端；调用 OpenAI 兼容 HTTP API
    └── LocalLLaVAWrapper  ──  离线占位，不主动加载模型（供单元测试）

  VLMWrapper()  ── 工厂函数，根据 backend 参数或 VLM_BACKEND 环境变量分派

向后兼容：
  - models/__init__.py 仍导出 `VLMWrapper`
  - 调用方写法 `vlm.decompose_task(...)` 不变
  - `isinstance(vlm, nn.Module)` 仍然为真（基类继承 nn.Module）
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import httpx
except ImportError:  # pragma: no cover - 让本地无 httpx 时仍可 import
    httpx = None  # type: ignore

try:
    from utils.prompt_templates import PromptTemplate
except ImportError:
    # 支持 `python -m models.vlm_wrapper` 这类直接执行
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.prompt_templates import PromptTemplate  # type: ignore


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VLMError(RuntimeError):
    """VLM 模块通用错误"""


class VLMJSONParseError(VLMError):
    """VLM 输出无法解析为 JSON（已重试仍失败）"""


# ---------------------------------------------------------------------------
# 图像编码工具
# ---------------------------------------------------------------------------


def _pil_from_any(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """统一转为 PIL.Image（RGB）。"""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            # CHW -> HWC
            arr = arr.permute(1, 2, 0)
        if arr.dtype == torch.float32 or arr.dtype == torch.float16:
            arr = (arr.clamp(0, 1) * 255).to(torch.uint8)
        arr = arr.numpy()
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_data_url(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    image_max_side: int = 512,
    fmt: str = "PNG",
) -> str:
    """PIL/np/torch 图像 → base64 PNG data URL。"""
    pil = _pil_from_any(image)
    if max(pil.size) > image_max_side:
        scale = image_max_side / max(pil.size)
        new_size = (int(pil.size[0] * scale), int(pil.size[1] * scale))
        pil = pil.resize(new_size, Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# JSON 抽取
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    鲁棒地从 VLM 输出里抽取 JSON 对象。
    策略：直接 loads → 找 ```json``` 围栏 → 找首对花括号。
    """
    text = text.strip()

    # 1. 整体直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. ```json ... ``` 围栏
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 找首对花括号（贪婪）
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start: end + 1]
        # 单引号 -> 双引号（容错 VLM 偶尔返回 Python 字面量风格）
        candidate2 = re.sub(r"(?<!\")\'", '"', candidate)
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# 体素归一化
# ---------------------------------------------------------------------------


def _flatten_nested(obj: Any) -> List[float]:
    """递归把嵌套 list / tuple / numpy 标量拍平成一维 float 列表。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().flatten().tolist()
    if isinstance(obj, np.ndarray):
        return obj.flatten().tolist()
    if isinstance(obj, (list, tuple)):
        out: List[float] = []
        for x in obj:
            out.extend(_flatten_nested(x))
        return out
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [float(obj)]
    raise VLMError(f"Cannot flatten object of type {type(obj)}")


def _coerce_to_voxel(arr: Any, grid_size: int = 6) -> torch.Tensor:
    """
    把 VLM 返回的体素数组归一化成 (grid_size, grid_size, grid_size) 的 0/1 torch.Tensor。

    接受的形态：
      - 216 长度扁平列表
      - 6×6 列表套列表（z 维度退化 → 塞到 z=0 层）
      - 6×6×6 三维嵌套
      - 6×(6×6) → 即 6 行，每行 6×6 矩阵
    """
    target = grid_size ** 3
    flat = _flatten_nested(arr)

    if len(flat) == target:
        data = flat
    elif len(flat) == grid_size * grid_size:
        # 6×6 退化形态 → 把 36 个数塞到 z=0 层，其余 5 层全 0
        data = list(flat) + [0.0] * (target - len(flat))
    else:
        raise VLMError(
            f"voxel_trajectory length {len(flat)} != {target} and != {grid_size**2} (and no known reshape)"
        )

    tensor = torch.tensor(data, dtype=torch.float32).reshape(grid_size, grid_size, grid_size)
    return (tensor > 0.5).float()


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class VLMWrapperBase(nn.Module, ABC):
    """所有 VLM 后端的统一接口。"""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> str:
        """基础文本生成；如有图则一并传入。"""

    def decompose_task(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        task_description: str,
        completed_subtasks: Optional[List[str]] = None,
    ) -> Tuple[str, torch.Tensor]:
        """任务拆解：返回 (subtask_desc, voxel_trajectory 6×6×6)。"""
        raise NotImplementedError

    def check_state(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        current_subtask: str,
        deviation_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """状态检查：返回 {subtask_done, need_replan, reason, confidence}。"""
        raise NotImplementedError

    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """初始任务分析。"""
        prompt = PromptTemplate.format_initial_analysis(task_description)
        text = self.generate_text(prompt, image=None)
        return _extract_json(text) or {}

    def forward(self, *args, **kwargs):  # pragma: no cover - 占位
        raise NotImplementedError("VLMWrapperBase is abstract; use concrete methods.")


# ---------------------------------------------------------------------------
# OpenAI 兼容实现
# ---------------------------------------------------------------------------


class OpenAIVLMWrapper(VLMWrapperBase):
    """
    通过 OpenAI 兼容 HTTP API 调用 VLM。
    支持任意 OpenAI 兼容服务：OpenAI / Azure / OpenRouter / 本地 vLLM / llama.cpp server。

    默认模型 gpt-4o-mini；视觉理解足够且成本低。
    """

    SYSTEM_PROMPT = (
        "You are an expert robot task planner and state evaluator. "
        "Always reply with strict JSON only — no prose, no markdown fences."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        image_max_side: int = 512,
        temperature: float = 0.0,
        timeout: int = 30,
        max_retries: int = 2,
        max_new_tokens: int = 512,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        if httpx is None:
            raise ImportError(
                "httpx is required for OpenAIVLMWrapper. `pip install httpx>=0.27.0`"
            )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env or pass api_key=."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.image_max_side = image_max_side
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self._client = httpx.Client(timeout=timeout)

    # ---------- HTTP -----------------------------------------------------

    def _call_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """发请求，带 retry/backoff；返回 content 字符串。"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_new_tokens if max_new_tokens is None else max_new_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(url, headers=headers, json=body)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise VLMError(f"transient {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise VLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, VLMError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                import time
                time.sleep(0.5 * (2 ** attempt))
        raise VLMError(f"OpenAI call failed after {self.max_retries + 1} attempts: {last_err}")

    # ---------- 文本生成 --------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url(image, self.image_max_side)},
            })
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self._call_chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)

    # ---------- 任务拆解 --------------------------------------------------

    def decompose_task(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        task_description: str,
        completed_subtasks: Optional[List[str]] = None,
    ) -> Tuple[str, torch.Tensor]:
        if completed_subtasks is None:
            completed_subtasks = []

        prompt = PromptTemplate.format_task_decomposition(task_description, completed_subtasks)
        prompt += (
            "\n\nREMINDER: voxel_trajectory must be a 6×6×6 nested list "
            "(216 numbers: 6 z-slices, each 6 rows of 6 cells, values 0/1)."
        )

        text = self.generate_text(prompt, image=image)
        parsed = self._parse_or_retry(text, prompt, image)

        subtask = str(parsed.get("subtask", "")).strip()
        voxel_raw = parsed.get("voxel_trajectory")
        if not subtask:
            raise VLMJSONParseError("decompose_task: missing 'subtask'")
        if voxel_raw is None:
            raise VLMJSONParseError("decompose_task: missing 'voxel_trajectory'")
        voxel = _coerce_to_voxel(voxel_raw, grid_size=6)
        return subtask, voxel

    # ---------- 状态检查 --------------------------------------------------

    def check_state(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        current_subtask: str,
        deviation_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        prompt = PromptTemplate.format_state_check(current_subtask)
        prompt += (
            f"\n\nTolerance threshold for replan: deviation_threshold={deviation_threshold}."
        )
        text = self.generate_text(prompt, image=image)
        parsed = self._parse_or_retry(text, prompt, image)
        return {
            "subtask_done": bool(parsed.get("subtask_done", False)),
            "need_replan": bool(parsed.get("need_replan", False)),
            "reason": str(parsed.get("reason", "")),
            "confidence": float(parsed.get("confidence", 0.0)),
        }

    # ---------- 内部：JSON 解析 + 一次重试 ------------------------------

    def _parse_or_retry(
        self,
        text: str,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]],
    ) -> Dict[str, Any]:
        parsed = _extract_json(text)
        if parsed is not None:
            return parsed

        retry_prompt = prompt + "\n\nREPLY WITH ONLY VALID JSON, NO PROSE."
        text2 = self.generate_text(retry_prompt, image=image)
        parsed2 = _extract_json(text2)
        if parsed2 is not None:
            return parsed2

        raise VLMJSONParseError(
            f"Failed to parse JSON after retry.\n--- attempt 1 ---\n{text}\n--- attempt 2 ---\n{text2}"
        )


# ---------------------------------------------------------------------------
# 旧实现的最小占位（不主动加载模型，供单元测试用）
# ---------------------------------------------------------------------------


class LocalLLaVAWrapper(VLMWrapperBase):
    """
    离线占位 VLM。默认不下载任何模型权重。
    单元测试可通过 monkeypatch `generate_text` 来注入固定响应。
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        load_in_4bit: bool = True,
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        super().__init__()
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        # 不立即加载模型；只在真正调用 generate_text 时才报错/懒加载
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):  # pragma: no cover - 留给未来扩展
        if self._model is not None:
            return
        # 显式延迟 import，避免无 transformers 的环境 import 失败
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        kwargs = dict(cache_dir=self.cache_dir, trust_remote_code=True)
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", **kwargs
        )

    def generate_text(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> str:
        # 离线默认行为：抛错，避免在没接 LLaVA 时静默成功
        raise NotImplementedError(
            "LocalLLaVAWrapper.generate_text is a stub. "
            "Implement the inference loop or monkeypatch this method in tests."
        )

    def decompose_task(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        task_description: str,
        completed_subtasks: Optional[List[str]] = None,
    ) -> Tuple[str, torch.Tensor]:
        text = self.generate_text("dummy", image=image)
        parsed = _extract_json(text) or {}
        return str(parsed.get("subtask", "")), _coerce_to_voxel(
            parsed.get("voxel_trajectory", []), grid_size=6
        )

    def check_state(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        current_subtask: str,
        deviation_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        text = self.generate_text("dummy", image=image)
        parsed = _extract_json(text) or {}
        return {
            "subtask_done": bool(parsed.get("subtask_done", False)),
            "need_replan": bool(parsed.get("need_replan", False)),
            "reason": str(parsed.get("reason", "")),
            "confidence": float(parsed.get("confidence", 0.0)),
        }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def VLMWrapper(  # noqa: N802 (保留大写名以兼容旧调用)
    *args,
    backend: Optional[str] = None,
    **kwargs,
) -> VLMWrapperBase:
    """
    VLM 工厂函数。`backend` 决定具体实现：
        - "openai"（默认）→ OpenAIVLMWrapper
        - "local"          → LocalLLaVAWrapper
    也可由 VLM_BACKEND 环境变量覆盖。
    """
    chosen = (backend or os.environ.get("VLM_BACKEND") or "openai").lower()
    if chosen == "openai":
        return OpenAIVLMWrapper(*args, **kwargs)
    if chosen == "local":
        return LocalLLaVAWrapper(*args, **kwargs)
    raise ValueError(f"Unknown VLM backend: {chosen!r}")


def create_vlm_wrapper(config: Dict[str, Any]) -> VLMWrapperBase:
    """从配置 dict 创建 VLM 实例。"""
    vlm_cfg = dict(config.get("vlm", {}))
    backend = vlm_cfg.pop("backend", None)
    # 把 ${ENV_VAR} 占位符解析为实际值
    api_key = vlm_cfg.get("api_key")
    if isinstance(api_key, str) and api_key.startswith("${"):
        env_name = api_key[2:-1]
        api_key = os.environ.get(env_name)
        if api_key is None:
            raise ValueError(f"Env var {env_name} not set for vlm.api_key")
        vlm_cfg["api_key"] = api_key
    return VLMWrapper(backend=backend, **vlm_cfg)


# ---------------------------------------------------------------------------
# 直接执行自检
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("VLM wrapper module. Quick smoke checks:")

    # 1) JSON 抽取鲁棒性
    samples = [
        '{"a": 1, "b": [1,2,3]}',
        '```json\n{"a": 1}\n```',
        "noise\n```\n{\"x\": [1,2]}\n```\nmore noise",
        "{'a': 1, 'b': 'hi'}",
    ]
    for s in samples:
        out = _extract_json(s)
        print(f"  _extract_json({s!r}) -> {out}")

    # 2) 体素归一化各形态
    flat = [1] * 216
    nested = [[[1] * 6 for _ in range(6)] for _ in range(6)]
    degenerated = [1] * 36
    mixed = [[[0] * 6 for _ in range(6)] for _ in range(6)]  # 全 0
    mixed[2][3][4] = 1  # 单点
    for arr, desc in [
        (flat, "flat 216"),
        (nested, "6x6x6"),
        (degenerated, "6x6 degenerated"),
        (mixed, "6x6x6 single voxel"),
    ]:
        v = _coerce_to_voxel(arr)
        print(f"  _coerce_to_voxel({desc}) -> shape={tuple(v.shape)} sum={int(v.sum())}")

    # 3) 工厂函数（不需要 API key，因为只是路由测试）
    print(f"  Factory function exists: {callable(VLMWrapper)}")
    print(f"  Base class: {VLMWrapperBase.__name__}")
    print(f"  Concrete classes: OpenAIVLMWrapper, LocalLLaVAWrapper")