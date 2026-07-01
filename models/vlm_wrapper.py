"""
VLM Wrapper Module
封装视觉-语言模型，用于任务拆解、轨迹生成和状态验证
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from PIL import Image
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


class VLMWrapper(nn.Module):
    """
    VLM封装模块

    功能：
    1. 任务拆解：将长时序任务分解为原子子任务
    2. 轨迹生成：根据当前场景生成6x6x6体素轨迹
    3. 状态验证：判断子任务执行结果是否符合预期

    输入：
        - RGB图像 (B, 3, 600, 800)
        - 全局任务描述 (str)
        - 已完成子任务列表 (List[str])
        - 当前子任务描述 (str, 状态检查时)

    输出：
        - 子任务描述 (str)
        - 体素轨迹 (B, 6, 6, 6)
        - 状态检查结果 (Dict)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        cache_dir: Optional[str] = None
    ):
        super().__init__()

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # 量化配置
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            dtype = torch.float16
        else:
            bnb_config = None
            dtype = torch.float32

        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        # 图像编码器（使用CLIP Vision Encoder作为基础）
        # 这里简化处理，实际应该使用视觉-语言模型
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码图像为特征向量

        Args:
            image: (B, 3, H, W) RGB图像

        Returns:
            image_features: (B, 256) 图像特征
        """
        return self.image_encoder(image).squeeze(-1).squeeze(-1)

    def generate_text(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        生成文本回复

        Args:
            prompt: 输入提示
            image: 可选的图像张量
            temperature: 采样温度
            top_p: nucleus采样参数

        Returns:
            生成的文本
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        if image is not None:
            # 将图像特征融入prompt（简化处理）
            image_features = self.encode_image(image)
            # 实际实现中需要更复杂的融合方式

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def decompose_task(
        self,
        image: torch.Tensor,
        task_description: str,
        completed_subtasks: Optional[List[str]] = None
    ) -> Tuple[str, torch.Tensor]:
        """
        任务拆解：生成下一个子任务及其对应的体素轨迹

        Args:
            image: 当前场景RGB图像 (B, 3, 600, 800)
            task_description: 全局任务描述
            completed_subtasks: 已完成的子任务列表

        Returns:
            subtask_description: 子任务描述
            voxel_trajectory: 体素轨迹 (B, 6, 6, 6)
        """
        if completed_subtasks is None:
            completed_subtasks = []

        # 构建任务拆解的prompt
        prompt = f"""You are a robot task planner. Given the current scene and the overall task, generate the next subtask and its corresponding voxel trajectory.

Overall task: {task_description}

Completed subtasks: {', '.join(completed_subtasks) if completed_subtasks else 'None'}

Please output the next subtask and a 6x6x6 voxel trajectory in JSON format:
{{
    "subtask": "description of the next subtask",
    "voxel_trajectory": [[...6 rows...], ...6 elements...]
}}"""

        response = self.generate_text(prompt, image)

        # 解析JSON响应
        try:
            # 提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)

            subtask_description = parsed["subtask"]
            voxel_trajectory = torch.tensor(parsed["voxel_trajectory"], dtype=torch.float32)

        except (json.JSONDecodeError, KeyError):
            # 解析失败时返回默认值
            subtask_description = "pick up the nearest object"
            voxel_trajectory = torch.zeros(6, 6, 6)

        return subtask_description, voxel_trajectory

    def check_state(
        self,
        image: torch.Tensor,
        current_subtask: str,
        deviation_threshold: float = 0.15
    ) -> Dict[str, bool]:
        """
        状态检查：判断子任务是否完成或需要重规划

        Args:
            image: 当前场景RGB图像 (B, 3, 600, 800)
            current_subtask: 当前子任务描述
            deviation_threshold: 偏差阈值

        Returns:
            {
                "subtask_done": bool,  # 子任务是否完成
                "need_replan": bool    # 是否需要重规划
            }
        """
        prompt = f"""You are a robot state checker. Given the current scene and the current subtask, determine if the subtask is completed or if replanning is needed.

Current subtask: {current_subtask}

Please output your assessment in JSON format:
{{
    "subtask_done": true/false,
    "need_replan": true/false,
    "reason": "brief explanation"
}}"""

        response = self.generate_text(prompt, image)

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)

            return {
                "subtask_done": parsed.get("subtask_done", False),
                "need_replan": parsed.get("need_replan", False),
                "reason": parsed.get("reason", "")
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "subtask_done": False,
                "need_replan": False,
                "reason": "parsing error"
            }

    def forward(self, *args, **kwargs):
        """Forward pass for compatibility with PyTorch model interface"""
        raise NotImplementedError("VLMWrapper is a wrapper, use specific methods instead")


def create_vlm_wrapper(config: Dict) -> VLMWrapper:
    """
    工厂函数：根据配置创建VLMWrapper实例

    Args:
        config: 配置字典，包含vlm相关配置

    Returns:
        VLMWrapper实例
    """
    return VLMWrapper(
        model_name=config.get("model_name", "meta-llama/Llama-2-7b-chat-hf"),
        load_in_4bit=config.get("load_in_4bit", True),
        max_new_tokens=config.get("max_new_tokens", 512),
        cache_dir=config.get("cache_dir", None)
    )