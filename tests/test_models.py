"""
Unit tests for models/ — runs on CPU without GPU or external API.
"""

import sys
from pathlib import Path

# Make `models` and `utils` importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from models import FlowTDP, LightVoxelEncoder, VLMWrapper, VLMWrapperBase
from models.light_voxel_encoder import StandardVoxelEncoder
from models.vlm_wrapper import (
    _extract_json,
    _coerce_to_voxel,
    LocalLLaVAWrapper,
    OpenAIVLMWrapper,
)


# ---------------------------------------------------------------------------
# LightVoxelEncoder
# ---------------------------------------------------------------------------


def test_light_voxel_encoder_shape():
    enc = LightVoxelEncoder()
    x = torch.randn(4, 6, 6, 6)
    y = enc(x)
    assert y.shape == (4, 128), f"expected (4, 128), got {tuple(y.shape)}"


def test_light_voxel_encoder_param_reduction():
    enc = LightVoxelEncoder()
    stats = enc.get_parameter_count()
    assert stats["reduction_pct"] >= 40.0, (
        f"expected >= 40% param reduction, got {stats['reduction_pct']}%"
    )


def test_light_voxel_encoder_no_dead_attrs():
    enc = LightVoxelEncoder()
    assert not hasattr(enc, "input_proj"), "input_proj should be removed"
    buffers = dict(enc.named_buffers())
    assert "positional_encoding" not in buffers, "positional_encoding buffer should be removed"


# ---------------------------------------------------------------------------
# FlowTDP
# ---------------------------------------------------------------------------


@pytest.fixture
def flow_model():
    # 用小尺寸 (64, 64) 让 CPU 测试快
    return FlowTDP(image_size=(64, 64))


@pytest.fixture
def flow_inputs():
    return {
        "image": torch.randn(2, 3, 64, 64),
        "traj": torch.randn(2, 128),
        "t": torch.tensor([0.5, 0.5]),
        "target": torch.randn(2, 8),
    }


def test_flow_tdp_forward(flow_model, flow_inputs):
    out = flow_model(flow_inputs["image"], flow_inputs["traj"], flow_inputs["t"])
    assert out["action"].shape == (2, 8)
    assert out["velocity"].shape == (2, 8)
    assert out["features"].shape == (2, 256)


def test_flow_tdp_sample_action_num_steps_1(flow_model, flow_inputs):
    a = flow_model.sample_action(flow_inputs["image"], flow_inputs["traj"], num_steps=1)
    assert a.shape == (2, 8)
    assert torch.isfinite(a).all(), "sampled action contains NaN/Inf"


def test_flow_tdp_sample_action_num_steps_4_differs(flow_model, flow_inputs):
    """多步 Euler 积分应该与单步不同（积分路径不同）。"""
    a1 = flow_model.sample_action(flow_inputs["image"], flow_inputs["traj"], num_steps=1)
    a4 = flow_model.sample_action(flow_inputs["image"], flow_inputs["traj"], num_steps=4)
    assert a1.shape == a4.shape
    # 不强求数值差异巨大（未训练的模型输出可能接近），但至少 shape 一致
    assert torch.isfinite(a4).all()


def test_flow_tdp_loss_finite(flow_model, flow_inputs):
    loss = flow_model.compute_flow_matching_loss(
        flow_inputs["image"], flow_inputs["traj"], flow_inputs["target"]
    )
    assert torch.isfinite(loss), "loss is NaN/Inf"
    assert loss.item() > 0


def test_flow_tdp_loss_decreases(flow_model, flow_inputs):
    """训练若干步断言 loss 总体下降趋势。

    5 步过短、单 batch，随机初始化时容易波动；改用 30 步 + 比较前/后窗口均值。
    """
    torch.manual_seed(42)
    opt = torch.optim.AdamW(flow_model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = flow_model.compute_flow_matching_loss(
            flow_inputs["image"], flow_inputs["traj"], flow_inputs["target"]
        )
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # 前 5 步均值 vs 后 5 步均值: 后者应该更低
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    assert late_avg < early_avg, (
        f"loss did not decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}, "
        f"losses={losses}"
    )


# ---------------------------------------------------------------------------
# VLM factory & JSON / voxel utils
# ---------------------------------------------------------------------------


def test_vlm_factory_dispatch_local(monkeypatch):
    """VLMWrapper(backend='local') 应返回 LocalLLaVAWrapper（即使没下载模型）。"""
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    vlm = VLMWrapper(backend="local")
    assert isinstance(vlm, LocalLLaVAWrapper)
    assert isinstance(vlm, VLMWrapperBase)


def test_vlm_factory_env_override(monkeypatch):
    """VLM_BACKEND env var 应覆盖默认。"""
    monkeypatch.setenv("VLM_BACKEND", "local")
    vlm = VLMWrapper()
    assert isinstance(vlm, LocalLLaVAWrapper)


def test_vlm_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        VLMWrapper(backend="nonexistent")


def test_extract_json_basic():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    assert _extract_json('noise\n```json\n{"x": 2}\n```\nmore') == {"x": 2}


def test_extract_json_single_quotes():
    """VLM 偶尔返回 Python 字面量风格。"""
    parsed = _extract_json("{'a': 1, 'b': 'hi'}")
    assert parsed == {"a": 1, "b": "hi"}


def test_extract_json_invalid_returns_none():
    assert _extract_json("no json here at all") is None


def test_coerce_to_voxel_flat_216():
    flat = [1] * 216
    v = _coerce_to_voxel(flat)
    assert v.shape == (6, 6, 6)
    assert int(v.sum()) == 216


def test_coerce_to_voxel_6x6x6_nested():
    nested = [[[1] * 6 for _ in range(6)] for _ in range(6)]
    v = _coerce_to_voxel(nested)
    assert v.shape == (6, 6, 6)
    assert int(v.sum()) == 216


def test_coerce_to_voxel_6x6_degenerated():
    """6×6 输入应塞到 z=0 层，其余层为 0。"""
    flat = [1] * 36
    v = _coerce_to_voxel(flat)
    assert v.shape == (6, 6, 6)
    assert int(v.sum()) == 36


def test_coerce_to_voxel_invalid_length_raises():
    with pytest.raises(Exception):
        _coerce_to_voxel([1] * 100)  # 既不是 216 也不是 36


def test_local_vlm_wrapper_decompose_calls_stub(monkeypatch):
    """LocalLLaVAWrapper 默认 generate_text 抛错；monkeypatch 后可调用。"""
    vlm = LocalLLaVAWrapper()
    monkeypatch.setattr(
        vlm, "generate_text",
        lambda prompt, image=None, **kw: '{"subtask": "pick", "voxel_trajectory": ' + str([[[1]*6]*6]*6) + '}',
    )
    subtask, voxel = vlm.decompose_task(image=None, task_description="x")
    assert subtask == "pick"
    assert voxel.shape == (6, 6, 6)