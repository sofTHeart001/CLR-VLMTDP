"""
End-to-end smoke test for the closed-loop controller with MockEnvironment
and a stub VLM. Verifies that A+B+C integrations don't crash and the
controller actually advances through subtasks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pytest

from models import FlowTDP, LightVoxelEncoder, VLMWrapperBase
from models import vlm_wrapper as vlm_module


# ---------------------------------------------------------------------------
# Stub VLM：返回固定 JSON，不联网
# ---------------------------------------------------------------------------


class StubVLMWrapper(VLMWrapperBase):
    def __init__(self):
        super().__init__()
        self._call_count = 0

    def generate_text(self, prompt, image=None, **kw):
        self._call_count += 1
        return '{"subtask": "pick", "voxel_trajectory": ' + str([[[1] * 6] * 6] * 6) + '}'

    def decompose_task(self, image, task_description, completed_subtasks=None):
        self._call_count += 1
        return "pick the block", torch.ones(6, 6, 6)

    def check_state(self, image, current_subtask, deviation_threshold=0.15):
        self._call_count += 1
        # 第一次调用返回"完成"，让闭环前进
        return {
            "subtask_done": self._call_count > 1,
            "need_replan": False,
            "reason": "stub says done",
            "confidence": 1.0,
        }


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_vlm(monkeypatch):
    """monkeypatch VLMWrapper 工厂返回 StubVLMWrapper。"""
    stub = StubVLMWrapper()
    # 同时 patch vlm_wrapper 模块内的 VLMWrapper 和 models 包 re-export 的那个
    monkeypatch.setattr(vlm_module, "VLMWrapper", lambda *a, **kw: stub)
    import models as models_pkg
    monkeypatch.setattr(models_pkg, "VLMWrapper", lambda *a, **kw: stub)
    return stub


def test_factory_returns_stub(stub_vlm):
    """monkeypatch 后 VLMWrapper() 应返回 StubVLMWrapper。"""
    from models import VLMWrapper
    vlm = VLMWrapper(backend="openai")  # backend 实际无效，因为已 monkeypatch
    assert isinstance(vlm, StubVLMWrapper)


def test_controller_runs_n_steps(stub_vlm):
    """
    不导入 ClosedLoopController（避免拉起 test_closed_loop 的 import 链）。
    我们自己写一个最小闭环验证 A+B+C 集成。
    """
    flow = FlowTDP(image_size=(64, 64))
    enc = LightVoxelEncoder()
    vlm = stub_vlm

    # Mock 环境：返回固定 image，每次 step 计数 +1
    class MiniEnv:
        def __init__(self, n_steps=10):
            self.n_steps = n_steps
            self.count = 0
        def reset(self):
            self.count = 0
            return {"rgb": torch.randn(3, 64, 64)}
        def step(self, action):
            self.count += 1
            done = self.count >= self.n_steps
            return {"rgb": torch.randn(3, 64, 64)}, 0.0, done, {}

    env = MiniEnv(n_steps=8)
    raw_image = env.reset()["rgb"]  # (3, 64, 64) — 不带 batch
    initial = raw_image.unsqueeze(0)  # (1, 3, 64, 64) — flow 期望 batched
    completed = []
    total_steps = 0

    for subtask_round in range(2):  # 最多 2 个子任务
        subtask, voxel = vlm.decompose_task(image=initial, task_description="stack")
        # encoder 期望 (B, 6, 6, 6)，stub 返回 (6, 6, 6) → 加 batch 维
        traj_feat = enc(voxel.unsqueeze(0))
        for _ in range(4):  # 每个子任务 4 步
            action = flow.sample_action(initial, traj_feat, num_steps=1)
            obs, _, done, _ = env.step(action)
            initial = obs["rgb"].unsqueeze(0)
            total_steps += 1
            if done:
                break
        check = vlm.check_state(image=initial, current_subtask=subtask)
        if check["subtask_done"]:
            completed.append(subtask)
        if done:
            break

    assert total_steps > 0, "controller did not advance any step"
    assert len(completed) >= 1, "no subtask marked done by stub"
    assert vlm._call_count >= 2, "VLM was not consulted"


def test_sample_action_1_and_4_steps_both_work(stub_vlm):
    flow = FlowTDP(image_size=(64, 64))
    enc = LightVoxelEncoder()
    image = torch.randn(1, 3, 64, 64)
    traj = enc(torch.ones(1, 6, 6, 6))
    for ns in (1, 4):
        a = flow.sample_action(image, traj, num_steps=ns)
        assert a.shape == (1, 8)
        assert torch.isfinite(a).all()


def test_voxel_extraction_and_projection_roundtrip():
    """voxel_extraction 输出格式兼容 LightVoxelEncoder 的输入。"""
    from utils import extract_voxel_trajectory
    line = np.zeros((15, 3))
    line[:, 0] = np.linspace(-0.2, 0.2, 15)
    grid = extract_voxel_trajectory(line, interpolate=True, interp_step_m=0.02)
    assert grid.shape == (6, 6, 6)
    enc = LightVoxelEncoder()
    feat = enc(grid.float().unsqueeze(0))  # encoder 期望 (B, 6, 6, 6)
    assert feat.shape == (1, 128)