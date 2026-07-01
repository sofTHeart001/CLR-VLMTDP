"""
Franka Forward Kinematics

主路径：调用 ManiSkill 内置 agent（基于 SAPIEN 物理引擎）的 FK，
跟 ManiSkill 仿真内部保持完全一致。

回退路径：如果 ManiSkill / SAPIEN 没装，用一个最小 DH 表手算（仅用于单元测试）。

典型用法:
    from utils.franka_fk import fk_panda_batch

    qpos = np.array([...])  # (T, 9)
    positions = fk_panda_batch(qpos)  # (T, 3) EE in robot base frame
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 主路径: ManiSkill / SAPIEN 内置 FK
# ---------------------------------------------------------------------------


_MANISKILL_AGENT = None  # 全局缓存 agent (FK 计算很贵，避免反复构造)


def _get_maniskill_agent():
    """惰性加载 ManiSkill Panda agent。

    ManiSkill3 的 agent 构造需要 scene + control_freq; 没有 env 时通常需要手动构造。
    如果构造失败, 抛出 RuntimeError 让调用方走 fallback。
    """
    global _MANISKILL_AGENT
    if _MANISKILL_AGENT is not None:
        return _MANISKILL_AGENT
    try:
        import sapien  # type: ignore
        from mani_skill.agents.robots.panda import Panda  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ManiSkill / SAPIEN is required for ManiSkill-FK. Install with "
            "`pip install mani-skill sapien`."
        ) from e

    # 构造一个 minimal scene 让 agent 有 context
    try:
        engine = sapien.Engine()
        scene = engine.create_scene()
        scene.set_timestep(1 / 100)
        agent = Panda(scene=scene, control_freq=100)
        _MANISKILL_AGENT = agent
        return agent
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ManiSkill Panda agent: {e}") from e


def fk_panda_batch_maniskill(qpos_batch: np.ndarray) -> np.ndarray:
    """
    用 ManiSkill 内置 Panda agent 做 FK。

    Args:
        qpos_batch: (T, 9) — 前 7 维 arm joint angles (rad)，后 2 维 gripper (忽略)

    Returns:
        positions: (T, 3) — panda_hand link origin 在 robot base 坐标系下的位置
    """
    if qpos_batch.ndim != 2 or qpos_batch.shape[1] != 9:
        raise ValueError(f"qpos_batch must be (T, 9), got {qpos_batch.shape}")

    agent = _get_maniskill_agent()
    robot = agent.robot

    positions = np.zeros((qpos_batch.shape[0], 3), dtype=np.float64)
    for i in range(qpos_batch.shape[0]):
        # ManiSkill 的 set_qpos 期望完整 qpos (含 gripper)
        robot.set_qpos(qpos_batch[i])
        # 获取末端 link 的 pose; ManiSkill 用 "panda_hand" 作为 EE link name
        ee_link = robot.get_link("panda_hand")
        pose = ee_link.get_pose()  # sapien.Pose
        positions[i] = [float(pose.p.x), float(pose.p.y), float(pose.p.z)]
    return positions


# ---------------------------------------------------------------------------
# 回退路径: 极简手写 FK (仅供单元测试和 ManiSkill 缺失时)
# ---------------------------------------------------------------------------


# Franka Panda modified DH (来源: franka_ros URDF, 8 行: 7 arm + 1 fixed EE)
# 注意: 这套参数不一定跟 ManiSkill 内部完全一致 (SAPIEN 可能用更精确的 mesh FK)，
# 所以生产代码必须走 ManiSkill 路径
PANDA_DH_FALLBACK = np.array([
    [0.0,    0.333,   0.0],
    [0.0,    0.0,    -np.pi/2],
    [0.0,    0.316,    np.pi/2],
    [0.0825, 0.0,      np.pi/2],
    [-0.0825, 0.0,    -np.pi/2],
    [0.0,    0.384,    np.pi/2],
    [0.088,  0.0,      np.pi/2],
    [0.0,    0.107,    0.0],
], dtype=np.float64)


def _dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,      d],
        [0.0,    0.0,     0.0,    1.0],
    ], dtype=np.float64)


def fk_panda_batch_fallback(qpos_batch: np.ndarray) -> np.ndarray:
    """回退手写 FK, 仅供测试."""
    positions = np.zeros((qpos_batch.shape[0], 3), dtype=np.float64)
    for i in range(qpos_batch.shape[0]):
        T = np.eye(4, dtype=np.float64)
        for j in range(PANDA_DH_FALLBACK.shape[0]):
            a, d, alpha = PANDA_DH_FALLBACK[j]
            theta = qpos_batch[i, j] if j < 7 else 0.0
            T = T @ _dh_transform(a, d, alpha, theta)
        positions[i] = T[:3, 3]
    return positions


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


def fk_panda_batch(qpos_batch: np.ndarray, prefer: str = "maniskill") -> np.ndarray:
    """
    统一 FK 接口。

    Args:
        qpos_batch: (T, 9) array
        prefer: "maniskill" (默认, 优先用 ManiSkill 内置) 或 "fallback" (手算)

    Returns:
        positions: (T, 3) array
    """
    if prefer == "maniskill":
        try:
            return fk_panda_batch_maniskill(qpos_batch)
        except (ImportError, RuntimeError) as e:
            import warnings
            warnings.warn(
                f"ManiSkill FK unavailable ({type(e).__name__}: {e}); "
                f"falling back to handwritten FK. "
                f"Results may differ slightly from ManiSkill's internal computation."
            )
            return fk_panda_batch_fallback(qpos_batch)
    elif prefer == "fallback":
        return fk_panda_batch_fallback(qpos_batch)
    else:
        raise ValueError(f"prefer must be 'maniskill' or 'fallback', got {prefer}")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_test():
    """Smoke check.

    1. 如果 ManiSkill 可用: 验证 zero pose FK 有限 + 形状正确
    2. Fallback: 验证 zero pose FK 形状正确
    """
    qpos_zero = np.zeros((1, 9))
    try:
        p = fk_panda_batch_maniskill(qpos_zero)
        print(f"  ManiSkill zero pose FK: {p[0].round(3)} (shape={p.shape})")
        assert p.shape == (1, 3)
        assert np.isfinite(p).all()
    except ImportError:
        print("  ManiSkill not available; running fallback only")

    # Fallback
    p = fk_panda_batch_fallback(qpos_zero)
    print(f"  Fallback zero pose FK: {p[0].round(3)} (shape={p.shape})")
    assert p.shape == (1, 3)
    assert np.isfinite(p).all()


if __name__ == "__main__":
    print("Franka FK module. Smoke checks:")
    self_test()

    q_batch = np.random.uniform(-1, 1, (10, 9))
    p = fk_panda_batch(q_batch)
    print(f"  Unified batch: {q_batch.shape} -> {p.shape}")
    assert p.shape == (10, 3)
    print("  All smoke checks passed.")