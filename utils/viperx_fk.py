"""
ViperX 6-DoF Forward Kinematics (for Aloha robot)

ViperX 是 Interbotix 的 6 自由度机械臂. Aloha 用两个 ViperX + 一个 gripper.

DH 参数 (基于 Interbotix ViperX 300 / 官方文档):
  https://www.trossenrobotics.com/vx300
  https://github.com/Interbotix/interbotix_ros_manipulators

Joint order (right arm, 7 motors):
  0: waist (基座旋转)
  1: shoulder (肩部俯仰)
  2: elbow (肘部)
  3: forearm_roll (前臂滚转)
  4: wrist_angle (腕部俯仰)
  5: wrist_rotate (腕部滚转)
  6: gripper (夹爪, 不用)

DH 参数 (modified convention, 米):
  Joint | a (m)   | d (m)    | alpha (rad)
  ------|---------|----------|------------
  waist | 0       | 0.1076   | -π/2
  shoulder| 0     | 0        | π/2
  elbow  | 0.1301 | 0        | -π/2
  forearm_roll | 0.1241 | 0 | π/2
  wrist_angle | 0 | 0.1338 | -π/2
  wrist_rotate | 0 | 0.0597 | 0
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ViperX 6-DoF DH parameters (modified convention)
# 参考: Interbotix ViperX 300 官方规格
VIPERX_DH = np.array([
    # a (m)     d (m)      alpha (rad)
    [0.0,       0.1076,    -np.pi / 2],   # 1: waist
    [0.0,       0.0,        np.pi / 2],   # 2: shoulder
    [0.1301,    0.0,       -np.pi / 2],   # 3: elbow
    [0.1241,    0.0,        np.pi / 2],   # 4: forearm_roll
    [0.0,       0.1338,    -np.pi / 2],   # 5: wrist_angle
    [0.0,       0.0597,     0.0],         # 6: wrist_rotate
], dtype=np.float64)


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """单个 modified DH 关节的 4×4 齐次变换."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,      d],
        [0.0,    0.0,     0.0,    1.0],
    ], dtype=np.float64)


def fk_viperx(joint_angles_6: np.ndarray) -> np.ndarray:
    """
    ViperX 6-DoF 正向运动学: 给 6 个关节角, 算末端 (panda_hand) 在 base 坐标系下的位置.

    Args:
        joint_angles_6: (6,) 6 个关节角 (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
                          单位: 弧度

    Returns:
        ee_pos: (3,) EE 在 base 坐标系下的 xyz
    """
    if joint_angles_6.shape != (6,):
        raise ValueError(f"joint_angles_6 must be (6,), got {joint_angles_6.shape}")

    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        a, d, alpha = VIPERX_DH[i]
        T = T @ _dh_transform(a, d, alpha, joint_angles_6[i])
    return T[:3, 3]


def fk_viperx_batch(joint_angles_batch: np.ndarray) -> np.ndarray:
    """
    批量 FK.

    Args:
        joint_angles_batch: (T, 6) 每行 6 个关节角

    Returns:
        positions: (T, 3) EE 位置
    """
    if joint_angles_batch.ndim != 2 or joint_angles_batch.shape[1] != 6:
        raise ValueError(f"Expected (T, 6), got {joint_angles_batch.shape}")

    positions = np.zeros((joint_angles_batch.shape[0], 3), dtype=np.float64)
    for t in range(joint_angles_batch.shape[0]):
        positions[t] = fk_viperx(joint_angles_batch[t])
    return positions


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_test():
    print("ViperX FK self-test:")

    # Zero pose: 应该伸直向上
    pos0 = fk_viperx(np.zeros(6))
    expected_z = sum([row[1] for row in VIPERX_DH])  # 累加 d 值
    print(f"  Zero pose: {pos0.round(4)}")
    print(f"  Expected (zero d sum): ({0:.3f}, {0:.3f}, {expected_z:.3f})")

    # Aloha "ready" pose 估计 (从 demo 数据):
    # waist: 0, shoulder: -0.96, elbow: 1.16, forearm_roll: 0, wrist_angle: -0.3, wrist_rotate: 0
    ready = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    pos1 = fk_viperx(ready)
    print(f"  Ready pose: {pos1.round(4)}")

    # Batch
    batch = np.array([np.zeros(6), ready, ready + 0.1])
    pos_batch = fk_viperx_batch(batch)
    print(f"  Batch: shape={pos_batch.shape}")
    print(f"  {pos_batch.round(4)}")

    print("  Done.")


if __name__ == "__main__":
    self_test()