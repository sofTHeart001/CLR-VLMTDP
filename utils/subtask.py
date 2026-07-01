"""
Sub-task Segmentation
按夹爪开/关状态变化，把一条长演示切分成多个 sub-task。

论文 §III.B 定义:
  "We define a sub-task as a discrete phase of the manipulation task that begins
   with the opening or closing the gripper and ends with closing or opening the
   gripper, typically indicating the robot's complete interaction with an object."

我们的实现:
  - 检测 gripper_open 布尔状态的变化点
  - 变化点 = sub-task 边界
  - 边界点之间 = 一个 sub-task
  - 退化处理: 如果整条 demo 没有任何夹爪变化（极端情况）, 把整条 demo 作为一个 sub-task
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SubTask:
    """一个子任务的索引区间 [start, end) (左闭右开)。"""
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def get_gripper_open(qpos: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    """
    把 (T, 9) qpos 转成 (T,) 的 gripper_open 布尔数组。

    Franka 在 ManiSkill 中 qpos = [arm(7), gripper(2)]，
    后两维是两个 gripper finger 的距离或角度。值 > threshold 视为张开。

    Args:
        qpos: (T, 9) or (9,) array
        threshold: 阈值, 默认 0.04 (Franka gripper 默认张开宽度约 0.04m)

    Returns:
        (T,) bool array, True = gripper open
    """
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    # Franka gripper: 两个 finger, qpos[-2] 或 qpos[-1] 都行
    # ManiSkill 用 panda 的 qpos, gripper 在最后两维, 通常取均值
    gripper_width = qpos[:, -2:].mean(axis=-1)
    return gripper_width > threshold


def segment_subtasks(
    qpos: np.ndarray,
    threshold: float = 0.04,
    min_length: int = 5,
) -> List[SubTask]:
    """
    按夹爪状态变化切分 sub-task。

    Args:
        qpos: (T, 9) array
        threshold: gripper 开/关阈值
        min_length: 每个 sub-task 最少步数 (过滤过短的瞬变噪声)

    Returns:
        List[SubTask], 每个 SubTask 有 start/end 索引 (左闭右开)

    示例 (PickCube-v0 一条典型 demo):
        qpos gripper_open 序列: [T, T, T, F, F, F, F, T, T, T, T]
                                                    ↑↑↑ (变化点)
        segment_subtasks 返回 3 个 SubTask:
            [0, 3)  - 起始 (open)
            [3, 7)  - 抓取 (close)
            [7, 11) - 释放后 (open)
    """
    if qpos.ndim != 2 or qpos.shape[1] != 9:
        raise ValueError(f"qpos must be (T, 9), got {qpos.shape}")

    gripper_open = get_gripper_open(qpos, threshold=threshold)
    T = len(gripper_open)
    if T == 0:
        return []

    # 找变化点
    transitions = []
    for i in range(1, T):
        if gripper_open[i] != gripper_open[i - 1]:
            transitions.append(i)

    # 切分: [0, t1), [t1, t2), ..., [tn, T)
    boundaries = [0] + transitions + [T]

    subtasks = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= min_length:
            subtasks.append(SubTask(start=s, end=e))

    # 退化情况: 没有任何 sub-task 满足 min_length
    if not subtasks and T > 0:
        # 把整条 demo 当作一个 sub-task
        subtasks.append(SubTask(start=0, end=T))

    return subtasks


def voxel_trajectory_for_timestep(
    subtasks: List[SubTask],
    timestep: int,
) -> int:
    """
    给定 timestep, 返回它属于哪个 sub-task 的索引。
    返回 -1 表示不在任何 sub-task 内 (例如在 sub-task 边界过渡区)。
    """
    for i, st in enumerate(subtasks):
        if st.start <= timestep < st.end:
            return i
    return -1


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_test():
    print("Subtask segmentation smoke checks:")

    # 模拟 PickCube-v0 风格的 demo
    # 0~9 open, 10~29 close (grasp), 30~49 open (move/place)
    qpos = np.zeros((50, 9))
    qpos[:, -2:] = 0.08  # gripper 全部张开
    qpos[10:30, -2:] = 0.0  # 20 步夹爪关闭
    # 其他维度无所谓

    subs = segment_subtasks(qpos)
    print(f"  Demo length=50; sub-tasks found: {len(subs)}")
    for i, st in enumerate(subs):
        print(f"    [{i}] {st.start}-{st.end} (length={st.length})")
    assert len(subs) == 3, f"Expected 3 sub-tasks, got {len(subs)}"
    assert subs[0].start == 0 and subs[0].end == 10
    assert subs[1].start == 10 and subs[1].end == 30
    assert subs[2].start == 30 and subs[2].end == 50

    # 退化情况: 全程开
    qpos2 = np.zeros((20, 9))
    qpos2[:, -2:] = 0.08
    subs2 = segment_subtasks(qpos2)
    print(f"  All-open demo: sub-tasks = {len(subs2)} (expected 1)")
    assert len(subs2) == 1

    # 退化情况: 频繁切换 (噪声)
    qpos3 = np.zeros((30, 9))
    qpos3[:, -2:] = 0.08
    # 让它频繁切换 (min_length 应该过滤)
    for i in range(0, 30, 2):
        qpos3[i, -1] = 0.0
    subs3 = segment_subtasks(qpos3, min_length=5)
    print(f"  Noisy demo (min_length=5): sub-tasks = {len(subs3)}")
    # min_length 过滤后应该只剩下少量 sub-task

    # voxel_trajectory_for_timestep 测试
    idx = voxel_trajectory_for_timestep(subs, 15)
    print(f"  timestep=15 -> sub-task index {idx}")
    assert idx == 1

    print("  All smoke checks passed.")


if __name__ == "__main__":
    self_test()