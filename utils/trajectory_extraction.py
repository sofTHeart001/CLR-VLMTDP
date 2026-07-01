"""
Trajectory Extraction Utilities
从演示数据的末端执行器 (EE) 位姿序列中提取 6×6×6 体素轨迹。

论文里体素轨迹的语义：轨迹经过的 voxel 按出现顺序标记 1, 2, 3, ...；
其他 voxel 为 0。本模块实现了这个语义。

典型用法：
    ee_poses = np.load('demo_ee_poses.npy')   # (T, 7) xyz+quat 或 (T, 3) xyz
    voxel_traj = extract_voxel_trajectory(ee_poses)
    # voxel_traj: (6, 6, 6) tensor，值是 0/1/2/... 整数
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    from utils.voxel_trajectory import DEFAULT_WORKSPACE_BOUNDS
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.voxel_trajectory import DEFAULT_WORKSPACE_BOUNDS  # type: ignore


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _quat_to_xyz(ee_poses: np.ndarray) -> np.ndarray:
    """
    接受 (T, 7) (xyz+quat) 或 (T, 3) (xyz) 输入，返回 (T, 3) xyz。
    其他 shape 抛错。
    """
    if ee_poses.ndim != 2:
        raise ValueError(f"ee_poses must be 2D (T, D); got shape {ee_poses.shape}")
    T, D = ee_poses.shape
    if D == 3:
        return ee_poses[:, :3]
    if D == 7:
        return ee_poses[:, :3]
    if D == 6:
        # xyz + rpy（或 xyz + 6D rotation）
        return ee_poses[:, :3]
    raise ValueError(f"ee_poses last dim must be 3/6/7; got {D}")


def _interpolate_path(waypoints: np.ndarray, step_m: float = 0.01) -> np.ndarray:
    """
    沿折线线性插值，使相邻点距离 ≤ step_m。
    返回新的 (M, 3) 点云。
    """
    if len(waypoints) < 2:
        return waypoints
    segments = np.diff(waypoints, axis=0)
    seg_lens = np.linalg.norm(segments, axis=1)
    total = seg_lens.sum()
    if total < 1e-9:
        return waypoints

    n_steps = max(int(np.ceil(total / step_m)), 1)
    out = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        n = max(int(np.ceil(seg_lens[i] / step_m)), 1)
        for k in range(1, n + 1):
            t = k / n
            out.append(waypoints[i] * (1 - t) + waypoints[i + 1] * t)
    return np.array(out)


def _world_to_voxel_idx(
    xyz: np.ndarray,
    bounds: Tuple[float, ...],
    grid_size: int,
) -> Tuple[int, int, int]:
    """单个 (3,) 点 → (i, j, k)；超出范围裁剪到最近 cell。"""
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    x = float(np.clip(xyz[0], xmin, xmax - 1e-9))
    y = float(np.clip(xyz[1], ymin, ymax - 1e-9))
    z = float(np.clip(xyz[2], zmin, zmax - 1e-9))
    fx = (x - xmin) / (xmax - xmin)
    fy = (y - ymin) / (ymax - ymin)
    fz = (z - zmin) / (zmax - zmin)
    i = min(int(fx * grid_size), grid_size - 1)
    j = min(int(fy * grid_size), grid_size - 1)
    k = min(int(fz * grid_size), grid_size - 1)
    return i, j, k


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def extract_voxel_trajectory(
    ee_poses: Union[np.ndarray, torch.Tensor],
    workspace_bounds: Optional[Tuple[float, ...]] = None,
    grid_size: int = 6,
    interpolate: bool = True,
    interp_step_m: float = 0.01,
    return_sequence: bool = True,
) -> torch.Tensor:
    """
    从末端执行器轨迹提取 6×6×6 体素轨迹。

    算法：
      1. 提取 xyz（忽略姿态）
      2. 可选线性插值到 ~1cm 步长
      3. 每个 waypoint → voxel cell (i,j,k)
      4. 按出现顺序写入 1, 2, 3, ...

    Args:
        ee_poses: (T, 7) xyz+quat 或 (T, 3) xyz
        workspace_bounds: (xmin, ymin, zmin, xmax, ymax, zmax)
        grid_size: 体素每边格数
        interpolate: 是否线性插值
        interp_step_m: 插值步长（米）
        return_sequence: True → 返回按时序整数标记的 (6,6,6) tensor
                        False → 返回 0/1 占用 grid

    Returns:
        torch.Tensor (grid_size, grid_size, grid_size)，整数或 0/1
    """
    if isinstance(ee_poses, torch.Tensor):
        ee = ee_poses.detach().cpu().numpy()
    else:
        ee = np.asarray(ee_poses)

    if ee.ndim == 1:
        # 单点 (D,)
        ee = ee[None, :]

    xyz = _quat_to_xyz(ee)  # (T, 3)
    bounds = workspace_bounds if workspace_bounds is not None else DEFAULT_WORKSPACE_BOUNDS

    if interpolate and len(xyz) >= 2:
        xyz = _interpolate_path(xyz, step_m=interp_step_m)

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)

    for step, pt in enumerate(xyz):
        try:
            i, j, k = _world_to_voxel_idx(pt, bounds, grid_size)
        except (ValueError, IndexError):
            continue
        # 同一 cell 已写入过：跳过（保留首次出现的次序）
        if grid[i, j, k] == 0:
            grid[i, j, k] = step + 1

    if not return_sequence:
        grid = (grid > 0).astype(np.int64)

    return torch.from_numpy(grid).long() if return_sequence else torch.from_numpy(grid).float()


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Trajectory extraction module. Smoke checks:")

    # 1) 水平直线：x 从 -0.2 到 +0.2，y=z=0
    line = np.zeros((20, 3))
    line[:, 0] = np.linspace(-0.2, 0.2, 20)
    grid = extract_voxel_trajectory(line, interpolate=True, interp_step_m=0.01)
    print(f"  Horizontal line: shape={tuple(grid.shape)}, nonzero cells={int((grid > 0).sum())}")
    # 期望：沿 x 一行 cells 被填（即某个 j,k 固定的一行 cells）
    row_jk_nonzero = {tuple(np.array(np.where(grid > 0))[1:3].T[i]) for i in range((grid > 0).sum())}
    print(f"  Distinct (j,k) layers: {row_jk_nonzero}")

    # 2) (T, 7) 输入
    line7 = np.zeros((20, 7))
    line7[:, 0] = np.linspace(-0.2, 0.2, 20)
    grid7 = extract_voxel_trajectory(line7)
    assert torch.equal(grid, grid7), "(T, 7) input should match (T, 3)"
    print(f"  (T, 7) input parity: OK")

    # 3) 超出范围点 → 裁剪
    out_box = np.array([
        [-1.0, 0.0, 0.0],   # x < xmin
        [+1.0, 0.0, 0.0],   # x > xmax
        [0.0, 0.0, 0.0],    # 中心
    ])
    grid_clip = extract_voxel_trajectory(out_box, interpolate=False)
    print(f"  Out-of-bounds clipping: shape={tuple(grid_clip.shape)}, nonzero={int((grid_clip > 0).sum())}")

    # 4) return_sequence=False → 0/1 占用 grid
    grid_bin = extract_voxel_trajectory(line, interpolate=True, interp_step_m=0.01, return_sequence=False)
    print(f"  Binary mode: dtype={grid_bin.dtype}, unique values={sorted(set(grid_bin.flatten().tolist()))}")

    # 5) 序列编号：从 1 开始单调
    grid_seq = extract_voxel_trajectory(line, interpolate=False, return_sequence=True)
    occupied_values = sorted([int(v) for v in grid_seq.flatten().tolist() if v > 0])
    print(f"  Sequence labels (sorted): {occupied_values}")

    print("  All smoke checks passed.")