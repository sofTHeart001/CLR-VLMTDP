"""
Voxel Trajectory Utilities
把前视图 + 深度图投影到 6×6×6 体素网格；这是 VLM-TDP 论文的核心数据预处理步骤。

核心函数：
    - get_default_intrinsics(image_size) -> 3x3 K
    - get_default_extrinsics()           -> 4x4 T (cam2world)
    - project_to_voxel_grid(image, depth, K, T, bounds, grid_size=6) -> (6,6,6) tensor
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# 默认相机参数（Franka / RLBench 桌面设置）
# ---------------------------------------------------------------------------

# 前视图相机：800x600 图像，fx=fy=400, cx=400, cy=300
DEFAULT_FRONT_INTRINSICS: np.ndarray = np.array([
    [400.0,   0.0, 400.0],
    [  0.0, 400.0, 300.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

# 相机外参：相机在 (0, 0, 1) 朝下看原点（cam2world）。
# 旋转 180° 绕 X 轴：相机 z+ 指向 -world z，使正 depth 落在桌面上方。
DEFAULT_FRONT_EXTRINSICS: np.ndarray = np.array([
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  1.0],
    [ 0.0,  0.0,  0.0,  1.0],
], dtype=np.float64)

# 工作空间边界（Franka 桌面）：x,y ∈ [-0.3, 0.3], z ∈ [0, 0.5]
DEFAULT_WORKSPACE_BOUNDS: Tuple[float, ...] = (-0.3, -0.3, 0.0, 0.3, 0.3, 0.5)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def get_default_intrinsics(image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
    """
    根据图像尺寸生成默认相机内参。
    对默认 800×600 图像直接返回 DEFAULT_FRONT_INTRINSICS；
    其他尺寸按比例缩放 fx/fy/cx/cy。
    """
    W, H = image_size
    base_W, base_H = 800, 600
    sx = W / base_W
    sy = H / base_H
    K = DEFAULT_FRONT_INTRINSICS.copy()
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy
    return K


def get_default_extrinsics() -> np.ndarray:
    """返回默认 cam2world 4×4 变换。"""
    return DEFAULT_FRONT_EXTRINSICS.copy()


# ---------------------------------------------------------------------------
# 投影到体素网格
# ---------------------------------------------------------------------------


def _to_depth_np(depth: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """统一 depth 到 (H, W) float32 numpy。"""
    if isinstance(depth, torch.Tensor):
        d = depth.detach().cpu().numpy()
    else:
        d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"depth must be (H, W), got shape {d.shape}")
    return d.astype(np.float32)


def project_to_voxel_grid(
    image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    depth: Optional[Union[np.ndarray, torch.Tensor]] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_extrinsics: Optional[np.ndarray] = None,
    workspace_bounds: Optional[Tuple[float, ...]] = None,
    grid_size: int = 6,
    occupancy_threshold: int = 3,
    depth_scale: float = 1.0,
) -> torch.Tensor:
    """
    把 RGB + 深度图投影到 (grid_size, grid_size, grid_size) 的 0/1 体素网格。

    算法：
      1. 背投影每个像素到世界坐标系
      2. 裁剪到 workspace_bounds
      3. 栅格化为 grid_size**3 体素
      4. 每个 cell 内点数 >= occupancy_threshold 视为占用

    Args:
        image: (H, W, 3) 图像，仅用于推断图像尺寸；可传 None（按 depth 推）
        depth: (H, W) 深度图（米）
        camera_intrinsics: 3×3 K；若为 None 用默认
        camera_extrinsics: 4×4 cam2world T；若为 None 用默认
        workspace_bounds: (xmin, ymin, zmin, xmax, ymax, zmax)
        grid_size: 体素每边格数（论文固定 6）
        occupancy_threshold: 体素 cell 占用阈值
        depth_scale: 把深度像素值乘以这个系数得到米（默认 1.0）

    Returns:
        torch.Tensor (grid_size, grid_size, grid_size)，float32，0/1
    """
    if depth is None:
        raise ValueError("depth is required")

    d = _to_depth_np(depth) * depth_scale
    H, W = d.shape

    # 推断图像尺寸（image 可选）
    if image is None:
        iw, ih = W, H
    elif isinstance(image, Image.Image):
        iw, ih = image.size  # (W, H)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            ih, iw = image.shape[1], image.shape[2]
        else:
            ih, iw = image.shape[0], image.shape[1]
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            ih, iw = arr.shape[0], arr.shape[1]
        else:
            ih, iw = arr.shape

    if (iw, ih) != (W, H):
        raise ValueError(
            f"image size {(iw, ih)} != depth size {(W, H)}; "
            "resize depth or pass matching image."
        )

    K = camera_intrinsics if camera_intrinsics is not None else get_default_intrinsics((W, H))
    T = camera_extrinsics if camera_extrinsics is not None else get_default_extrinsics()
    bounds = workspace_bounds if workspace_bounds is not None else DEFAULT_WORKSPACE_BOUNDS

    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # 像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
    uv1 = np.stack([u.ravel(), v.ravel(), np.ones_like(u).ravel()], axis=0)  # (3, H*W)
    z = d.ravel().astype(np.float64)

    # 过滤有效深度
    valid = (z > 0) & np.isfinite(z)
    uv1 = uv1[:, valid]
    z = z[valid]

    # 相机系点
    pts_cam = (K_inv @ uv1) * z  # (3, N)
    pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])  # (4, N)

    # 世界系点
    pts_world = (T @ pts_cam_h)[:3, :]  # (3, N)

    # 裁剪到工作空间
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    in_box = (
        (pts_world[0] >= xmin) & (pts_world[0] < xmax) &
        (pts_world[1] >= ymin) & (pts_world[1] < ymax) &
        (pts_world[2] >= zmin) & (pts_world[2] < zmax)
    )
    pts_world = pts_world[:, in_box]

    if pts_world.shape[1] == 0:
        return torch.zeros(grid_size, grid_size, grid_size, dtype=torch.float32)

    # 栅格化
    edges = [
        np.linspace(xmin, xmax, grid_size + 1),
        np.linspace(ymin, ymax, grid_size + 1),
        np.linspace(zmin, zmax, grid_size + 1),
    ]
    hist, _ = np.histogramdd(pts_world.T, bins=edges)
    occupied = (hist >= occupancy_threshold).astype(np.float32)

    # axis 顺序：hist 返回 (x_bin, y_bin, z_bin)，即 (i, j, k)
    # 但语义上 voxel[i, j, k] 应该对应 (x, y, z)，所以保持这个顺序
    return torch.from_numpy(occupied).reshape(grid_size, grid_size, grid_size)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Voxel trajectory module. Smoke checks:")

    # 1) 默认相机参数
    K = get_default_intrinsics((800, 600))
    T = get_default_extrinsics()
    print(f"  K shape={K.shape}, fx={K[0,0]:.0f}, cx={K[0,2]:.0f}")
    print(f"  T shape={T.shape}, camera_z={T[2,3]:.1f}")

    # 2) 合成深度图：背景深度 1.0（→ world z=0），中心高台深度 0.7（→ world z=0.3）
    #    这意味着 background 填满 z_bin=0 整层（36 cells），raised 加到 z_bin=3
    H, W = 600, 800
    depth = np.full((H, W), 1.0, dtype=np.float32)
    depth[280:320, 380:420] = 0.7  # 中心 40×40 凸起

    grid = project_to_voxel_grid(depth=depth, grid_size=6)
    print(f"  Synthetic grid: shape={tuple(grid.shape)}, occupied={int(grid.sum())}")
    occupied_cells = [(int(i), int(j), int(k))
                      for i in range(6) for j in range(6) for k in range(6)
                      if grid[i, j, k] > 0]
    print(f"  Occupied cells: {len(occupied_cells)} total; sample: {occupied_cells[:5]}")

    # 3) 接受 torch 输入
    depth_t = torch.from_numpy(depth)
    grid2 = project_to_voxel_grid(depth=depth_t, grid_size=6)
    assert torch.equal(grid, grid2), "torch/np input mismatch"
    print(f"  torch input parity: OK")

    # 4) 纯平深度 (z=1.0 → world z=0)：应填满 z_bin=0 整层（36 cells）
    grid3 = project_to_voxel_grid(depth=np.full((600, 800), 1.0, dtype=np.float32), grid_size=6)
    print(f"  Flat depth (z=1.0 → world z=0): occupied={int(grid3.sum())} (expect 36)")
    assert grid3.sum() == 36, f"Expected 36 cells at z=0, got {int(grid3.sum())}"

    # 5) 高 z 深度（depth=0.55 → world z=0.45）→ 全部落在 z_bin=5 顶面
    grid4 = project_to_voxel_grid(depth=np.full((600, 800), 0.55, dtype=np.float32), grid_size=6)
    top_layer_occ = int(grid4[:, :, 5].sum())
    print(f"  Flat depth z=0.55 (→ world z=0.45): top z-bin filled {top_layer_occ}/36 cells")
    assert top_layer_occ == 36, f"Expected all 36 top cells, got {top_layer_occ}"