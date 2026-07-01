"""
Visual Prompt Utilities
为 VLM 生成 mask-based 视觉提示：
  - draw_voxel_grid_overlay: 在俯视图上画 6×6 编号网格
  - draw_voxel_grid_on_image: 把占据的体素 cell 按相机投影画到原图

设计灵感来自 VLM-TDP 论文：把 6×6×6 工作空间压缩为 6×6 俯视图，
用 mask 作为视觉提示让 VLM 选择哪些 cell 是轨迹的一部分。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from utils.voxel_trajectory import (
        get_default_intrinsics,
        get_default_extrinsics,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.voxel_trajectory import (  # type: ignore
        get_default_intrinsics,
        get_default_extrinsics,
    )


# ---------------------------------------------------------------------------
# 通用：图像 → PIL
# ---------------------------------------------------------------------------


def _to_pil(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.permute(1, 2, 0)
        if arr.dtype == torch.float32 or arr.dtype == torch.float16:
            arr = (arr.clamp(0, 1) * 255).to(torch.uint8)
        arr = arr.numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# 工具：在 6×6 网格上画一行 cell index
# ---------------------------------------------------------------------------


def _grid_label(i: int, j: int) -> str:
    """给 (row, col) 网格生成 label，例如 (0,0)→'A1', (5,5)→'F6'。"""
    return f"{chr(ord('A') + i)}{j + 1}"


# ---------------------------------------------------------------------------
# 主要函数 1：在俯视图上画 6×6 编号网格
# ---------------------------------------------------------------------------


def draw_voxel_grid_overlay(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    grid_size: int = 6,
    line_width: int = 2,
    show_numbers: bool = True,
    cell_color: Tuple[int, int, int] = (255, 215, 0),  # 金色
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    在图像上叠一个 grid_size × grid_size 的均匀网格，每格中心标一个字母+数字。
    常用于把 RGB 图像的"桌面区域"切成 6×6 个候选 cell 让 VLM 选择。

    Args:
        image: 原图（H×W×3 或 PIL）
        grid_size: 每边格数
        line_width: 网格线宽
        show_numbers: 是否画 cell 编号
        cell_color: 网格线颜色 (R,G,B)
        text_color: 编号文字颜色 (R,G,B)

    Returns:
        PIL.Image，画好网格的副本
    """
    pil = _to_pil(image).copy()
    W, H = pil.size
    cell_w = W / grid_size
    cell_h = H / grid_size

    draw = ImageDraw.Draw(pil)

    # 横线
    for i in range(grid_size + 1):
        y = int(i * cell_h)
        draw.line([(0, y), (W, y)], fill=cell_color, width=line_width)
    # 竖线
    for j in range(grid_size + 1):
        x = int(j * cell_w)
        draw.line([(x, 0), (x, H)], fill=cell_color, width=line_width)

    if show_numbers:
        try:
            font = ImageFont.load_default(size=max(12, int(min(cell_w, cell_h) / 4)))
        except (TypeError, AttributeError):
            font = ImageFont.load_default()
        for i in range(grid_size):
            for j in range(grid_size):
                cx = int((j + 0.5) * cell_w)
                cy = int((i + 0.5) * cell_h)
                label = _grid_label(i, j)
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((cx - tw // 2, cy - th // 2), label, fill=text_color, font=font)

    return pil


# ---------------------------------------------------------------------------
# 主要函数 2：把占据的 voxel cell 投影到原图上画彩色块
# ---------------------------------------------------------------------------


def _world_cell_corners(
    i: int, j: int, k: int,
    grid_size: int,
    bounds: Tuple[float, ...],
) -> np.ndarray:
    """
    返回 cell (i,j,k) 的 8 个世界坐标角点 (8, 3)。
    i→x, j→y, k→z。
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    xs = np.linspace(xmin, xmax, grid_size + 1)
    ys = np.linspace(ymin, ymax, grid_size + 1)
    zs = np.linspace(zmin, zmax, grid_size + 1)
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[j], ys[j + 1]
    z0, z1 = zs[k], zs[k + 1]
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    return corners


def _project_points(pts_world: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """世界点 (N, 3) → 像素点 (N, 2)。"""
    pts_h = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])  # (N, 4)
    pts_cam = (np.linalg.inv(T) @ pts_h.T).T[:, :3]  # (N, 3) in camera frame
    uv = np.zeros((pts_world.shape[0], 2), dtype=np.float64)
    valid = pts_cam[:, 2] > 0
    if valid.sum() == 0:
        return uv
    proj = (K @ pts_cam[valid].T).T  # (M, 3)
    z = proj[:, 2:3]
    z = np.where(z > 0, z, 1.0)  # 避免除零
    uv[valid] = proj[:, :2] / z
    return uv


def draw_voxel_grid_on_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    occupied_cells: List[Tuple[int, int, int]],
    grid_size: int = 6,
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_extrinsics: Optional[np.ndarray] = None,
    workspace_bounds: Optional[Tuple[float, ...]] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 3,
    show_labels: bool = True,
) -> Image.Image:
    """
    把列表中的体素 cell 投影到原图上画彩色 3D 边框。

    Args:
        image: 原图
        occupied_cells: [(i, j, k), ...] 要标记的 cell
        grid_size: 体素每边格数
        camera_intrinsics / camera_extrinsics / workspace_bounds: 与 voxel_trajectory 一致
        color: 边框颜色
        line_width: 边框线宽
        show_labels: 是否在第一个 cell 中心画 cell label

    Returns:
        PIL.Image，画好标记的副本
    """
    pil = _to_pil(image).copy()
    W, H = pil.size

    if not occupied_cells:
        return pil

    K = camera_intrinsics if camera_intrinsics is not None else get_default_intrinsics((W, H))
    T = camera_extrinsics if camera_extrinsics is not None else get_default_extrinsics()
    if workspace_bounds is None:
        from utils.voxel_trajectory import DEFAULT_WORKSPACE_BOUNDS
        workspace_bounds = DEFAULT_WORKSPACE_BOUNDS

    draw = ImageDraw.Draw(pil)

    for cell_idx, (i, j, k) in enumerate(occupied_cells):
        if not (0 <= i < grid_size and 0 <= j < grid_size and 0 <= k < grid_size):
            continue
        corners_world = _world_cell_corners(i, j, k, grid_size, workspace_bounds)
        uv = _project_points(corners_world, K, T)
        if np.any(~np.isfinite(uv)):
            continue

        # 12 条边的索引
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 立柱
        ]
        for a, b in edges:
            x1, y1 = uv[a]
            x2, y2 = uv[b]
            if 0 <= x1 < W and 0 <= x2 < W and 0 <= y1 < H and 0 <= y2 < H:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

        if show_labels and cell_idx == 0:
            cx, cy = uv.mean(axis=0)
            if 0 <= cx < W and 0 <= cy < H:
                label = _grid_label(i, j)
                try:
                    font = ImageFont.load_default(size=14)
                except (TypeError, AttributeError):
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.rectangle(
                    [cx - tw // 2 - 2, cy - th // 2 - 2, cx + tw // 2 + 2, cy + th // 2 + 2],
                    fill=(255, 255, 255),
                )
                draw.text((cx - tw // 2, cy - th // 2), label, fill=(0, 0, 0), font=font)

    return pil


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Visual prompt module. Smoke checks:")

    # 1) 合成灰图 + 网格叠加
    img = np.full((600, 800, 3), 200, dtype=np.uint8)  # 浅灰背景
    overlay = draw_voxel_grid_overlay(img, grid_size=6)
    print(f"  draw_voxel_grid_overlay: size={overlay.size}, mode={overlay.mode}")
    assert overlay.size == (800, 600)

    # 2) 把占据 cell 投影画到图上
    cells = [(2, 2, 0), (3, 3, 1), (4, 4, 2)]
    marked = draw_voxel_grid_on_image(img, cells, grid_size=6)
    print(f"  draw_voxel_grid_on_image: size={marked.size}, n_cells={len(cells)}")

    # 3) 接受 torch 输入
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    overlay_t = draw_voxel_grid_overlay(img_t, grid_size=6)
    print(f"  torch input -> PIL: size={overlay_t.size}")

    # 4) 空 list 不崩溃
    no_cells = draw_voxel_grid_on_image(img, [])
    print(f"  empty occupied_cells: OK, size={no_cells.size}")

    print("  All smoke checks passed.")