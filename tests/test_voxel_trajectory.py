"""
Unit tests for utils/voxel_trajectory.py and utils/trajectory_extraction.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
from PIL import Image

from utils import (
    project_to_voxel_grid,
    get_default_intrinsics,
    get_default_extrinsics,
    DEFAULT_WORKSPACE_BOUNDS,
    extract_voxel_trajectory,
    draw_voxel_grid_overlay,
    draw_voxel_grid_on_image,
)


# ---------------------------------------------------------------------------
# intrinsics / extrinsics
# ---------------------------------------------------------------------------


def test_default_intrinsics_shape():
    K = get_default_intrinsics((800, 600))
    assert K.shape == (3, 3)
    assert K[0, 0] > 0 and K[1, 1] > 0  # fx, fy


def test_default_intrinsics_rescale():
    K1 = get_default_intrinsics((800, 600))
    K2 = get_default_intrinsics((400, 300))
    assert K2[0, 0] == K1[0, 0] / 2  # fx 缩放


def test_default_extrinsics_shape():
    T = get_default_extrinsics()
    assert T.shape == (4, 4)
    assert T[3, 3] == 1.0


# ---------------------------------------------------------------------------
# project_to_voxel_grid
# ---------------------------------------------------------------------------


def test_project_synthetic_blob_center():
    """中心放 40×40 凸起，期望至少产生一些 cell。"""
    H, W = 600, 800
    depth = np.full((H, W), 1.0, dtype=np.float32)
    depth[280:320, 380:420] = 0.7
    grid = project_to_voxel_grid(depth=depth, grid_size=6)
    assert grid.shape == (6, 6, 6)
    assert int(grid.sum()) > 0


def test_project_flat_depth_fills_one_layer():
    """全 1.0 深度（→ world z=0）应填满 z=0 整层 36 cells。"""
    depth = np.full((600, 800), 1.0, dtype=np.float32)
    grid = project_to_voxel_grid(depth=depth, grid_size=6)
    assert int(grid.sum()) == 36


def test_project_torch_input_parity():
    depth_np = np.full((600, 800), 0.7, dtype=np.float32)
    depth_t = torch.from_numpy(depth_np)
    g_np = project_to_voxel_grid(depth=depth_np, grid_size=6)
    g_t = project_to_voxel_grid(depth=depth_t, grid_size=6)
    assert torch.equal(g_np, g_t)


def test_project_image_size_mismatch_raises():
    depth = np.zeros((600, 800), dtype=np.float32)
    img = Image.new("RGB", (400, 300))
    with pytest.raises(ValueError):
        project_to_voxel_grid(image=img, depth=depth)


def test_project_returns_torch_tensor():
    depth = np.zeros((600, 800), dtype=np.float32)
    grid = project_to_voxel_grid(depth=depth, grid_size=6)
    assert isinstance(grid, torch.Tensor)
    assert grid.dtype == torch.float32


# ---------------------------------------------------------------------------
# extract_voxel_trajectory
# ---------------------------------------------------------------------------


def test_extract_trajectory_horizontal_line():
    """水平直线 x∈[-0.2,0.2], y=z=0 应填充一行 cells。"""
    line = np.zeros((20, 3))
    line[:, 0] = np.linspace(-0.2, 0.2, 20)
    grid = extract_voxel_trajectory(line, interpolate=True, interp_step_m=0.01)
    assert grid.shape == (6, 6, 6)
    n_occ = int((grid > 0).sum())
    assert n_occ >= 3, f"expected ≥3 cells along x, got {n_occ}"


def test_extract_trajectory_quaternion_input():
    """(T, 7) 输入与 (T, 3) 等价（忽略姿态）。"""
    line3 = np.zeros((20, 3))
    line3[:, 0] = np.linspace(-0.2, 0.2, 20)
    line7 = np.zeros((20, 7))
    line7[:, :3] = line3
    g3 = extract_voxel_trajectory(line3, interpolate=False)
    g7 = extract_voxel_trajectory(line7, interpolate=False)
    assert torch.equal(g3, g7)


def test_extract_trajectory_out_of_bounds_clipped():
    out_box = np.array([
        [-1.0, 0.0, 0.0],
        [+1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    grid = extract_voxel_trajectory(out_box, interpolate=False)
    assert grid.shape == (6, 6, 6)
    # 三个不同点至少填充 2 个不同 cells（被裁剪后）
    assert int((grid > 0).sum()) >= 2


def test_extract_trajectory_return_sequence_off():
    line = np.zeros((20, 3))
    line[:, 0] = np.linspace(-0.2, 0.2, 20)
    grid_bin = extract_voxel_trajectory(line, interpolate=True, interp_step_m=0.01, return_sequence=False)
    unique = sorted(set(grid_bin.flatten().tolist()))
    assert unique == [0.0, 1.0], f"binary mode unique: {unique}"


def test_extract_trajectory_sequence_starts_at_one():
    line = np.zeros((20, 3))
    line[:, 0] = np.linspace(-0.2, 0.2, 20)
    grid_seq = extract_voxel_trajectory(line, interpolate=False, return_sequence=True)
    occ_vals = sorted([int(v) for v in grid_seq.flatten().tolist() if v > 0])
    assert occ_vals[0] == 1, f"sequence should start at 1, got {occ_vals[0]}"


# ---------------------------------------------------------------------------
# visual_prompt
# ---------------------------------------------------------------------------


def test_draw_voxel_grid_overlay_shape():
    img = np.full((600, 800, 3), 200, dtype=np.uint8)
    overlay = draw_voxel_grid_overlay(img, grid_size=6)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (800, 600)


def test_draw_voxel_grid_on_image_with_cells():
    img = np.full((600, 800, 3), 200, dtype=np.uint8)
    cells = [(2, 2, 0), (3, 3, 1)]
    marked = draw_voxel_grid_on_image(img, cells, grid_size=6)
    assert isinstance(marked, Image.Image)
    assert marked.size == (800, 600)


def test_draw_voxel_grid_on_image_empty_list():
    img = np.full((600, 800, 3), 200, dtype=np.uint8)
    marked = draw_voxel_grid_on_image(img, [])
    assert marked.size == (800, 600)


def test_draw_voxel_grid_overlay_torch_input():
    img_np = np.full((600, 800, 3), 200, dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    overlay = draw_voxel_grid_overlay(img_t, grid_size=6)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (800, 600)