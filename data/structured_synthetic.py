"""
Structured Synthetic Data Generator
生成"有因果关系"的合成 ManiSkill 数据。

设计原则:
  - 图像包含视觉信息（目标位置用红色块表示）
  - 动作 = 归一化方向向量（从 EE 到目标）
  - 体素轨迹 = 6×6×6 grid 中目标位置 cell 标 1

这样:
  - 没有体素时: 模型需要从图像里识别红色块位置 → 计算方向
  - 有体素时: 模型直接拿到目标位置 cell → 计算方向
  - 两者都能学, 但有体素应该更简单 / 更准
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

from utils.trajectory_extraction import extract_voxel_trajectory
from utils.voxel_trajectory import DEFAULT_WORKSPACE_BOUNDS, project_to_voxel_grid


# ---------------------------------------------------------------------------
# 任务: 抓一个随机位置的方块
# ---------------------------------------------------------------------------


GRID_SIZE = 6


def make_top_down_image(
    ee_pos: np.ndarray,
    target_pos: np.ndarray,
    image_size: Tuple[int, int] = (128, 128),
    grid_size: int = GRID_SIZE,
    bounds: Tuple[float, ...] = DEFAULT_WORKSPACE_BOUNDS,
) -> np.ndarray:
    """
    渲染一张俯视图:
      - 黑色背景
      - 绿色点表示 EE 当前位置
      - 红色方块表示目标
      - 浅灰色网格线

    Args:
        ee_pos: (2,) EE 的 (x, y) 坐标
        target_pos: (2,) 目标的 (x, y) 坐标
        image_size: (H, W)
        grid_size: 每边格数
        bounds: workspace_bounds (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    H, W = image_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, :] = 30  # 深灰背景

    xmin, ymin, _, xmax, ymax, _ = bounds

    # 网格线
    for i in range(1, grid_size):
        y = int(i * H / grid_size)
        x = int(i * W / grid_size)
        img[y, :, :] = 60
        img[:, x, :] = 60

    def world_to_pixel(x, y):
        """(x, y) 世界坐标 → 像素坐标"""
        px = int((x - xmin) / (xmax - xmin) * W)
        py = int((y - ymin) / (ymax - ymin) * H)
        return np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)

    # 绿色 EE 点
    ex, ey = world_to_pixel(ee_pos[0], ee_pos[1])
    cv2_circle = lambda cx, cy, r, color: [
        (cx + dx, cy + dy)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        if dx * dx + dy * dy <= r * r
    ]
    for px, py in cv2_circle(ex, ey, 4, (0, 255, 0)):
        if 0 <= px < W and 0 <= py < H:
            img[py, px, :] = (0, 255, 0)  # BGR 但我们只关心相对

    # 红色目标方块 (8x8 像素)
    tx, ty = world_to_pixel(target_pos[0], target_pos[1])
    for px in range(tx - 5, tx + 6):
        for py in range(ty - 5, ty + 6):
            if 0 <= px < W and 0 <= py < H:
                img[py, px, :] = (0, 0, 255)  # 红色 (in RGB: BGR format is reverse)

    return img


def make_action(ee_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """action = 归一化方向向量 (从 EE 到 target), 8 维"""
    delta = target_pos - ee_pos
    direction = delta / (np.linalg.norm(delta) + 1e-8)
    # 8 维动作: 前 2 维是方向, 后面填充 0 (论文 8 维 = 7 joint + 1 gripper)
    action = np.zeros(8)
    action[:2] = direction
    return action


def make_voxel_trajectory(ee_pos: np.ndarray, target_pos: np.ndarray,
                          grid_size: int = GRID_SIZE,
                          bounds: Tuple[float, ...] = DEFAULT_WORKSPACE_BOUNDS) -> np.ndarray:
    """生成体素轨迹: 在 6×6×6 grid 中标记 EE→target 路径。

    用直线轨迹从 EE 到 target, 然后 extract_voxel_trajectory 转成 grid。
    """
    # 生成直线 waypoints (50 步)
    t = np.linspace(0, 1, 50)[:, None]
    path = ee_pos[None, :] * (1 - t) + target_pos[None, :] * t  # (50, 2)
    # 补 z 维度
    path_3d = np.concatenate([path, np.full((50, 1), 0.05)], axis=1)
    # 提取体素
    grid = extract_voxel_trajectory(
        path_3d,
        workspace_bounds=bounds,
        grid_size=grid_size,
        interpolate=True,
        interp_step_m=0.01,
        return_sequence=True,
    ).numpy()
    return grid  # (6, 6, 6) int


def make_structured_demo(
    T: int = 60,
    image_size: Tuple[int, int] = (128, 128),
    grid_size: int = GRID_SIZE,
    seed: int = 0,
) -> dict:
    """
    生成一条"抓随机方块"演示。

    Args:
        T: 时间步数
        image_size: 图像 (H, W)
        grid_size: 体素网格每边格数
        seed: 随机种子

    Returns:
        dict with: qpos, actions, rgb, voxel_trajectories
    """
    rng = np.random.default_rng(seed)
    bounds = DEFAULT_WORKSPACE_BOUNDS

    # 随机选 EE 起点和目标
    ee_start = rng.uniform([-0.15, -0.15, 0.20], [0.15, 0.15, 0.40])
    target = rng.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 0.0])

    # 生成轨迹: EE 从 start 沿直线到 target
    t = np.linspace(0, 1, T)[:, None]
    ee_trajectory = ee_start[None, :] * (1 - t) + target[None, :] * t  # (T, 3)

    # 9-D qpos (7 arm + 2 gripper) - 直接用 ee_trajectory 作为 qpos (简化)
    qpos = np.zeros((T, 9), dtype=np.float32)
    qpos[:, :7] = ee_trajectory[:, :3]  # arm
    qpos[:, -2:] = 0.04  # gripper 一直开 (简化)

    # actions: 每个 timestep 的 delta direction + 小噪声
    actions = np.zeros((T, 8), dtype=np.float32)
    for i in range(T):
        actions[i] = make_action(ee_trajectory[i], target)
        actions[i] += rng.normal(0, 0.05, size=8).astype(np.float32)

    # rgb: 每个 timestep 渲染俯视图
    rgb = np.zeros((T, image_size[0], image_size[1], 3), dtype=np.uint8)
    for i in range(T):
        rgb[i] = make_top_down_image(ee_trajectory[i, :2], target[:2],
                                       image_size=image_size, grid_size=grid_size,
                                       bounds=bounds)

    # voxel_trajectories: 每个 timestep 的 6×6×6 grid
    voxel_trajs = np.zeros((T, grid_size, grid_size, grid_size), dtype=np.int64)
    for i in range(T):
        voxel_trajs[i] = make_voxel_trajectory(ee_trajectory[i], target,
                                                  grid_size=grid_size, bounds=bounds)

    return {
        "qpos": qpos,
        "actions": actions,
        "rgb": rgb,
        "voxel_trajectories": voxel_trajs,
        "ee_trajectory": ee_trajectory,
        "target": target,
    }


def save_demo_to_h5(demo: dict, path: str, image_size: Tuple[int, int],
                    grid_size: int) -> None:
    """把生成的演示保存到 ManiSkill 风格的 .h5 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("obs/agent/qpos", data=demo["qpos"])
        f.create_dataset("actions", data=demo["actions"])
        f.create_dataset(f"obs/sensor_data/base_camera/rgb", data=demo["rgb"])

        # 简单相机参数 (俯视)
        param = f.create_group("obs/sensor_param/base_camera")
        H, W = image_size
        K = np.array([
            [W, 0, W / 2],
            [0, W, H / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        param.create_dataset("intrinsic_cv", data=K)
        # 相机在 (0, 0, 1) 朝下看
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        param.create_dataset("cam2world_gl", data=T)


def make_structured_dataset(
    num_demos: int = 20,
    T: int = 60,
    image_size: Tuple[int, int] = (128, 128),
    output_dir: str = None,
) -> List[str]:
    """
    生成多个结构化演示并保存到 .h5 文件。

    Returns:
        List of .h5 file paths
    """
    import tempfile
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="structured_")
    output_dir = Path(output_dir)

    paths = []
    for i in range(num_demos):
        demo = make_structured_demo(T=T, image_size=image_size, seed=i * 1000 + 42)
        path = str(output_dir / f"structured_demo_{i:04d}.h5")
        save_demo_to_h5(demo, path, image_size=image_size, grid_size=GRID_SIZE)
        paths.append(path)
        # 打印示例
        if i == 0:
            occupied = int((demo["voxel_trajectories"][0] > 0).sum())
            print(f"  Sample demo 0: target={demo['target'].round(3)}, "
                  f"image shape={demo['rgb'][0].shape}, "
                  f"voxel occupied cells (t=0)={occupied}")

    return paths


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_test():
    print("Structured synthetic data smoke checks:")
    demo = make_structured_demo(T=60, image_size=(128, 128), seed=42)
    print(f"  qpos shape: {demo['qpos'].shape}")
    print(f"  actions shape: {demo['actions'].shape}")
    print(f"  rgb shape: {demo['rgb'].shape}")
    print(f"  voxel_trajectories shape: {demo['voxel_trajectories'].shape}")
    assert demo["qpos"].shape == (60, 9)
    assert demo["actions"].shape == (60, 8)
    assert demo["rgb"].shape == (60, 128, 128, 3)
    assert demo["voxel_trajectories"].shape == (60, 6, 6, 6)

    # 体素应该有非零 cell
    occ0 = int((demo["voxel_trajectories"][0] > 0).sum())
    occ30 = int((demo["voxel_trajectories"][30] > 0).sum())
    print(f"  voxel occupied cells: t=0 → {occ0}, t=30 → {occ30}")
    assert occ0 > 0 and occ30 > 0

    # 验证动作方向正确
    ee = demo["ee_trajectory"][0]
    target = demo["target"]
    expected_dir = (target[:2] - ee[:2]) / np.linalg.norm(target[:2] - ee[:2])
    actual_dir = demo["actions"][0, :2] / (np.linalg.norm(demo["actions"][0, :2]) + 1e-8)
    cos_sim = np.dot(expected_dir, actual_dir)
    print(f"  Action direction cosine similarity: {cos_sim:.3f} (should be close to 1)")
    assert cos_sim > 0.7, f"Action direction wrong: cos_sim={cos_sim}"

    print("  All smoke checks passed.")


if __name__ == "__main__":
    self_test()