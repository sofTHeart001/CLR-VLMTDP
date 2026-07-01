"""
从 ManiSkill motionplanning 演示提取状态，渲染简化图像。
绕开 SAPIEN 渲染问题 (Windows headless 卡死)。

输入: data/maniskill/PickCube-v1/motionplanning/trajectory.h5 (含 qpos + actions)
输出: data/maniskill/PickCube-v1/with_images/trajectory.h5 (含 rgb + voxel + actions)

策略:
  - 从 env_states/articulations/panda 提取 qpos
  - 从 env_states/actors/cube 提取 cube 位置
  - 用我们的 utils.voxel_trajectory 算 EE 位置和体素
  - 渲染简化俯视图 (跟之前 structured_synthetic 一样)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# 确保能找到项目根目录的 utils 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np


def parse_panda_state(flat_state: np.ndarray) -> Dict[str, np.ndarray]:
    """
    解析 (31,) 的 panda flat state vector。

    ManiSkill 的 env_states/articulations/panda shape (T, 31) 包含:
      - root pose (7): x, y, z, qw, qx, qy, qz
      - root velocity (6): vx, vy, vz, wx, wy, wz
      - joint positions (9): 7 arm + 2 gripper
      - joint velocities (9)
    """
    return {
        "root_pose": flat_state[:7],      # (7,)
        "root_vel": flat_state[7:13],     # (6,)
        "qpos": flat_state[13:22],        # (9,) - arm(7) + gripper(2)
        "qvel": flat_state[22:31],        # (9,)
    }


def parse_cube_state(flat_state: np.ndarray) -> np.ndarray:
    """
    解析 (13,) 的 cube state vector。

    ManiSkill 的 cube state 通常:
      - pose (7): x, y, z, qw, qx, qy, qz
      - vel (6): vx, vy, vz, wx, wy, wz
    """
    return flat_state[:7]  # (7,)


def make_top_down_image(
    ee_pos_xy: tuple,
    cube_pos_xy: tuple,
    image_size: tuple = (128, 128),
    grid_size: int = 6,
    bounds: tuple = (-0.3, -0.3, 0.0, 0.3, 0.3, 0.5),
) -> np.ndarray:
    """
    渲染简化俯视图:
      - 深灰背景 + 网格线
      - 绿色 EE 点
      - 红色方块 = cube 位置
    """
    H, W = image_size
    img = np.zeros((H, W, 3), dtype=np.uint8) + 30  # 深灰背景

    xmin, ymin, _, xmax, ymax, _ = bounds

    # 网格
    for i in range(1, grid_size):
        y = int(i * H / grid_size)
        x = int(i * W / grid_size)
        img[y, :, :] = 60
        img[:, x, :] = 60

    def to_px(x, y):
        px = int((x - xmin) / (xmax - xmin) * W)
        py = int((y - ymin) / (ymax - ymin) * H)
        return np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)

    # 绿色 EE
    ex, ey = to_px(*ee_pos_xy)
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            if dx * dx + dy * dy <= 16:
                px, py = ex + dx, ey + dy
                if 0 <= px < W and 0 <= py < H:
                    img[py, px, :] = (0, 255, 0)

    # 红色 cube 方块
    cx, cy = to_px(*cube_pos_xy)
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            px, py = cx + dx, cy + dy
            if 0 <= px < W and 0 <= py < H:
                img[py, px, :] = (0, 0, 255)

    return img


def compute_voxel_from_ee_path(
    ee_positions: np.ndarray,
    grid_size: int = 6,
    bounds: tuple = (-0.3, -0.3, 0.0, 0.3, 0.3, 0.5),
) -> np.ndarray:
    """
    给定 EE 路径, 用 utils.extract_voxel_trajectory 算体素。
    """
    from utils.trajectory_extraction import extract_voxel_trajectory
    if len(ee_positions) < 2:
        return np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)

    voxel = extract_voxel_trajectory(
        ee_positions,
        workspace_bounds=bounds,
        grid_size=grid_size,
        interpolate=True,
        interp_step_m=0.01,
        return_sequence=True,
    ).numpy()
    return voxel


def compute_ee_position(qpos: np.ndarray) -> np.ndarray:
    """
    用 fallback FK 计算 EE 位置（直接走手算，避免 ManiSkill agent 卡死）。
    """
    from utils.franka_fk import fk_panda_batch_fallback
    return fk_panda_batch_fallback(qpos[None])[0]


def process_demo(
    in_h5: h5py.File,
    out_h5: h5py.File,
    traj_key: str,
    image_size: tuple = (128, 128),
) -> bool:
    """处理单条轨迹; 返回是否成功"""
    try:
        # 读取数据
        qpos_flat = in_h5[f"{traj_key}/env_states/articulations/panda"][:]  # (T+1, 31)
        cube_flat = in_h5[f"{traj_key}/env_states/actors/cube"][:]  # (T+1, 13)
        actions = in_h5[f"{traj_key}/actions"][:]  # (T, 8)
        terminated = in_h5[f"{traj_key}/terminated"][:] if "terminated" in in_h5[traj_key] else None
        success = in_h5[f"{traj_key}/success"][:] if "success" in in_h5[traj_key] else None
    except KeyError as e:
        print(f"  Missing key: {e}")
        return False

    # T = actions.shape[0], qpos has T+1 frames (初始 + T 步)
    T = actions.shape[0]

    # 解析 qpos (用前 9 维: arm(7) + gripper(2))
    qpos = qpos_flat[:T, 13:22]  # (T, 9)

    # 解析 cube 位置 (前 3 维 = xyz)
    cube_xyz = cube_flat[:T, :3]  # (T, 3)
    print(f"  Parsed cube_xyz shape={cube_xyz.shape}", flush=True)

    # 计算 EE 位置
    print(f"  Computing EE positions for {T} steps...", flush=True)
    from utils.franka_fk import fk_panda_batch_fallback
    ee_positions = fk_panda_batch_fallback(qpos)  # batch version (fast)
    print(f"  EE positions: shape={ee_positions.shape}", flush=True)
    print(f"  EE range: x=[{ee_positions[:,0].min():.3f}, {ee_positions[:,0].max():.3f}], "
          f"y=[{ee_positions[:,1].min():.3f}, {ee_positions[:,1].max():.3f}], "
          f"z=[{ee_positions[:,2].min():.3f}, {ee_positions[:,2].max():.3f}]", flush=True)

    # 渲染图像
    images = np.zeros((T, image_size[0], image_size[1], 3), dtype=np.uint8)
    for t in range(T):
        images[t] = make_top_down_image(
            ee_pos_xy=(float(ee_positions[t, 0]), float(ee_positions[t, 1])),
            cube_pos_xy=(float(cube_xyz[t, 0]), float(cube_xyz[t, 1])),
            image_size=image_size,
        )

    # 计算体素轨迹 (每步用完整 EE 路径)
    voxel_traj = compute_voxel_from_ee_path(ee_positions)

    # 保存
    grp = out_h5.create_group(traj_key)
    grp.create_dataset("actions", data=actions)
    grp.create_dataset("qpos", data=qpos)
    grp.create_dataset("obs/agent/qpos", data=qpos)  # 兼容 ManiSkillDemoDataset
    grp.create_dataset("obs/sensor_data/base_camera/rgb", data=images)
    grp.create_dataset("obs/sensor_data/base_camera/depth",
                      data=np.zeros((T, image_size[0], image_size[1]), dtype=np.float32))
    grp.create_dataset("ee_positions", data=ee_positions)
    grp.create_dataset("cube_xyz", data=cube_xyz)
    if terminated is not None:
        grp.create_dataset("terminated", data=terminated)
    if success is not None:
        grp.create_dataset("success", data=success)
    # 预存"全 demo 共用的体素轨迹" (EE 完整路径)
    grp.create_dataset("full_voxel_trajectory", data=voxel_traj)

    print(f"  Saved: images shape={images.shape}, voxel shape={voxel_traj.shape}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="D:/Desktop/github_project/CLR-VLMTDP/data/maniskill/PickCube-v1/motionplanning/trajectory.h5")
    parser.add_argument("--output_dir",
                        default="D:/Desktop/github_project/CLR-VLMTDP/data/maniskill/PickCube-v1/with_images")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = output_dir / "trajectory.h5"

    # 删除旧文件,避免叠加
    if output_h5.exists():
        output_h5.unlink()

    in_h5 = h5py.File(args.input, "r")
    out_h5 = h5py.File(str(output_h5), "w")

    # 复制全局 metadata
    if "env_info" in in_h5.attrs:
        out_h5.attrs["env_info"] = in_h5.attrs["env_info"]

    # 找轨迹
    traj_keys = sorted([k for k in in_h5.keys() if k.startswith("traj_")],
                        key=lambda x: int(x.split("_")[1]))[:args.num_episodes]

    print(f"Processing {len(traj_keys)} episodes from {args.input}")
    print(f"Output: {output_h5}")

    # 收集所有 episode 数据,最后在 root 级写 flat 数据
    all_qpos, all_actions, all_rgb, all_depth = [], [], [], []
    all_voxel_traj, all_ee, all_cube = [], [], []
    all_success, all_terminated = [], []

    success_count = 0
    for i, traj_key in enumerate(traj_keys):
        print(f"\n[{i+1}/{len(traj_keys)}] {traj_key}")
        if process_demo(in_h5, out_h5, traj_key, tuple(args.image_size)):
            success_count += 1
            ep = out_h5[traj_key]
            all_qpos.append(ep["qpos"][:])
            all_actions.append(ep["actions"][:])
            all_rgb.append(ep["obs/sensor_data/base_camera/rgb"][:])
            all_depth.append(ep["obs/sensor_data/base_camera/depth"][:])
            T_ep = ep["actions"].shape[0]
            all_voxel_traj.append(np.broadcast_to(
                ep["full_voxel_trajectory"][:], (T_ep, 6, 6, 6)
            ).copy())
            all_ee.append(ep["ee_positions"][:])
            all_cube.append(ep["cube_xyz"][:])
            if "success" in ep:
                all_success.append(ep["success"][:])
            if "terminated" in ep:
                all_terminated.append(ep["terminated"][:])
            # 清理 per-trajectory 数据,只留 root 拼接结果
            del out_h5[traj_key]

    # 写 root 级 flat 数据 (兼容 ManiSkillDemoDataset)
    print(f"\n=== Writing flat root-level data ===")
    out_h5.create_dataset("obs/agent/qpos", data=np.concatenate(all_qpos, axis=0))
    out_h5.create_dataset("actions", data=np.concatenate(all_actions, axis=0))
    out_h5.create_dataset("obs/sensor_data/base_camera/rgb", data=np.concatenate(all_rgb, axis=0))
    out_h5.create_dataset("obs/sensor_data/base_camera/depth", data=np.concatenate(all_depth, axis=0))
    out_h5.create_dataset("voxel_trajectories_per_t", data=np.concatenate(all_voxel_traj, axis=0))
    out_h5.create_dataset("ee_positions", data=np.concatenate(all_ee, axis=0))
    out_h5.create_dataset("cube_xyz", data=np.concatenate(all_cube, axis=0))
    if all_success:
        out_h5.create_dataset("success", data=np.concatenate(all_success, axis=0))
    if all_terminated:
        out_h5.create_dataset("terminated", data=np.concatenate(all_terminated, axis=0))

    print(f"\nDone. {success_count}/{len(traj_keys)} episodes processed.")
    print(f"Output: {output_h5}")
    print(f"  Total steps: {sum(a.shape[0] for a in all_actions)}")
    print(f"  Image shape: {all_rgb[0].shape}")

    in_h5.close()
    out_h5.close()


if __name__ == "__main__":
    main()