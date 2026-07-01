"""
Replay ManiSkill demos to generate RGB images
把 obs_mode="none" 的演示重放一遍，渲染 RGB + 深度，保存为新 h5 文件。

ManiSkill 演示默认只保存 state (no images)，需要 replay 重新渲染。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np


def replay_demo_to_h5(
    demo_h5_path: str,
    output_h5_path: str,
    num_episodes: int = 5,
    image_size: tuple = (128, 128),
):
    """
    Replay `num_episodes` trajectories from demo_h5_path and save with RGB.

    Args:
        demo_h5_path: 输入演示 (obs_mode="none")
        output_h5_path: 输出新 h5 (含 rgb)
        num_episodes: 处理多少条
        image_size: 渲染分辨率
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  注册 envs

    print(f"Loading env PickCube-v1 with obs_mode='rgbd'...")
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
    )

    # 找到 trajectory IDs
    with h5py.File(demo_h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                            key=lambda x: int(x.split("_")[1]))
        traj_keys = traj_keys[:num_episodes]

    print(f"Replaying {len(traj_keys)} episodes...")

    out_h5 = h5py.File(output_h5_path, "w")

    for i, traj_key in enumerate(traj_keys):
        print(f"\n[{i+1}/{len(traj_key)}] Replaying {traj_key}...")

        # Load and replay this trajectory
        try:
            episode = env.unwrapped.load_trajectory(demo_h5_path, traj_key)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Reset env with the trajectory's seed
        obs, info = env.reset(seed=int(traj_key.split("_")[1]))

        # Get camera intrinsics from env
        # (ManiSkill exposes this via env.unwrapped.scene)

        # Replay actions
        with h5py.File(demo_h5_path, "r") as f:
            actions = f[f"{traj_key}/actions"][:]

        images = []
        depths = []
        for t, action in enumerate(actions):
            # Render
            rgb = env.unwrapped.render_rgb_array()  # (H, W, 3) uint8
            depth = env.unwrapped.render_depth_array()  # (H, W) float
            images.append(rgb)
            depths.append(depth)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

        images = np.stack(images, axis=0)
        depths = np.stack(depths, axis=0)

        # 保存到新 h5
        ep_group = out_h5.create_group(traj_key)
        ep_group.create_dataset("actions", data=actions)
        ep_group.create_dataset("obs/sensor_data/base_camera/rgb", data=images)
        ep_group.create_dataset("obs/sensor_data/base_camera/depth", data=depths)

        print(f"  Saved {len(actions)} frames with images shape={images.shape}, "
              f"depths shape={depths.shape}")

    out_h5.close()
    env.close()
    print(f"\nDone. Saved to {output_h5_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input demo h5 path")
    parser.add_argument("--output", required=True, help="Output h5 path with images")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128])
    args = parser.parse_args()

    replay_demo_to_h5(
        args.input,
        args.output,
        num_episodes=args.num_episodes,
        image_size=tuple(args.image_size),
    )


if __name__ == "__main__":
    main()