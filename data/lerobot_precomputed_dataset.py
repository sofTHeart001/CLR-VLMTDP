"""
LeRobot Aloha Dataset (pre-computed version)
直接读预计算 h5 (含 images + voxels + actions), 训练时无需 FK 计算.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LeRobotPrecomputedDataset(Dataset):
    """
    从预计算 h5 直接读 (image, voxel, action).

    优点: 训练时无 CPU 计算瓶颈 (FK + voxel 都已预计算)
    """

    def __init__(
        self,
        h5_path: str = "D:/Desktop/github_project/CLR-VLMTDP/data/lerobot_precomputed/with_voxels.h5",
        image_size: tuple = (96, 96),
        prediction_horizon: int = 12,
        voxel_repr: str = "sequence",
        num_episodes: int = 50,
    ):
        self.h5_path = h5_path
        self.image_size = image_size
        self.prediction_horizon = prediction_horizon
        self.voxel_repr = voxel_repr

        # 加载所有数据到内存
        with h5py.File(h5_path, "r") as f:
            self.images = f["images"][:]  # (T, H, W, 3) uint8
            self.voxels = f["voxels"][:]  # (T, 6, 6, 6) int32
            self.actions = f["actions"][:]  # (T, 8) float32 (右臂 7 + 1 gripper)
            self.episode_index = f["episode_index"][:]  # (T,)

        T = len(self.images)
        self.T = T

        # 限制 episode 数
        if num_episodes > 0:
            unique_eps = sorted(set(self.episode_index.tolist()))
            keep_eps = set(unique_eps[:num_episodes])
            self.mask = np.array([ep in keep_eps for ep in self.episode_index])
            self.images = self.images[self.mask]
            self.voxels = self.voxels[self.mask]
            self.actions = self.actions[self.mask]
            self.episode_index = self.episode_index[self.mask]
            self.T = len(self.images)
            print(f"  Filtered to {num_episodes} episodes, {self.T} frames")

        # 建索引
        self.index_map = []
        for t in range(self.T - prediction_horizon + 1):
            self.index_map.append(t)

        print(f"  Loaded {self.T} frames, {len(self.index_map)} samples")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        t = self.index_map[idx]
        # Image (96, 96, 3) uint8 -> (3, 96, 96) float
        img = self.images[t]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Voxel
        v = self.voxels[t]
        if self.voxel_repr == "sequence":
            voxel_t = torch.from_numpy(v).long()
        else:
            voxel_t = (torch.from_numpy(v) > 0).float()

        # Action window
        T_h = self.prediction_horizon
        action_window = self.actions[t: t + T_h, 7:14]  # 右臂 7-DoF
        if action_window.shape[0] < T_h:
            pad = np.zeros((T_h - action_window.shape[0], 7), dtype=np.float32)
            action_window = np.concatenate([action_window, pad], axis=0)
        action_t = torch.from_numpy(action_window).float()

        return {
            "image": img_t,
            "voxel_trajectory": voxel_t,
            "action": action_t,
        }


if __name__ == "__main__":
    print("LeRobotPrecomputedDataset smoke check:")
    ds = LeRobotPrecomputedDataset(num_episodes=10)
    print(f"  Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"  Sample shapes: image={tuple(sample['image'].shape)}, "
          f"voxel={tuple(sample['voxel_trajectory'].shape)}, "
          f"action={tuple(sample['action'].shape)}")
    print(f"  Voxel occupied cells: {int((sample['voxel_trajectory'] > 0).sum())}")
    print(f"  Action[0]: {sample['action'][0].numpy()}")