"""
LeRobot Aloha Dataset Adapter
把 LeRobot aloha 仿真演示转成 FlowTDP 训练数据 (image, voxel, action)。

LeRobot aloha 数据格式:
  data/chunk-XXX/file-XXX.parquet: 每行 = 一帧, 字段:
    - observation.state: (14,)  左右臂关节角度
    - observation.images.top: (480, 640, 3) 来自视频
    - action: (14,) 目标关节角度
    - episode_index, frame_index, timestamp, next.done
  videos/observation.images.top/chunk-XXX/file-XXX.mp4: 对应视频

我们做:
  - image: resize 到 96x96
  - voxel: 6x6x6 grid, 从右臂 EE 路径编码 (用简化的网格)
  - action: 右臂 7-DoF (或全 14-DoF)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


# LeRobot 14-dim state 索引 (基于 aloha sim_transfer_cube_human)
# 通常: left_arm(7) + right_arm(7)
LEFT_ARM_IDX = slice(0, 7)
RIGHT_ARM_IDX = slice(7, 14)

# 我们只预测右臂的 action (右臂做抓取)
ACTION_DIM = 7


def get_default_aloha_paths(cache_dir: str = "D:/Desktop/github_project/CLR-VLMTDP/data/lerobot_cache") -> dict:
    """返回 LeRobot aloha 数据的标准路径."""
    base = Path(cache_dir) / "datasets--lerobot--aloha_sim_transfer_cube_human"
    snapshots = list((base / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {base}/snapshots")
    snap = snapshots[0]
    return {
        "parquet": snap / "data",
        "videos": snap / "videos",
        "meta": snap / "meta",
    }


def list_parquet_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("chunk-*/file-*.parquet"))


def list_video_files(video_dir: Path) -> List[Path]:
    """LeRobot 视频在 videos/<camera_name>/chunk-XYZ/file-XYZ.mp4"""
    return sorted(video_dir.glob("*/chunk-*/file-*.mp4"))


def load_video_frames(mp4_path: Path) -> np.ndarray:
    """加载整个 mp4 到 (T, H, W, 3) uint8."""
    reader = iio.imiter(str(mp4_path), plugin='pyav')
    frames = list(reader)
    return np.stack(frames, axis=0)


def resize_image(img: np.ndarray, target_size: tuple = (96, 96)) -> np.ndarray:
    pil = Image.fromarray(img).resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def make_voxel_from_joints(joint_seq: np.ndarray, grid_size: int = 6) -> np.ndarray:
    """
    把关节序列编码成 6×6×6 voxel grid.

    简化的方法: 把 7-D joint 序列 reduce 到 3-D (取前 3 dim 离散化).
    这是 VLA 中常见的"无物理意义"voxel, 只为测试框架.
    """
    # joint_seq: (T, 7)
    if len(joint_seq) < 2:
        return np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)

    # 用前 3 个关节当 3D 坐标 (简化, 但能给体素条件 grid)
    coords = joint_seq[:, :3]  # (T, 3)

    # 离散化到 6×6×6 grid
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # 避免除 0

    normalized = (coords - mins) / ranges  # (T, 3) in [0, 1]
    grid_indices = np.floor(normalized * (grid_size - 1e-6)).astype(int)  # (T, 3) in [0, grid_size-1]

    voxel = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)
    for t, (i, j, k) in enumerate(grid_indices):
        voxel[i, j, k] = t + 1  # 标记时序

    return voxel


class LeRobotAlohaDataset(Dataset):
    """
    LeRobot aloha 数据集 adapter.
    返回 (image, voxel, action) per timestep.
    """

    def __init__(
        self,
        cache_dir: str = "D:/Desktop/github_project/CLR-VLMTDP/data/lerobot_cache",
        image_size: tuple = (96, 96),
        prediction_horizon: int = 12,
        voxel_repr: str = "sequence",
        num_episodes: int = 5,
    ):
        self.paths = get_default_aloha_paths(cache_dir)
        self.image_size = image_size
        self.prediction_horizon = prediction_horizon
        self.voxel_repr = voxel_repr

        # 加载所有 parquet + 视频
        self.episodes = []  # list of (parquet_df, video_frames, episode_id)
        parquet_files = list_parquet_files(self.paths["parquet"])[:num_episodes]
        video_files = list_video_files(self.paths["videos"])[:num_episodes]

        print(f"Loading {len(parquet_files)} episodes from LeRobot aloha...")
        for i, (pq, vid) in enumerate(zip(parquet_files, video_files)):
            try:
                df = pd.read_parquet(pq)
                frames = load_video_frames(vid)
                # 视频可能比 parquet 长 (拼接多个 episode), 截到 parquet 长度
                if len(frames) > len(df):
                    frames = frames[: len(df)]
                elif len(frames) < len(df):
                    # 如果视频短了, 截断 df
                    df = df.iloc[: len(frames)]
                if len(df) < 2:
                    print(f"  ep {i}: too few frames ({len(df)})")
                    continue
                self.episodes.append({
                    "df": df,
                    "frames": frames,
                    "id": i,
                    "length": len(df),
                })
            except Exception as e:
                print(f"  ep {i}: failed ({e})")

        # 建索引
        self.index_map = []  # (episode_idx, timestep)
        for ei, ep in enumerate(self.episodes):
            usable = max(0, ep["length"] - prediction_horizon + 1)
            for t in range(usable):
                self.index_map.append((ei, t))

        print(f"  Loaded {len(self.episodes)} episodes, {len(self.index_map)} samples total")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ei, t = self.index_map[idx]
        ep = self.episodes[ei]
        df = ep["df"]
        frames = ep["frames"]

        # Image
        img = resize_image(frames[t], self.image_size)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        # Action: 用右臂 7-DoF
        actions_full = np.stack([np.array(a) for a in df["action"].values])
        # 取右臂 (索引 7-13)
        actions_right = actions_full[:, 7:14]  # (T, 7)

        # Voxel: 用右臂 joint 序列
        states_full = np.stack([np.array(s) for s in df["observation.state"].values])
        right_arm_seq = states_full[:, 7:14]  # (T, 7)
        voxel = make_voxel_from_joints(right_arm_seq)  # (6, 6, 6)

        if self.voxel_repr == "sequence":
            voxel_t = torch.from_numpy(voxel).long()
        else:
            voxel_t = (torch.from_numpy(voxel) > 0).float()

        # Action window
        T_h = self.prediction_horizon
        action_window = actions_right[t: t + T_h]
        if action_window.shape[0] < T_h:
            pad = np.zeros((T_h - action_window.shape[0], action_window.shape[1]))
            action_window = np.concatenate([action_window, pad], axis=0)
        action_t = torch.from_numpy(action_window).float()

        return {
            "image": img_t,
            "voxel_trajectory": voxel_t,
            "action": action_t,
        }


if __name__ == "__main__":
    print("LeRobotAlohaDataset smoke check:")
    ds = LeRobotAlohaDataset(num_episodes=3)
    print(f"  Dataset size: {len(ds)}")

    sample = ds[0]
    print(f"  Sample shapes: image={tuple(sample['image'].shape)}, "
          f"voxel={tuple(sample['voxel_trajectory'].shape)}, "
          f"action={tuple(sample['action'].shape)}")
    print(f"  Image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    print(f"  Voxel occupied cells: {int((sample['voxel_trajectory'] > 0).sum())}")
    print(f"  Action range: [{sample['action'].min():.3f}, {sample['action'].max():.3f}]")
    print(f"  Action[0]: {sample['action'][0].numpy()}")

    # 保存一张示例图
    import os
    from PIL import Image as PILImage
    pil = PILImage.fromarray((sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    pil.save("results/figures/lerobot_sample.png")
    print(f"  Saved sample image to results/figures/lerobot_sample.png")
