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


def make_voxel_from_ee_path(
    ee_positions: np.ndarray,
    workspace_bounds: tuple = (-0.3, 0.0, 0.0, 0.3, 0.4, 0.4),
    grid_size: int = 6,
    interpolate: bool = True,
    interp_step_m: float = 0.02,
) -> np.ndarray:
    """
    把 EE 位置序列编码成 6×6×6 voxel grid (按时序标记).

    这才是 "VLM-TDP 体素轨迹" 的正确做法:
    - 末端执行器的 3D 位置 (世界坐标)
    - 经过的 cell 按时间顺序标记 1, 2, 3, ...
    - 路径上的 cell 都被标记

    Args:
        ee_positions: (T, 3) EE 在 base 坐标系下的位置
        workspace_bounds: (xmin, ymin, zmin, xmax, ymax, zmax)
        grid_size: 每边 cell 数
        interpolate: 是否插值 (让路径更密)
        interp_step_m: 插值步长 (米)
    """
    if len(ee_positions) < 2:
        return np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)

    if interpolate:
        from utils.trajectory_extraction import _interpolate_path
        ee_positions = _interpolate_path(ee_positions, step_m=interp_step_m)

    xmin, ymin, zmin, xmax, ymax, zmax = workspace_bounds
    voxel = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)

    for t, (x, y, z) in enumerate(ee_positions):
        # 裁剪到工作空间
        x = np.clip(x, xmin, xmax - 1e-9)
        y = np.clip(y, ymin, ymax - 1e-9)
        z = np.clip(z, zmin, zmax - 1e-9)
        # 离散化
        i = int((x - xmin) / (xmax - xmin) * (grid_size - 1e-9))
        j = int((y - ymin) / (ymax - ymin) * (grid_size - 1e-9))
        k = int((z - zmin) / (zmax - zmin) * (grid_size - 1e-9))
        if voxel[i, j, k] == 0:
            voxel[i, j, k] = t + 1  # 标记时序 (第一个到的 cell 标 1)

    return voxel


class LeRobotAlohaDataset(Dataset):
    """
    LeRobot aloha 数据集 adapter.
    返回 (image, voxel, action) per timestep.

    把多个 parquet + 1 个 video 当作一个连续序列处理
    (parquet 行和 video 帧 1:1 对应).
    """

    def __init__(
        self,
        cache_dir: str = "D:/Desktop/github_project/CLR-VLMTDP/data/lerobot_cache",
        image_size: tuple = (96, 96),
        prediction_horizon: int = 12,
        voxel_repr: str = "sequence",
        num_episodes: int = 50,
    ):
        self.paths = get_default_aloha_paths(cache_dir)
        self.image_size = image_size
        self.prediction_horizon = prediction_horizon
        self.voxel_repr = voxel_repr

        parquet_files = list_parquet_files(self.paths["parquet"])
        video_files = list_video_files(self.paths["videos"])

        print(f"Loading LeRobot aloha data ({len(parquet_files)} parquet files, {len(video_files)} video files)...")

        # 合并所有 parquet
        all_dfs = []
        for pq in parquet_files:
            df = pd.read_parquet(pq)
            all_dfs.append(df)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"  Combined parquet: {len(combined_df)} frames, {combined_df['episode_index'].nunique()} unique episodes")

        if not video_files:
            print("  No video files found")
            return

        vid = video_files[0]  # top camera (first available)
        all_frames = load_video_frames(vid)
        print(f"  Loaded video: {len(all_frames)} frames from {vid.name}")

        # 截断到 parquet 长度
        n = min(len(all_frames), len(combined_df))
        all_frames = all_frames[:n]
        combined_df = combined_df.iloc[:n].reset_index(drop=True)

        # 把整个序列当作 1 个 entry
        self.episodes = [{
            "df": combined_df,
            "frames": all_frames,
            "id": "all",
            "length": n,
        }]

        # 限制使用的 episode 数 (用 num_episodes 选前面 N 个 episode)
        if num_episodes > 0 and num_episodes < combined_df['episode_index'].nunique():
            # 找到 num_episodes 个 episode 包含的帧数
            keep_eps = sorted(combined_df['episode_index'].unique())[:num_episodes]
            mask = combined_df['episode_index'].isin(keep_eps)
            n_keep = int(mask.sum())
            self.episodes[0]["df"] = combined_df[mask].reset_index(drop=True)
            self.episodes[0]["frames"] = all_frames[:n_keep]
            self.episodes[0]["length"] = n_keep
            print(f"  Filtered to {num_episodes} episodes = {n_keep} frames")

        # 建索引
        self.index_map = []
        for ei, ep in enumerate(self.episodes):
            usable = max(0, ep["length"] - prediction_horizon + 1)
            for t in range(usable):
                self.index_map.append((ei, t))

        print(f"  Final: {len(self.episodes)} entries, {len(self.index_map)} samples")

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

        # Voxel: 用右臂 FK 算 EE 位置 → 离散化
        states_full = np.stack([np.array(s) for s in df["observation.state"].values])
        right_arm_joints = states_full[:, 7:13]  # (T, 6) — 拿前 6 维 (丢夹爪)
        # FK → EE 位置 (T, 3)
        from utils.viperx_fk import fk_viperx_batch
        ee_positions = fk_viperx_batch(right_arm_joints)
        # 用 EE 位置做 voxel
        # Aloha sim 工作空间: x ∈ [-0.2, 0.3], y ∈ [0, 0.4], z ∈ [0, 0.4]
        voxel = make_voxel_from_ee_path(
            ee_positions,
            workspace_bounds=(-0.2, 0.0, 0.0, 0.3, 0.4, 0.4),
            grid_size=6,
            interpolate=True,
            interp_step_m=0.02,
        )

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
