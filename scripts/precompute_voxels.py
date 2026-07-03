"""
Pre-compute voxel trajectories for LeRobot aloha dataset.

Saves to a new h5 file with all data + pre-computed voxel grids.
This way training doesn't need to recompute FK on every __getitem__.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd

from utils.viperx_fk import fk_viperx_batch
from data.lerobot_dataset import (
    get_default_aloha_paths,
    list_parquet_files,
    list_video_files,
    make_voxel_from_ee_path,
)


def precompute(output_path: str = "D:/Desktop/github_project/CLR-VLMTDP/data/lerobot_precomputed/with_voxels.h5"):
    """合并 parquet + video + 预计算 voxel, 存到一个 h5."""
    paths = get_default_aloha_paths()
    parquet_files = list_parquet_files(paths["parquet"])
    video_files = list_video_files(paths["videos"])

    print(f"Loading {len(parquet_files)} parquet files + 1 video...")
    all_dfs = []
    for pq in parquet_files:
        all_dfs.append(pd.read_parquet(pq))
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Combined: {len(combined_df)} frames, "
          f"{combined_df['episode_index'].nunique()} episodes")

    if not video_files:
        print("  No video files found")
        return

    # Load full video
    vid = video_files[0]
    all_frames = load_video_resized(vid, target_size=(96, 96))
    print(f"  Loaded video: {len(all_frames)} frames @ 96x96")

    n = min(len(all_frames), len(combined_df))
    all_frames = all_frames[:n]
    combined_df = combined_df.iloc[:n].reset_index(drop=True)

    # 预计算每帧的 voxel
    print("Pre-computing voxels (FK on CPU)...")
    n = len(combined_df)

    # 批量算 EE 位置 (按 episode 分组, 每组 batch 算)
    ee_positions_all = np.zeros((n, 3), dtype=np.float32)
    voxel_all = np.zeros((n, 6, 6, 6), dtype=np.int32)

    for ep_id in combined_df['episode_index'].unique():
        mask = (combined_df['episode_index'] == ep_id).values
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue

        # 算这一 episode 的 FK
        states = np.stack([np.array(s) for s in combined_df.iloc[idx]['observation.state'].values])
        right_arm_joints = states[:, 7:13]  # (T, 6)
        ee_pos = fk_viperx_batch(right_arm_joints)
        ee_positions_all[idx] = ee_pos.astype(np.float32)

        # 算 voxel (整 episode 的路径, 每个 timestep 共享)
        voxel = make_voxel_from_ee_path(
            ee_pos,
            workspace_bounds=(-0.2, 0.0, 0.0, 0.3, 0.4, 0.4),
            grid_size=6,
        )
        voxel_all[idx] = voxel[None].astype(np.int32).repeat(len(idx), axis=0)

        if (ep_id % 5) == 0:
            print(f"  ep {ep_id}: {len(idx)} frames, "
                  f"EE range x=[{ee_pos[:,0].min():.2f},{ee_pos[:,0].max():.2f}], "
                  f"voxel occupied={int((voxel > 0).sum())}")

    # 写到一个 h5
    print(f"\nWriting pre-computed h5 to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        # 图像
        f.create_dataset("images", data=all_frames, compression="gzip", compression_opts=4)
        # voxel
        f.create_dataset("voxels", data=voxel_all, compression="gzip", compression_opts=4)
        # 状态和动作
        states = np.stack([np.array(s) for s in combined_df['observation.state'].values]).astype(np.float32)
        actions = np.stack([np.array(a) for a in combined_df['action'].values]).astype(np.float32)
        f.create_dataset("states", data=states)
        f.create_dataset("actions", data=actions)
        f.create_dataset("ee_positions", data=ee_positions_all)
        f.create_dataset("episode_index", data=combined_df['episode_index'].values.astype(np.int32))
        f.create_dataset("frame_index", data=combined_df['frame_index'].values.astype(np.int32))

    print(f"  Done. File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")


def load_video_resized(mp4_path: Path, target_size=(96, 96)) -> np.ndarray:
    """Load mp4 + resize to (T, H, W, 3) uint8."""
    from PIL import Image
    print(f"  Loading and resizing {mp4_path.name} to {target_size}...")
    reader = iio.imiter(str(mp4_path), plugin='pyav')
    frames = []
    for i, frame in enumerate(reader):
        pil = Image.fromarray(frame).resize(
            (target_size[1], target_size[0]), Image.BILINEAR
        )
        frames.append(np.array(pil, dtype=np.uint8))
        if i % 5000 == 0 and i > 0:
            print(f"    loaded {i} frames")
    print(f"    loaded {len(frames)} frames total")
    return np.stack(frames, axis=0)


if __name__ == "__main__":
    precompute()