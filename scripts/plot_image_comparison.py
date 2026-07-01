"""
Side-by-side comparison:
  - Real ManiSkill frames (from official sample.mp4)
  - Our simplified top-down images (from extract_states.py)
诚实对比, 让用户清楚知道数据来源差异。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    # ManiSkill 官方 sample.mp4 路径
    mp4_path = "D:/Desktop/github_project/CLR-VLMTDP/data/maniskill/PickCube-v1/motionplanning/sample.mp4"
    # 我们的 flat h5 路径
    h5_path = "D:/Desktop/github_project/CLR-VLMTDP/data/maniskill/PickCube-v1/with_images/trajectory.h5"

    # 1. 加载 5 帧 sample.mp4
    import imageio.v3 as iio
    mp4_frames = list(iio.imiter(mp4_path, plugin='pyav'))
    print(f"Loaded {len(mp4_frames)} real ManiSkill frames from sample.mp4")

    # 2. 加载 5 帧我们的 h5
    with h5py.File(h5_path, "r") as f:
        our_frames = f["obs/sensor_data/base_camera/rgb"][:]  # (T, 128, 128, 3)
    print(f"Loaded {len(our_frames)} simplified frames from h5")

    # 选 5 个均匀间隔
    indices = np.linspace(0, min(len(mp4_frames), len(our_frames)) - 1, 5, dtype=int)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))

    for col, idx in enumerate(indices):
        # 上排: 真 ManiSkill 帧
        ax_real = axes[0, col]
        ax_real.imshow(mp4_frames[idx])
        ax_real.set_title(f"Real ManiSkill frame {idx}", fontsize=10)
        ax_real.axis("off")

        # 下排: 我们的简化帧
        ax_ours = axes[1, col]
        ax_ours.imshow(our_frames[idx])
        ax_ours.set_title(f"Our simplified image (same step)", fontsize=10)
        ax_ours.axis("off")

    axes[0, 0].text(-0.15, 0.5, "Real\n(ManiSkill render)",
                     transform=axes[0, 0].transAxes, fontsize=11, fontweight="bold",
                     rotation=90, va="center", ha="center")
    axes[1, 0].text(-0.15, 0.5, "Synthetic\n(simplified)",
                     transform=axes[1, 0].transAxes, fontsize=11, fontweight="bold",
                     rotation=90, va="center", ha="center")

    fig.suptitle(
        "Image Comparison: Real ManiSkill Render vs Our Simplified Top-Down\n"
        "(Note: ManiSkill rendering blocked on this Windows machine — see README for workaround plan)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = "results/figures/image_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison: {out}")


if __name__ == "__main__":
    main()