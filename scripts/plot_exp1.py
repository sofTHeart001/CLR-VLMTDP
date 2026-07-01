"""
Experiment 1 可视化
读取 results/exp1_*.json 生成训练曲线 + 性能对比柱状图。

Usage:
    python scripts/plot_exp1.py --input results/exp1_traj_ablation_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非 GUI 后端
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(results: dict, output_path: Path):
    """画训练损失曲线: With-Voxel vs Without-Voxel"""
    # 注意: 当前 JSON 只存 final loss, 不是逐步 loss。
    # 如果未来 run 实验时也存了逐步 loss, 这里可以画曲线。
    # 现在用 bar chart 替代。
    pass


def plot_summary_bar(results: dict, output_path: Path):
    """画性能对比柱状图"""
    metrics = {
        "Final Loss": (results["with_voxel"]["final_loss"], results["without_voxel"]["final_loss"]),
        "Action MSE": (results["with_voxel"]["mse"], results["without_voxel"]["mse"]),
        "Inference (ms)": (results["with_voxel"]["inference_ms_mean"], results["without_voxel"]["inference_ms_mean"]),
        "Train Time (s)": (results["with_voxel"]["train_time_s"] / 10, results["without_voxel"]["train_time_s"] / 10),  # 缩放到同一 y 轴
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    colors = ["#2ecc71", "#e74c3c"]  # 绿=With-Voxel, 红=Without
    labels = ["With Voxel\n(TDP)", "Without Voxel\n(DP baseline)"]

    for ax, (name, (with_v, without_v)) in zip(axes, metrics.items()):
        bars = ax.bar(labels, [with_v, without_v], color=colors, alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, [with_v, without_v]):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                    f"{v:.3f}" if v < 100 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        # 标注 delta
        delta = without_v - with_v
        delta_pct = (delta / without_v) * 100 if without_v != 0 else 0
        ax.set_title(f"{name}\nΔ = {delta:+.3f} ({delta_pct:+.1f}%)", fontsize=10)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(with_v, without_v) * 1.15)

    fig.suptitle(
        f"Experiment 1: Voxel Trajectory Ablation\n"
        f"Voxel condition improves Action MSE by {abs(results['summary']['delta_mse'] / results['without_voxel']['mse'] * 100):.1f}%",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved summary bar chart to {output_path}")


def plot_loss_curve_from_log(log_path: Path, output_path: Path):
    """如果未来 experiment 输出 per-step loss, 这里画曲线。当前 log 不存在, 占位."""
    if not log_path.exists():
        print(f"No log file at {log_path}, skipping loss curve plot")
        return

    # 解析 log (假设每行 "step N loss=X.XXXX")
    steps, losses = [], []
    for line in log_path.read_text().splitlines():
        if "loss=" not in line:
            continue
        try:
            parts = line.strip().split()
            step = int([p for p in parts if p.startswith("step")][0].split("/")[0].replace("step", ""))
            loss = float([p for p in parts if p.startswith("loss=")][0].split("=")[1])
            steps.append(step)
            losses.append(loss)
        except Exception:
            continue

    if not steps:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, "b-", alpha=0.7, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Flow Matching Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved loss curve to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to exp1 JSON result file")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        results = json.load(f)

    stem = input_path.stem  # e.g. exp1_traj_ablation_v2
    plot_summary_bar(results, output_dir / f"{stem}_summary.png")
    plot_loss_curve_from_log(Path(f"results/{stem}_log.txt"),
                             output_dir / f"{stem}_loss_curve.png")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()