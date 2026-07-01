"""
Experiment 1 可视化（增强版）
- 训练损失曲线 (从 jsonl)
- 性能对比柱状图
- 综合 dashboard
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_losses(jsonl_path: Path):
    """从 jsonl 加载每步 loss, 按 model 分组"""
    by_model = defaultdict(list)
    if not jsonl_path.exists():
        return by_model
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            by_model[d["model"]].append((d["step"], d["running_avg"]))
    return by_model


def plot_loss_curves(jsonl_path: Path, output_path: Path):
    """训练损失曲线"""
    by_model = load_losses(jsonl_path)
    if not by_model:
        print(f"  No loss data in {jsonl_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"with_voxel": "#2ecc71", "without_voxel": "#e74c3c"}
    labels = {"with_voxel": "With Voxel (TDP)",
              "without_voxel": "Without Voxel (DP baseline)"}

    for model, data in by_model.items():
        if not data:
            continue
        steps, losses = zip(*data)
        ax.plot(steps, losses, label=labels.get(model, model),
                color=colors.get(model, "gray"), linewidth=2, alpha=0.85)

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Loss (running avg)", fontsize=11)
    ax.set_title("Experiment 1 (Real ManiSkill data): Training Loss Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss curves: {output_path}")


def plot_summary_bar(results: dict, output_path: Path):
    """4 指标柱状图"""
    metrics = {
        "Final Loss": (results["with_voxel"]["final_loss"], results["without_voxel"]["final_loss"]),
        "Action MSE": (results["with_voxel"]["mse"], results["without_voxel"]["mse"]),
        "Inference (ms)": (results["with_voxel"]["inference_ms_mean"], results["without_voxel"]["inference_ms_mean"]),
        "Train Time (s) / 10": (results["with_voxel"]["train_time_s"] / 10, results["without_voxel"]["train_time_s"] / 10),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.2))
    colors = ["#2ecc71", "#e74c3c"]
    labels = ["With Voxel\n(TDP)", "Without Voxel\n(DP)"]

    for ax, (name, (with_v, without_v)) in zip(axes, metrics.items()):
        bars = ax.bar(labels, [with_v, without_v], color=colors, alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, [with_v, without_v]):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                    f"{v:.3f}" if v < 100 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        delta = without_v - with_v
        delta_pct = (delta / without_v) * 100 if without_v != 0 else 0
        ax.set_title(f"{name}\nΔ={delta:+.3f} ({delta_pct:+.1f}%)", fontsize=10)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(with_v, without_v) * 1.18)

    delta_mse = results["summary"]["delta_mse"]
    fig.suptitle(
        f"Experiment 1 (Real ManiSkill data): Voxel Trajectory Ablation\n"
        f"Voxel condition improves Action MSE by {abs(delta_mse / results['without_voxel']['mse'] * 100):.1f}%",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary bar: {output_path}")


def plot_dashboard(results: dict, jsonl_path: Path, output_path: Path,
                   sample_frames_dir: Path = None):
    """综合 dashboard: loss curves + bar chart + 关键数字"""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # Top: loss curves (2 cols)
    ax_loss = fig.add_subplot(gs[0, :3])
    by_model = load_losses(jsonl_path)
    colors = {"with_voxel": "#2ecc71", "without_voxel": "#e74c3c"}
    labels = {"with_voxel": "With Voxel (TDP)",
              "without_voxel": "Without Voxel (DP baseline)"}
    for model, data in by_model.items():
        if not data:
            continue
        steps, losses = zip(*data)
        ax_loss.plot(steps, losses, label=labels.get(model, model),
                     color=colors.get(model, "gray"), linewidth=2)
    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss (Real ManiSkill data)", fontweight="bold")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # Top-right: 关键数字
    ax_text = fig.add_subplot(gs[0, 3])
    ax_text.axis("off")
    delta_mse = results["summary"]["delta_mse"]
    delta_pct = abs(delta_mse / results["without_voxel"]["mse"] * 100)
    txt = (
        "Key Results\n"
        "================\n"
        f"Data: ManiSkill PickCube-v1\n"
        f"Demos used: {results.get('num_episodes', 'N/A')}\n\n"
        f"With Voxel:\n"
        f"  Action MSE:  {results['with_voxel']['mse']:.4f}\n"
        f"  Final Loss:  {results['with_voxel']['final_loss']:.4f}\n\n"
        f"Without Voxel:\n"
        f"  Action MSE:  {results['without_voxel']['mse']:.4f}\n"
        f"  Final Loss:  {results['without_voxel']['final_loss']:.4f}\n\n"
        f"Voxel improves MSE by:\n"
        f"  {delta_pct:.1f}% (Δ={delta_mse:+.4f})\n"
    )
    ax_text.text(0.05, 0.5, txt, fontsize=10, family="monospace",
                 verticalalignment="center")

    # Bottom: 4 指标 bar
    metrics = {
        "Final Loss": (results["with_voxel"]["final_loss"], results["without_voxel"]["final_loss"]),
        "Action MSE": (results["with_voxel"]["mse"], results["without_voxel"]["mse"]),
        "Infer (ms)": (results["with_voxel"]["inference_ms_mean"], results["without_voxel"]["inference_ms_mean"]),
        "Train (s)": (results["with_voxel"]["train_time_s"], results["without_voxel"]["train_time_s"]),
    }
    for i, (name, (with_v, without_v)) in enumerate(metrics.items()):
        ax = fig.add_subplot(gs[1, i])
        bars = ax.bar(["With", "Without"], [with_v, without_v],
                       color=["#2ecc71", "#e74c3c"], alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, [with_v, without_v]):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                    f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(with_v, without_v) * 1.18)

    fig.suptitle("Experiment 1: Voxel Trajectory Ablation (Real ManiSkill data)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--losses_jsonl", default="results/exp1_real_data_losses.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        results = json.load(f)

    stem = Path(args.input).stem
    plot_loss_curves(Path(args.losses_jsonl), output_dir / f"{stem}_loss_curve.png")
    plot_summary_bar(results, output_dir / f"{stem}_summary.png")
    plot_dashboard(results, Path(args.losses_jsonl),
                   output_dir / f"{stem}_dashboard.png")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()