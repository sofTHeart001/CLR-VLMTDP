"""
Plot Experiment 1 results: loss curves + summary bar + dashboard.
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
    by_model = defaultdict(list)
    if not jsonl_path.exists():
        return by_model
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            by_model[d["model"]].append((d["step"], d["running_avg"]))
    return by_model


def plot_loss_curves_from_json(results: dict, output_path: Path):
    """Plot loss curves from in-memory results."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"with_voxel": "#2ecc71", "without_voxel": "#e74c3c"}
    labels = {"with_voxel": "With Voxel (TDP)", "without_voxel": "Without Voxel (DP baseline)"}

    for model_key in ["with_voxel", "without_voxel"]:
        if model_key not in results:
            continue
        m = results[model_key]
        if "loss_curve" in m:
            steps = list(range(0, len(m["loss_curve"]) * 100, 100))[:len(m["loss_curve"])]
            losses = m["loss_curve"]
            ax.plot(steps, losses, label=labels[model_key], color=colors[model_key],
                    linewidth=2, alpha=0.85)

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Loss (running avg)", fontsize=11)
    ax.set_title("Experiment 1: Training Loss Curve (50 episodes, 3000 steps)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve: {output_path}")


def plot_summary_bar(results: dict, output_path: Path):
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
        ax.set_title(f"{name}\nDelta={delta:+.3f} ({delta_pct:+.1f}%)", fontsize=10)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(with_v, without_v) * 1.18)

    delta_mse = results["summary"]["delta_mse"]
    fig.suptitle(
        f"Experiment 1: Voxel Trajectory Ablation (LeRobot Aloha, 50 ep, 3000 steps, GPU)\n"
        f"Voxel condition improves Action MSE by {abs(delta_mse / results['without_voxel']['mse'] * 100):.1f}%",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary bar: {output_path}")


def plot_dashboard(results: dict, output_path: Path):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)
    ax_loss = fig.add_subplot(gs[0, :3])
    for model_key, color, label in [
        ("with_voxel", "#2ecc71", "With Voxel (TDP)"),
        ("without_voxel", "#e74c3c", "Without Voxel (DP baseline)"),
    ]:
        m = results.get(model_key, {})
        if "loss_curve" in m:
            steps = list(range(0, len(m["loss_curve"]) * 100, 100))[:len(m["loss_curve"])]
            ax_loss.plot(steps, m["loss_curve"], label=label, color=color, linewidth=2)
    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss (Real LeRobot Aloha data)", fontweight="bold")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    # 自动调整 y 轴以显示数据
    all_losses = []
    for model_key in ["with_voxel", "without_voxel"]:
        m = results.get(model_key, {})
        if "loss_curve" in m:
            all_losses.extend(m["loss_curve"])
    if all_losses:
        ax_loss.set_ylim(min(all_losses) * 0.95, max(all_losses) * 1.05)

    ax_text = fig.add_subplot(gs[0, 3])
    ax_text.axis("off")
    delta_mse = results["summary"]["delta_mse"]
    delta_pct = abs(delta_mse / results["without_voxel"]["mse"] * 100)
    txt = (
        "Key Results\n"
        "================\n"
        f"Data: LeRobot Aloha sim\n"
        f"Episodes: 50\n"
        f"Steps: 3000 (best)\n\n"
        f"With Voxel:\n"
        f"  Action MSE:  {results['with_voxel']['mse']:.4f}\n"
        f"  Final Loss:  {results['with_voxel']['final_loss']:.4f}\n\n"
        f"Without Voxel:\n"
        f"  Action MSE:  {results['without_voxel']['mse']:.4f}\n"
        f"  Final Loss:  {results['without_voxel']['final_loss']:.4f}\n\n"
        f"Voxel improves MSE by:\n"
        f"  {delta_pct:.1f}% (Delta={delta_mse:+.4f})\n"
    )
    ax_text.text(0.05, 0.5, txt, fontsize=10, family="monospace",
                 verticalalignment="center")

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

    fig.suptitle("Experiment 1: Voxel Trajectory Ablation (Real LeRobot Aloha, GPU)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/exp1.json")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.input).stem
    plot_loss_curves_from_json(results, out_dir / f"{stem}_loss_curve.png")
    plot_summary_bar(results, out_dir / f"{stem}_summary.png")
    plot_dashboard(results, out_dir / f"{stem}_dashboard.png")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()