"""
Experiment 1: 体素轨迹条件 vs 无条件消融

论文 Table I 的核心对比：
  - Diffusion Policy (DP)       — 仅图像条件
  - Trajectory-conditioned DP   — 图像 + 体素轨迹条件
  - VLM-TDP                     — VLM 生成轨迹 + 上述

我们要验证 (论文 Table I 的核心发现):
  "Both TDP and VLM-TDP outperformed diffusion policy in all tasks"
  即: 给策略加体素轨迹条件 → 成功率提升

本实验:
  - 训两个模型: use_voxel=True (with-traj) vs use_voxel=False (img-only)
  - 同数据、同超参、同训练步数
  - 评测在 ManiSkill PickCube-v0 (或 mock 环境)

数据来源:
  - 真实: ManiSkill PickCube-v0 演示 (需 ManiSkill)
  - 假数据: make_synthetic_h5 自动生成 (默认)

Usage:
  # 用合成数据快速跑通
  python scripts/exp1_traj_ablation.py --synthetic

  # 用真实 ManiSkill 数据
  python scripts/exp1_traj_ablation.py --h5_dir data/maniskill/PickCube-v0/

  # 调整训练规模
  python scripts/exp1_traj_ablation.py --synthetic --steps 2000 --eval_episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 确保 src 在 path 里
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import FlowTDP, LightVoxelEncoder
from data import ManiSkillDemoDataset, DatasetConfig, make_synthetic_h5
from models.flow_tdp import create_flow_tdp


# ---------------------------------------------------------------------------
# 训练 / 评测 工具
# ---------------------------------------------------------------------------


def _safe_device() -> torch.device:
    """Pick best available device. Skip CUDA if sm capability not supported."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        # Run a real-ish kernel chain including backward to confirm GPU works
        a = torch.randn(64, 64, device="cuda", requires_grad=True)
        b = (a @ a).sum()
        b.backward()
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as e:
        import warnings
        warnings.warn(f"CUDA present but backward failed ({e}); falling back to CPU")
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def train_one_model(
    model: FlowTDP,
    encoder: LightVoxelEncoder,
    dataset: ManiSkillDemoDataset,
    *,
    steps: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: torch.device,
    log_every: int = 200,
    use_voxel: bool = True,
) -> List[float]:
    """
    训练 FlowTDP 模型; 返回每 step 的 loss 列表。
    """
    model.to(device).train()
    encoder.to(device).eval()  # 体素编码器冻结

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []

    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            images = batch["image"].to(device)               # (B, 3, H, W)
            voxel = batch["voxel_trajectory"].to(device)      # (B, 6, 6, 6)
            actions = batch["action"].to(device)             # (B, T, action_dim)

            # 体素编码
            with torch.no_grad():
                if use_voxel:
                    voxel_feat = encoder(voxel.float())      # (B, 128)
                else:
                    voxel_feat = torch.zeros(images.shape[0], 128, device=device)

            # 把 (B, T, action_dim) reshape 为 (B*T, action_dim) — 论文的做法:
            # 对每个预测窗口内的每个 timestep 独立求 loss
            B, T, D = actions.shape
            actions_flat = actions.reshape(B * T, D)

            # Flow Matching loss
            timestep = torch.rand(B * T, device=device)
            loss = model.compute_flow_matching_loss(
                images.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, *images.shape[1:]),
                voxel_feat.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1),
                actions_flat,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % log_every == 0:
                avg = np.mean(losses[-log_every:])
                print(f"    step {step:5d}/{steps}  loss={avg:.4f}")

    return losses


@torch.no_grad()
def evaluate_model(
    model: FlowTDP,
    encoder: LightVoxelEncoder,
    dataset: ManiSkillDemoDataset,
    *,
    num_episodes: int = 20,
    device: torch.device,
    use_voxel: bool = True,
) -> Dict[str, float]:
    """
    简单评测: 让策略在 dataset 的前 N 条 demo 上做预测, 用 action 预测的 MSE 作为 proxy。
    (在 ManiSkill 真实环境上跑 episode 需要 env 接口, 这里用 proxy metric)
    """
    model.to(device).eval()
    encoder.to(device).eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    total_mse = 0.0
    n = 0
    inference_times = []

    for i, batch in enumerate(loader):
        if i >= num_episodes:
            break
        images = batch["image"].to(device)
        voxel = batch["voxel_trajectory"].to(device)
        gt_action = batch["action"].to(device)              # (1, T, D)

        with torch.no_grad():
            if use_voxel:
                voxel_feat = encoder(voxel.float())
            else:
                voxel_feat = torch.zeros(1, 128, device=device)

            # 测单步推理时间
            t0 = time.time()
            pred_action = model.sample_action(images, voxel_feat, num_steps=1)
            inference_times.append((time.time() - t0) * 1000)  # ms

        mse = ((pred_action - gt_action[:, 0]) ** 2).mean().item()
        total_mse += mse
        n += 1

    return {
        "mse": total_mse / max(n, 1),
        "inference_ms_mean": float(np.mean(inference_times)) if inference_times else 0.0,
        "inference_ms_std": float(np.std(inference_times)) if inference_times else 0.0,
        "num_evaluated": n,
    }


def run_experiment(
    h5_paths: List[str],
    *,
    steps: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-4,
    eval_episodes: int = 20,
    image_size: tuple = (128, 128),
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    跑完整实验 (训两个模型, 评测对比)。
    """
    set_seed(seed)
    if device is None:
        device = _safe_device()

    print(f"\n{'='*70}")
    print(f"Experiment 1: Voxel Trajectory Ablation")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Steps: {steps}, batch_size: {batch_size}, lr: {lr}")
    print(f"  Image size: {image_size}")
    print(f"  Demos: {len(h5_paths)}")
    print(f"  Eval episodes: {eval_episodes}")

    # 配置
    cfg = DatasetConfig(
        h5_paths=h5_paths,
        image_size=image_size,
        prediction_horizon=12,
        workspace_bounds=(-0.3, -0.3, 0.0, 0.3, 0.3, 0.5),
        voxel_repr="sequence",
        use_fk="fallback",  # 自动降级到 fallback (ManiSkill 可能未装)
    )

    dataset = ManiSkillDemoDataset(cfg)
    print(f"  Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot train.")

    # 模型
    encoder = LightVoxelEncoder()

    results = {}

    # ---- 实验组 1: With voxel trajectory ----
    print(f"\n--- Training: FlowTDP WITH voxel trajectory ---")
    set_seed(seed)
    model_with = create_flow_tdp({"image": {"height": image_size[0], "width": image_size[1]},
                                    "voxel": {"feature_dim": 128},
                                    "robot": {"action_dim": 7, "gripper_dim": 1}},
                                   use_voxel=True)
    t0 = time.time()
    losses_with = train_one_model(
        model_with, encoder, dataset,
        steps=steps, batch_size=batch_size, lr=lr,
        device=device, use_voxel=True,
    )
    train_time_with = time.time() - t0

    print(f"\n  Evaluating with-voxel model on {eval_episodes} episodes...")
    eval_with = evaluate_model(model_with, encoder, dataset,
                                num_episodes=eval_episodes, device=device, use_voxel=True)
    results["with_voxel"] = {
        "final_loss": float(np.mean(losses_with[-100:])) if losses_with else None,
        "train_time_s": train_time_with,
        **eval_with,
    }

    # ---- 对照组: No voxel trajectory (image only) ----
    print(f"\n--- Training: FlowTDP WITHOUT voxel trajectory (image only baseline) ---")
    set_seed(seed)
    model_without = create_flow_tdp({"image": {"height": image_size[0], "width": image_size[1]}},
                                      use_voxel=False)
    t0 = time.time()
    losses_without = train_one_model(
        model_without, encoder, dataset,
        steps=steps, batch_size=batch_size, lr=lr,
        device=device, use_voxel=False,
    )
    train_time_without = time.time() - t0

    print(f"\n  Evaluating no-voxel model on {eval_episodes} episodes...")
    eval_without = evaluate_model(model_without, encoder, dataset,
                                   num_episodes=eval_episodes, device=device, use_voxel=False)
    results["without_voxel"] = {
        "final_loss": float(np.mean(losses_without[-100:])) if losses_without else None,
        "train_time_s": train_time_without,
        **eval_without,
    }

    # ---- 汇总 ----
    delta_loss = results["without_voxel"]["final_loss"] - results["with_voxel"]["final_loss"]
    delta_mse = results["without_voxel"]["mse"] - results["with_voxel"]["mse"]

    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}")
    print(f"  Metric            With-Voxel    Without-Voxel    Delta")
    print(f"  ----------------  ------------  ---------------  ----------")
    print(f"  Final Loss        {results['with_voxel']['final_loss']:.4f}        "
          f"{results['without_voxel']['final_loss']:.4f}        "
          f"{delta_loss:+.4f}")
    print(f"  Action MSE        {results['with_voxel']['mse']:.4f}        "
          f"{results['without_voxel']['mse']:.4f}        "
          f"{delta_mse:+.4f}")
    print(f"  Inference (ms)    {results['with_voxel']['inference_ms_mean']:.2f}          "
          f"{results['without_voxel']['inference_ms_mean']:.2f}          ")
    print(f"  Train Time (s)    {results['with_voxel']['train_time_s']:.1f}        "
          f"{results['without_voxel']['train_time_s']:.1f}")

    results["summary"] = {
        "delta_loss": delta_loss,
        "delta_mse": delta_mse,
        "voxel_helps": delta_mse > 0,  # with-voxel MSE 应该更低 (小更好)
    }

    return results


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: voxel trajectory ablation")
    parser.add_argument("--h5_dir", type=str, default=None,
                        help="Directory containing ManiSkill .h5 demos. "
                             "If not set, uses --synthetic.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic .h5 demos for testing (no ManiSkill needed).")
    parser.add_argument("--num_synthetic", type=int, default=10,
                        help="Number of synthetic demos to generate.")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Training steps per model.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/exp1_traj_ablation.json")
    args = parser.parse_args()

    # 准备数据
    if args.h5_dir:
        h5_paths = sorted(str(p) for p in Path(args.h5_dir).glob("*.h5"))
        if not h5_paths:
            print(f"No .h5 files found in {args.h5_dir}; falling back to synthetic.")
            args.synthetic = True

    if args.synthetic or not args.h5_dir:
        tmpdir = Path(tempfile.mkdtemp(prefix="exp1_synth_"))
        h5_paths = []
        for i in range(args.num_synthetic):
            p = tmpdir / f"synth_demo_{i:04d}.h5"
            make_synthetic_h5(str(p), T=80, image_hw=(200, 300))
            h5_paths.append(str(p))
        print(f"Generated {len(h5_paths)} synthetic demos at {tmpdir}")

    # 跑实验
    image_size = tuple(args.image_size)
    results = run_experiment(
        h5_paths,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_episodes=args.eval_episodes,
        image_size=image_size,
        seed=args.seed,
    )

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()