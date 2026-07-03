"""
Fast Experiment 1: voxel ablation on pre-computed LeRobot Aloha data.
使用预计算 h5 (FK 已算好, 训练时直接读).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from models import LightVoxelEncoder
from models.flow_tdp import create_flow_tdp
from data.lerobot_precomputed_dataset import LeRobotPrecomputedDataset
from scripts.exp1_real_data import _safe_device, set_seed


def train_one_model(
    model, encoder, dataset, *, steps, batch_size, lr, device, use_voxel, log_every=100,
):
    model.to(device).train()
    encoder.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []
    step = 0
    t0 = time.time()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            images = batch["image"].to(device)
            voxel = batch["voxel_trajectory"].to(device)
            actions = batch["action"].to(device)
            with torch.no_grad():
                if use_voxel:
                    voxel_feat = encoder(voxel.float())
                else:
                    voxel_feat = torch.zeros(images.shape[0], 128, device=device)
            B, T, D = actions.shape
            actions_flat = actions.reshape(B * T, D)
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
                print(f"    step {step:5d}/{steps}  loss={avg:.4f}  ({time.time()-t0:.1f}s)",
                      flush=True)
    return losses, time.time() - t0


@torch.no_grad()
def evaluate_model(model, encoder, dataset, *, num_episodes, device, use_voxel):
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
        gt_action = batch["action"].to(device)
        with torch.no_grad():
            if use_voxel:
                voxel_feat = encoder(voxel.float())
            else:
                voxel_feat = torch.zeros(1, 128, device=device)
            t0 = time.time()
            pred_action = model.sample_action(images, voxel_feat, num_steps=1)
            inference_times.append((time.time() - t0) * 1000)
        mse = ((pred_action - gt_action[:, 0]) ** 2).mean().item()
        total_mse += mse
        n += 1
    return {
        "mse": total_mse / max(n, 1),
        "inference_ms_mean": float(np.mean(inference_times)) if inference_times else 0.0,
        "num_evaluated": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", default="D:/Desktop/github_project/CLR-VLMTDP/data/lerobot_precomputed/with_voxels.h5")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/exp1_lerobot_v3.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = _safe_device()
    print(f"Device: {device}")

    print("=" * 70)
    print("Experiment 1 v3: LeRobot Aloha PRE-COMPUTED voxels (FAST)")
    print("=" * 70)
    print(f"  Steps: {args.steps}, batch: {args.batch_size}, num_episodes: {args.num_episodes}")

    dataset = LeRobotPrecomputedDataset(
        h5_path=args.h5_path, num_episodes=args.num_episodes,
    )

    encoder = LightVoxelEncoder()
    config = {
        "image": {"channels": 3, "height": 96, "width": 96},
        "voxel": {"feature_dim": 128},
        "robot": {"action_dim": 6, "gripper_dim": 1},
    }
    results = {"config": vars(args), "steps": args.steps}

    print("\n--- Training: FlowTDP WITH voxel ---")
    set_seed(args.seed)
    model_with = create_flow_tdp(config, use_voxel=True)
    losses_with, time_with = train_one_model(
        model_with, encoder, dataset,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        device=device, use_voxel=True,
    )
    print(f"  Final loss: {np.mean(losses_with[-100:]):.4f}  time: {time_with:.1f}s")
    eval_with = evaluate_model(model_with, encoder, dataset,
                                num_episodes=args.eval_episodes, device=device, use_voxel=True)
    results["with_voxel"] = {
        "final_loss": float(np.mean(losses_with[-100:])),
        "train_time_s": time_with,
        **eval_with,
    }

    print("\n--- Training: FlowTDP WITHOUT voxel ---")
    set_seed(args.seed)
    model_without = create_flow_tdp(config, use_voxel=False)
    losses_without, time_without = train_one_model(
        model_without, encoder, dataset,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        device=device, use_voxel=False,
    )
    print(f"  Final loss: {np.mean(losses_without[-100:]):.4f}  time: {time_without:.1f}s")
    eval_without = evaluate_model(model_without, encoder, dataset,
                                   num_episodes=args.eval_episodes, device=device, use_voxel=False)
    results["without_voxel"] = {
        "final_loss": float(np.mean(losses_without[-100:])),
        "train_time_s": time_without,
        **eval_without,
    }

    delta_mse = eval_without["mse"] - eval_with["mse"]
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"  Metric            With-Voxel    Without-Voxel    Delta")
    print(f"  ----------------  ------------  ---------------  ----------")
    print(f"  Action MSE        {eval_with['mse']:.4f}        "
          f"{eval_without['mse']:.4f}        {delta_mse:+.4f}")
    print(f"  Final Loss        {results['with_voxel']['final_loss']:.4f}        "
          f"{results['without_voxel']['final_loss']:.4f}        "
          f"{results['without_voxel']['final_loss']-results['with_voxel']['final_loss']:+.4f}")
    print(f"  Train Time (s)    {time_with:.0f}          {time_without:.0f}          "
          f"{(time_without-time_with)/max(time_with,0.1)*100:+.0f}%")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()