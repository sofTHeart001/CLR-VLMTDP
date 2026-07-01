"""
Run Experiment 1 with REAL ManiSkill data (already extracted by scripts/extract_states.py).

用法:
    python scripts/exp1_real_data.py \
        --h5_path data/maniskill/PickCube-v1/with_images/trajectory.h5 \
        --steps 3000 \
        --eval_episodes 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from models import FlowTDP, LightVoxelEncoder
from models.flow_tdp import create_flow_tdp


def _safe_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        a = torch.randn(64, 64, device="cuda", requires_grad=True)
        _ = (a @ a).sum()
        _ = torch.autograd.grad(_ , a, retain_graph=False)[0]
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


class FlatH5Dataset(Dataset):
    """
    读取 ManiSkill 风格的 flat h5 文件:
      obs/agent/qpos:                  (T, 9)
      actions:                         (T, action_dim)
      obs/sensor_data/base_camera/rgb: (T, H, W, 3)
      voxel_trajectories_per_t:        (T, 6, 6, 6) [可选]

    Returns:
        {image, voxel_trajectory, action_window}
    """

    def __init__(
        self,
        h5_path: str,
        image_size: tuple = (96, 96),
        prediction_horizon: int = 12,
        voxel_repr: str = "sequence",
    ):
        with h5py.File(h5_path, "r") as f:
            self.qpos = f["obs/agent/qpos"][:]            # (T, 9)
            self.actions = f["actions"][:]                 # (T, 8)
            self.rgb = f["obs/sensor_data/base_camera/rgb"][:]  # (T, H, W, 3)
            if "voxel_trajectories_per_t" in f:
                self.voxel_traj = f["voxel_trajectories_per_t"][:]  # (T, 6, 6, 6)
            else:
                # 没有就全零
                T = self.qpos.shape[0]
                self.voxel_traj = np.zeros((T, 6, 6, 6), dtype=np.int64)

        self.T = self.qpos.shape[0]
        self.image_size = image_size
        self.prediction_horizon = prediction_horizon
        self.voxel_repr = voxel_repr

        # 样本数 = T - T + 1 (边界外不能取 T 长度的动作窗口)
        self.num_samples = max(0, self.T - prediction_horizon + 1)

    def __len__(self):
        return self.num_samples

    def _resize(self, img):
        from PIL import Image
        pil = Image.fromarray(img).resize(
            (self.image_size[1], self.image_size[0]), Image.BILINEAR
        )
        return np.array(pil, dtype=np.uint8)

    def __getitem__(self, idx):
        img = self._resize(self.rgb[idx])
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        vt = self.voxel_traj[idx]
        if self.voxel_repr == "sequence":
            vt_t = torch.from_numpy(vt).long()
        else:
            vt_t = (torch.from_numpy(vt) > 0).float()

        T_h = self.prediction_horizon
        action_window = self.actions[idx: idx + T_h]
        if action_window.shape[0] < T_h:
            pad = np.zeros((T_h - action_window.shape[0], action_window.shape[1]))
            action_window = np.concatenate([action_window, pad], axis=0)
        action_t = torch.from_numpy(action_window).float()

        return {
            "image": img_t,
            "voxel_trajectory": vt_t,
            "action": action_t,
        }


def set_seed(seed: int = 42):
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
    dataset: FlatH5Dataset,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_voxel: bool,
    log_every: int = 50,
) -> List[float]:
    model.to(device).train()
    encoder.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []
    step = 0
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
                print(f"    step {step:5d}/{steps}  loss={avg:.4f}", flush=True)
                # Save per-step loss to JSONL for plotting
                with open("results/exp1_real_data_losses.jsonl", "a") as f:
                    f.write(json.dumps({
                        "model": "with_voxel" if use_voxel else "without_voxel",
                        "step": step,
                        "loss": loss.item(),
                        "running_avg": float(avg),
                    }) + "\n")
    return losses


@torch.no_grad()
def evaluate_model(
    model: FlowTDP,
    encoder: LightVoxelEncoder,
    dataset: FlatH5Dataset,
    *,
    num_episodes: int,
    device: torch.device,
    use_voxel: bool,
) -> dict:
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
        "inference_ms_std": float(np.std(inference_times)) if inference_times else 0.0,
        "num_evaluated": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path",
                        default="D:/Desktop/github_project/CLR-VLMTDP/data/maniskill/PickCube-v1/with_images/trajectory.h5")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--image_size", type=int, nargs=2, default=[96, 96])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/exp1_real_data.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = _safe_device()

    print("=" * 70)
    print("Experiment 1 (Real ManiSkill data)")
    print("=" * 70)
    print(f"  Data: {args.h5_path}")
    print(f"  Steps: {args.steps}, batch: {args.batch_size}, lr: {args.lr}")
    print(f"  Image: {args.image_size}, Device: {device}")

    dataset = FlatH5Dataset(
        args.h5_path,
        image_size=tuple(args.image_size),
        prediction_horizon=12,
    )
    print(f"  Dataset: {len(dataset)} samples (from {dataset.T} timesteps)")

    encoder = LightVoxelEncoder()
    results = {}

    print("\n--- Training: FlowTDP WITH voxel trajectory ---")
    set_seed(args.seed)
    model_with = create_flow_tdp(
        {"image": {"height": args.image_size[0], "width": args.image_size[1]},
         "voxel": {"feature_dim": 128},
         "robot": {"action_dim": 7, "gripper_dim": 1}},
        use_voxel=True,
    )
    t0 = time.time()
    losses_with = train_one_model(model_with, encoder, dataset,
                                   steps=args.steps, batch_size=args.batch_size,
                                   lr=args.lr, device=device, use_voxel=True)
    train_time_with = time.time() - t0

    print("\n  Evaluating with-voxel...")
    eval_with = evaluate_model(model_with, encoder, dataset,
                                num_episodes=args.eval_episodes, device=device, use_voxel=True)
    results["with_voxel"] = {
        "final_loss": float(np.mean(losses_with[-100:])) if losses_with else None,
        "train_time_s": train_time_with,
        **eval_with,
    }

    print("\n--- Training: FlowTDP WITHOUT voxel trajectory ---")
    set_seed(args.seed)
    model_without = create_flow_tdp(
        {"image": {"height": args.image_size[0], "width": args.image_size[1]},
         "voxel": {"feature_dim": 128},
         "robot": {"action_dim": 7, "gripper_dim": 1}},
        use_voxel=False,
    )
    t0 = time.time()
    losses_without = train_one_model(model_without, encoder, dataset,
                                      steps=args.steps, batch_size=args.batch_size,
                                      lr=args.lr, device=device, use_voxel=False)
    train_time_without = time.time() - t0

    print("\n  Evaluating without-voxel...")
    eval_without = evaluate_model(model_without, encoder, dataset,
                                   num_episodes=args.eval_episodes, device=device, use_voxel=False)
    results["without_voxel"] = {
        "final_loss": float(np.mean(losses_without[-100:])) if losses_without else None,
        "train_time_s": train_time_without,
        **eval_without,
    }

    delta_mse = results["without_voxel"]["mse"] - results["with_voxel"]["mse"]
    delta_loss = results["without_voxel"]["final_loss"] - results["with_voxel"]["final_loss"]
    delta_inf = results["without_voxel"]["inference_ms_mean"] - results["with_voxel"]["inference_ms_mean"]

    print("\n" + "=" * 70)
    print("Results (Real ManiSkill data)")
    print("=" * 70)
    print(f"  Metric            With-Voxel    Without-Voxel    Delta")
    print(f"  ----------------  ------------  ---------------  ----------")
    print(f"  Final Loss        {results['with_voxel']['final_loss']:.4f}        "
          f"{results['without_voxel']['final_loss']:.4f}        "
          f"{delta_loss:+.4f}")
    print(f"  Action MSE        {results['with_voxel']['mse']:.4f}        "
          f"{results['without_voxel']['mse']:.4f}        "
          f"{delta_mse:+.4f}")
    print(f"  Inference (ms)    {results['with_voxel']['inference_ms_mean']:.2f}          "
          f"{results['without_voxel']['inference_ms_mean']:.2f}          "
          f"{delta_inf:+.2f}")
    print(f"  Train Time (s)    {results['with_voxel']['train_time_s']:.1f}        "
          f"{results['without_voxel']['train_time_s']:.1f}")

    results["summary"] = {
        "delta_loss": delta_loss,
        "delta_mse": delta_mse,
        "voxel_helps": delta_mse > 0,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()