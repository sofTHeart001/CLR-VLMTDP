"""
FlowTDP Training Script
使用条件流匹配训练策略模型
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from pathlib import Path

from models import FlowTDP, LightVoxelEncoder
from utils import PromptTemplate


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    """
    创建训练和验证数据集

    TODO: 根据实际数据集格式实现
    """
    from torch.utils.data import Dataset

    class RLBenchDataset(Dataset):
        """RLBench演示数据集"""

        def __init__(self, data_dir: str, split: str = "train"):
            self.data_dir = Path(data_dir)
            self.split = split
            # TODO: 加载实际数据

        def __len__(self):
            # TODO: 返回实际数据集大小
            return 1000

        def __getitem__(self, idx):
            # TODO: 返回 (image, voxel_trajectory, action)
            return {
                "image": torch.randn(3, 600, 800),
                "voxel_trajectory": torch.randn(6, 6, 6),
                "action": torch.randn(8)
            }

    data_dir = config["paths"]["data_dir"]
    train_dataset = RLBenchDataset(data_dir, split="train")
    val_dataset = RLBenchDataset(data_dir, split="val")

    return train_dataset, val_dataset


def train_epoch(
    model: FlowTDP,
    voxel_encoder: LightVoxelEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    config: dict
) -> dict:
    """训练一个epoch"""
    model.train()
    voxel_encoder.eval()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # 转移数据到设备
        images = batch["image"].to(device)
        voxel_trajectories = batch["voxel_trajectory"].to(device)
        actions = batch["action"].to(device)

        # 编码体素轨迹
        with torch.no_grad():
            trajectory_features = voxel_encoder(voxel_trajectories)

        # 计算流匹配损失
        loss = model.compute_flow_matching_loss(
            images,
            trajectory_features,
            actions
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录
        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({"loss": loss.item()})

        # 日志记录
        if num_batches % config["logging"]["log_every"] == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/epoch": epoch
            })

    # 更新学习率
    scheduler.step()

    return {
        "avg_loss": total_loss / num_batches
    }


@torch.no_grad()
def validate(
    model: FlowTDP,
    voxel_encoder: LightVoxelEncoder,
    dataloader: DataLoader,
    device: torch.device,
    config: dict
) -> dict:
    """验证模型"""
    model.eval()
    voxel_encoder.eval()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        images = batch["image"].to(device)
        voxel_trajectories = batch["voxel_trajectory"].to(device)
        actions = batch["action"].to(device)

        trajectory_features = voxel_encoder(voxel_trajectories)
        loss = model.compute_flow_matching_loss(
            images,
            trajectory_features,
            actions
        )

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"val_loss": loss.item()})

    return {
        "avg_loss": total_loss / num_batches
    }


def save_checkpoint(
    model: FlowTDP,
    voxel_encoder: LightVoxelEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    config: dict
):
    """保存检查点"""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "voxel_encoder_state_dict": voxel_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": config
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train FlowTDP model")
    parser.add_argument("--config", type=str, default="config/train.yaml",
                        help="Path to training config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    device = torch.device(f"cuda:{config['device']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化wandb
    if not args.no_wandb:
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"].get("wandb_entity"),
            config=config,
            name="flow_tdp_training"
        )

    # 创建模型
    model = FlowTDP().to(device)
    voxel_encoder = LightVoxelEncoder().to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Voxel encoder parameters: {sum(p.numel() for p in voxel_encoder.parameters()):,}")

    # 创建数据集和数据加载器
    train_dataset, val_dataset = create_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["flow_matching"]["batch_size"],
        shuffle=True,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["flow_matching"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )

    # 创建优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["flow_matching"]["learning_rate"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
        betas=config["training"]["optimizer"]["betas"]
    )

    num_training_steps = config["training"]["flow_matching"]["num_training_steps"]
    warmup_steps = config["training"]["flow_matching"]["warmup_steps"]

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=config["training"]["flow_matching"]["learning_rate"] * 0.01
    )

    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        voxel_encoder.load_state_dict(checkpoint["voxel_encoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # 训练循环
    num_epochs = num_training_steps // len(train_loader)
    best_val_loss = float("inf")

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs):
        # 训练
        train_metrics = train_epoch(
            model, voxel_encoder, train_loader,
            optimizer, scheduler, device, epoch, config
        )

        print(f"Epoch {epoch} - Train Loss: {train_metrics['avg_loss']:.4f}")

        # 验证
        if config["validation"]["enabled"] and epoch % (config["validation"]["val_interval"] // len(train_loader)) == 0:
            val_metrics = validate(
                model, voxel_encoder, val_loader, device, config
            )
            print(f"Epoch {epoch} - Val Loss: {val_metrics['avg_loss']:.4f}")

            if not args.no_wandb:
                wandb.log({
                    "val/loss": val_metrics["avg_loss"],
                    "val/epoch": epoch
                })

            # 保存最佳模型
            if val_metrics["avg_loss"] < best_val_loss:
                best_val_loss = val_metrics["avg_loss"]
                save_checkpoint(
                    model, voxel_encoder, optimizer, scheduler, epoch,
                    best_val_loss, Path(config["paths"]["checkpoint_dir"]) / "best.pt",
                    config
                )

        # 保存检查点
        if epoch % (config["checkpointing"]["save_every"] // len(train_loader)) == 0:
            save_checkpoint(
                model, voxel_encoder, optimizer, scheduler, epoch,
                train_metrics["avg_loss"], config["paths"]["checkpoint_dir"],
                config
            )

    # 保存最终模型
    save_checkpoint(
        model, voxel_encoder, optimizer, scheduler, num_epochs - 1,
        train_metrics["avg_loss"], Path(config["paths"]["checkpoint_dir"]) / "final.pt",
        config
    )

    print("Training completed!")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()