"""
FlowTDP Module
使用Conditional Flow Matching替代DDPM的策略模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
from einops import rearrange


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) 时间步
        Returns:
            pos_embed: (B, dim) 位置编码
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CrossAttention(nn.Module):
    """交叉注意力模块"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) query
            context: (B, M, D) key/value
            mask: (B, N, M) attention mask
        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape
        if context is None:
            context = x
        M = context.shape[1]

        Q = self.q_proj(x).view(B, N, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(context).view(B, M, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(context).view(B, M, self.num_heads, -1).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = (attn @ V).transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)

        return self.norm(output + x)


class FlowTDPBlock(nn.Module):
    """FlowTDP的Transformer块"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = CrossAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) 输入特征
            condition: (B, C, D) 条件特征（轨迹特征）
        Returns:
            output: (B, N, D) 输出特征
        """
        # 自注意力
        x = self.self_attn(x)

        # 条件交叉注意力
        if condition is not None:
            x = self.cross_attn(x, condition)

        # MLP
        x = x + self.mlp(self.norm(x))

        return x


class FlowTDP(nn.Module):
    """
    Conditional Flow Matching策略模型

    使用条件流匹配替代DDPM扩散模型，实现单步推理

    核心改进：
    1. 预测速度场而非噪声
    2. 条件流匹配损失
    3. 单步推理生成动作

    输入：
        - 图像: (B, 3, 600, 800)
        - 体素轨迹特征: (B, 128)
        - 时间步t: (B,)

    输出：
        - 机器人动作: (B, action_dim)

    Args:
        use_voxel: 是否使用体素轨迹作为条件输入。
          - True (默认): 等价于论文 TDP，使用 6×6×6 voxel 编码作为额外条件
          - False: 退化为 baseline diffusion policy（仅图像条件），用于消融实验
    """

    def __init__(
        self,
        image_channels: int = 3,
        image_size: tuple = (600, 800),
        trajectory_dim: int = 128,
        action_dim: int = 8,  # 7关节 + 1夹爪
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_voxel: bool = True,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_voxel = use_voxel

        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 轨迹特征投影（仅在 use_voxel=True 时使用）
        self.trajectory_proj = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 时间编码
        self.time_embedding = SinusoidalPositionEmbedding(hidden_dim)

        # Transformer处理层
        self.transformer = nn.ModuleList([
            FlowTDPBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 动作投影头
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 限制动作范围到[-1, 1]
        )

        # 速度场预测头
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码图像

        Args:
            image: (B, C, H, W)

        Returns:
            image_features: (B, hidden_dim)
        """
        return self.image_encoder(image)

    def forward(
        self,
        image: torch.Tensor,
        trajectory_features: torch.Tensor,
        timestep: torch.Tensor,
        target_action: Optional[torch.Tensor] = None,
        mode: str = "action"
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            image: (B, C, H, W) 输入图像
            trajectory_features: (B, trajectory_dim) 轨迹特征
            timestep: (B,) 时间步（Flow Matching需要）
            target_action: (B, action_dim) 目标动作（训练时需要）
            mode: "action" 或 "velocity"

        Returns:
            {
                "action": (B, action_dim) 预测动作,
                "velocity": (B, action_dim) 预测速度场,
                "features": (B, hidden_dim) 特征
            }
        """
        batch_size = image.shape[0]

        # 编码图像
        image_features = self.encode_image(image)  # (B, hidden_dim)

        # 投影轨迹特征（如果启用）
        if self.use_voxel:
            trajectory_proj = self.trajectory_proj(trajectory_features)  # (B, hidden_dim)
        else:
            # 不使用 voxel 时, 用零向量代替 condition (保留模块结构)
            trajectory_proj = torch.zeros_like(image_features)

        # 时间编码
        time_embed = self.time_embedding(timestep)  # (B, hidden_dim)

        # 融合特征
        x = image_features + time_embed
        x = x.unsqueeze(1)  # (B, 1, hidden_dim)
        condition = trajectory_proj.unsqueeze(1)  # (B, 1, hidden_dim)

        # Transformer处理
        for block in self.transformer:
            x = block(x, condition)

        features = x.squeeze(1)  # (B, hidden_dim)

        # 预测
        action = self.action_head(features)
        velocity = self.velocity_head(features)

        return {
            "action": action,
            "velocity": velocity,
            "features": features
        }

    def sample_action(
        self,
        image: torch.Tensor,
        trajectory_features: torch.Tensor,
        num_steps: int = 1,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Flow Matching采样（Euler 积分）

        从 t=1 的高斯噪声出发，沿速度场反向 Euler 积分到 t=0，得到动作。

        Args:
            image: (B, C, H, W)
            trajectory_features: (B, trajectory_dim)
            num_steps: Euler 步数（论文默认 1 步即可）
            guidance_scale: CFG 强度（当前实现为占位，恒为 1.0）

        Returns:
            action: (B, action_dim)
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        batch_size = image.shape[0]
        device = image.device

        # 从 x_1 ~ N(0, I) 出发
        x = torch.randn(batch_size, self.action_dim, device=device)
        dt = 1.0 / num_steps

        with torch.no_grad():
            for k in range(num_steps, 0, -1):
                t = torch.full((batch_size,), k * dt, device=device)
                v = self.forward(
                    image, trajectory_features, t, mode="velocity"
                )["velocity"]
                # x_{t - dt} = x_t - v(x_t, t) * dt
                x = x - v * dt * guidance_scale

        return x

    def compute_flow_matching_loss(
        self,
        image: torch.Tensor,
        trajectory_features: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        计算条件流匹配损失

        Loss = E[t, x0, y] || v_theta(x_t, t, y) - u_t(x_t|x0) ||^2

        其中 u_t(x_t|x0) = x_t - x0 是最优速度场

        Args:
            image: (B, C, H, W)
            trajectory_features: (B, trajectory_dim)
            action: (B, action_dim) 目标动作

        Returns:
            loss: 流匹配损失
        """
        batch_size = action.shape[0]
        device = action.device

        # 采样时间步
        timestep = torch.rand(batch_size, device=device)

        # 生成噪声动作
        noise = torch.randn_like(action)
        noisy_action = (1 - timestep.view(-1, 1)) * action + timestep.view(-1, 1) * noise

        # 最优速度场
        optimal_velocity = noise - action

        # 预测速度场（fp32；fp16 autocast 在某些 GPU 上会触发 dtype 不匹配, 故默认关闭）
        output = self.forward(image, trajectory_features, timestep, mode="velocity")
        predicted_velocity = output["velocity"]

        # MSE损失
        loss = F.mse_loss(predicted_velocity, optimal_velocity)

        return loss


def create_flow_tdp(config: Dict, use_voxel: bool = True) -> FlowTDP:
    """
    工厂函数：根据配置创建FlowTDP模型

    Args:
        config: 配置字典
        use_voxel: 是否使用体素轨迹条件（消融实验时设为 False）

    Returns:
        FlowTDP实例
    """
    image_config = config.get("image", {})
    voxel_config = config.get("voxel", {})
    robot_config = config.get("robot", {})

    return FlowTDP(
        image_channels=image_config.get("channels", 3),
        image_size=(image_config.get("height", 600), image_config.get("width", 800)),
        trajectory_dim=voxel_config.get("feature_dim", 128),
        action_dim=robot_config.get("action_dim", 7) + robot_config.get("gripper_dim", 1),
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_voxel=use_voxel,
    )


if __name__ == "__main__":
    # 测试代码
    model = FlowTDP()

    # 测试输入
    batch_size = 4
    image = torch.randn(batch_size, 3, 600, 800)
    trajectory_features = torch.randn(batch_size, 128)
    timestep = torch.rand(batch_size)
    action = torch.randn(batch_size, 8)

    # 前向传播
    output = model(image, trajectory_features, timestep, action)
    print(f"Action shape: {output['action'].shape}")
    print(f"Velocity shape: {output['velocity'].shape}")
    print(f"Features shape: {output['features'].shape}")

    # 测试流匹配损失
    loss = model.compute_flow_matching_loss(image, trajectory_features, action)
    print(f"Flow matching loss: {loss.item()}")

    # 测试采样
    sampled_action = model.sample_action(image, trajectory_features)
    print(f"Sampled action shape: {sampled_action.shape}")
    print(f"Action range: [{sampled_action.min():.3f}, {sampled_action.max():.3f}]")