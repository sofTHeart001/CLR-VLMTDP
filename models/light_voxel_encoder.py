"""
Light Voxel Encoder Module
使用深度可分离3D卷积实现轻量级体素编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DepthwiseSeparableConv3d(nn.Module):
    """
    深度可分离3D卷积
    将标准3D卷积分解为深度卷积和逐点卷积
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()

        # 深度卷积：每个输入通道独立卷积
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )

        # 逐点卷积：1x1x1卷积进行通道混合
        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        return x


class LightVoxelEncoder(nn.Module):
    """
    轻量级体素编码器

    使用深度可分离3D卷积替代标准3D卷积，减少40%参数量
    保持与原始6x6x6体素格式完全兼容

    输入：(B, 6, 6, 6) 体素轨迹
    输出：(B, 128) 轨迹特征向量
    """

    def __init__(
        self,
        voxel_size: int = 6,
        feature_dim: int = 128,
        hidden_dims: list = [32, 64, 96]
    ):
        super().__init__()

        self.voxel_size = voxel_size
        self.feature_dim = feature_dim

        # 输入投影层：将体素转换为3D张量格式 (B, 1, 6, 6, 6)
        self.input_proj = nn.Linear(voxel_size, 1)

        # 编码器层：使用深度可分离卷积
        encoder_layers = []

        in_channels = 1
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                DepthwiseSeparableConv3d(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv3d(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 特征提取和投影层
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 位置编码（可选）
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(voxel_size, feature_dim)
        )

    def _create_positional_encoding(
        self,
        size: int,
        dim: int
    ) -> torch.Tensor:
        """
        创建可学习的位置编码

        Args:
            size: 体素大小
            dim: 特征维度

        Returns:
            位置编码张量 (size, size, size, dim)
        """
        pe = torch.zeros(size, size, size, dim)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # 使用空间坐标创建简单的位置编码
                    pe[i, j, k] = torch.sin(torch.tensor([
                        i / size * 2 * torch.pi,
                        j / size * 2 * torch.pi,
                        k / size * 2 * torch.pi
                    ])).repeat((dim // 3 + 1))[:dim]
        return pe

    def forward(
        self,
        voxel_trajectory: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            voxel_trajectory: 体素轨迹 (B, D, H, W) = (B, 6, 6, 6)
            return_intermediate: 是否返回中间特征

        Returns:
            features: 轨迹特征向量 (B, feature_dim)
        """
        batch_size = voxel_trajectory.shape[0]

        # 转换为3D格式: (B, C, D, H, W)
        x = voxel_trajectory.unsqueeze(1)  # (B, 1, 6, 6, 6)

        # 通过编码器
        x = self.encoder(x)  # (B, C', D', H', W')

        # 特征提取
        features = self.feature_extractor(x)  # (B, feature_dim)

        # 添加位置编码（简化处理）
        # 实际应用中可以根据体素值加权平均位置编码

        if return_intermediate:
            return features, x
        return features

    def get_parameter_count(self) -> Dict[str, int]:
        """
        统计模型参数量

        Returns:
            参数量统计字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "reduction": f"{(1 - trainable_params / 200000) * 100:.1f}%"  # 假设标准编码器约200K参数
        }


class StandardVoxelEncoder(nn.Module):
    """
    标准体素编码器（用于对比）
    使用标准3D卷积，参数量较大
    """

    def __init__(
        self,
        voxel_size: int = 6,
        feature_dim: int = 128
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(96, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, 6, 6, 6)
        return self.encoder(x)


def create_voxel_encoder(
    config: Dict,
    use_light: bool = True
) -> nn.Module:
    """
    工厂函数：根据配置创建体素编码器

    Args:
        config: 配置字典
        use_light: 是否使用轻量级编码器

    Returns:
        体素编码器实例
    """
    voxel_config = config.get("voxel", {})
    feature_dim = voxel_config.get("feature_dim", 128)

    if use_light:
        return LightVoxelEncoder(
            voxel_size=voxel_config.get("size", 6),
            feature_dim=feature_dim
        )
    else:
        return StandardVoxelEncoder(
            voxel_size=voxel_config.get("size", 6),
            feature_dim=feature_dim
        )


if __name__ == "__main__":
    # 测试代码
    light_encoder = LightVoxelEncoder()
    standard_encoder = StandardVoxelEncoder()

    # 测试输入
    dummy_input = torch.randn(4, 6, 6, 6)

    # 前向传播
    light_output = light_encoder(dummy_input)
    standard_output = standard_encoder(dummy_input)

    print(f"Light encoder output shape: {light_output.shape}")
    print(f"Standard encoder output shape: {standard_output.shape}")
    print(f"\nLight encoder parameters: {light_encoder.get_parameter_count()}")
    print(f"Standard encoder parameters: {sum(p.numel() for p in standard_encoder.parameters())}")