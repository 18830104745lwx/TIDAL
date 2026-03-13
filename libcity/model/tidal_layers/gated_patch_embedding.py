"""
模块② — 门控Patch嵌入
(Gated Patch Embedding)

核心思想:
  1. Patch 化: 沿时间维度将 T 个时间步分成 T/P 个 patch, 每个包含 P 步的信息
     → 降低序列长度, 每个 token 包含更丰富的局部时间上下文
  2. Cross Connection: 拼接最新时刻特征
  3. 门控机制: 自动学习哪些特征通道更重要
"""

import torch
import torch.nn as nn
from einops import rearrange


class GatedPatchEmbedding(nn.Module):
    """
    门控Patch嵌入

    参数:
        in_dim (int): 原始特征维度 (如 C=1)
        hid_dim (int): 隐藏层维度
        patch_size (int): Patch 大小, 将 P 个连续时间步合并为一个 token
        activation (str): 激活函数 'gelu' 或 'relu'

    输入:
        x: (B, T, N, C) - 原始输入
        latestX: (B, T, N, C) - 最新时刻特征 (复制 T 次)

    输出:
        out: (B, T', N, hid_dim) - Patch 嵌入, T' = T // patch_size
    """

    def __init__(self, in_dim, hid_dim, patch_size=2, activation='gelu'):
        super().__init__()
        self.patch_size = patch_size
        assert activation in ['gelu', 'relu']

        # Cross Connection: 拼接原始特征和最新时刻特征 → 2 × in_dim
        # Patch 化: 每个 patch 包含 P 个时间步 → 2 × in_dim × P
        patch_in_dim = in_dim * 2 * patch_size

        # Patch 投影: 将展平的 patch 映射到隐藏空间
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_in_dim, hid_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        # 通道门控: 自适应选择重要特征
        self.gate_net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Sigmoid()
        )

    def forward(self, x, latestX):
        """
        Args:
            x: (B, T, N, C) - 历史输入
            latestX: (B, T, N, C) - 最新时刻特征 (repeat T 次)

        Returns:
            out: (B, T', N, hid_dim) - T' = T // patch_size
        """
        B, T, N, C = x.shape
        P = self.patch_size

        # 处理 T 不是 patch_size 整数倍的情况: 尾部补零
        remainder = T % P
        if remainder != 0:
            pad_len = P - remainder
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
            latestX = torch.nn.functional.pad(latestX, (0, 0, 0, 0, 0, pad_len))

        # Cross Connection: 拼接原始和最新时刻
        x_cross = torch.cat([x, latestX], dim=-1)  # (B, T_padded, N, 2C)

        # Patch 化: 沿时间维度分组
        # (B, T, N, 2C) → (B, T', N, P*2C)
        patches = rearrange(x_cross, 'b (t p) n c -> b t n (p c)', p=P)

        # 投影到隐藏空间
        emb = self.patch_proj(patches)  # (B, T', N, hid_dim)

        # 门控: 自适应通道选择
        gate = self.gate_net(emb)  # (B, T', N, hid_dim), 值域 (0, 1)

        return emb * gate  # 门控输出
