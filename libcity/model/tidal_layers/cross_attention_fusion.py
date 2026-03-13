"""
模块③ — 交叉注意力门控融合
(Cross-Attention Gated Fusion)

核心思想:
  TIDAL 用可学习的门控网络为每种嵌入自适应分配权重:
    gate_i = sigmoid(W_i [data; emb_i])  ∈ (0,1)
    data = data + Σ gate_i * emb_i
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力门控融合

    为 TimeEmbedding、TimeLagEmbedding、SpatialEmbedding 各自学习一个门控权重,
    实现数据驱动的自适应嵌入融合.

    参数:
        hid_dim (int): 隐藏维度

    输入:
        data: (B, T', N, C) - Patch 编码后的主特征
        te:   (B, T', N, C) - 时间嵌入 (已扩展到节点维度)
        tle:  (B, T', N, C) - 时间滞后嵌入 (已扩展到节点维度)
        se:   (B, T', N, C) - 空间嵌入 (已广播)

    输出:
        fused: (B, T', N, C) - 融合后的特征
    """

    def __init__(self, hid_dim, use_gating=True):
        super().__init__()
        self.use_gating = use_gating  # 消融开关: False → 简单相加

        # 每种嵌入一个独立的门控网络
        self.gate_te = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )
        self.gate_tle = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )
        self.gate_se = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )

        # 融合后的 LayerNorm
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, data, te, tle, se):
        """
        Args:
            data: (B, T', N, C) - 主特征
            te:   (B, T', N, C) - 时间嵌入
            tle:  (B, T', N, C) - 时间滞后嵌入
            se:   (B, T', N, C) - 空间嵌入 (可广播)

        Returns:
            fused: (B, T', N, C)
        """
        if self.use_gating:
            # 完整: 门控加权融合
            g_te = self.gate_te(torch.cat([data, te], dim=-1))
            g_tle = self.gate_tle(torch.cat([data, tle], dim=-1))
            g_se = self.gate_se(torch.cat([data, se], dim=-1))
            fused = data + g_te * te + g_tle * tle + g_se * se
        else:
            # 消融 A6: 简单 element-wise 相加
            fused = data + te + tle + se

        return self.norm(fused)
