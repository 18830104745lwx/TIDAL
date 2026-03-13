"""
模块⑦ — 渐进式精炼预测器
(Progressive Refinement Predictor)

核心思想:
  TIDAL: Stage 1 粗预测 (Stage 1) → Stage 2 注意力精炼 (Query粗预测, Key编码序列)

  优势:
    1. 粗预测提供一个合理的初始猜测
    2. 精炼网络用注意力回看编码序列, 修正细节误差
    3. 残差结构: final = coarse + refine_delta
"""

import torch
import torch.nn as nn
from einops import rearrange


class ProgressivePredictor(nn.Module):
    """
    渐进式精炼预测器

    参数:
        num_nodes (int): 节点数
        input_length (int): 输入 (patch后的) 时间步数 T'
        predict_length (int): 预测时间步数 T_out
        in_dim (int): 输入特征维度 (= hid_dim)
        pre_dim (int): 预测输出维度 (如 1=流量)
        num_of_filters (int): 粗预测中间层维度
        refine_heads (int): 精炼注意力头数
        activation (str): 激活函数

    输入:
        data: (B, T', N, in_dim) - 编码后的特征序列

    输出:
        final_out: (B, T_out, N, pre_dim) - 预测输出
    """

    def __init__(self, num_nodes, input_length, predict_length, in_dim, pre_dim,
                 num_of_filters=128, refine_heads=2, activation='gelu',
                 use_refine=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.predict_length = predict_length
        self.in_dim = in_dim
        self.pre_dim = pre_dim
        self.use_refine = use_refine  # 消融开关: False → 仅粗预测
        assert activation in ['gelu', 'relu']

        # ===== Stage 1: 粗预测 (MLP 粗预测) =====
        self.coarse_encoder = nn.Sequential(
            nn.Linear(input_length * in_dim, num_of_filters),
            nn.GELU() if activation == 'gelu' else nn.ReLU()
        )
        self.coarse_heads = nn.ModuleList([
            nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)
        ])

        # ===== Stage 2: 精炼 =====
        # 粗预测的嵌入 (将 pre_dim 映射回 in_dim 用于 attention)
        self.coarse_embed = nn.Linear(pre_dim, in_dim)
        # 精炼注意力: 粗预测 Query → 编码序列 Key/Value
        self.refine_attn = nn.MultiheadAttention(
            in_dim, num_heads=refine_heads, batch_first=True, dropout=0.1
        )
        self.refine_norm = nn.LayerNorm(in_dim)
        # 精炼输出投影
        self.refine_proj = nn.Linear(in_dim, pre_dim)

    def forward(self, data):
        """
        Args:
            data: (B, T', N, in_dim)

        Returns:
            final_out: (B, T_out, N, pre_dim)
            coarse: (B, T_out, N, pre_dim) - 粗预测 (用于辅助损失)
        """
        B, T, N, C = data.shape

        # ===== Stage 1: 粗预测 =====
        # 展平时间维度
        data_flat = rearrange(data, 'b t n c -> b n (t c)')  # (B, N, T'*C)
        data_encoded = self.coarse_encoder(data_flat)          # (B, N, num_filters)

        # 每个时间步独立预测
        coarse_outs = []
        for head in self.coarse_heads:
            out_j = head(data_encoded)  # (B, N, pre_dim)
            coarse_outs.append(out_j.unsqueeze(1))
        coarse = torch.cat(coarse_outs, dim=1)  # (B, T_out, N, pre_dim)

        # ===== 消融 A4: 跳过精炼, 仅粗预测 =====
        if not self.use_refine:
            return coarse, coarse

        # ===== Stage 2: 精炼 (每个节点独立, 沿时间维度) =====
        coarse_emb = self.coarse_embed(coarse)
        query = rearrange(coarse_emb, 'b t n c -> (b n) t c')
        kv = rearrange(data, 'b t n c -> (b n) t c')
        query_normed = self.refine_norm(query)
        refine_out, _ = self.refine_attn(query_normed, kv, kv)
        refine_out = rearrange(refine_out, '(b n) t c -> b t n c', b=B, n=N)
        refine_delta = self.refine_proj(refine_out)
        final_out = coarse + refine_delta

        return final_out, coarse
