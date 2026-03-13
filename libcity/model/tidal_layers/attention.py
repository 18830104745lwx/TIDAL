"""
注意力层 — 多头注意力 + 注意力封装层
(Multi-Head Attention + Attention Layer Wrapper)

TIDAL 自有模块
"""

import torch
import torch.nn as nn
import math

from libcity.model.tidal_utils.ProbAttention import ProbAttention


class MultiHeadsAttention(nn.Module):
    """
    标准多头注意力

    参数:
        hid_dim (int): 隐藏维度
        n_heads (int): 注意力头数
        dropout (float): Dropout 概率
    """
    def __init__(self, hid_dim, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = hid_dim
        self.n_heads = n_heads
        self.d_k = hid_dim // n_heads

        self.W_Q = nn.Linear(hid_dim, hid_dim)
        self.W_K = nn.Linear(hid_dim, hid_dim)
        self.W_V = nn.Linear(hid_dim, hid_dim)
        self.out_proj = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: (B, N_q, hid_dim)
            key:   (B, N_k, hid_dim)
            value: (B, N_k, hid_dim)

        Returns:
            out: (B, N_q, hid_dim)
            attn: (B, n_heads, N_q, N_k)
        """
        B = query.size(0)

        Q = self.W_Q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.out_proj(context)

        return out, attn


class AttentionLayer(nn.Module):
    """
    注意力层封装 — 支持 full / prob / proxy 三种模式

    参数:
        hid_dim (int): 隐藏维度
        n_heads (int): 注意力头数
        dropout (float): Dropout 概率
        att_type (str): 'full' / 'prob' / 'proxy'
        returnA (bool): 是否返回注意力权重
        prob_factor (int): ProbAttention 的 factor 参数
    """
    def __init__(self, hid_dim, n_heads, dropout=0.1, att_type='full',
                 returnA=False, prob_factor=5):
        super().__init__()
        self.att_type = att_type
        self.returnA = returnA

        if att_type == 'prob':
            self.attention = ProbAttention(
                mask_flag=False, factor=prob_factor,
                attention_dropout=dropout, output_attention=returnA
            )
            self.W_Q = nn.Linear(hid_dim, hid_dim)
            self.W_K = nn.Linear(hid_dim, hid_dim)
            self.W_V = nn.Linear(hid_dim, hid_dim)
            self.out_proj = nn.Linear(hid_dim, hid_dim)
            self.n_heads = n_heads
            self.d_k = hid_dim // n_heads
        else:
            # full attention or proxy attention
            self.attention = MultiHeadsAttention(hid_dim, n_heads, dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: (B, N_q, hid_dim)
            key:   (B, N_k, hid_dim)
            value: (B, N_k, hid_dim)

        Returns:
            out: (B, N_q, hid_dim)
            attn: attention weights or None
        """
        if self.att_type == 'prob':
            B = query.size(0)
            Q = self.W_Q(query).view(B, -1, self.n_heads, self.d_k)
            K = self.W_K(key).view(B, -1, self.n_heads, self.d_k)
            V = self.W_V(value).view(B, -1, self.n_heads, self.d_k)

            out, attn = self.attention(Q, K, V)
            out = self.out_proj(out)
            return out, attn
        else:
            out, attn = self.attention(query, key, value)
            if not self.returnA:
                attn = None
            return out, attn
