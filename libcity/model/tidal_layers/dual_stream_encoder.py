"""
模块⑥ — 双流交替时空编码器
(Dual-Stream Interleaved Spatio-Temporal Encoder)

核心思想:
  TIDAL: 空间流 → 时间流 → 跨流交互, 交替堆叠

  数据流 (每个 DualStreamSTBlock):
    输入 x(B,T',N,C)
      → 空间注意力: 节点间交互 (保留 Query Cross Time, Proxy 等)
      → 时间注意力: 时间步间交互 (新增!)
      → 跨流交互: 门控融合空间流和时间流
    输出 x(B,T',N,C)
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from libcity.model.tidal_layers.attention import (
    AttentionLayer, MultiHeadsAttention
)
from libcity.model.tidal_utils.DIFFormer import DIFFormerConv
from libcity.model.tidal_utils.ProbAttention import ProbAttention


class DropPath(nn.Module):
    """随机深度 (Stochastic Depth): 训练时按概率跳过整个残差分支"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # 按 batch 维度随机 mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ============================================================================
# 时间自注意力 (时间自注意力组件)
# ============================================================================
class TemporalSelfAttention(nn.Module):
    """
    沿时间维度的自注意力

    每个节点独立地对其时间序列做 self-attention,
    捕捉时间步之间的依赖关系

    参数:
        hid_dim (int): 隐藏维度
        n_heads (int): 注意力头数
        dropout (float): Dropout 概率
    """

    def __init__(self, hid_dim, n_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hid_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T', N, C)

        Returns:
            out: (B, T', N, C)
        """
        B, T, N, C = x.shape

        # 重排: 每个节点独立做时间 attention
        x_tn = rearrange(x, 'b t n c -> (b n) t c')  # (B*N, T', C)

        # Pre-LN + Self-Attention
        x_normed = self.norm(x_tn)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)

        # 残差连接
        x_tn = x_tn + self.dropout(attn_out)

        return rearrange(x_tn, '(b n) t c -> b t n c', b=B, n=N)


# ============================================================================
# 跨流交互
# ============================================================================
class CrossStreamInteraction(nn.Module):
    """
    空间流和时间流之间的门控交互

    使用可学习的门控网络自适应融合两个流的信息

    参数:
        hid_dim (int): 隐藏维度
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x_spatial, x_temporal):
        """
        Args:
            x_spatial: (B, T', N, C) - 空间流输出
            x_temporal: (B, T', N, C) - 时间流输出

        Returns:
            fused: (B, T', N, C)
        """
        gate = self.gate_net(torch.cat([x_spatial, x_temporal], dim=-1))
        fused = gate * x_spatial + (1 - gate) * x_temporal
        return self.norm(fused)


# ============================================================================
# 空间注意力 (多种空间注意力机制)
# ============================================================================
class SpatialAttentionLayer(nn.Module):
    """
    空间注意力层

    支持 4 种空间注意力 + Query Cross Time 机制,
    但独立封装以适应双流架构

    参数:
        hid_dim (int): 隐藏维度
        n_heads (int): 注意力头数
        att_type (str): 注意力类型 'full'/'prob'/'proxy'/'difformer'
        query_cross_time (bool): 是否使用最新时刻查询历史
        num_nodes (int): 节点数 (proxy 类型需要)
        M (int): 代理节点数 (proxy 类型需要)
        dropout (float): Dropout 概率
    """

    def __init__(self, hid_dim, n_heads, att_type='proxy', query_cross_time=True,
                 num_nodes=None, M=8, dropout=0.1, att_dropout=0.1,
                 prob_factor=5, return_att=False):
        super().__init__()
        self.att_type = att_type
        self.query_cross_time = query_cross_time
        self.return_att = return_att

        if att_type == 'difformer':
            self.att_layer = DIFFormerConv(
                hid_dim, hid_dim, num_heads=n_heads, output_attn=return_att
            )
        elif att_type in ['full', 'prob']:
            self.att_layer = AttentionLayer(
                hid_dim, n_heads, dropout=att_dropout, att_type=att_type,
                returnA=return_att, prob_factor=prob_factor
            )
        elif att_type == 'proxy':
            assert num_nodes is not None
            self.readout_fc = nn.Linear(num_nodes, M)
            self.node2proxy = AttentionLayer(
                hid_dim, n_heads, dropout=att_dropout, att_type='proxy',
                returnA=return_att
            )
            self.proxy2node = AttentionLayer(
                hid_dim, n_heads, dropout=att_dropout, att_type='proxy',
                returnA=return_att
            )

        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 4),
            nn.GELU(),
            nn.Linear(hid_dim * 4, hid_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T', N, C)

        Returns:
            out: (B, T', N, C)
            A: 注意力权重 (可选)
        """
        B, T, N, C = x.shape

        # Query Cross Time: 用最新时刻查询历史
        if self.query_cross_time:
            now = x[:, -1, :, :]                         # (B, N, C)
            query = repeat(now, 'b n c -> (b t) n c', t=T)
        else:
            query = None

        data = rearrange(x, 'b t n c -> (b t) n c')
        data_normed = self.norm1(data)

        if self.att_type == 'difformer':
            q = self.norm1(query) if query is not None else data_normed
            attn_out, A = self.att_layer(q, data_normed)
        elif self.att_type in ['full', 'prob']:
            q = self.norm1(query) if query is not None else data_normed
            attn_out, A = self.att_layer(q, data_normed, data_normed)
        elif self.att_type == 'proxy':
            if query is not None:
                temp = rearrange(now, 'b n c -> b c n')
                z_proxy = self.readout_fc(temp)
                z_proxy = repeat(z_proxy, 'b c k -> (b t) c k', t=T)
                z_proxy = rearrange(z_proxy, 'bt c K -> bt K c')
            else:
                temp = rearrange(x, 'b t n c -> b t c n')
                z_proxy = self.readout_fc(temp)
                z_proxy = rearrange(z_proxy, 'b t c K -> (b t) K c')

            z_proxy = self.norm1(z_proxy)
            proxy_feat, _ = self.node2proxy(z_proxy, data_normed, data_normed)
            attn_out, A = self.proxy2node(data_normed, proxy_feat, proxy_feat)

        # 残差 + MLP
        x_out = data + self.dropout(attn_out)
        x_out = x_out + self.dropout(self.mlp(self.norm2(x_out)))

        out = rearrange(x_out, '(b t) n c -> b t n c', b=B)
        return out, A


# ============================================================================
# 双流时空编码块 (核心)
# ============================================================================
class DualStreamSTBlock(nn.Module):
    """
    双流交替时空编码块

    一个 Block 包含:
      Step 1: 空间注意力 (节点间交互, Query Cross Time)
      Step 2: 时间注意力 (时间步间交互, TIDAL 新增)
      Step 3: 跨流交互 (空间+时间门控融合, TIDAL 新增)

    参数:
        hid_dim (int): 隐藏维度
        n_heads (int): 注意力头数
        att_type (str): 空间注意力类型
        query_cross_time (bool): 是否使用 Query Cross Time
        num_nodes (int): 节点数
        M (int): 代理节点数
        dropout (float): Dropout 概率
    """

    def __init__(self, hid_dim, n_heads=2, att_type='proxy',
                 query_cross_time=True, num_nodes=None, M=8,
                 dropout=0.1, att_dropout=0.1, prob_factor=5,
                 return_att=False, drop_path_rate=0.0,
                 use_graph_conv=True):
        super().__init__()
        self.use_graph_conv = use_graph_conv  # 消融开关: False → 不使用图卷积

        # 空间流: 空间注意力机制
        self.spatial_attn = SpatialAttentionLayer(
            hid_dim, n_heads, att_type=att_type,
            query_cross_time=query_cross_time,
            num_nodes=num_nodes, M=M,
            dropout=dropout, att_dropout=att_dropout,
            prob_factor=prob_factor, return_att=return_att
        )

        # 时间流: TIDAL 新增
        self.temporal_attn = TemporalSelfAttention(
            hid_dim, n_heads=n_heads, dropout=dropout
        )

        # 跨流交互: TIDAL 新增
        self.cross_stream = CrossStreamInteraction(hid_dim)

        # DropPath (随机深度正则化)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # 逐层图卷积: 空间信息在每层迭代精炼
        if self.use_graph_conv:
            self.graph_proj = nn.Linear(hid_dim, hid_dim)
            self.graph_norm = nn.LayerNorm(hid_dim)

    def forward(self, x, adj=None):
        """
        Args:
            x: (B, T', N, C)
            adj: (N, N) 邻接矩阵 (可选, 用于逐层图卷积)

        Returns:
            out: (B, T', N, C)
            A: 空间注意力权重 (可选)
        """
        # Step 1: 空间注意力 (保留 Query Cross Time)
        x_s, A = self.spatial_attn(x)

        # Step 2: 时间注意力 (新增)
        x_t = self.temporal_attn(x)

        # Step 3: 跨流交互 (新增)
        x_out = self.cross_stream(x_s, x_t)

        # Step 4: 逐层图卷积 (空间信息逐层精炼)
        if self.use_graph_conv and adj is not None:
            x_graph = torch.einsum('mn,btnd->btmd', adj, x_out)
            x_out = self.graph_norm(x_out + self.graph_proj(x_graph))

        # 残差连接 + DropPath
        x_out = x + self.drop_path(x_out - x)

        return x_out, A

