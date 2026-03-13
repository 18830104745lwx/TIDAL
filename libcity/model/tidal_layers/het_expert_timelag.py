"""
模块④ — 异质Expert时间滞后嵌入
(Heterogeneous Expert Time-Lag Embedding)

核心思想:
  TIDAL 使用 4 种归纳偏置完全不同的异质 Expert:
    1. WaveletExpert     — 多尺度时频分析, 捕捉局部突变
    2. PolynomialExpert  — 多项式拟合, 建模平滑趋势
    3. AttentionPoolExpert — 注意力池化, 发现关键时刻
    4. PeriodicExpert    — 可学习周期函数, 检测周期模式

  采用三视图 (时域+频域+语义MoE) 框架.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 四种异质 Expert
# ============================================================================

class WaveletExpert(nn.Module):
    """
    小波Expert: 使用可学习的类小波基函数做多尺度变换

    原理: 用不同尺度的高斯窗口 × 余弦调制, 在特征空间做局部时频分析
    """

    def __init__(self, dim, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        # 可学习的尺度参数 (初始化为不同量级)
        self.log_scales = nn.Parameter(torch.linspace(-1.0, 2.0, num_scales))
        # 每个尺度的投影
        self.scale_proj = nn.Linear(dim * num_scales, dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C) - 时间滞后差值

        Returns:
            out: (B, T, C)
        """
        B, T, C = x.shape
        outs = []
        for i in range(self.num_scales):
            scale = self.log_scales[i].exp() + 0.1  # 确保正值
            # 构造类小波核: Gabor-like
            t = torch.arange(C, device=x.device, dtype=x.dtype)
            t = (t - C / 2.0) / scale
            kernel = torch.exp(-0.5 * t ** 2) * torch.cos(2 * math.pi * t / (scale + 1))
            kernel = kernel / (kernel.norm() + 1e-8)
            # 在特征维度做卷积 (等价于特征空间的小波变换)
            out_i = x * kernel.unsqueeze(0).unsqueeze(0)  # (B, T, C) element-wise
            outs.append(out_i)
        # 拼接多尺度 → 投影回原维度
        multi_scale = torch.cat(outs, dim=-1)  # (B, T, C*num_scales)
        return self.scale_proj(multi_scale)     # (B, T, C)


class PolynomialExpert(nn.Module):
    """
    多项式Expert: 用多项式基拟合平滑的时间滞后关系

    原理: 生成 [x, x², x³] 多项式特征, 用可学习权重加权组合
    """

    def __init__(self, dim, degree=3):
        super().__init__()
        self.degree = degree
        # 为每个阶的多项式学习权重
        self.coeff_net = nn.Linear(dim * (degree + 1), dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)

        Returns:
            out: (B, T, C)
        """
        x_normed = self.norm(x)
        # 生成多项式特征: [x⁰, x¹, x², x³]
        poly_features = [torch.ones_like(x_normed)]  # x⁰ = 1
        for d in range(1, self.degree + 1):
            poly_features.append(x_normed ** d)
        # 拼接 → 加权融合
        poly_cat = torch.cat(poly_features, dim=-1)  # (B, T, C*(degree+1))
        return self.coeff_net(poly_cat)                # (B, T, C)


class AttentionPoolExpert(nn.Module):
    """
    注意力池化Expert: 用自注意力发现最关键的时间滞后时刻

    原理: dim 维特征作为 Q/K/V, 通过 2-head attention 交互
    """

    def __init__(self, dim, n_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)

        Returns:
            out: (B, T, C)
        """
        x_normed = self.norm(x)
        out, _ = self.attn(x_normed, x_normed, x_normed)
        return out


class PeriodicExpert(nn.Module):
    """
    周期检测Expert: 用可学习的周期函数捕捉周期性滞后模式

    原理: 多组可学习 (频率, 相位, 振幅), 输出 Σ A_i * sin(2π*f_i*x + φ_i)
    """

    def __init__(self, dim, num_periods=4):
        super().__init__()
        self.num_periods = num_periods
        # 可学习的周期参数
        self.frequencies = nn.Parameter(torch.randn(num_periods, dim) * 0.1)
        self.phases = nn.Parameter(torch.zeros(num_periods, dim))
        self.amplitudes = nn.Parameter(torch.ones(num_periods, dim) * 0.1)
        # 融合投影
        self.proj = nn.Linear(dim * num_periods, dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)

        Returns:
            out: (B, T, C)
        """
        periodic_outs = []
        for i in range(self.num_periods):
            freq = self.frequencies[i]   # (C,)
            phase = self.phases[i]       # (C,)
            amp = self.amplitudes[i]     # (C,)
            # sin(2π * freq * x + phase) * amplitude
            p = amp * torch.sin(2 * math.pi * freq * x + phase)
            periodic_outs.append(p)
        periodic_cat = torch.cat(periodic_outs, dim=-1)  # (B, T, C*num_periods)
        return self.proj(periodic_cat)                    # (B, T, C)


# ============================================================================
# 异质Expert时间滞后嵌入主模块
# ============================================================================

class HeterogeneousTimeLagEmbedding(nn.Module):
    """
    异质Expert多视图时间滞后嵌入

    三视图:
      - 时域视图: MLP(diff)
      - 频率视图: FFT(diff).real → Linear
      - 语义视图: 异质MoE(diff) — 门控选择4种异质Expert

    融合: concat([time, freq, semantic]) → MLP → out_dim

    参数:
        in_dim (int): 输入维度 (= hid_dim)
        out_dim (int): 输出维度 (= hid_dim)
        emb_dropout (float): Dropout概率

    输入:
        time_emb: (B, T, C) - 所有时间步的时间嵌入
        ref_emb: (B, 1, C) - 参考时刻 (最新) 的时间嵌入

    输出:
        out: (B, T, out_dim) - 时间滞后嵌入
    """

    def __init__(self, in_dim, out_dim, emb_dropout=0.1, use_moe=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_moe = use_moe  # 消融开关: False → 单MLP替代MoE

        # 时域视图
        self.time_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        # 频率视图: FFT + 投影
        self.freq_proj = nn.Linear(in_dim, in_dim)

        # 语义视图
        if self.use_moe:
            # 完整: 异质 MoE (4 Expert)
            self.experts = nn.ModuleList([
                WaveletExpert(in_dim),
                PolynomialExpert(in_dim),
                AttentionPoolExpert(in_dim),
                PeriodicExpert(in_dim),
            ])
            self.num_experts = len(self.experts)
            self.moe_gate = nn.Linear(in_dim, self.num_experts)
        else:
            # 消融 A3: 单MLP替代
            self.semantic_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, in_dim)
            )

        # 三视图融合
        self.fusion = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, time_emb, ref_emb):
        """
        Args:
            time_emb: (B, T, C) - 各时间步的时间嵌入
            ref_emb: (B, 1, C) - 参考时刻嵌入

        Returns:
            out: (B, T, out_dim)
        """
        # 计算时间滞后 diff
        diff = time_emb - ref_emb  # (B, T, C)

        # 视图1: 时域
        view_time = self.time_mlp(diff)  # (B, T, C)

        # 视图2: 频率域 (FFT + 线性投影)
        # 频率域特征提取 (使用统计量替代FFT, 避免CUDA JIT兼容性问题)
        diff_var = diff.var(dim=1, keepdim=True).expand_as(diff)  # 方差作为频率代理
        view_freq_raw = diff * diff_var  # (B, T, C)
        view_freq = self.freq_proj(view_freq_raw)          # (B, T, C)

        # 视图3: 语义
        if self.use_moe:
            # 完整: MoE (异质Expert)
            gate_logits = self.moe_gate(diff)
            gates = torch.softmax(gate_logits, dim=-1)
            expert_outs = []
            for exp in self.experts:
                expert_outs.append(exp(diff))
            expert_stack = torch.stack(expert_outs, dim=-1)
            view_sem = torch.sum(
                expert_stack * gates.unsqueeze(-2), dim=-1
            )
        else:
            # 消融 A3: 单MLP替代
            view_sem = self.semantic_mlp(diff)

        # 三视图融合
        fused = torch.cat([view_time, view_freq, view_sem], dim=-1)  # (B, T, 3C)
        out = self.fusion(fused)  # (B, T, out_dim)
        out = self.dropout(out)
        return out
