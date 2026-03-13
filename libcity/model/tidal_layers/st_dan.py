"""
模块① — ST-DAN: 时空感知分布自适应归一化
(Spatio-Temporal Distribution-Adaptive Normalization)

核心思想:
  编码端: 逐实例 Instance Normalization
  解码端: 用时间滞后语义 + 空间语义生成条件反归一化参数
  与 RevIN 的区别: 反归一化参数不是简单复用输入统计量,
    而是用模型学到的时空语义来生成, 更好地适应未来时段的分布偏移
"""

import torch
import torch.nn as nn


class STDAN(nn.Module):
    """
    时空感知分布自适应归一化

    参数:
        num_features (int): 输入特征维度 (如 C=1 表示单变量流量)
        hid_dim (int): 隐藏维度
        affine (bool): 是否使用可学习的仿射参数
        eps (float): 数值稳定性

    用法:
        dan = STDAN(num_features=1, hid_dim=64)
        x_norm = dan.normalize(x)       # 编码前
        ...模型编码...
        y = dan.denormalize(y, tle, se)  # 解码后
    """

    def __init__(self, num_features, hid_dim, eps=1e-5, use_condition=True):
        super().__init__()
        self.num_features = num_features
        self.hid_dim = hid_dim
        self.eps = eps
        self.use_condition = use_condition  # 消融开关: False → RevIN

        # 条件网络: 根据时空语义生成反归一化参数 (μ', σ')
        self.condition_net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, num_features * 2)  # 输出 (μ_pred, log_σ_pred)
        )

        # 可学习的混合门: 控制原始统计量 vs 条件统计量的比例
        self.blend_gate = nn.Parameter(torch.zeros(1))  # 初始化为0 → sigmoid=0.5

        # 缓存归一化统计量
        self._cached_mu = None
        self._cached_sigma = None

    def normalize(self, x):
        """
        编码端归一化: 逐实例 Instance Normalization

        Args:
            x: (B, T, N, C) - 原始输入

        Returns:
            x_norm: (B, T, N, C) - 归一化后的输入
        """
        # 沿时间维度计算统计量 (每个batch、每个节点独立)
        mu = x.mean(dim=1, keepdim=True)       # (B, 1, N, C)
        sigma = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, N, C)

        # 缓存, 供 denormalize 使用
        self._cached_mu = mu.detach()
        self._cached_sigma = sigma.detach()

        return (x - mu) / sigma

    def denormalize(self, y, time_lag_emb, spatial_emb):
        """
        解码端条件反归一化

        Args:
            y: (B, T_out, N, C_out) - 模型预测输出
            time_lag_emb: (B, T, hid_dim) - 时间滞后嵌入 (取最后时刻)
            spatial_emb: (1, 1, N, hid_dim) - 空间嵌入

        Returns:
            y_denorm: (B, T_out, N, C_out) - 反归一化后的输出
        """
        assert self._cached_mu is not None, "必须先调用 normalize()"

        C_out = y.shape[-1]
        mu_cached = self._cached_mu[..., :C_out]
        sigma_cached = self._cached_sigma[..., :C_out]

        # ===== 消融 A1: RevIN 模式 (不使用条件反归一化) =====
        if not self.use_condition:
            return y * sigma_cached + mu_cached

        # ===== 完整 ST-DAN 条件反归一化 =====
        # 构造条件输入: 时间语义 + 空间语义
        tl = time_lag_emb[:, -1:, :]              # (B, 1, hid_dim)
        sp = spatial_emb.squeeze(0).squeeze(0)     # (N, hid_dim)
        condition = tl + sp.unsqueeze(0)           # (B, N, hid_dim)
        condition = condition.unsqueeze(1)         # (B, 1, N, hid_dim)

        # 生成条件反归一化参数
        params = self.condition_net(condition)     # (B, 1, N, 2*C)
        mu_pred, log_sigma_pred = params.chunk(2, dim=-1)
        sigma_pred = log_sigma_pred.exp()

        # 混合: 可学习比例融合原始统计量和条件统计量
        alpha = torch.sigmoid(self.blend_gate)
        mu_pred = mu_pred[..., :C_out]
        sigma_pred = sigma_pred[..., :C_out]

        mu_final = alpha * mu_cached + (1 - alpha) * mu_pred
        sigma_final = alpha * sigma_cached + (1 - alpha) * sigma_pred

        return y * sigma_final + mu_final
