"""
TIDAL — Time-lag Informed Distribution-Adaptive Learning
for Traffic Flow Prediction
基于时间滞后信息的分布自适应学习方法：面向交通流量预测

核心创新点 (Contributions):

  1. 异质Expert时间滞后嵌入机制 (HetTLE):
     首次提出融合时域、频域、语义三视图的时间滞后建模方法. 语义视图
     引入 4 种归纳偏置完全不同的异质 Expert (Wavelet 小波变换 /
     Polynomial 多项式拟合 / AttentionPool 注意力池化 / Periodic
     周期检测), 通过门控 MoE 路由自适应选择, 突破传统位置编码对
     时间滞后效应建模能力的局限.

  2. 时空感知分布自适应归一化 (ST-DAN):
     提出基于时空语义条件的反归一化策略. 编码端采用 Instance
     Normalization 消除分布差异; 解码端利用时间滞后语义与空间语义
     动态生成条件反归一化参数, 通过可学习混合门平衡原始统计量与
     条件统计量, 有效缓解交通流量预测中的时序分布偏移问题.

  3. 渐进式精炼预测框架:
     设计"粗预测→注意力精炼"的两阶段预测架构: Stage1 MLP 生成
     初始预测, Stage2 以粗预测为 Query 对编码序列做交叉注意力
     精炼 (final = coarse + refine_delta). 结合门控 Patch 嵌入、
     交叉注意力多源融合、自适应邻接矩阵 (网格距离 SVD 初始化)
     双流时空编码器与辅助损失联合优化, 提升预测精度与训练效率.

模型流程:
  输入(B,T,N,C)
  → ① ST-DAN 自适应归一化
  → ② 门控 Patch 嵌入 (含 Cross Connection)
  → ③ 交叉注意力门控融合 (TE + HetTLE + SE)
  → ④ 双流编码器×L (自适应邻接矩阵图卷积 + Skip 连接融合)
  → ⑤ 渐进式精炼预测器 (粗预测 + 精细预测)
  → ① ST-DAN 条件反归一化 → 残差连接
  → 输出(B,T_out,N,C_out)

训练损失: L = L_main(精细预测) + 0.1 × L_aux(粗预测)
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops import rearrange
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

# TIDAL 自有模块
from libcity.model.tidal_layers.st_dan import STDAN
from libcity.model.tidal_layers.gated_patch_embedding import GatedPatchEmbedding
from libcity.model.tidal_layers.cross_attention_fusion import CrossAttentionFusion
from libcity.model.tidal_layers.het_expert_timelag import HeterogeneousTimeLagEmbedding
from libcity.model.tidal_layers.dual_stream_encoder import DualStreamSTBlock
from libcity.model.tidal_layers.progressive_predictor import ProgressivePredictor

# 时空嵌入模块 (TIDAL 自有)
from libcity.model.tidal_layers.embeddings import (
    getTimeEmbedding, getSpatialEmbedding
)


# ============================================================================
# TIDAL Core Model
# ============================================================================
class TIDALModel(nn.Module):
    """
    TIDAL核心模型

    参数 (通过 args 传入):
        input_length (int): 原始输入时间步数 (如 12)
        predict_length (int): 预测时间步数 (如 12)
        in_dim (int): 原始特征维度 (如 1)
        pre_dim (int): 预测输出维度
        num_nodes (int): 节点数
        hid_dim (int): 隐藏维度
        n_heads (int): 注意力头数
        M (int): Proxy 代理节点数
        d_out (int): 预测器中间层维度
        num_layers (int): 编码器层数
        patch_size (int): Patch 大小
        att_type (str): 空间注意力类型
        query_cross_time (bool): 是否使用 Query Cross Time
        return_att (bool): 是否返回注意力权重
        norm_flag (str): 归一化策略
        slice_size_per_day (int): 每天时间片数
        各种 dropout 参数...
    """

    def __init__(self, args):
        super().__init__()
        self.input_length = args.input_length
        self.predict_length = args.predict_length
        self.in_dim = args.in_dim
        self.pre_dim = args.pre_dim
        self.num_nodes = args.num_nodes
        self.hid_dim = args.hid_dim
        self.n_heads = args.n_heads
        self.M = args.M
        self.d_out = args.d_out
        self.num_layers = args.num_layers
        self.patch_size = args.patch_size
        self.query_cross_time = bool(args.query_cross_time)
        self.return_att = bool(args.return_att)
        self.att_type = args.att_type
        self.slice_size_per_day = args.slice_size_per_day
        self.drop_path_rate = getattr(args, 'drop_path_rate', 0.0)
        # 网格尺寸 (用于距离初始化)
        self.grid_rows = getattr(args, 'grid_rows', 20)
        self.grid_cols = getattr(args, 'grid_cols', 20)

        # ===== 消融实验开关 (默认 True = 完整模型) =====
        self.use_stdan = getattr(args, 'use_stdan', True)              # A1
        self.use_hettle = getattr(args, 'use_hettle', True)            # A2
        self.use_moe_experts = getattr(args, 'use_moe_experts', True)  # A3
        self.use_progressive_refine = getattr(args, 'use_progressive_refine', True)  # A4
        self.use_gated_fusion = getattr(args, 'use_gated_fusion', True)  # A6
        self.use_adaptive_adj = getattr(args, 'use_adaptive_adj', True)  # A8

        # Patch 后的时间步数
        self.patched_length = (self.input_length + self.patch_size - 1) // self.patch_size

        # ===== ① ST-DAN 归一化 =====
        self.st_dan = STDAN(
            num_features=self.in_dim,
            hid_dim=self.hid_dim,
            use_condition=self.use_stdan,  # A1 消融开关
        )

        # ===== ② 门控 Patch 嵌入 =====
        self.gated_patch_emb = GatedPatchEmbedding(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            patch_size=self.patch_size,
            activation=args.activation_data,
        )

        # ===== 嵌入层 (时间 + 空间) =====
        self.getTemb = getTimeEmbedding(
            self.hid_dim, self.slice_size_per_day, args.te_emb_dropout
        )
        self.getSemb = getSpatialEmbedding(
            self.hid_dim, self.num_nodes, args.se_emb_dropout
        )

        # ===== ④ 异质Expert时间滞后嵌入 =====
        if self.use_hettle:
            self.getTimeLagEmb = HeterogeneousTimeLagEmbedding(
                in_dim=self.hid_dim,
                out_dim=self.hid_dim,
                emb_dropout=args.tle_emb_dropout,
                use_moe=self.use_moe_experts,  # A3 消融开关
            )
        else:
            # A2 消融: 可学习位置编码替代 HetTLE
            self.tle_fallback = nn.Parameter(
                torch.randn(1, self.patched_length, 1, self.hid_dim) * 0.02
            )

        # ===== ③ 交叉注意力门控融合 =====
        self.fusion = CrossAttentionFusion(
            self.hid_dim,
            use_gating=self.use_gated_fusion,  # A6 消融开关
        )

        # ===== ⑥ 单流编码器 (用统一编码器) =====
        # DropPath rate 按层递增 (越深的层 drop 越多)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.num_layers)]
        self.encoder_blocks = nn.ModuleList([
            DualStreamSTBlock(
                hid_dim=self.hid_dim,
                n_heads=self.n_heads,
                att_type=self.att_type,
                query_cross_time=self.query_cross_time,
                num_nodes=self.num_nodes,
                M=self.M,
                dropout=args.enc_dropout,
                att_dropout=args.att_dropout,
                prob_factor=args.prob_factor,
                return_att=self.return_att,
                drop_path_rate=dpr[i],
                use_graph_conv=self.use_adaptive_adj,  # A8 消融开关
            ) for i in range(self.num_layers)
        ])

        # ===== ⑧ 自适应邻接矩阵 (网格距离初始化, 逐层使用) =====
        if self.use_adaptive_adj:
            adj_emb_dim = 16
            self.adj_emb1, self.adj_emb2 = self._init_adj_with_distance(adj_emb_dim)

        # ===== Skip 连接融合 (类似 TrafficFormer) =====
        self.skip_projs = nn.ModuleList([
            nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.num_layers)
        ])
        self.skip_norm = nn.LayerNorm(self.hid_dim)

        # ===== ⑦ 渐进式精炼预测器 =====
        self.predictor = ProgressivePredictor(
            num_nodes=self.num_nodes,
            input_length=self.patched_length,
            predict_length=self.predict_length,
            in_dim=self.hid_dim,
            pre_dim=self.pre_dim,
            num_of_filters=self.d_out,
            activation=args.activation_dec,
            use_refine=self.use_progressive_refine,  # A4 消融开关
        )

    def _init_adj_with_distance(self, emb_dim):
        """用网格距离先验初始化邻接矩阵嵌入 (而非随机)"""
        N = self.num_nodes
        rows, cols = self.grid_rows, self.grid_cols

        # 计算每个节点在网格中的坐标
        coords = np.array([(i // cols, i % cols) for i in range(N)], dtype=np.float32)

        # 计算节点间欧氏距离
        diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
        dist = np.sqrt((diff ** 2).sum(axis=-1))  # (N, N)

        # 转为高斯相似度: 近的节点 → 相似度高
        sigma = dist.std()
        sim = np.exp(-dist ** 2 / (2 * sigma ** 2))  # (N, N)

        # SVD 分解为两个低秩矩阵
        U, S, Vt = np.linalg.svd(sim)
        sqrt_S = np.sqrt(S[:emb_dim])
        emb1 = U[:, :emb_dim] * sqrt_S[None, :]  # (N, emb_dim)
        emb2 = Vt[:emb_dim, :].T * sqrt_S[None, :]  # (N, emb_dim)

        return (
            nn.Parameter(torch.from_numpy(emb1).float()),
            nn.Parameter(torch.from_numpy(emb2).float()),
        )

    def forward(self, x, x_time):
        """
        TIDAL 前向传播

        Args:
            x: (B, T, N, C) - 交通特征
            x_time: (B, T, 2) - 时间特征 [时间片索引, 星期索引]

        Returns:
            main_output: (B, T_out, N, pre_dim)
            A_list: 注意力权重列表
        """
        B, T, N, _ = x.shape

        # ===== ① ST-DAN 归一化 =====
        x_norm = self.st_dan.normalize(x)

        # 准备最新时刻特征 (用于 Cross Connection 和残差连接)
        latestX = x[:, -1:, :, :].repeat([1, self.input_length, 1, 1])
        # ⚠️ 不能再次调用 st_dan.normalize(), 否则会覆盖 cached_mu/sigma
        # 使用已缓存的统计量手动归一化
        latestX_norm = (latestX - self.st_dan._cached_mu) / self.st_dan._cached_sigma

        # ===== ② 门控 Patch 嵌入 =====
        data = self.gated_patch_emb(x_norm, latestX_norm)  # (B, T', N, hid_dim)
        T_patched = data.shape[1]

        # ===== 时间嵌入 =====
        x_time_emb = self.getTemb(x_time)  # (B, T, hid_dim)
        # Patch 化时间嵌入: 取每个 patch 最后一步的时间嵌入
        if T_patched < T:
            # 对时间嵌入按 patch 取最后一个
            P = self.patch_size
            remainder = T % P
            if remainder != 0:
                # pad 后再 reshape
                x_time_emb_padded = torch.nn.functional.pad(
                    x_time_emb, (0, 0, 0, P - remainder)
                )
            else:
                x_time_emb_padded = x_time_emb
            x_time_emb_patched = rearrange(
                x_time_emb_padded, 'b (t p) c -> b t p c', p=P
            )[:, :, -1, :]  # 取每个 patch 的最后一步
        else:
            x_time_emb_patched = x_time_emb
        # 扩展到节点维度
        te_expanded = x_time_emb_patched.unsqueeze(2).expand(B, T_patched, N, -1)

        # ===== ⑤ 异质Expert时间滞后嵌入 =====
        if self.use_hettle:
            tle = self.getTimeLagEmb(
                x_time_emb_patched,        # (B, T', hid_dim)
                x_time_emb_patched[:, -1:]  # (B, 1, hid_dim) - 最新时刻
            )  # (B, T', hid_dim)
            tle_expanded = tle.unsqueeze(2).expand(B, T_patched, N, -1)
        else:
            # A2 消融: 可学习位置编码替代
            tle_expanded = self.tle_fallback.expand(B, T_patched, N, -1)

        # ===== 空间嵌入 =====
        se = self.getSemb(data)  # (1, 1, N, hid_dim)
        se_expanded = se.expand(B, T_patched, N, -1)

        # ===== ③ 交叉注意力门控融合 =====
        data = self.fusion(data, te_expanded, tle_expanded, se_expanded)

        # ===== ⑦ 编码器 + 图卷积 + Skip 连接 =====
        if self.use_adaptive_adj:
            adj = torch.softmax(self.adj_emb1 @ self.adj_emb2.T, dim=-1)  # (N, N) 计算一次复用
        else:
            adj = None  # A8 消融: 不使用图卷积
        A_list = []
        skip_sum = torch.zeros_like(data)  # Skip 连接累加器
        for i, enc_block in enumerate(self.encoder_blocks):
            data, A = enc_block(data, adj=adj)  # 每层都做图卷积
            A_list.append(A)
            # 累加 skip: 每层输出都贡献给最终结果
            skip_sum = skip_sum + self.skip_projs[i](data)

        # 多层 Skip 融合
        data = self.skip_norm(skip_sum / self.num_layers)

        # ===== ⑦ 渐进式精炼预测器 =====
        main_output, coarse_output = self.predictor(data)  # 同时返回粗预测

        # 残差基线
        if self.input_length == self.predict_length:
            res_base = latestX[:, :, :, :self.pre_dim]
        else:
            res_base = latestX[:, 0:self.predict_length, :, :self.pre_dim]

        # ===== ① ST-DAN 条件反归一化 =====
        # 使用编码器最后时间步的输出作为时间条件 (比原始TLE更丰富)
        enc_temporal_cond = data.mean(dim=2)  # (B, T', hid_dim) - 节点均值
        main_output = self.st_dan.denormalize(
            main_output,
            enc_temporal_cond,  # (B, T', hid_dim) - 编码器输出作为时间条件
            se,   # (1, 1, N, hid_dim) - 空间嵌入作为条件
        )
        coarse_output = self.st_dan.denormalize(coarse_output, enc_temporal_cond, se)

        # ===== 残差连接 =====
        main_output = main_output + res_base
        coarse_output = coarse_output + res_base

        if self.return_att:
            return main_output, coarse_output, A_list
        return main_output, coarse_output, None


# ============================================================================
# LibCity Wrapper
# ============================================================================
class TIDAL(AbstractTrafficStateModel):
    """
    TIDAL 模型的 LibCity 框架适配器

    论文: TIDAL: Time-lag Informed Distribution-Adaptive Learning
    for Traffic Flow Prediction
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger(__name__)
        self._scaler = self.data_feature.get('scaler')

        # 数据参数
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.output_dim = config.get('output_dim', 1)

        # 时间参数
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)

        # 模型参数
        self.hid_dim = config.get('hid_dim', 64)
        self.d_out = config.get('d_out', 1024)
        self.n_heads = config.get('n_heads', 2)
        self.M = config.get('M', 8)
        self.num_layers = config.get('num_layers', 1)
        self.patch_size = config.get('patch_size', 2)

        # 注意力参数
        self.att_type = config.get('att_type', 'proxy')
        self.query_cross_time = config.get('query_cross_time', 1)
        self.return_att = config.get('return_att', 0)
        self.prob_factor = config.get('prob_factor', 5)

        # 正则化参数
        self.enc_dropout = config.get('enc_dropout', 0.1)
        self.att_dropout = config.get('att_dropout', 0.1)
        self.te_emb_dropout = config.get('te_emb_dropout', 0.1)
        self.se_emb_dropout = config.get('se_emb_dropout', 0.1)
        self.tle_emb_dropout = config.get('tle_emb_dropout', 0.1)

        # 激活函数
        self.activation_data = config.get('activation_data', 'gelu')
        self.activation_dec = config.get('activation_dec', 'gelu')

        # 时间参数
        self.add_time_in_day = config.get("add_time_in_day", True)
        self.add_day_in_week = config.get("add_day_in_week", True)
        self.slice_size_per_day = config.get('slice_size_per_day', 288)

        # 损失函数
        self.huber_delta = config.get('huber_delta', 2)
        self.set_loss = config.get('set_loss', 'huber')
        self.device = config.get('device', torch.device('cpu'))

        # ===== 消融实验开关 =====
        self.use_stdan = config.get('use_stdan', True)
        self.use_hettle = config.get('use_hettle', True)
        self.use_moe_experts = config.get('use_moe_experts', True)
        self.use_progressive_refine = config.get('use_progressive_refine', True)
        self.use_gated_fusion = config.get('use_gated_fusion', True)
        self.use_adaptive_adj = config.get('use_adaptive_adj', True)

        # 构建模型
        self._build_model()

        self._logger.info('✨ TIDAL模型初始化完成')
        self._logger.info(f'   - 节点数: {self.num_nodes}')
        self._logger.info(f'   - 输入窗口: {self.input_window}, 输出窗口: {self.output_window}')
        self._logger.info(f'   - 隐藏维度: {self.hid_dim}, 注意力头数: {self.n_heads}')
        self._logger.info(f'   - 注意力类型: {self.att_type}, Patch大小: {self.patch_size}')
        self._logger.info(f'   - 编码器层数: {self.num_layers}')

        # 打印消融开关状态
        ablation_flags = {
            'use_stdan': self.use_stdan, 'use_hettle': self.use_hettle,
            'use_moe_experts': self.use_moe_experts, 'use_progressive_refine': self.use_progressive_refine,
            'use_gated_fusion': self.use_gated_fusion, 'use_adaptive_adj': self.use_adaptive_adj,
        }
        disabled = [k for k, v in ablation_flags.items() if not v]
        if disabled:
            self._logger.info(f'   ⚠️ 消融模式: 已禁用 {disabled}')
        else:
            self._logger.info(f'   - 消融开关: 全部启用 (完整模型)')

    def _build_model(self):
        """构建 TIDAL 模型"""
        class Args:
            pass

        args = Args()
        args.input_length = self.input_window
        args.predict_length = self.output_window
        args.in_dim = self.feature_dim
        args.pre_dim = self.output_dim
        args.num_nodes = self.num_nodes
        args.hid_dim = self.hid_dim
        args.d_out = self.d_out
        args.n_heads = self.n_heads
        args.M = self.M
        args.num_layers = self.num_layers
        args.patch_size = self.patch_size
        args.query_cross_time = self.query_cross_time
        args.return_att = self.return_att
        args.att_type = self.att_type
        args.activation_data = self.activation_data
        args.activation_dec = self.activation_dec
        args.te_emb_dropout = self.te_emb_dropout
        args.se_emb_dropout = self.se_emb_dropout
        args.tle_emb_dropout = self.tle_emb_dropout
        args.enc_dropout = self.enc_dropout
        args.att_dropout = self.att_dropout
        args.prob_factor = self.prob_factor
        args.slice_size_per_day = self.slice_size_per_day

        # ===== 传递消融开关 =====
        args.use_stdan = self.use_stdan
        args.use_hettle = self.use_hettle
        args.use_moe_experts = self.use_moe_experts
        args.use_progressive_refine = self.use_progressive_refine
        args.use_gated_fusion = self.use_gated_fusion
        args.use_adaptive_adj = self.use_adaptive_adj

        self.model = TIDALModel(args)

        # Xavier 初始化
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        """前向传播"""
        x = batch['X']  # Grid: (B, T, row, col, F) 或 Point: (B, T, N, F)

        # Grid 数据集是 5D, 需要 reshape 为 4D
        self._grid_shape = None
        if x.dim() == 5:
            B, T, R, C_grid, F = x.shape
            self._grid_shape = (R, C_grid)
            x = x.reshape(B, T, R * C_grid, F)  # (B, T, N, F)
        B, T, N, F = x.shape

        x_time = self._extract_time_features(batch, B, T)
        x_traffic = x[..., 0:self.feature_dim]

        main_output, coarse_output, _ = self.model(x_traffic, x_time)
        # 将 coarse 保存用于辅助损失
        self._last_coarse = coarse_output
        return main_output

    def _extract_time_features(self, batch, B, T):
        """
        从 batch 中提取时间特征

        Returns:
            x_time: (B, T, 2) - [:,:,0]=时间片索引, [:,:,1]=星期索引
        """
        x = batch['X']
        # Grid 数据是 5D (B,T,row,col,F), 先展平
        if x.dim() == 5:
            B_, T_, R, C_g, F_ = x.shape
            x = x.reshape(B_, T_, R * C_g, F_)
        N = x.shape[2]

        def _to_bt(tensor, expect_nodes):
            if tensor.dim() == 2:
                return tensor
            if tensor.dim() == 3:
                if tensor.shape[-1] == 1:
                    return tensor.squeeze(-1)
                if tensor.shape[-1] == expect_nodes:
                    return tensor[:, :, 0]
                if tensor.shape[-1] == 7:
                    return tensor.argmax(dim=-1).long()
            if tensor.dim() == 4:
                if tensor.shape[-1] == 7:
                    return tensor[:, :, 0, :].argmax(dim=-1).long()
                if tensor.shape[-1] == 1:
                    return tensor.squeeze(-1)[:, :, 0]
            out = tensor
            while out.dim() > 2:
                out = out.squeeze(-1)
            return out

        if hasattr(batch, 'keys') and 'time_in_day_feat' in batch and 'day_in_week_feat' in batch:
            time_in_day = batch['time_in_day_feat']
            day_in_week = batch['day_in_week_feat']

            time_in_day_bt = _to_bt(time_in_day, N)
            day_in_week_bt = _to_bt(day_in_week, N)

            if time_in_day_bt.dtype.is_floating_point:
                time_idx = torch.floor(time_in_day_bt * self.slice_size_per_day).long()
            else:
                time_idx = time_in_day_bt.long()
            time_idx = torch.clamp(time_idx, 0, self.slice_size_per_day - 1)

            if day_in_week_bt.dtype.is_floating_point:
                day_idx = torch.round(day_in_week_bt).long()
            else:
                day_idx = day_in_week_bt.long()
            day_idx = torch.clamp(day_idx, 0, 6)

            x_time = torch.stack([time_idx.to(x.device), day_idx.to(x.device)], dim=-1)
        else:
            if x.shape[-1] > self.feature_dim:
                if x.shape[-1] >= self.feature_dim + 1:
                    time_feat = x[..., self.feature_dim]
                    base = time_feat[:, :, 0]
                    time_idx = torch.floor(base * self.slice_size_per_day).long()
                else:
                    time_idx = torch.zeros(B, T, dtype=torch.long, device=x.device)

                if x.shape[-1] >= self.feature_dim + 8:
                    day_onehot = x[..., self.feature_dim + 1:self.feature_dim + 8]
                    day_idx = day_onehot[:, :, 0, :].argmax(dim=-1).long()
                else:
                    day_idx = torch.zeros(B, T, dtype=torch.long, device=x.device)

                time_idx = torch.clamp(time_idx, 0, self.slice_size_per_day - 1)
                day_idx = torch.clamp(day_idx, 0, 6)

                x_time = torch.stack([time_idx, day_idx], dim=-1)
            else:
                x_time = torch.zeros(B, T, 2, dtype=torch.long, device=x.device)

        return x_time

    def get_loss_func(self, set_loss='huber'):
        """获取损失函数"""
        if set_loss == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        else:
            self._logger.warning(f'Unrecognized loss function {set_loss}, using huber loss')
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        return lf

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='huber'):
        """不执行预测直接计算损失"""
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None):
        """计算损失 (主损失 + 辅助粗预测损失)"""
        y_true = batch['y']
        # Grid 数据 y_true 是 5D, 展平为 4D
        if y_true.dim() == 5:
            B_, T_, R, C_g, F_ = y_true.shape
            y_true = y_true.reshape(B_, T_, R * C_g, F_)

        y_predicted = self.forward(batch)

        # 主损失
        main_loss = self.calculate_loss_without_predict(
            y_true, y_predicted, batches_seen, set_loss=self.set_loss
        )

        # 辅助损失: 粗预测也应接近真实值 (加速收敛)
        aux_loss = self.calculate_loss_without_predict(
            y_true, self._last_coarse, batches_seen, set_loss=self.set_loss
        )

        return main_loss + 0.1 * aux_loss

    def predict(self, batch):
        """预测"""
        output = self.forward(batch)
        # Grid 数据需要 reshape 回 5D 给 evaluator
        if self._grid_shape is not None:
            R, C_g = self._grid_shape
            B_, T_ = output.shape[:2]
            output = output.reshape(B_, T_, R, C_g, -1)
        return output
