"""
时空嵌入层 — 时间嵌入 + 空间嵌入
(Time Embedding + Spatial Embedding)

TIDAL 自有模块
"""

import torch
import torch.nn as nn


class getTimeEmbedding(nn.Module):
    """
    时间嵌入层 (Time Embedding)

    将离散的时间信息（时间片索引、星期索引）编码为连续的向量表示

    时间特征包含两个维度:
    1. 一天中的时间 (Time-of-Day): 0~287 (假设5分钟间隔，共288个时间片)
    2. 一周中的星期 (Day-of-Week): 0~6 (周一到周日)

    参数:
        hid_dim (int): 嵌入向量维度
        slice_size_per_day (int): 每天的时间片数量（例如: 5分钟间隔=288）
        emb_dropout (float): 嵌入层Dropout概率

    输入:
        time_data: (B, T, 2) - [:,:,0]=时间片索引, [:,:,1]=星期索引

    输出:
        time_emb: (B, T, hid_dim) - 时间嵌入向量
    """
    def __init__(self, hid_dim, slice_size_per_day, emb_dropout=0.1):
        super().__init__()
        self.time_in_day_embedding = nn.Embedding(slice_size_per_day, hid_dim)
        self.day_in_week_embedding = nn.Embedding(7, hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, time_data):
        t_hour = time_data[..., 0:1].squeeze(-1).long()
        t_day = time_data[..., 1:2].squeeze(-1).long()

        time_in_day_emb = self.time_in_day_embedding(t_hour)
        day_in_week_emb = self.day_in_week_embedding(t_day)

        time_emb = time_in_day_emb + day_in_week_emb
        time_emb = self.dropout(time_emb)
        return time_emb


class getSpatialEmbedding(nn.Module):
    """
    空间嵌入层 (Spatial Embedding)

    为每个交通节点学习一个唯一的空间位置嵌入

    参数:
        hid_dim (int): 嵌入向量维度
        num_nodes (int): 节点总数
        emb_dropout (float): Dropout概率

    输入:
        x: (B, T, N, C) - 用于获取节点数
        spatial_indexs: (N,) - 节点索引，默认为[0,1,...,N-1]

    输出:
        spatial_emb: (1, 1, N, hid_dim) - 空间嵌入
    """
    def __init__(self, hid_dim, num_nodes, emb_dropout=0.1):
        super().__init__()
        self.spatial_embedding = nn.Embedding(num_nodes, hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, spatial_indexs=None):
        if spatial_indexs is None:
            batch, _, num_nodes, _ = x.shape
            spatial_indexs = torch.LongTensor(torch.arange(num_nodes)).to(x.device)

        spatial_emb = self.spatial_embedding(spatial_indexs).unsqueeze(0).unsqueeze(1)
        spatial_emb = self.dropout(spatial_emb)
        return spatial_emb
