import torch
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def log_cosh_loss(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return torch.mean(loss)


def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


def masked_huber_loss(preds, labels, delta=1.0, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def quantile_loss(preds, labels, delta=0.25):
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))


def masked_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask_mean = torch.mean(mask)
    if mask_mean < 1e-8:
        return torch.tensor(0.0, device=preds.device)
    mask /= mask_mean
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val))


def r2_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return r2_score(labels, preds)


def explained_variance_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return explained_variance_score(labels, preds)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels,
                   null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def r2_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return r2_score(labels, preds)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)


# ========== E1方案：复合损失函数实现 ==========
from typing import Optional
import torch.nn as nn

def _apply_log1p(x: torch.Tensor) -> torch.Tensor:
    """数值安全：预测若出现极少数负值，先截断到0再log1p"""
    return torch.log1p(torch.clamp(x, min=0.0))

def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """计算带掩码的MAE"""
    if mask is not None:
        diff = (pred - target).abs() * mask
        denom = mask.sum().clamp_min(1.0)
        return diff.sum() / denom
    return (pred - target).abs().mean()

def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """计算带掩码的MSE"""
    if mask is not None:
        diff2 = ((pred - target) ** 2) * mask
        denom = mask.sum().clamp_min(1.0)
        return diff2.sum() / denom
    return ((pred - target) ** 2).mean()

class CompositeMAERMSE(nn.Module):
    """
    复合损失：w1·MAE + w2·RMSE（直接在原空间计算）。
    """
    def __init__(self, w_mae: float = 0.7, w_rmse: float = 0.3):
        super().__init__()
        self.w_mae = float(w_mae)
        self.w_rmse = float(w_rmse)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        l1 = _masked_mae(pred, target, mask)
        rmse = torch.sqrt(_masked_mse(pred, target, mask) + 1e-8)
        return self.w_mae * l1 + self.w_rmse * rmse

class CompositeMAERMSELog1p(nn.Module):
    """
    在损失内部对 pred/target 同时做 log1p，再计算 w1·MAE + w2·RMSE。
    无需改 Trainer 即可实现"log 空间训练"，MAPE 通常更稳。
    """
    def __init__(self, w_mae: float = 0.7, w_rmse: float = 0.3):
        super().__init__()
        self.w_mae = float(w_mae)
        self.w_rmse = float(w_rmse)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_t = _apply_log1p(pred)
        target_t = _apply_log1p(target)
        l1 = _masked_mae(pred_t, target_t, mask)
        rmse = torch.sqrt(_masked_mse(pred_t, target_t, mask) + 1e-8)
        return self.w_mae * l1 + self.w_rmse * rmse

# 为了兼容现有的函数式调用方式，提供函数包装器
def composite_mae_rmse_torch(preds, labels, w_mae=0.7, w_rmse=0.3, null_val=np.nan):
    """函数式复合损失：MAE + RMSE"""
    # 创建掩码（复用现有逻辑）
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # 计算复合损失
    loss_fn = CompositeMAERMSE(w_mae=w_mae, w_rmse=w_rmse)
    return loss_fn(preds, labels, mask)

def composite_mae_rmse_log1p_torch(preds, labels, w_mae=0.7, w_rmse=0.3, null_val=np.nan):
    """函数式复合损失：log1p空间的MAE + RMSE"""
    # 创建掩码（复用现有逻辑）
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # 计算log1p空间复合损失
    loss_fn = CompositeMAERMSELog1p(w_mae=w_mae, w_rmse=w_rmse)
    return loss_fn(preds, labels, mask)


# ========== MAPE优化方案 ==========

def smape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    """
    对称平均绝对百分比误差 (Symmetric MAPE)
    优势：
    1. 数值稳定，避免除零问题
    2. 对称性：对高估和低估同等惩罚
    3. 范围有界：0-200%
    
    公式：SMAPE = mean(2 * |pred - true| / (|pred| + |true| + epsilon))
    """
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    epsilon = 1e-5
    numerator = 2.0 * torch.abs(preds - labels)
    denominator = torch.abs(preds) + torch.abs(labels) + epsilon
    loss = numerator / denominator
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def weighted_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    """
    加权MAPE - 对不同流量范围给予不同权重
    
    权重策略：
    - 低流量 (0-10): 权重 0.3
    - 中流量 (10-50): 权重 1.0  
    - 高流量 (>50): 权重 1.5
    
    优势：更关注高流量时段的预测准确性
    """
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    
    # 计算流量权重
    flow_weights = torch.ones_like(labels)
    flow_weights[labels < 10] = 0.3
    flow_weights[(labels >= 10) & (labels < 50)] = 1.0
    flow_weights[labels >= 50] = 1.5
    
    # 结合mask和流量权重
    combined_mask = mask * flow_weights
    combined_mask /= torch.mean(combined_mask)
    combined_mask = torch.where(torch.isnan(combined_mask), torch.zeros_like(combined_mask), combined_mask)
    
    epsilon = 1e-5
    loss = torch.abs((preds - labels) / (labels + epsilon))
    loss = loss * combined_mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def log_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    """
    对数空间MAPE
    
    公式：mean(|log(pred+1) - log(true+1)| / |log(true+1)| + epsilon)
    
    优势：
    1. 对极小值和极大值都稳定
    2. 适合流量变化范围大的场景
    3. 减少异常值影响
    """
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # log1p变换
    log_preds = torch.log1p(torch.clamp(preds, min=0))
    log_labels = torch.log1p(torch.clamp(labels, min=0))
    
    epsilon = 1e-5
    loss = torch.abs((log_preds - log_labels) / (log_labels + epsilon))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def wape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    """
    加权绝对百分比误差 (WAPE)
    
    公式：WAPE = sum(|pred - true|) / sum(true)
    
    优势：
    1. 更关注总体预测准确性
    2. 对个别异常值不敏感
    3. 业务解释性强
    """
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    
    epsilon = 1e-5
    abs_error = torch.abs(preds - labels) * mask
    true_sum = (labels * mask).sum() + epsilon
    
    wape = abs_error.sum() / true_sum
    return wape


def rmspe_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    """
    均方根百分比误差 (RMSPE)
    
    公式：RMSPE = sqrt(mean((pred - true)^2 / true^2))
    
    优势：
    1. 对大误差更敏感
    2. 常用于评估时间序列预测
    """
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    epsilon = 1e-5
    squared_percentage_error = torch.square((preds - labels) / (labels + epsilon))
    loss = squared_percentage_error * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))


def composite_mae_mape_torch(preds, labels, w_mae=0.7, w_mape=0.3, null_val=np.nan):
    """
    MAE + MAPE 复合损失（针对交通流量预测优化）
    
    推荐配置：
    - w_mae=0.7, w_mape=0.3：主要优化绝对误差，辅助优化相对误差
    - w_mae=0.6, w_mape=0.4：更平衡的优化
    
    优势：
    1. 直接优化MAPE目标
    2. MAE提供稳定的梯度
    3. MAPE引导模型关注相对误差
    
    注意：MAPE损失会自动过滤低流量时段（<5.0）
    """
    mae_loss = masked_mae_torch(preds, labels, null_val)
    # masked_mape_torch会自动使用threshold=5.0过滤低流量
    mape_loss = masked_mape_torch(preds, labels, null_val)
    
    return w_mae * mae_loss + w_mape * mape_loss


def composite_mae_smape_torch(preds, labels, w_mae=0.7, w_smape=0.3, null_val=np.nan, mask_val=np.nan):
    """
    MAE + SMAPE 复合损失
    
    推荐配置：
    - w_mae=0.7, w_smape=0.3：关注绝对误差，同时优化相对误差
    - w_mae=0.5, w_smape=0.5：平衡绝对和相对误差
    
    优势：
    1. 同时优化MAE和SMAPE
    2. SMAPE更稳定，适合作为辅助损失
    3. 提升整体预测质量
    """
    mae_loss = masked_mae_torch(preds, labels, null_val, mask_val)
    smape_loss = smape_torch(preds, labels, null_val, mask_val)
    
    return w_mae * mae_loss + w_smape * smape_loss
