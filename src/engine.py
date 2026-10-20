"""引擎层

训练一个 epoch / 验证 / 测试集预测。
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import per_class_auc


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss，多标签类别不均衡的业界标准方案。

    参考: https://arxiv.org/abs/2009.14119
    - gamma_neg: 负样本 focusing 指数（越大越抑制 easy negatives）
    - gamma_pos: 正样本 focusing 指数
    - clip(margin): 对负样本概率做 shift，p < clip 的负样本完全不惩罚
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        # 正样本：-y * log(p) * (1-p)^gamma_pos
        pos_loss = -targets * torch.log(p.clamp(min=self.eps)) * (1 - p).pow(self.gamma_pos)
        # 负样本：-(1-y) * log(1-p_m) * p_m^gamma_neg，其中 p_m = max(p - clip, 0)
        p_shift = (p - self.clip).clamp(min=0.0)
        neg_loss = -(1 - targets) * torch.log((1 - p_shift).clamp(min=self.eps)) * p_shift.pow(self.gamma_neg)
        return (pos_loss + neg_loss).mean()


def build_criterion(loss_name: str, pos_weight=None, device=None):
    """构造损失函数。loss_name: bce / asl。"""
    if loss_name == "asl":
        return AsymmetricLoss()
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_epoch(model: nn.Module, loader, criterion, optimizer, device) -> float:
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> Tuple[np.ndarray, np.ndarray, list]:
    """验证：返回 (y, probs, study_ids)。probs 为 sigmoid 后的概率。"""
    model.eval()
    all_y, all_p, all_s = [], [], []
    for x, y, s in loader:
        x = x.to(device)
        logits = model(x)
        all_y.append(y.numpy())
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_s.extend(s)
    return (
        np.concatenate(all_y, axis=0),
        np.concatenate(all_p, axis=0),
        all_s,
    )


@torch.no_grad()
def predict(model: nn.Module, loader, device, tta: bool = False) -> Tuple[np.ndarray, list]:
    """测试集预测：返回 (probs, study_ids)。

    tta=True 时做水平翻转 TTA（原图 + 翻转图，logits 平均），通常 +0.3~0.5% AUC。
    """
    model.eval()
    all_p, all_s = [], []
    for x, _, s in loader:
        x = x.to(device)
        logits = model(x)
        if tta:
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_s.extend(s)
    return np.concatenate(all_p, axis=0), all_s


def eval_study_metric(model: nn.Module, loader, device, study_labels,
                      class_ids, pool: str = "mean", threshold: float = 0.5):
    """在验证集上算检查级（Study）macro AUC 与 macro F1。

    study_labels: {study_id: np.array[C]}
    返回 (val_img_macro, val_study_macro, per_class_aucs, val_study_f1, y_study, p_study)
    """
    from .metrics import aggregate_by_study, macro_f1

    y_img, p_img, s_img = evaluate(model, loader, device)
    _, img_macro = per_class_auc(y_img, p_img, class_ids)

    studies, p_study = aggregate_by_study(s_img, p_img, pool=pool)
    y_study = np.array([study_labels[s] for s in studies])
    aucs, study_macro = per_class_auc(y_study, p_study, class_ids)
    study_f1 = macro_f1(y_study, p_study, threshold=threshold)
    return img_macro, study_macro, aucs, study_f1, y_study, p_study


# --------------------------------------------------------------------------- #
# 多图 Study 级模型（联合建模）训练/验证/预测
# --------------------------------------------------------------------------- #
def train_epoch_multi(model: nn.Module, loader, criterion, optimizer, device) -> float:
    """训练一个 epoch（Study 级多图），返回平均 loss（按 Study 计）。"""
    model.train()
    total_loss, n = 0.0, 0
    for images, sizes, ys, _ in loader:
        images, ys = images.to(device), ys.to(device)
        optimizer.zero_grad()
        logits = model(images, sizes)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ys.size(0)
        n += ys.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_multi(model: nn.Module, loader, device) -> Tuple[np.ndarray, np.ndarray, list]:
    """验证（Study 级多图）：返回 (y, probs, study_ids)。probs 为 sigmoid 概率。"""
    model.eval()
    all_y, all_p, all_s = [], [], []
    for images, sizes, ys, sids in loader:
        images = images.to(device)
        logits = model(images, sizes)
        all_y.append(ys.numpy())
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_s.extend(sids)
    return (
        np.concatenate(all_y, axis=0),
        np.concatenate(all_p, axis=0),
        all_s,
    )


@torch.no_grad()
def predict_multi(model: nn.Module, loader, device, tta: bool = False) -> Tuple[np.ndarray, list]:
    """测试集预测（Study 级多图）：返回 (probs, study_ids)。"""
    model.eval()
    all_p, all_s = [], []
    for images, sizes, _, sids in loader:
        images = images.to(device)
        logits = model(images, sizes)
        if tta:
            logits = (logits + model(torch.flip(images, dims=[3]), sizes)) / 2.0
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_s.extend(sids)
    return np.concatenate(all_p, axis=0), all_s
