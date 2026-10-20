"""指标层

多标签分类相关指标：每类 AUC、macro AUC、macro F1、检查级（Study）聚合。
全部纯 numpy 实现，不依赖 scikit-learn。
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """对数值做平均秩（处理并列值），1-based。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, y_score) -> float:
    """单类二分类 AUC（Mann-Whitney U 统计量）；正负样本只有一类时返回 nan。"""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(y_score)
    sum_pos_ranks = float(ranks[y_true == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def per_class_auc(y_true, y_score, class_ids: List[int]) -> Tuple[np.ndarray, float]:
    """返回每类 AUC 数组 + macro AUC（忽略单类别列）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    aucs = np.array([roc_auc(y_true[:, c], y_score[:, c]) for c in range(y_true.shape[1])],
                    dtype=float)
    valid = aucs[~np.isnan(aucs)]
    macro = float(valid.mean()) if len(valid) else float("nan")
    return aucs, macro


def macro_f1(y_true, y_score, threshold: float = 0.5) -> float:
    """宏平均 F1（辅助指标）。y_score 为概率，按 threshold 二值化。"""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    fs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c].astype(int)
        yp = y_pred[:, c].astype(int)
        if yt.sum() == 0:
            # 该类别在验证集无正样本，跳过
            continue
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        if tp == 0:
            fs.append(0.0)
        else:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            fs.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(fs)) if fs else float("nan")


def best_threshold_per_class(y_true, y_score, thresholds=None) -> List[float]:
    """对每个类别在验证集上搜索最大化 F1 的阈值。

    返回长度为 C 的阈值列表；某类无正样本时回退 0.5。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)
    best = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c].astype(int)
        if yt.sum() == 0:
            best.append(0.5)
            continue
        best_t, best_f = 0.5, -1.0
        for t in thresholds:
            yp = (y_score[:, c] >= t).astype(int)
            tp = int(((yp == 1) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            if tp == 0:
                f = 0.0
            else:
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f = 2 * prec * rec / max(prec + rec, 1e-12)
            if f > best_f:
                best_f, best_t = f, float(t)
        best.append(best_t)
    return best


def macro_f1_at_thresholds(y_true, y_score, thresholds) -> float:
    """给定每类阈值时的 macro F1。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c].astype(int)
        if yt.sum() == 0:
            continue
        yp = (y_score[:, c] >= float(thresholds[c])).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        if tp == 0:
            fs.append(0.0)
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        fs.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(fs)) if fs else float("nan")


def aggregate_by_study(study_ids, probs, pool: str = "mean") -> Tuple[List[str], np.ndarray]:
    """把图像级预测聚合成检查级。

    返回 (study_list, probs_matrix)，study_list 按 study_id 排序。
    """
    df = pd.DataFrame(np.asarray(probs), index=pd.Index(study_ids, name="study"))
    if pool == "max":
        agg = df.groupby("study").max()
    else:
        agg = df.groupby("study").mean()
    return list(agg.index), agg.values
