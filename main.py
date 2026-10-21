"""胸部 X 光片多标签分类 baseline 入口。

项目已拆分到 src/：
- src/config.py：配置数据类
- src/utils.py：随机种子、日志、JSON
- src/data.py：数据读取、标签解析、图片索引、增强、Dataset/DataLoader
- src/models.py：DenseNet121 / EfficientNet-B3 / ConvNeXt-Tiny 构造
- src/engine.py：训练 / 验证 / 预测循环
- src/metrics.py：每类 AUC、macro AUC、检查级聚合

用法：
  python main.py --mode check  --train_csv ... --valid_csv ... --image_dir ...
  python main.py --mode train  --train_csv ... --valid_csv ... --image_dir ...
  python main.py --mode predict --test_csv ... --image_dir ... --checkpoint ...
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler

from src.config import Config, CLASS_NAMES
from src.utils import log, save_json, seed_everything
from src.data import (
    build_labels,
    build_samples,
    compute_study_weights,
    detect_label_format,
    find_col,
    load_label_df,
    make_loader,
    resolve_study_images,
    resolve_split_image_dir,
    ChestXrayDataset,
    StudyDataset,
    make_study_loader,
    STUDY_COL_CANDIDATES,
    SUBJECT_COL_CANDIDATES,
)
from src.models import build_model, MultiImageModel
from src.engine import (
    train_epoch, evaluate, predict, eval_study_metric, build_criterion,
    train_epoch_multi, evaluate_multi, predict_multi,
)
from src.metrics import (
    aggregate_by_study, per_class_auc, macro_f1,
    best_threshold_per_class, macro_f1_at_thresholds,
)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="胸部 X 光多标签分类 baseline")
    p.add_argument("--mode", choices=["check", "train", "eval", "kfold", "predict"], default="check")
    p.add_argument("--train_csv", default=None)
    p.add_argument("--valid_csv", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--image_dir", default=None, help="图片根目录（含 pic_xxx.jpg）")
    p.add_argument("--output_dir", default="outputs/baseline")
    p.add_argument("--checkpoint", default=None,
                   help="predict 模式需要；可用逗号分隔多个 checkpoint 做集成（概率平均）")
    p.add_argument("--backbone", default="densenet121",
                   choices=["densenet121", "efficientnet_b3", "convnext_tiny"])
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--loss", choices=["bce", "asl"], default="bce",
                   help="bce=BCEWithLogits+pos_weight；asl=Asymmetric Loss（更适合类别不均衡）")
    p.add_argument("--asl_gamma_neg", type=float, default=4.0)
    p.add_argument("--asl_gamma_pos", type=float, default=1.0)
    p.add_argument("--asl_clip", type=float, default=0.05)
    p.add_argument("--sampler", choices=["none", "balanced"], default="none",
                   help="balanced=类别平衡过采样（缓解少数类）")
    p.add_argument("--sampler_boost", type=float, default=1.0,
                   help="采样加权强度，越大越偏向少数类")
    p.add_argument("--pool", choices=["mean", "max", "attn"], default="mean",
                   help="多图聚合方式：mean/max/attn（multi 模式用特征池化）")
    p.add_argument("--multi", action="store_true",
                   help="多图 Study 级联合建模（同一 Study 多张图特征池化后联合预测）")
    p.add_argument("--kfold", type=int, default=5,
                   help="K 折交叉验证折数（--mode kfold 用）")
    p.add_argument("--pos_weight_cap", type=float, default=20.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", default=None,
                   help="手动指定每类阈值（逗号分隔10个），覆盖 checkpoint 里的阈值")
    p.add_argument("--no_tune_threshold", action="store_false", dest="tune_threshold",
                   help="关闭验证集上的每类阈值搜索")
    p.add_argument("--eval_train", action="store_true",
                   help="每轮额外评估训练集（耗时，仅用于观察过拟合）")
    p.add_argument("--no_tta", action="store_false", dest="tta",
                   help="关闭测试时水平翻转 TTA")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    args = p.parse_args()
    cfg = Config(**vars(args))
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


# --------------------------------------------------------------------------- #
# 公共逻辑
# --------------------------------------------------------------------------- #
def prepare_data(cfg: Config, csv_path: str, mode: str, split: str = "train", class_ids=None):
    """加载并解析一份 CSV，返回统一结构。

    split: train/valid/test，用于定位图片子目录（若 image_dir 下有对应子目录）。

    返回 dict:
      df, study_col, dicom_col, labels(DataFrame|None), class_ids, images_map, samples, dropped, missing
    """
    df = load_label_df(csv_path, mode)
    study_col, dicom_col, class_col, detected_ids, has_label = detect_label_format(df)

    if class_ids is None:
        class_ids = detected_ids
    log(f"[{mode}] 标签格式解析: study_col={study_col}, dicom_col={dicom_col}, "
        f"类别数={len(class_ids) if class_ids else 0}, 类别id={class_ids}")

    # 图片按 Dicom_id 匹配；image_dir 支持直接指向 split 子目录，或指向含 train/valid/test 的父目录
    image_dir = resolve_split_image_dir(cfg.image_dir, split)
    log(f"[{mode}] 图片目录: {image_dir}")
    images, missing = resolve_study_images(df, study_col, dicom_col, image_dir)
    total_studies = len(set(df[study_col].astype(str).str.strip()))
    log(f"[{mode}] 图片解析: {len(images)}/{total_studies} 个 Study 有图，"
        f"缺失图片数={len(missing)}")
    if missing:
        log(f"[{mode}] 缺失示例: {missing[:10]}")

    labels = None
    if has_label and class_col is not None:
        labels = build_labels(df, study_col, class_col, class_ids)
        # 只保留有图片的 Study
        valid_studies = [s for s in labels.index if s in images]
        labels = labels.loc[valid_studies]
        log(f"[{mode}] 有标签且有图的 Study: {len(labels)}")

    valid_studies = [s for s in images.keys() if labels is None or s in labels.index]
    samples = build_samples(valid_studies, images)
    dropped = sum(1 for s in images if labels is not None and s not in labels.index)
    return {
        "df": df, "study_col": study_col, "dicom_col": dicom_col,
        "labels": labels, "class_ids": class_ids,
        "images": images, "samples": samples,
        "dropped": dropped, "missing": missing, "has_label": has_label,
    }


def compute_pos_weight(labels: pd.DataFrame, cap: float) -> torch.Tensor:
    """计算各类 pos_weight = neg/pos，并截断到 [1, cap]。"""
    pos = labels.values.sum(axis=0)
    neg = len(labels) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, cap), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# check / train / predict
# --------------------------------------------------------------------------- #
def cmd_check(cfg: Config):
    data = prepare_data(cfg, cfg.train_csv, "训练", split="train")
    labels, class_ids = data["labels"], data["class_ids"]
    log(f"训练 Study 数（有标签有图）: {len(labels)}")

    pos = labels.values.sum(axis=0)
    log("各类正样本分布:")
    for cid, pc in zip(class_ids, pos):
        name = CLASS_NAMES.get(cid, ("", ""))[1]
        log(f"  class {cid} ({name}): pos={int(pc)}  ({pc/len(labels):.4f})")

    total_imgs = len(data["samples"])
    n_studies = len(data["images"])
    log(f"图像总数: {total_imgs}  平均每 Study {total_imgs/max(n_studies,1):.2f} 张")
    multi = sum(1 for v in data["images"].values() if len(v) > 1)
    log(f"多图 Study 数: {multi}  ({multi/max(n_studies,1):.4f})")

    if data["images"]:
        sid0, paths0 = next(iter(data["images"].items()))
        log(f"示例: Study={sid0} -> {paths0[:3]}")
    log("\n自检通过。若统计与数据一致，即可执行 --mode train。")


def split_folds(study_ids, k: int, seed: int):
    """把 study 列表随机均分成 k 折（用于交叉验证）。"""
    rng = np.random.RandomState(seed)
    ids = np.array(study_ids)
    rng.shuffle(ids)
    folds = np.array_split(ids, k)
    return [list(f) for f in folds]


def _train_fold(cfg: Config, device, class_ids, train_labels, train_samples, train_images,
                valid_labels=None, valid_samples=None, valid_images=None,
                output_dir=None, fold_tag=""):
    """训练一个 fold（或单次训练），早停按验证集 val_study_macro_auc。

    返回 (best_auc, best_epoch, thresholds)。
    """
    seed_everything(cfg.seed)
    output_dir = output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pos_weight = compute_pos_weight(train_labels, cfg.pos_weight_cap).to(device)
    valid_studies = list(valid_labels.index) if valid_labels is not None else None
    multi = cfg.multi

    train_label_lookup = {s: train_labels.loc[s].values for s in train_labels.index}
    train_study_images = {s: train_images[s] for s in train_labels.index}

    # 少数类平衡采样（类别平衡过采样）
    sampler = None
    if cfg.sampler == "balanced":
        study_w = compute_study_weights(train_labels, cfg.sampler_boost)
        w_map = {s: float(w) for s, w in zip(train_labels.index, study_w)}
        if multi:
            study_list = sorted(train_study_images.keys())
            weights = torch.tensor([w_map[s] for s in study_list], dtype=torch.float)
        else:
            weights = torch.tensor([w_map[s] for s, _ in train_samples], dtype=torch.float)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        log(f"启用类别平衡采样（boost={cfg.sampler_boost}）")

    if multi:
        train_loader = make_study_loader(
            StudyDataset(train_study_images, train_label_lookup, class_ids,
                         cfg.image_size, train=True),
            cfg.batch_size, shuffle=True, workers=cfg.workers, sampler=sampler)
        model = MultiImageModel(cfg.backbone, len(class_ids), pretrained=cfg.pretrained,
                                dropout=cfg.dropout, pool=cfg.pool).to(device)
    else:
        train_loader = make_loader(
            ChestXrayDataset(train_samples, train_label_lookup, class_ids,
                             cfg.image_size, train=True),
            cfg.batch_size, shuffle=True, workers=cfg.workers, sampler=sampler)
        model = build_model(cfg.backbone, len(class_ids), pretrained=cfg.pretrained,
                            dropout=cfg.dropout).to(device)

    criterion = build_criterion(cfg.loss, pos_weight=pos_weight, device=device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_auc, best_epoch, patience = -1.0, -1, 0
    best_y_study, best_p_study = None, None
    history = []

    tag = f" [{fold_tag}]" if fold_tag else ""
    log(f"\n开始训练{tag}: backbone={cfg.backbone}  size={cfg.image_size}  "
        f"batch={cfg.batch_size}  epochs={cfg.epochs}  lr={cfg.lr}  "
        f"train_studies={len(train_labels)}  valid_studies={len(valid_studies or [])}\n")

    for epoch in range(1, cfg.epochs + 1):
        import time
        t0 = time.time()
        if multi:
            tr_loss = train_epoch_multi(model, train_loader, criterion, optimizer, device)
        else:
            tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        tr_macro = float("nan")
        if cfg.eval_train:
            if multi:
                tr_y, tr_p, _ = evaluate_multi(model, train_loader, device)
            else:
                tr_y, tr_p, _ = evaluate(model, train_loader, device)
            _, tr_macro = per_class_auc(tr_y, tr_p, class_ids)

        val_msg = "（无验证集）"
        monitor = -tr_loss
        va_aucs = None
        va_study_macro = float("nan")
        va_study_f1 = float("nan")
        if valid_samples is not None:
            valid_label_lookup = {s: valid_labels.loc[s].values for s in valid_studies}
            if multi:
                valid_study_images = {s: valid_images[s] for s in valid_studies}
                valid_loader = make_study_loader(
                    StudyDataset(valid_study_images, valid_label_lookup, class_ids,
                                 cfg.image_size, train=False),
                    cfg.batch_size, shuffle=False, workers=cfg.workers)
                y_st, p_st, _ = evaluate_multi(model, valid_loader, device)
                va_aucs, va_study_macro = per_class_auc(y_st, p_st, class_ids)
                va_study_f1 = macro_f1(y_st, p_st, cfg.threshold)
                va_img_macro = va_study_macro
            else:
                valid_loader = make_loader(
                    ChestXrayDataset(valid_samples, valid_label_lookup, class_ids,
                                     cfg.image_size, train=False),
                    cfg.batch_size, shuffle=False, workers=cfg.workers)
                va_img_macro, va_study_macro, va_aucs, va_study_f1, y_st, p_st = eval_study_metric(
                    model, valid_loader, device, valid_label_lookup, class_ids,
                    pool=cfg.pool, threshold=cfg.threshold)
            val_msg = (f"val_img_macro={va_img_macro:.4f}  "
                       f"val_study_macro={va_study_macro:.4f}  val_study_f1={va_study_f1:.4f}")
            monitor = va_study_macro

        scheduler.step()
        dt = time.time() - t0
        train_part = f"train_macro={tr_macro:.4f}  " if cfg.eval_train else ""
        log(f"epoch {epoch:02d}/{cfg.epochs}  loss={tr_loss:.4f}  "
            f"{train_part}{val_msg}  ({dt:.1f}s)")
        if va_aucs is not None:
            per_cls = "  ".join(f"c{cid}={a:.3f}" for cid, a in zip(class_ids, va_aucs))
            log(f"        每类AUC: {per_cls}")

        row = {"epoch": epoch, "train_loss": tr_loss}
        if cfg.eval_train:
            row["train_macro_auc"] = tr_macro
        history.append(row)
        if va_aucs is not None:
            history[-1]["val_study_macro_auc"] = va_study_macro
            history[-1]["val_study_macro_f1"] = va_study_f1
            history[-1]["val_img_macro_auc"] = va_img_macro
            for cid, a in zip(class_ids, va_aucs):
                history[-1][f"auc_class_{cid}"] = a

        if monitor > best_auc:
            best_auc, best_epoch, patience = monitor, epoch, 0
            if va_aucs is not None:
                best_y_study, best_p_study = y_st, p_st
            ckpt_name = f"best_model_{fold_tag}.pt" if fold_tag else "best_model.pt"
            torch.save({
                "model_state": model.state_dict(),
                "backbone": cfg.backbone,
                "image_size": cfg.image_size,
                "class_ids": class_ids,
                "pool": cfg.pool,
                "multi": multi,
                "dropout": cfg.dropout,
                "thresholds": [cfg.threshold] * len(class_ids),
                "best_auc": best_auc,
                "epoch": epoch,
            }, os.path.join(output_dir, ckpt_name))
            log(f"  >> 保存最佳模型 (auc={best_auc:.4f})")
        else:
            patience += 1
            if patience >= cfg.patience:
                log(f"早停：连续 {cfg.patience} 个 epoch 未提升，停止于 epoch {epoch}")
                break

    # ---- 用最佳 epoch 的验证集预测，搜索每类最优阈值 ----
    thresholds = [cfg.threshold] * len(class_ids)
    tuned_f1 = float("nan")
    if cfg.tune_threshold and best_y_study is not None:
        thresholds = best_threshold_per_class(best_y_study, best_p_study)
        tuned_f1 = macro_f1_at_thresholds(best_y_study, best_p_study, thresholds)
        log("每类最优阈值（验证集搜索，最大化 F1）:")
        for cid, t in zip(class_ids, thresholds):
            name = CLASS_NAMES.get(cid, ("", ""))[1]
            log(f"  class {cid} ({name}): threshold={t:.2f}")
        log(f"阈值调优后 macro F1: {tuned_f1:.4f}")
        ckpt_name = f"best_model_{fold_tag}.pt" if fold_tag else "best_model.pt"
        ckpt_path = os.path.join(output_dir, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["thresholds"] = thresholds
        torch.save(ckpt, ckpt_path)

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)
    return best_auc, best_epoch, thresholds


def cmd_train(cfg: Config):
    device = torch.device(cfg.device)
    log(f"使用设备: {device}")

    tr = prepare_data(cfg, cfg.train_csv, "训练", split="train")
    class_ids = tr["class_ids"]
    labels = tr["labels"]

    pos = labels.values.sum(axis=0)
    log("训练集各类正样本数:")
    for cid, pc in zip(class_ids, pos):
        name = CLASS_NAMES.get(cid, ("", ""))[1]
        log(f"  class {cid} ({name}): pos={int(pc)}  ({pc/len(labels):.4f})")

    valid_labels = valid_samples = valid_images = None
    if cfg.valid_csv:
        va = prepare_data(cfg, cfg.valid_csv, "验证", split="valid", class_ids=class_ids)
        valid_labels = va["labels"]
        valid_samples = va["samples"]
        valid_images = va["images"]

    best_auc, best_epoch, thresholds = _train_fold(
        cfg, device, class_ids, labels, tr["samples"], tr["images"],
        valid_labels, valid_samples, valid_images, cfg.output_dir, "")

    save_json({
        "backbone": cfg.backbone,
        "image_size": cfg.image_size,
        "class_ids": class_ids,
        "num_train_studies": len(labels),
        "num_train_images": len(tr["samples"]),
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "loss": cfg.loss,
        "dropout": cfg.dropout,
        "weight_decay": cfg.weight_decay,
        "multi": cfg.multi,
        "pool": cfg.pool,
        "thresholds": [round(float(t), 2) for t in thresholds],
    }, cfg.output_path / "summary.json")
    log(f"训练完成。best_study_auc={best_auc:.4f} @ epoch {best_epoch}")
    log(f"输出目录: {cfg.output_dir}")


def cmd_kfold(cfg: Config):
    device = torch.device(cfg.device)
    log(f"使用设备: {device}")

    tr = prepare_data(cfg, cfg.train_csv, "训练", split="train")
    class_ids = tr["class_ids"]
    labels = tr["labels"]
    images = tr["images"]
    all_studies = sorted(labels.index)

    k = cfg.kfold
    folds = split_folds(all_studies, k, cfg.seed)
    log(f"\n开始 {k} 折交叉验证：训练集 {len(all_studies)} 个 Study，"
        f"每折验证约 {len(folds[0])} 个")

    results = []
    for i in range(k):
        fold_dir = os.path.join(cfg.output_dir, f"fold_{i}")
        done_flag = os.path.join(fold_dir, "DONE")
        ckpt_file = os.path.join(fold_dir, f"best_model_fold_{i}.pt")

        # 断点续训：该折已有 checkpoint 则跳过，读回 best_auc 计入汇总
        # 注意：关机中断的折若已保存了 best_model（半成品），也会被跳过；
        # 如需强制重训某个折，手动删除对应的 best_model_fold_i.pt 即可。
        if os.path.isfile(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location="cpu")
            prev_auc = ckpt.get("best_auc", float("nan"))
            prev_epoch = ckpt.get("epoch", -1)
            results.append({"fold": i, "best_auc": round(float(prev_auc), 4),
                            "best_epoch": prev_epoch})
            log(f"Fold {i} 已有 checkpoint（best_auc={prev_auc:.4f} @ epoch {prev_epoch}），跳过")
            continue

        valid_studies = folds[i]
        train_studies = [s for j in range(k) if j != i for s in folds[j]]

        train_labels = labels.loc[train_studies]
        valid_labels = labels.loc[valid_studies]
        train_images = {s: images[s] for s in train_studies}
        valid_images = {s: images[s] for s in valid_studies}
        train_samples = [(s, p) for s in train_studies for p in images[s]]
        valid_samples = [(s, p) for s in valid_studies for p in images[s]]

        log(f"\n========== Fold {i + 1}/{k} ==========")
        os.makedirs(fold_dir, exist_ok=True)  # 确保 fold 目录存在（防御 _train_fold 内部遗漏）
        best_auc, best_epoch, _ = _train_fold(
            cfg, device, class_ids, train_labels, train_samples, train_images,
            valid_labels, valid_samples, valid_images, fold_dir, f"fold_{i}")
        results.append({"fold": i, "best_auc": round(float(best_auc), 4),
                        "best_epoch": best_epoch})
        # 写完成标记，供下次断点续训识别
        with open(done_flag, "w") as f:
            f.write(f"fold={i} best_auc={best_auc:.4f} epoch={best_epoch}\n")
        log(f"Fold {i} 完成: best_auc={best_auc:.4f} @ epoch {best_epoch}")

    aucs = [r["best_auc"] for r in results]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    log(f"\n========== K 折交叉验证完成 ==========")
    log(f"每折 AUC: {[round(a, 4) for a in aucs]}")
    log(f"平均 val_study_macro_auc: {mean_auc:.4f} ± {std_auc:.4f}")

    save_json({
        "k": k,
        "backbone": cfg.backbone,
        "image_size": cfg.image_size,
        "loss": cfg.loss,
        "multi": cfg.multi,
        "pool": cfg.pool,
        "fold_results": results,
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
    }, cfg.output_path / "kfold_summary.json")
    log(f"K 折汇总已保存: {os.path.join(cfg.output_dir, 'kfold_summary.json')}")
    log(f"各折模型保存在 {cfg.output_dir}/fold_*/best_model_fold_*.pt，可用于集成预测")


def _predict_one(ckpt_path: str, cfg: Config, device, test_samples, test_study_images, class_ids):
    """用单个 checkpoint 对测试集预测，返回 (studies, probs, thresholds)。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    multi = ckpt.get("multi", False)
    if multi:
        model = MultiImageModel(ckpt["backbone"], len(class_ids), pretrained=False,
                                dropout=ckpt.get("dropout", 0.0),
                                pool=ckpt.get("pool", "mean")).to(device)
    else:
        model = build_model(ckpt["backbone"], len(class_ids), pretrained=False,
                            dropout=ckpt.get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log(f"加载 checkpoint: {ckpt_path}  "
        f"(backbone={ckpt['backbone']}, multi={multi}, image_size={ckpt.get('image_size')}, "
        f"best_auc={ckpt.get('best_auc')})")

    if multi:
        loader = make_study_loader(
            StudyDataset(test_study_images, None, class_ids, ckpt["image_size"], train=False),
            cfg.batch_size, shuffle=False, workers=cfg.workers)
        probs, studies = predict_multi(model, loader, device, tta=cfg.tta)
    else:
        loader = make_loader(
            ChestXrayDataset(test_samples, None, class_ids, ckpt["image_size"], train=False),
            cfg.batch_size, shuffle=False, workers=cfg.workers)
        probs_img, s_img = predict(model, loader, device, tta=cfg.tta)
        studies, probs = aggregate_by_study(s_img, probs_img, pool=ckpt.get("pool", "mean"))
    thresholds = ckpt.get("thresholds") or [cfg.threshold] * len(class_ids)
    return studies, probs, thresholds


def cmd_evaluate(cfg: Config):
    """在验证集上评估单个或多个 checkpoint，并给出集成后的指标对比。"""
    device = torch.device(cfg.device)
    ckpt_paths = [p.strip() for p in cfg.checkpoint.split(",") if p.strip()]

    first_ckpt = torch.load(ckpt_paths[0], map_location=device)
    class_ids = first_ckpt["class_ids"]

    va = prepare_data(cfg, cfg.valid_csv, "验证", split="valid", class_ids=class_ids)
    valid_labels = va["labels"]
    valid_studies = list(valid_labels.index)
    valid_label_lookup = {s: valid_labels.loc[s].values for s in valid_studies}
    valid_samples = va["samples"]
    valid_study_images = va["images"]

    log(f"在验证集上评估 {len(ckpt_paths)} 个模型\n")

    probs_list, names = [], []
    y_ref = None
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device)
        multi = ckpt.get("multi", False)
        if multi:
            model = MultiImageModel(ckpt["backbone"], len(class_ids), pretrained=False,
                                    dropout=ckpt.get("dropout", 0.0),
                                    pool=ckpt.get("pool", "mean")).to(device)
        else:
            model = build_model(ckpt["backbone"], len(class_ids), pretrained=False,
                                dropout=ckpt.get("dropout", 0.0)).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        if multi:
            simgs = {s: valid_study_images[s] for s in valid_studies}
            loader = make_study_loader(
                StudyDataset(simgs, valid_label_lookup, class_ids, ckpt["image_size"], train=False),
                cfg.batch_size, shuffle=False, workers=cfg.workers)
            y, prob, _ = evaluate_multi(model, loader, device)
        else:
            loader = make_loader(
                ChestXrayDataset(valid_samples, valid_label_lookup, class_ids,
                                 ckpt["image_size"], train=False),
                cfg.batch_size, shuffle=False, workers=cfg.workers)
            y_img, p_img, s_img = evaluate(model, loader, device)
            studies, prob = aggregate_by_study(s_img, p_img, pool=ckpt.get("pool", "mean"))
            y = np.array([valid_label_lookup[s] for s in studies])

        if y_ref is None:
            y_ref = y
        aucs, macro = per_class_auc(y, prob, class_ids)
        f1 = macro_f1(y, prob, cfg.threshold)
        name = os.path.basename(os.path.dirname(os.path.abspath(p))) or p
        log(f"[{name}] val_study_macro_auc={macro:.4f}  f1@0.5={f1:.4f}")
        probs_list.append(prob)
        names.append(name)

    # ---- 集成（概率平均） ----
    ens = np.mean(probs_list, axis=0)
    aucs, macro = per_class_auc(y_ref, ens, class_ids)
    f1 = macro_f1(y_ref, ens, cfg.threshold)
    thr = best_threshold_per_class(y_ref, ens)
    f1_tuned = macro_f1_at_thresholds(y_ref, ens, thr)
    log(f"\n[集成 {len(probs_list)} 个模型] val_study_macro_auc={macro:.4f}  "
        f"f1@0.5={f1:.4f}  f1@最优阈值={f1_tuned:.4f}")
    log("每类AUC: " + "  ".join(f"c{cid}={a:.3f}" for cid, a in zip(class_ids, aucs)))
    log(f"每类最优阈值: {[round(float(t), 2) for t in thr]}")


def cmd_predict(cfg: Config):
    device = torch.device(cfg.device)
    ckpt_paths = [p.strip() for p in cfg.checkpoint.split(",") if p.strip()]

    # 用第一个 checkpoint 的 class_ids 作为基准
    first_ckpt = torch.load(ckpt_paths[0], map_location=device)
    class_ids = first_ckpt["class_ids"]

    te = prepare_data(cfg, cfg.test_csv, "测试", split="test", class_ids=class_ids)
    test_samples = te["samples"]
    test_study_images = te["images"]

    if cfg.tta:
        log("已启用水平翻转 TTA（原图 + 翻转图 logits 平均）")

    # ---- 逐个模型预测并集成（概率平均） ----
    studies_list, probs_list, thresholds_list = [], [], []
    for p in ckpt_paths:
        studies, probs, thr = _predict_one(p, cfg, device, test_samples, test_study_images, class_ids)
        studies_list.append(studies)
        probs_list.append(probs)
        thresholds_list.append(thr)

    agg_studies = studies_list[0]
    # 所有模型的 study 顺序应一致（都按 study 排序）
    for st in studies_list[1:]:
        assert list(st) == list(agg_studies), "不同模型的 study 顺序不一致，无法集成"

    agg_probs = np.mean(probs_list, axis=0)
    thresholds = list(np.mean(np.array(thresholds_list, dtype=float), axis=0))

    # 手动阈值覆盖（用于把 eval 搜出的最优阈值写进 Gate 列）
    if cfg.thresholds:
        manual = [float(x) for x in cfg.thresholds.split(",") if x.strip() != ""]
        if len(manual) == len(class_ids):
            thresholds = manual
            log(f"使用手动阈值（覆盖 checkpoint）: {[round(t, 2) for t in thresholds]}")
        else:
            log(f"警告: --thresholds 给了 {len(manual)} 个值，应为 {len(class_ids)} 个，已忽略")

    n_models = len(ckpt_paths)
    log(f"集成 {n_models} 个模型的概率（简单平均）")
    log(f"最终每类阈值: {[round(float(t), 2) for t in thresholds]}")

    # ---- 长格式提交：每个 Study 展开成 10 行（10 个类别） ----
    # 从测试 CSV 拿到 Study_id -> Subject_id 映射
    test_df = load_label_df(cfg.test_csv, "测试")
    study_col = find_col(test_df, STUDY_COL_CANDIDATES)
    subject_col = find_col(test_df, SUBJECT_COL_CANDIDATES)
    study2subject = {}
    if subject_col is not None:
        study2subject = dict(zip(
            test_df[study_col].astype(str).str.strip(),
            test_df[subject_col].astype(str).str.strip(),
        ))

    rows = []
    for study, pvec in zip(agg_studies, agg_probs):
        subject = study2subject.get(study, "")
        for cid, prob, thr in zip(class_ids, pvec, thresholds):
            rows.append({
                "Subject_id": subject,
                "Study_id": study,
                "Predict_class": int(cid),
                "Probability": round(float(prob), 6),
                "Gate": round(float(thr), 2),
            })
    sub = pd.DataFrame(rows,
                       columns=["Subject_id", "Study_id", "Predict_class", "Probability", "Gate"])
    # 按 Study_id 和 Predict_class 排序，保证稳定
    sub = sub.sort_values(["Study_id", "Predict_class"]).reset_index(drop=True)

    os.makedirs(cfg.output_dir, exist_ok=True)
    sub_path = os.path.join(cfg.output_dir, "submission.csv")
    sub.to_csv(sub_path, index=False, encoding="utf-8")
    log(f"提交文件已生成: {sub_path}  shape={sub.shape}")
    log(f"每个 Study 展开 10 行，共 {len(sub)} 行")
    log(f"前 12 行:\n{sub.head(12).to_string()}")


def main():
    cfg = parse_args()
    if cfg.mode == "check":
        if not cfg.train_csv:
            raise SystemExit("check 模式需要 --train_csv")
        if not cfg.image_dir:
            raise SystemExit("check 模式需要 --image_dir")
        cmd_check(cfg)
    elif cfg.mode == "train":
        if not cfg.train_csv:
            raise SystemExit("train 模式需要 --train_csv")
        if not cfg.image_dir:
            raise SystemExit("train 模式需要 --image_dir")
        cmd_train(cfg)
    elif cfg.mode == "kfold":
        if not cfg.train_csv:
            raise SystemExit("kfold 模式需要 --train_csv")
        if not cfg.image_dir:
            raise SystemExit("kfold 模式需要 --image_dir")
        cmd_kfold(cfg)
    elif cfg.mode == "eval":
        if not cfg.valid_csv:
            raise SystemExit("eval 模式需要 --valid_csv")
        if not cfg.image_dir:
            raise SystemExit("eval 模式需要 --image_dir")
        if not cfg.checkpoint:
            raise SystemExit("eval 模式需要 --checkpoint（可逗号分隔多个）")
        cmd_evaluate(cfg)
    elif cfg.mode == "predict":
        if not cfg.test_csv:
            raise SystemExit("predict 模式需要 --test_csv")
        if not cfg.checkpoint:
            raise SystemExit("predict 模式需要 --checkpoint")
        if not cfg.image_dir:
            raise SystemExit("predict 模式需要 --image_dir")
        cmd_predict(cfg)


if __name__ == "__main__":
    main()
