"""数据层

多标签数据管线，适配比赛真实格式：
- 每行一张图（Dicom_id 唯一），标签列 Predict_class 为分号分隔的多值，如 '7;9'
- 类别 id 范围 1~10
- 同一 Study 的多张图标签一致，标签按 Study 聚合
- 图片按 Dicom_id 匹配文件名（pic_000000 -> pic_000000.jpg）
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import log

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STUDY_COL_CANDIDATES = ["study_id", "studyid", "study", "检查id", "studyno"]
SUBJECT_COL_CANDIDATES = ["subject_id", "subjectid", "subject", "患者id", "patient_id", "patientid"]
DICOM_COL_CANDIDATES = ["dicom_id", "dicomid", "dicom", "image_id", "imageid",
                        "image_name", "imagename", "图片id", "图像id"]
CLASS_COL_CANDIDATES = ["predict_class", "predictclass", "class", "class_id", "classid",
                        "label", "label_id", "disease", "disease_id", "患病类别id", "类别id"]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def norm_col(name: str) -> str:
    return str(name).strip().lower()


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def to_int_id(v) -> Optional[int]:
    try:
        f = float(v)
        if f.is_integer():
            return int(f)
    except (TypeError, ValueError):
        pass
    return None


def parse_class_ids(value) -> List[int]:
    """解析类别值：'7;9' -> [7, 9]，'10' -> [10]。"""
    ids = []
    for part in str(value).split(";"):
        i = to_int_id(part.strip())
        if i is not None:
            ids.append(i)
    return ids


# --------------------------------------------------------------------------- #
# 标签加载与解析
# --------------------------------------------------------------------------- #
def load_label_df(csv_path: str, mode: str) -> pd.DataFrame:
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{mode} 标签文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"读取 {mode} 标签: {csv_path}  shape={df.shape}  columns={list(df.columns)}")
    return df


def detect_label_format(df: pd.DataFrame):
    """解析比赛 CSV，返回 (study_col, dicom_col, class_col, class_ids, has_label)。

    - class_ids：所有出现的类别 id（升序，如 [1..10]）
    - has_label：是否有标签（test 集无 Predict_class 列）
    """
    study_col = find_col(df, STUDY_COL_CANDIDATES)
    if study_col is None:
        raise ValueError(
            f"找不到 Study_id 列。候选: {STUDY_COL_CANDIDATES}；当前列: {list(df.columns)}"
        )
    dicom_col = find_col(df, DICOM_COL_CANDIDATES)
    if dicom_col is None:
        raise ValueError(
            f"找不到图片 id 列（Dicom_id）。候选: {DICOM_COL_CANDIDATES}；当前列: {list(df.columns)}"
        )
    class_col = find_col(df, CLASS_COL_CANDIDATES)
    if class_col is None:
        return study_col, dicom_col, None, None, False

    all_ids = set()
    for v in df[class_col]:
        for i in parse_class_ids(v):
            all_ids.add(i)
    class_ids = sorted(all_ids)
    return study_col, dicom_col, class_col, class_ids, True


def build_labels(df: pd.DataFrame, study_col: str, class_col: str, class_ids: List[int]) -> pd.DataFrame:
    """把标签整理成「一行一个 Study」的宽表，列=类别 id，值 0/1（float）。

    同一 Study 多行（多张图）的类别取并集。
    返回 DataFrame：index=study_id(str)，columns=class_ids。
    """
    study_str = df[study_col].astype(str).str.strip()
    study2cls: Dict[str, set] = {}
    for s, v in zip(study_str, df[class_col]):
        s = s.strip()
        study2cls.setdefault(s, set()).update(parse_class_ids(v))

    label_df = pd.DataFrame(index=sorted(study2cls.keys()))
    for cid in class_ids:
        label_df[cid] = [1.0 if cid in study2cls[s] else 0.0 for s in label_df.index]
    return label_df.astype(np.float32)


# --------------------------------------------------------------------------- #
# 图片路径解析
# --------------------------------------------------------------------------- #
def build_image_index(image_dir: str):
    """扫描图片目录，建立 (文件名 stem -> 相对路径) 与 (任意目录段 -> 相对路径) 索引。"""
    image_dir = Path(image_dir)
    by_stem: Dict[str, List[str]] = {}
    by_segment: Dict[str, List[str]] = {}
    for root, _, files in os.walk(image_dir):
        for f in files:
            if not f.lower().endswith(IMG_EXTS):
                continue
            full = Path(root) / f
            rel = str(full.relative_to(image_dir))
            by_stem.setdefault(full.stem, []).append(rel)
            for seg in [full.stem] + [p.name for p in full.parents if p != image_dir]:
                by_segment.setdefault(seg, []).append(rel)
    return by_stem, by_segment


def resolve_split_image_dir(image_dir: str, split: str) -> str:
    """解析某个 split 的实际图片目录。

    支持两种布局：
    - image_dir 直接就是该 split 的图片目录（如 ./cxr_first/train）
    - image_dir 是父目录，其下有 train/valid/test 子目录（如 ./cxr_first）

    若 image_dir/split 存在则用子目录，否则用 image_dir 本身。
    """
    sub = Path(image_dir) / split
    if sub.is_dir():
        return str(sub)
    return image_dir


def resolve_study_images(df: pd.DataFrame, study_col: str, dicom_col: str, image_dir: str):
    """按 Dicom_id 匹配图片文件名，聚合成 {study_id: [image_path]}。

    图片命名约定：文件名 stem == Dicom_id（如 pic_000000.jpg）。
    返回 (resolved, missing_dicoms)。
    """
    by_stem, by_segment = build_image_index(image_dir)
    resolved: Dict[str, List[str]] = {}
    missing: List[str] = []

    for s, d in zip(df[study_col].astype(str).str.strip(),
                    df[dicom_col].astype(str).str.strip()):
        s = s.strip()
        d = d.strip()
        hits = by_stem.get(d, []) or by_segment.get(d, [])
        paths = [str(Path(image_dir) / h) for h in hits if os.path.isfile(Path(image_dir) / h)]
        if not paths:
            missing.append(d)
            continue
        bucket = resolved.setdefault(s, [])
        for p in paths:
            if p not in bucket:
                bucket.append(p)
    return resolved, missing


# --------------------------------------------------------------------------- #
# 图像增强
# --------------------------------------------------------------------------- #
class Transform:
    """图像预处理与增强。灰度图统一转 RGB，归一化用 ImageNet 均值方差。"""

    def __init__(self, size: int, train: bool):
        self.size = size
        self.train = train
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __call__(self, image) -> np.ndarray:
        import torchvision.transforms as T

        size = self.size
        if self.train:
            # 医学图像增强保持克制：只做几何微扰 + 亮度对比度微调，
            # 避免破坏病灶特征（不做强色彩扰动、不做大角度旋转）。
            ops = [
                T.Resize((size, size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.92, 1.08)),
                T.RandomRotation(degrees=8),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        else:
            ops = [
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        return T.Compose(ops)(image)


# --------------------------------------------------------------------------- #
# Dataset / DataLoader
# --------------------------------------------------------------------------- #
class ChestXrayDataset:
    """多标签图像 Dataset。样本单位 = 单张图像，标签继承自所属 Study。"""

    def __init__(self, samples: List[Tuple[str, str]], label_lookup,
                 class_ids: List[int], image_size: int, train: bool):
        self.samples = samples          # [(study_id, image_path)]
        self.label_lookup = label_lookup  # {study_id: np.array[C]} 或 None（测试集）
        self.class_ids = class_ids
        self.transform = Transform(image_size, train)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str):
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int):
        import torch

        study_id, path = self.samples[idx]
        img = self._load_image(path)
        x = self.transform(img)
        if self.label_lookup is not None:
            y = torch.tensor(np.asarray(self.label_lookup[study_id]), dtype=torch.float32)
        else:
            y = torch.zeros(len(self.class_ids), dtype=torch.float32)
        return x, y, study_id


def make_loader(dataset: ChestXrayDataset, batch_size: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True)


def build_samples(studies: List[str], images: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """把 {study_id: [paths]} 展开成 [(study_id, path)]。"""
    samples = []
    for s in studies:
        for p in images[s]:
            samples.append((s, p))
    return samples


# --------------------------------------------------------------------------- #
# 多图 Study 级 Dataset（联合建模）
# --------------------------------------------------------------------------- #
class StudyDataset:
    """多图联合建模：样本单位 = 一个 Study（含 1~k 张图）。

    与 ChestXrayDataset 的区别：一次返回一个 Study 的全部图，让模型能利用
    多视角（AP/PA/LATERAL）图之间的信息联合预测，而不是逐图预测后聚合。
    """

    def __init__(self, study_images: Dict[str, List[str]], label_lookup,
                 class_ids: List[int], image_size: int, train: bool):
        self.studies = sorted(study_images.keys())
        self.study_images = study_images      # {study_id: [image_path]}
        self.label_lookup = label_lookup      # {study_id: np.array[C]} 或 None
        self.class_ids = class_ids
        self.transform = Transform(image_size, train)

    def __len__(self) -> int:
        return len(self.studies)

    def _load_image(self, path: str):
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int):
        import torch

        sid = self.studies[idx]
        paths = self.study_images[sid]
        imgs = torch.stack([self.transform(self._load_image(p)) for p in paths])
        if self.label_lookup is not None:
            y = torch.tensor(np.asarray(self.label_lookup[sid]), dtype=torch.float32)
        else:
            y = torch.zeros(len(self.class_ids), dtype=torch.float32)
        return imgs, y, sid


def study_collate(batch):
    """把变长的 Study 样本拼成一个 batch。

    batch: [(imgs[k,3,H,W], y[C], sid), ...]
    返回 (all_imgs[total,3,H,W], sizes[B], ys[B,C], sids[B])
    """
    import torch

    all_imgs = torch.cat([b[0] for b in batch], dim=0)
    sizes = [b[0].size(0) for b in batch]
    ys = torch.stack([b[1] for b in batch], dim=0)
    sids = [b[2] for b in batch]
    return all_imgs, sizes, ys, sids


def make_study_loader(dataset: StudyDataset, batch_size: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, collate_fn=study_collate)
