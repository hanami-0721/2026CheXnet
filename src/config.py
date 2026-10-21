"""配置层

定义多标签分类任务的配置数据类。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# 类别 id -> (中文名, 英文名)
CLASS_NAMES = {
    1: ("纵隔心影增大", "Enlarged Cardiomediastinum"),
    2: ("气胸", "Pneumothorax"),
    3: ("肺部实变", "Consolidation"),
    4: ("肺炎", "Pneumonia"),
    5: ("肺水肿", "Edema"),
    6: ("心影扩大", "Cardiomegaly"),
    7: ("肺不张", "Atelectasis"),
    8: ("肺部阴影", "Lung Opacity"),
    9: ("胸腔积液", "Pleural Effusion"),
    10: ("无异常发现", "No Finding"),
}


@dataclass
class Config:
    mode: str = "check"                      # check / train / predict
    train_csv: Optional[str] = None
    valid_csv: Optional[str] = None
    test_csv: Optional[str] = None
    image_dir: Optional[str] = None          # 图片根目录
    output_dir: str = "outputs/baseline"

    backbone: str = "densenet121"            # densenet121 / efficientnet_b3 / convnext_tiny
    pretrained: bool = True
    checkpoint: Optional[str] = None
    dropout: float = 0.2                     # 分类头 dropout，缓解过拟合

    # 损失函数：bce（BCEWithLogits + pos_weight）/ asl（Asymmetric Loss）
    loss: str = "bce"
    asl_gamma_neg: float = 4.0               # ASL 负样本 focusing（困难负样本挖掘强度）
    asl_gamma_pos: float = 1.0               # ASL 正样本 focusing
    asl_clip: float = 0.05                   # ASL 负样本 margin

    # 少数类专项：类别平衡采样
    sampler: str = "none"                    # none / balanced（类别平衡过采样）
    sampler_boost: float = 1.0               # 采样加权强度（越大越偏向少数类）

    image_size: int = 512
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 5e-4               # 从 1e-4 加大，抑制过拟合
    workers: int = 4
    patience: int = 3                        # 从 4 收紧，更早停止

    pool: str = "mean"                       # Study 多图聚合：mean / max / attn（multi 模式）
    multi: bool = False                      # True=多图 Study 级联合建模
    kfold: int = 5                           # K 折交叉验证的折数（--mode kfold 用）
    pos_weight_cap: float = 20.0             # 类别不均衡 pos_weight 的上限
    threshold: float = 0.5                   # 默认阈值（会被每类最优阈值覆盖）
    thresholds: Optional[str] = None         # 手动指定每类阈值（逗号分隔，predict 用，覆盖 checkpoint）
    tune_threshold: bool = True              # 是否在验证集上搜索每类最优阈值
    eval_train: bool = False                 # 是否每轮额外评估训练集（费时，仅诊断用）
    tta: bool = True                         # 测试时水平翻转 TTA（预测阶段）

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
