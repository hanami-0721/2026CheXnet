"""模型层

构造多标签分类模型：backbone 提取特征，head 输出 num_classes 个 logit。
支持两种粒度：
- 图像级（逐图预测，Study 级用概率聚合）
- Study 级多图联合（同一 Study 的多张图特征池化后联合预测）
"""

import torch
import torch.nn as nn

# 各 backbone 的全局池化后特征维度（用于构建多图模型的分类头）
FEAT_DIMS = {
    "densenet121": 1024,
    "efficientnet_b3": 1536,
    "convnext_tiny": 768,
}


def build_model(backbone: str = "densenet121", num_classes: int = 10,
                pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """按 backbone 名构造图像级分类模型。

    - densenet121: 走 torchvision（胸片任务经典 baseline）
    - efficientnet_b3 / convnext_tiny: 走 timm（需额外安装 timm）

    dropout > 0 时在分类头前插入 Dropout，用于缓解过拟合。
    """
    if backbone == "densenet121":
        from torchvision import models as tv_models

        try:
            model = tv_models.densenet121(
                weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            model = tv_models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        if dropout and dropout > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier = nn.Linear(in_features, num_classes)
        return model

    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            f"backbone '{backbone}' 需要 timm。请先 `pip install timm`，或改用 densenet121。"
        ) from exc
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes,
                             drop_rate=dropout)


def build_backbone_features(backbone: str = "densenet121", pretrained: bool = True) -> nn.Module:
    """构造只输出特征向量（不含分类头）的 backbone。

    - densenet121: classifier 置为 Identity，输出 [N, 1024]
    - timm 模型: num_classes=0，输出全局池化后的特征 [N, D]
    """
    if backbone == "densenet121":
        from torchvision import models as tv_models

        try:
            model = tv_models.densenet121(
                weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            model = tv_models.densenet121(pretrained=pretrained)
        model.classifier = nn.Identity()
        return model

    import timm
    return timm.create_model(backbone, pretrained=pretrained, num_classes=0)


class MultiImageModel(nn.Module):
    """多图 Study 级联合分类模型。

    把同一 Study 的 k 张图分别过共享 backbone 得到特征，再在 Study 内部做
    池化（mean / max / attention）得到 Study 级特征，最后过分类头输出 logits。

    forward 输入：
      images: [total_imgs, 3, H, W]  —— 一个 batch 内所有 Study 的图展平
      sizes:  list[int]               —— 每个 Study 的图数
    输出：[B, num_classes] 的 logits（B = 该 batch 的 Study 数）。
    """

    def __init__(self, backbone: str = "densenet121", num_classes: int = 10,
                 pretrained: bool = True, dropout: float = 0.2, pool: str = "mean"):
        super().__init__()
        self.encoder = build_backbone_features(backbone, pretrained)
        self.feat_dim = FEAT_DIMS.get(backbone, None)
        if self.feat_dim is None:
            self.feat_dim = getattr(self.encoder, "num_features", 1024)
        self.pool = pool
        if pool == "attn":
            self.attn = nn.Sequential(
                nn.Linear(self.feat_dim, 64), nn.Tanh(), nn.Linear(64, 1)
            )
        head = [nn.Dropout(dropout)] if dropout and dropout > 0 else []
        head.append(nn.Linear(self.feat_dim, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, images, sizes):
        feats = self.encoder(images)          # [total, D]
        pooled = []
        idx = 0
        for k in sizes:
            f = feats[idx:idx + k]            # [k, D]
            if self.pool == "max":
                p = f.max(dim=0)[0]
            elif self.pool == "attn":
                w = self.attn(f).softmax(dim=0)  # [k, 1]
                p = (f * w).sum(dim=0)
            else:                             # mean
                p = f.mean(dim=0)
            pooled.append(p)
            idx += k
        x = torch.stack(pooled, dim=0)        # [B, D]
        return self.head(x)