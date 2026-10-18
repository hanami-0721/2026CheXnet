# 胸部 X 光多标签分类 Baseline

比赛任务：给定胸部 X 光片，输出 **10 类**疾病的概率（多标签），按 **检查（Study）级别**评估。

- **主指标**：宏平均 AUC (Macro-averaged AUC)
- **辅助指标**：宏平均 F1 (Macro-F1)
- **提交格式**：长表，每个 Study 展开成 10 行（10 个类别各一行）

## 类别映射

| 类别 id | 中文名 | 英文名 |
|---------|--------|--------|
| 1 | 纵隔心影增大 | Enlarged Cardiomediastinum |
| 2 | 气胸 | Pneumothorax |
| 3 | 肺部实变 | Consolidation |
| 4 | 肺炎 | Pneumonia |
| 5 | 肺水肿 | Edema |
| 6 | 心影扩大 | Cardiomegaly |
| 7 | 肺不张 | Atelectasis |
| 8 | 肺部阴影 | Lung Opacity |
| 9 | 胸腔积液 | Pleural Effusion |
| 10 | 无异常发现 | No Finding |

## 项目结构

```
main.py                 # argparse 入口 + 流程编排
src/
  config.py             # 配置数据类 + 类别映射
  utils.py              # 随机种子 / 日志 / JSON
  data.py               # 数据读取、标签解析、图片索引、增强、Dataset/DataLoader
  models.py             # DenseNet121 / EfficientNet-B3 / ConvNeXt-Tiny 构造
  engine.py             # 训练 / 验证 / 预测循环
  metrics.py            # 每类 AUC、macro AUC、macro F1、检查级聚合
```

## 数据格式（已按比赛真实格式适配）

- 标签 CSV 每行一张图，列含 `Dicom_id / Subject_id / Study_id / ViewPosition / PatientOrientation / Rows / Columns / Predict_class`
- `Predict_class` 为**分号分隔的多值**（如 `7;9` 表示同时患类别 7 和 9），类别 id 范围 **1~10**
- 同一 Study 的多张图标签一致，脚本按 Study 聚合
- 图片按 `Dicom_id` 匹配文件名（`pic_000000` → `pic_000000.jpg`），递归扫描 `--image_dir`

## 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：`torch`、`torchvision`、`pandas`、`numpy`、`scikit-learn`、`Pillow`、`tqdm`。
> 默认 backbone `densenet121`（torchvision 自带），无需 `timm`。

## 使用流程

```bash
# 0) 数据自检（不训练）
python main.py --mode check --train_csv train.csv --valid_csv valid.csv --image_dir /图片目录

# 1) 训练
python main.py --mode train --train_csv train.csv --valid_csv valid.csv \
               --image_dir /图片目录 --output_dir outputs/baseline

# 2) 测试集推理与提交（自动生成长格式 submission.csv）
python main.py --mode predict --test_csv test.csv --image_dir /图片目录 \
               --checkpoint outputs/baseline/best_model.pt --output_dir outputs/baseline
```

## 主要可调参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--backbone` | `densenet121` | `efficientnet_b3` / `convnext_tiny`（需 timm） |
| `--image_size` | `512` | 胸片细节多，建议 ≥384；T4 16GB 显存不足降到 384 |
| `--batch_size` | `16` | T4 16GB 建议 16；更大显存可到 32 |
| `--epochs` | `20` | 有早停，可给大一点 |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--patience` | `4` | 连续多少 epoch 不涨就早停 |
| `--pool` | `mean` | 一 Study 多图时聚合：`mean` / `max` |
| `--pos_weight_cap` | `20` | 类别不均衡 pos_weight 上限 |
| `--threshold` | `0.5` | 验证 F1 与提交阈值（默认 0.5） |
| `--seed` | `42` | 随机种子 |

## 服务器约束（T4 16GB 单卡）

- 模型文件 ≤ 500MB：DenseNet121 约 28MB，远低于上限
- 单图延迟 ≤ 100ms：`densenet121@512` 在 T4 上单图推理约 20~40ms，满足
- 只允许用主办方训练数据 + 数据增强，**不允许外部标注数据**（本脚本只用 ImageNet 预训练权重，不含外部标注数据）

## 拿到结果后：如何决定下一步

看 `outputs/baseline/summary.json` 和 `history.csv`，重点看 **`val_study_macro_auc`**（主指标）。

| 现象 | 原因 | 下一步 |
|------|------|--------|
| 整体 AUC 低（< 0.75） | backbone 弱 / 分辨率低 | 换 `efficientnet_b3`，`--image_size` 提 768 |
| 少数类（气胸/纵隔心影增大）AUC 明显低 | 类别不均衡 | 调 `--pos_weight_cap`、改 Asymmetric Loss |
| train 高、val 明显低 | 过拟合 | 加强增广、加 weight_decay、降 lr |
| 多图 Study 聚合效果差 | 聚合方式不合适 | `--pool max` 对比；后面做多图联合建模 |
| 单模型到瓶颈 | — | K 折交叉验证 + 多模型集成 + TTA |
| macro F1 低但 AUC 高 | 阈值不合适 | 验证集上对每类单独搜阈值 |

> 本 baseline 的定位是「快速得到可复现的起点分数」，先把这一版跑通、记录 `val_study_macro_auc` 作为基准，再逐项改进。
"# 2026CheXnet" 
