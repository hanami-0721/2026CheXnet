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

# 3) 在验证集上评估（单模型或多个模型集成，看 AUC/F1 对比，不生成文件）
python main.py --mode eval --valid_csv valid.csv --image_dir /图片目录 \
               --checkpoint outputs/asl/best_model.pt,outputs/effb3_768/best_model.pt

# 4) K 折交叉验证（训练集内部分 K 折，训练 K 个模型 + 稳定评估）
python main.py --mode kfold --train_csv train.csv --image_dir /图片目录 \
               --kfold 5 --loss asl --output_dir outputs/kfold
```

## 主要可调参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--backbone` | `densenet121` | `efficientnet_b3` / `convnext_tiny`（需 timm） |
| `--image_size` | `512` | 胸片细节多，建议 ≥384；T4 16GB 显存不足降到 384 |
| `--batch_size` | `16` | T4 16GB 建议 16；更大显存可到 32 |
| `--epochs` | `20` | 有早停，可给大一点 |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--weight_decay` | `5e-4` | 正则强度，抑制过拟合 |
| `--patience` | `3` | 连续多少 epoch 不涨就早停 |
| `--dropout` | `0.2` | 分类头 dropout，抑制过拟合 |
| `--loss` | `bce` | `bce`=BCE+pos_weight；`asl`=Asymmetric Loss（类别不均衡首选） |
| `--asl_gamma_neg` | `4.0` | ASL 负样本 focusing |
| `--asl_gamma_pos` | `1.0` | ASL 正样本 focusing |
| `--asl_clip` | `0.05` | ASL 负样本 margin |
| `--pool` | `mean` | 多图聚合：`mean` / `max` / `attn`（`--multi` 模式用特征池化） |
| `--multi` | — | 多图 Study 级联合建模（同一 Study 多张图特征池化后联合预测） |
| `--pos_weight_cap` | `20` | 类别不均衡 pos_weight 上限（仅 `--loss bce` 生效） |
| `--threshold` | `0.5` | 默认阈值（会被每类最优阈值覆盖） |
| `--no_tune_threshold` | — | 关闭验证集上的每类阈值搜索 |
| `--eval_train` | — | 每轮额外评估训练集（耗时，仅诊断过拟合） |
| `--seed` | `42` | 随机种子 |

## 调优版说明（基于首轮 baseline 结果）

首轮 `densenet121@512` 训练结果：`val_study_macro_auc ≈ 0.793`（epoch 7 峰值），
但 `train_macro_auc` 冲到 0.93、`val_study_macro_f1` 仅 0.38~0.44，典型**过拟合 + 类别不均衡 + 阈值非最优**。

本轮改动（已默认启用）：

1. **Asymmetric Loss**（`--loss asl`）：解决少数类（纵隔心影增大 3.4%、肺炎 7.8%）AUC 偏低
2. **每类最优阈值**（`--tune_threshold`，默认开）：训练后在验证集上搜每类阈值，直接提升 macro F1
3. **Dropout 0.2 + weight_decay 5e-4 + 加强增强**：缓解过拟合
4. **早停 patience 收紧到 3**：避免峰值后继续过拟合

推荐调优命令：

```bash
# 方案 A：Asymmetric Loss（主攻类别不均衡）
python main.py --mode train --train_csv ./cxr_first/train.csv \
  --valid_csv ./cxr_first/valid.csv --image_dir ./cxr_first \
  --loss asl --output_dir outputs/asl

# 方案 B：保持 BCE，靠 dropout + 增强 + 阈值调优
python main.py --mode train --train_csv ./cxr_first/train.csv \
  --valid_csv ./cxr_first/valid.csv --image_dir ./cxr_first \
  --loss bce --output_dir outputs/bce_dropout
```

跑完后对比两个目录的 `summary.json` 里的 `best_auc`（val_study_macro_auc）和
`val_study_macro_f1_tuned`，选更好的方向继续。推理时脚本会自动用 checkpoint 里存的每类阈值，
并在 `submission.csv` 里输出 `Gate` 列（自定义阈值）。

### 多图 Study 联合建模（核心差异点）

一个 Study 有 1~7 张图（不同视角 AP/PA/LATERAL），图像级模式是「逐图预测 + mean 聚合」，
会丢掉视角和图间关系。`--multi` 模式把同一 Study 的多张图分别过共享 backbone 得到特征，
再在 Study 内部做特征池化（mean / max / attention），联合输出一个 Study 级预测：

```bash
# 多图联合 + Asymmetric Loss + attention 池化
python main.py --mode train --train_csv ./cxr_first/train.csv \
  --valid_csv ./cxr_first/valid.csv --image_dir ./cxr_first \
  --multi --pool attn --loss asl --output_dir outputs/multi_attn

# 或先试 mean 池化（更稳）
python main.py --mode train --train_csv ./cxr_first/train.csv \
  --valid_csv ./cxr_first/valid.csv --image_dir ./cxr_first \
  --multi --pool mean --loss asl --output_dir outputs/multi_mean
```

> `--multi` 模式的 `batch_size` 指**每个 batch 的 Study 数**，一个 Study 最多 7 张图时
> 总图数可能很大。T4 16GB 建议 `--batch_size 8`，显存不足再降到 4。

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
# baseline
0.7934

# 第一版优化
1.Asymmetric Loss（--loss asl）—— 多标签不均衡的业界标准方案，比 pos_weight 更能救少数类
2.每类最优阈值（--tune_threshold，默认开）—— 训练后在验证集上对每类搜阈值，直接提升 macro F1，并回写到 best_model.pt
3.Dropout 0.2 + weight_decay 5e-4 + 加强增强 —— 缓解过拟合
4.早停 patience 收紧到 3 —— 避免峰值后继续过拟合
推理输出 Gate 列 —— 用训练时搜出的每类阈值（提交格式允许的自定义阈值）
0.7967

# 第二轮优化
1.TTA（测试时增强） —— 已默认开启（--no_tta 可关），预测时原图 + 水平翻转取平均，通常 +0.3~0.5% AUC。
2.多图 Study 联合建模（--multi）—— 这是核心改动：
0.7879 没提升

# 第三轮改动
重新训练efficientnet_b3与densenet121集成（ASL backbone 不同、错误不相关，集成收益最大）
0.8042 

# 第4轮改动
densenet121 5折交叉验证+集成
0.8058

# 第五轮改动
densenet121+ efficientnet_b3 双模型集成，分别交叉验证+困难样本挖掘+少数类专项
（efficientnet_b3过拟合崩了，进行消融实验）
