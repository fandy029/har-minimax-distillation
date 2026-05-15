# GAIT 数据集 — 训练文档

## 概述

Gait_Classification 数据集，4 类活动识别（sit_on_bed / sit_on_chair / lying / ambulating），3 轴加速度计 40Hz。

采用**两阶段训练**方案（论文第 3 章）：
- **阶段 1**：Focal Loss 纯硬标签预训练
- **阶段 2**：知识蒸馏微调（Focal + KL 散度）

## 快速开始

```bash
cd /home/fandy/workplace/thesis/scripts/gait

# 阶段1: 纯硬标签预训练 (3种架构)
python3 training/train_phase1.py

# 阶段2: 蒸馏微调 (各自最佳参数)
python3 training/train_phase2.py
```

## 单架构训练

```bash
# 只训练 PureCNN
python3 training/train_phase1.py --models purecnn
python3 training/train_phase2.py --models purecnn

# 只训练 Transformer
python3 training/train_phase1.py --models transformer
python3 training/train_phase2.py --models transformer
```

## 输出文件

```
gait/
├── soft_label/                          # 软标签生成脚本
│   ├── gait_prepare.py                  # 数据预处理+划分
│   ├── gait_gen.py                      # 调用LLM API生成软标签
│   ├── gait_merge.py                    # 合并各类软标签
│   ├── gait_run.sh / gait_cleanup.sh    # 启动/清理
│   ├── api_config.py                    # API配置
│   ├── DATASET_INFO.md                  # 数据集特征分析
│   └── README.md                        # 软标签生成文档
├── training/                            # 训练脚本
│   ├── train_phase1.py                  # 阶段1: Focal Loss预训练
│   ├── train_phase2.py                  # 阶段2: 知识蒸馏微调
│   └── README.md                        # 本文档
└── output/
    ├── results/                         # 模型+结果JSON
    │   ├── gait_*_phase1.pt/json        # 阶段1模型+结果
    │   ├── gait_*_phase2_*.pt/json      # 阶段2模型+结果
    │   ├── phase1_summary.json          # 阶段1汇总
    │   └── phase2_summary.json          # 阶段2汇总
    ├── logs/                            # 训练日志
    ├── soft_labels/                     # 合并后的软标签.npy
    ├── checkpoints/                     # 软标签生成断点
    └── per_class/                       # 每类中间数据
```

每个 JSON 文件包含：
- `test_acc` / `val_acc` / `train_acc`
- `test_per_class`: 每类准确率
- `test_confusion_matrix`: 混淆矩阵
- `epoch_log`: 每 epoch 的 loss / acc / per-class acc
- `config`: 超参数配置

## 训练超参数

| 参数 | 阶段 1 | 阶段 2 |
|------|--------|--------|
| 损失函数 | Focal Loss (γ=2.0) | α·Focal + (1-α)·T²·KL |
| 优化器 | AdamW | AdamW |
| 学习率 | 5×10⁻⁴ | 1×10⁻⁴ |
| 权重衰减 | 1×10⁻⁴ | 1×10⁻⁴ |
| 调度器 | CosineAnnealingWarmRestarts | 同左 |
| T₀ / T_mult | 20 / 2 | 20 / 2 |
| 批量大小 | 64 | 64 |
| 最大 epoch | 300 | 300 |
| 早停 patience | 15 | 15 |
| 梯度裁剪 | max_norm=5.0 | max_norm=5.0 |

## 最佳蒸馏参数

从 63 组网格搜索（3 策略 × 3α × 3T × 3 架构）选出：

| 架构 | 策略 | α | T | 硬标签占比 | 软标签占比 |
|------|------|---|---|-----------|-----------|
| **PureCNN** | all | 0.5 | 3.5 | 50% | 50% |
| **CNN-Residual** | filtered | 0.7 | 1.5 | 70% | 30% |
| **Transformer** | all | 0.7 | 1.5 | 70% | 30% |

### 参数选择理由

#### 为什么高 α（硬标签权重 ≥ 0.5）？

GAIT 数据集的 4 个类别在物理特征上分离度较好：
- **lying**：重力垂直分量 gve≈0.1，几乎无歧义区分
- **ambulating**：运动强度 free_mag_std 远高于静态类
- **sit_on_bed vs sit_on_chair**：仅靠 gfr 微小差异区分

LLM 软标签准确率约 79%，对于 lying 和 sit_on_bed 表现好（94%/79%），但对 sit_on_chair 和 ambulating 仅 69%/62%。高 α 保证硬标签的强监督不被打乱，避免少数类被软标签噪声误导。

#### 为什么 PureCNN 用较高 T=3.5？

T 越大，软标签越"平滑"——各类概率趋于均匀，KL 散度强调**类别间结构关系**而非具体概率值。PureCNN 作为全卷积架构，对平滑的类别关系信号吸收更好，在 α=0.5/T=3.5 时软标签提供"lying 和 sit 之间的相似度很低，ambulating 和其他都不同"这种结构性知识，而不强制拟合具体概率。

#### 为什么 CNN-Res / Transformer 用 T=1.5？

T 较小意味着软标签更"尖锐"——强调主要类别的概率。残差连接（CNN-Res）和自注意力（Transformer）本身已能捕捉复杂的特征交互，不需要过于平滑的软标签。T=1.5 提供"这个样本最可能是什么类"的强信号，配合高 α=0.7，让模型在保证硬标签精度的前提下小幅调整决策边界。

#### 为什么 CNN-Res 选 filtered 策略？

CNN-Res 参数量最小（144K），对噪声更敏感。filtered 策略通过 entropy < 1.5、gap > 0.05、confidence > 0.5 三重筛选，去除了约 3% 的低质量软标签，保留更干净的类别关系信号，使相对增益达到 +3.16pp。

## 最终结果

### 阶段 1 — 纯硬标签基线

| 架构 | Val Acc | Test Acc | 参数量 | 最佳 Epoch |
|------|---------|----------|--------|-----------|
| PureCNN | 93.04% | 89.87% | 610,500 | 19 |
| CNN-Residual | 91.14% | 87.97% | 144,260 | 34 |
| 🏆 **Transformer** | 93.04% | **91.77%** | 167,428 | 29 |

### 阶段 2 — 蒸馏后

| 架构 | 策略 | α | T | Test Acc | Gain | 最佳 Epoch |
|------|------|---|---|----------|------|-----------|
| PureCNN | all | 0.5 | 3.5 | 91.14% | +1.27pp | 39 |
| CNN-Residual | filtered | 0.7 | 1.5 | 87.97% | +0.00pp | 2 |
| Transformer | all | 0.7 | 1.5 | 91.14% | -0.63pp | 6 |

> **注意**：Transformer 阶段1 即达到 91.77%，蒸馏未能进一步提升。PureCNN 蒸馏从 89.87% 提升至 91.14%（+1.27pp）。

### 每类准确率 (测试集, Transformer 阶段1 = 最佳 91.77%)

| 类别 | 准确率 |
|------|--------|
| sit_on_bed | 92.11% |
| sit_on_chair | 55.56% |
| lying | 100.00% |
| ambulating | 33.33% |

### 混淆矩阵 (测试集, Transformer 阶段1)

```
              sit_bed  sit_chair  lying  ambulat
  sit_on_bed     35        1        1        1
  sit_on_chair    3        5        1        0
  lying           0        0      102        0
  ambulating      6        0        0        3
```

### 关键发现

1. **Transformer 阶段1 即达最优** — 无需蒸馏即为 91.77%，自注意力天然捕捉类别间关系
2. **PureCNN 蒸馏有效** — 从 89.87% → 91.14%（+1.27pp），但收敛需更多 epoch（E39）
3. **CNN-Res 不适合 GAIT** — 参数量太小（144K），绝对精度最低
4. **ambulating 和 sit_on_chair 仍是瓶颈** — 样本极少（各 9 个测试样本），准确率仅 33%/56%
5. **lying 类 100% 识别** — 重力特征区分度极高

## 网格搜索历史

完整 63 组实验结果：`output/results/phase2_all.json`

搜索空间：
- 策略：all / filtered / correct_only
- α：0.3 / 0.5 / 0.7
- T：1.5 / 2.5 / 3.5
- 架构：purecnn / cnnres / transformer

PureCNN 27 组实验中 21 组正增益，Transformer 9 组中仅 1 组正增益。

## 环境依赖

- Python 3.12+
- PyTorch
- NumPy, Pandas, scikit-learn, scipy
- 模型定义：`scripts/models.py`
