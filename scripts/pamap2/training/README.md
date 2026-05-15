# PAMAP2 数据集 — 训练文档

## 概述

PAMAP2 (Physical Activity Monitoring for Aging People)，手部 IMU 100Hz，5 类活动识别。

**两阶段训练**（论文第 3 章）：阶段 1 Focal Loss 预训练 → 阶段 2 知识蒸馏微调。

## 快速开始

```bash
cd /home/fandy/workplace/thesis/scripts/pamap2

# 阶段1: 纯硬标签预训练
python3 training/train_phase1.py

# 阶段2: 蒸馏微调 (需先完成阶段1)
python3 training/train_phase2.py

# 单架构
python3 training/train_phase1.py --models purecnn
python3 training/train_phase2.py --models purecnn
```

## 目录结构

```
pamap2/
├── soft_label/                     # 软标签生成
│   ├── pamap2_prepare.py
│   ├── pamap2_gen.py
│   ├── pamap2_merge.py
│   └── pamap2_run.sh / cleanup.sh
├── training/                       # 训练
│   ├── train_phase1.py
│   ├── train_phase2.py
│   └── README.md
└── output/
    ├── results/                    # 模型+结果JSON
    ├── logs/                       # 训练日志
    └── soft_labels/                # 软标签.npy
```

## 数据集

| 类别 | 训练样本 |
|------|---------|
| lying | ~1,783 |
| sitting | ~1,697 |
| standing | ~1,700 |
| walking | ~803 |
| jogging | ~515 |

- 6 通道：加速度 3 轴 + 陀螺仪 3 轴
- 窗口 128×6，步进 64

## 训练超参数

| 参数 | 阶段 1 | 阶段 2 |
|------|--------|--------|
| 损失函数 | Focal Loss (γ=2.0) | α·Focal + (1-α)·T²·KL |
| 优化器 | AdamW (lr=5e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) |
| 调度器 | CosineAnnealingWarmRestarts (T₀=20) | 同左 |
| 批量/早停/梯度裁剪 | 64 / 15 / 5.0 | 同左 |

## 蒸馏参数

| 架构 | 策略 | α | T |
|------|------|---|---|
| PureCNN | all | 0.5 | 3.5 |
| CNN-Residual | filtered | 0.7 | 1.5 |
| Transformer | all | 0.7 | 1.5 |

> 参数参考 GAIT 实验选出。PAMAP2 物理特征分离度好（手部 IMU 对站/坐/躺区分度高），可考虑后续针对性网格搜索。

## 输出文件

每个阶段生成 `.pt`（模型权重）和 `.json`（结果详情），JSON 包含：
- `test_acc` / `val_acc` / `train_acc`
- `test_per_class`: 每类准确率
- `test_confusion_matrix`: 混淆矩阵
- `epoch_log`: 每 epoch 的 loss/acc
- `config`: 超参数
