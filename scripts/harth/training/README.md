# HARTH — 训练文档

## 快速开始

```bash
cd /home/fandy/workplace/thesis/scripts/harth
python3 training/train_phase1.py
python3 training/train_phase2.py
```

## 数据集

| 类别数 | 通道数 | 训练样本 |
|--------|--------|----------|
| 8 | 6 | — |

## 训练参数

| 参数 | 阶段1 | 阶段2 |
|------|--------|--------|
| 损失 | Focal Loss (γ=2.0) | α·Focal+(1-α)·T²·KL |
| 优化器 | AdamW(lr=5e-4,wd=1e-4) | AdamW(lr=1e-4,wd=1e-4) |
| 调度 | CosWR(T₀=20) | CosWR(T₀=20) |
| 批次/早停 | 64/15 | 64/15 |

## 蒸馏参数 (参考值, 需网格搜索)

| 架构 | α | T | 策略 |
|------|---|---|------|
| PureCNN | 0.5 | 2.5 | all |
| CNN-Res | 0.7 | 1.5 | filtered |
| Transformer | 0.7 | 1.5 | all |

> 参数值基于GAIT/PAMAP2实验经验, 各数据集实际最优值需网格搜索确定。
