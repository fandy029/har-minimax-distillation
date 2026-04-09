# 基于MiniMax大模型知识蒸馏的人体活动识别

## 项目概述

利用MiniMax大模型（M2.7）作为教师模型，基于IMU传感器窗口的物理特征生成软标签（Soft Labels），通过知识蒸馏提升CNN学生模型的HAR分类准确率。

---

## 📁 数据集路径

所有数据集位于 `/home/fandy/workplace/thesis/datasets/`：

| 数据集 | 路径 | 通道数 | 类别数 | 训练/测试样本 |
|--------|------|--------|--------|--------------|
| PAMAP2 | `datasets/PAMAP2/` | 6 | 5 | 2,137 / 535 |
| UCI-HAR | `datasets/UCI_HAR/` | 9 | 6 | 7,352 / 2,947 |
| MotionSense | `datasets/MotionSense/` | 6 | 6 | 17,492 / 4,373 |
| WISDM | `datasets/WISDM/` | 3 | 6 | 13,365 / 3,342 |
| KuHar | `datasets/KuHar/1.Raw_time_domian_data/` | 8 | 18 | 20,000 / 19,129 |
| UCI-HAR-New | `datasets/UCI_HAR_New/` | 561 | 12 | 6,213 / 3,162 |
| MotionSense-DM | `datasets/MotionSense_DeviceMotion/A_DeviceMotion_data/` | 12 | 6 | 13,777 / 4,306 |
| HARTH | `datasets/HARTH/harth/` | 6 | 6 | 20,000 / 17,678 |
| Gait | `datasets/Gait_Classification/` | 3 | 4 | 588 / 184 |

### 外部数据集路径

部分数据集位于其他目录（软链接或依赖）：

| 数据集 | 路径 |
|--------|------|
| PAMAP2 (原始) | `/home/fandy/workplace/simclr/datasets/PAMAP2/PAMAP2_Dataset/` |
| UCI-HAR (原始) | `/home/fandy/workplace/simclr/datasets/UCI-HAR/` |
| MotionSense (原始) | `/home/fandy/workplace/simclr/datasets/MotionSense/` |
| WISDM (原始) | `/home/fandy/workplace/simclr/datasets/WISDM/` |

---

## 🔬 软标签数据路径

所有软标签（.npy格式）：

| 数据集 | 文件路径 | Shape | 大小 |
|--------|---------|-------|------|
| PAMAP2 | `soft_labels_final.npy` | (2137, 5) | 44KB |
| UCI-HAR | `soft_labels_uci_har.npy` | (7352, 6) | 176KB |
| MotionSense | `soft_labels_motionsense.npy` | (17492, 6) | 412KB |
| WISDM | `soft_labels_wisdm.npy` | (13365, 6) | 316KB |
| KuHar | `new_results/kuhar_soft.npy` | (20000, 18) | 1.4MB |
| UCI-HAR-New | `new_results/uci_har_new_soft.npy` | (6213, 12) | 292KB |
| MotionSense-DM | `new_results/motion_sense_dm_soft.npy` | (13777, 6) | 324KB |
| HARTH | `checkpoints/harth_soft_labels.npy` | (20000, 6) | 472KB |

---

## 📊 实验结果

### 9数据集总览

| # | 数据集 | Pure CNN | +MiniMax KD | 提升 |
|---|--------|---------|------------|------|
| 1 | PAMAP2 | 92.70% | 93.10% | **+0.40%** |
| 2 | UCI-HAR | 96.20% | 96.50% | **+0.30%** |
| 3 | MotionSense | 99.20% | 99.40% | **+0.20%** |
| 4 | WISDM | 99.60% | 99.60% | ±0% |
| 5 | KuHar | 81.00% | 81.01% | **+0.01%** |
| 6 | UCI-HAR-New | 93.45% | 93.67% | **+0.22%** |
| 7 | MotionSense-DM | 99.58% | 99.58% | ±0% |
| 8 | HARTH | 95.73% | 95.83% | **+0.10%** |
| 9 | Gait | 96.74% | 98.37% | **+1.63%** |

**9数据集平均提升: +0.32%**

### 关键发现

✅ **有效场景**:
- 小数据集蒸馏效果最显著（Gait +1.63%，训练仅588样本）
- 物理特征清晰的动作（走路、慢跑、上下楼）提升明显
- PAMAP2走路类：90.00% → 93.33%（+3.33%）

❌ **失效场景**:
- 高饱和数据集无提升空间（WISDM 99.6%, MotionSense-DM 99.58%）
- 相似动作互相干扰（KuHar Stand: 81.34% → 40.09%）
- 过渡态动作灾难性下降（UCI-HAR-New SIT_TO_STAND: 90% → 0%）

---

## ✅ 成果

1. **完整的9数据集实验框架**: 从数据加载、模型训练到蒸馏的端到端流程
2. **CNN架构设计**: DeepCNN（1D时序信号）+ MLP（特征向量），参数量~1.4M
3. **MiniMax API集成**: Prompt工程设计，支持软标签批量生成
4. **训练策略优化**: Focal Loss + Mixup + CosineAnnealingWarmRestarts
5. **蒸馏效果验证**: 9数据集平均+0.32%提升，Gait最高+1.63%
6. **详细实验报告**: `report_full_9datasets.md` 包含各类别细粒度分析

---

## ❌ 不足与局限性

### 1. 相似动作区分失败
**问题**: Sit和Stand在物理特征上几乎相同（低频、低幅度），MiniMax软标签导致两者互相干扰。
- KuHar Sit: 33.84% → 71.40% (+37.56%)
- KuHar Stand: 81.34% → 40.09% (-41.25%)

**根因**: 1D CNN无法捕捉如此细微的差异，软标签也帮不上忙。

### 2. 过渡态动作灾难性下降
**问题**: UCI-HAR-New的SIT_TO_STAND/LIE_TO_STAND蒸馏后完全失效。
- SIT_TO_STAND: 90% → 0%
- LIE_TO_STAND: 81% → 0%

**根因**: 过渡态窗口太短（几十帧），物理特征不完整，MiniMax给出高置信度错误答案。

### 3. 高饱和数据集无提升空间
**问题**: WISDM、MotionSense-DM已达99%+，蒸馏几乎没有帮助。

### 4. 软标签质量依赖Prompt
**问题**: 当前Prompt对模糊场景（过渡态、相似动作）效果差，需人工设计物理特征描述。

### 5. 仅用时序数据，未融合多模态
**问题**: 当前方法仅用IMU传感器数据，未使用视觉或文本描述信息。

---

## 🔧 核心脚本说明

| 脚本 | 用途 |
|------|------|
| `step1_pure_cnn.py` | 新增数据集（KuHar, UCI-HAR-New, MotionSense-DM）Pure CNN训练 |
| `step2_distill.py` | 新增数据集MiniMax蒸馏 |
| `run_all_datasets.py` | 全部8个数据集统一训练框架 |
| `train_new_datasets.py` | 新数据集训练入口 |
| `pure_cnn_baseline.py` | 4个原始数据集Pure CNN基线 |
| `pamap2_distill_final.py` | PAMAP2蒸馏主方案 |
| `gen_soft_labels.py` | 独立软标签生成脚本 |
| `report_full_9datasets.md` | 完整实验报告（约18KB） |

---

## 🔮 未来改进方向

1. **时序建模**: 引入LSTM/Transformer处理过渡态动作和相似动作区分
2. **自适应温度**: 对高置信度错误样本降低温度T，减少误导
3. **主动学习**: 优先对难分类样本调用MiniMax API，优化成本
4. **课程蒸馏**: 从易到难逐步蒸馏（先简单类后复杂类）
5. **多模态融合**: 结合视觉（摄像头）或文本（活动描述）信息
6. **自蒸馏**: 用Pure CNN生成软标签进行迭代自训练

---

## 📝 实验配置

```
API: https://api.minimaxi.com/v1
Model: MiniMax-M2.7-highspeed
蒸馏温度 T: 3.0
损失函数: 0.6×Focal + 0.4×KL散度
优化器: AdamW (lr=5e-4, wd=1e-4)
Epochs: 60-100
Batch: 64
Max Train: 20000样本/数据集
```
