# MiniMax知识蒸馏人体活动识别 (HAR)

基于MiniMax大语言模型的知识蒸馏方案，用于人体活动识别（HAR）任务。

## 📋 项目概述

本项目使用**"LLM as Teacher"知识蒸馏框架**，利用MiniMax作为教师模型为HAR任务生成软标签，显著提升CNN学生模型的识别准确率。

### 方法亮点

- **MiniMax教师模型**：分析加速度信号特征，输出类别概率分布（软标签）
- **高效蒸馏**：仅需每类200个样本（约1200次API调用）即可带来显著提升
- **通用性强**：适用于不同数据格式和规模的HAR数据集

## 📊 实验结果

| 数据集 | Pure CNN | +MiniMax KD | 提升 |
|--------|----------|-------------|------|
| PAMAP2 | 92.7% | 93.1% | +0.4% |
| UCI-HAR | 96.2% | 96.5% | +0.3% |
| MotionSense | 99.2% | 99.4% | +0.2% |
| WISDM | 99.6% | 99.6% | ±0% |

## 📁 数据集目录结构

本项目使用的数据集存放在 `../simclr/datasets/` 目录下：

```
../simclr/datasets/
├── PAMAP2/
│   └── PAMAP2_Dataset/
│       └── Protocol/
│           ├── subject101.dat
│           ├── subject102.dat
│           └── ...
│
├── UCI-HAR/
│   ├── train/
│   │   ├── Inertial Signals/
│   │   │   ├── total_acc_x_train.txt
│   │   │   ├── total_acc_y_train.txt
│   │   │   ├── total_acc_z_train.txt
│   │   │   └── ...
│   │   ├── X_train.txt
│   │   ├── y_train.txt
│   │   └── subject_train.txt
│   └── test/
│       ├── Inertial Signals/
│       └── ...
│
├── MotionSense/
│   ├── dws_1/      # downstairs
│   │   ├── sub_10.csv
│   │   ├── sub_11.csv
│   │   └── ...
│   ├── jog_16/      # jogging
│   ├── sit_13/      # sitting
│   ├── std_14/      # standing
│   ├── ups_12/      # upstairs
│   └── wlk_15/      # walking
│
└── WISDM/
    └── WISDM_ar_v1.1/
        └── WISDM_ar_v1.1_raw.txt
```

### 数据集说明

| 数据集 | 原始样本数 | 窗口样本数 | 类别数 | 特点 |
|--------|-----------|-----------|--------|------|
| **PAMAP2** | ~2,672 | ~3,340 | 5 | 样本较少 |
| **UCI-HAR** | ~10,299 | ~10,299 | 6 | 标准学术数据集 |
| **MotionSense** | ~20,000 | ~21,865 | 6 | 手机传感器，数据干净 |
| **WISDM** | ~1,098,207 | ~16,707 | 6 | 严重不平衡(9:1) |

### ⚠️ 数据下载

数据集需要单独下载，不在本仓库中。推荐下载方式：

```bash
# 创建数据目录
mkdir -p ../simclr/datasets

# PAMAP2
# 来源: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

# UCI-HAR  
# 来源: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

# MotionSense
# 来源: https://github.com/mmalei/MotionSense

# WISDM
# 来源: https://www.cise.ufl.edu/helina/oject/WISDM_ar_v1.1_raw.txt
```

## 🗂️ 代码文件结构

```
thesis/
├── README.md                    # 本文件
├── .gitignore                   # Git忽略规则
│
├── 数据准备相关
├── pamap2_distill_final.py       # PAMAP2知识蒸馏完整版
├── uci_har_distill.py           # UCI-HAR知识蒸馏
├── motionsense_distill.py       # MotionSense知识蒸馏
├── wisdm_distill.py             # WISDM知识蒸馏
│
├── 基线对比实验
├── pure_cnn_baseline.py         # 三个数据集的Pure CNN基线
├── uci_har_baseline.py          # UCI-HAR Pure CNN基线
│
└── 报告
├── report_knowledge_distillation.md    # 完整报告(源码)
└── report_knowledge_distillation.pdf    # PDF版本
```

## 🔧 核心方法

### 软标签生成流程

```
输入: 加速度窗口 (128, 3)
         │
         ▼
┌─────────────────────────────────────┐
│     手工特征提取 (7个特征)            │
│  - Y趋势、幅度均值/方差/最大值       │
│  - Y轴范围、步态峰值、主频            │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      MiniMax教师模型推理             │
│  Prompt包含特征值+物理常识            │
│  输出类别概率分布JSON                 │
└─────────────────────────────────────┘
         │
         ▼
软标签: [0.05, 0.02, 0.70, 0.15, ...]
         ↑ 包含"上楼和下楼更相似"知识
```

### 混合损失函数

```python
Loss = α × FocalLoss(hard_labels) + (1-α) × KL_div(soft_teacher || soft_student) × T²
     = 0.6 × FocalLoss + 0.4 × KL_div × 9
```

### CNN模型结构

```
DeepCNN: 4层Conv1d + 3层FC + Dropout(0.4)
├── Conv1d(3, 64) → BN → ReLU
├── Conv1d(64, 128) → BN → ReLU  
├── Conv1d(128, 256) → BN → ReLU
├── Conv1d(256, 256) → BN → ReLU
├── AdaptiveAvgPool1d(8)
├── Flatten → FC(2048→128) → ReLU → Dropout
├── FC(128→64) → ReLU → Dropout
└── FC(64→num_classes)
```

## 🚀 快速开始

### 1. 环境依赖

```bash
pip install torch numpy pandas scikit-learn openai
```

### 2. 下载数据

将数据集放入 `../simclr/datasets/` 目录（见上文说明）

### 3. 运行实验

```bash
# PAMAP2知识蒸馏
python pamap2_distill_final.py

# 其他数据集类似
python uci_har_distill.py
python motionsense_distill.py
python wisdm_distill.py
```

## 📝 引用

如果你使用了本项目的代码或方法，请引用：

```
@misc{har_distillation_2026,
  title={MiniMax Knowledge Distillation for Human Activity Recognition},
  author={},
  year={2026}
}
```

## 📄 许可证

MIT License
