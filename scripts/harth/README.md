# HARTH 软标签生成

## 数据集概述

[HARTH](https://archive.ics.uci.edu/dataset/779/harth) 是一个人体活动识别数据集，使用背 IMU + 大腿 IMU 两个传感器（各 3 轴），采样率 50Hz。

原始数据包含 22 个被试的 CSV 文件，位于 `thesis/datasets/HARTH/harth/`。

## 类别映射

### 使用的 8 类（raw label → class id）

| Raw Label | Activity | Class ID | 训练集样本数 | 说明 |
|-----------|----------|----------|-------------|------|
| 1 | Walking | 0 | 11,950 | |
| 2 | Running | 1 | 2,912 | |
| 3 | Shuffling | 2 | 2,533 | 缓慢移动 |
| 4 | Stairs Up | 3 | 766 | 上楼 |
| 5 | Stairs Down | 4 | 682 | 下楼 |
| 6 | Standing | 5 | 7,449 | 站立静止 |
| 7 | Sitting | 6 | 29,017 | 坐姿静止 |
| 8 | Lying | 7 | 4,287 | 躺姿静止 |

**总计: 59,596 训练样本**

### 被过滤的类别

| Raw Label | Activity | 样本数 | 过滤原因 |
|-----------|----------|--------|----------|
| 13 | Cycling (sitting) | 393,963 | 骑车，不属于日常活动分类目标 |
| 14 | Cycling (standing) | 55,814 | 骑车变体 |
| 130 | Cycling (sitting, inactive) | 41,785 | 骑车变体 |
| 140 | Cycling (standing, inactive) | 7,865 | 骑车变体 |

骑行类（13/14/130/140）数据量大但与其他活动差异明显，不属于论文研究的日常活动范围，故过滤。

### label_map

```python
label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
```

**注意：** 不是顺序映射！raw label 8→class 7（lying），不是 raw label 8 被忽略。

## 数据划分

```python
# 第一次划分：80% train, 20% test
X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 第二次划分：80% of train = 64% total, 20% of train = 16% total = validation
X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

| 集合 | 样本数 | 占比 |
|------|--------|------|
| Train | 59,596 | 64% |
| Validation | 14,900 | 16% |
| Test | 18,624 | 20% |

**`random_state=42` 必须与训练代码 `run_distill.py` 保持一致！**

## 传感器通道

窗口 shape: `(128, 6)` = 128 步 × 6 通道

| 通道索引 | 传感器 | 轴 |
|----------|--------|-----|
| 0 | back | x (左右) |
| 1 | back | y (前后) |
| 2 | back | z (上下，重力轴) |
| 3 | thigh | x (左右) |
| 4 | thigh | y (前后) |
| 5 | thigh | z (上下，重力轴) |

**重力方向：**
- back_z: 站立时约 -0.20（朝下），躺下时约 -0.14
- thigh_z: 坐姿时约 +0.90（大腿水平，z 轴朝上），站立时约 -0.12（大腿垂直）

## 软标签生成

### 运行

```bash
cd thesis/scripts/harth

# 从头开始
python gen.py --force

# 断点续跑（默认）
python gen.py
```

### 输出

```
thesis/results/
├── soft_labels/
│   ├── harth_soft.npy                  # 全量软标签 (59596, 8)
│   └── harth_soft_correct_only.npy     # 仅正确预测的软标签
└── logs/
    ├── gen_harth.log                   # 主日志
    ├── gen_harth_errors.log            # 错误日志
    ├── gen_harth_correct.log           # 正确预测日志
    └── gen_harth_checkpoint.json       # 断点文件
```

### 配置

在 `gen.py` 中：
- `MAX_PER_CLASS = 3000` — 每类采样上限
- `SLEEP_SEC` — 从 `api_config.py` 导入，控制 API 调用间隔
- `TEMPERATURE` — 从 `api_config.py` 导入

### 准确率参考（LLM 分类器）

| Class | 准确率 | 说明 |
|-------|--------|------|
| walk | ~55% | 常被误判为 shuffle/stairs |
| run | ~88% | 特征明显 |
| shuffle | ~46% | 和 stand/lying 重叠 |
| stairs_up | ~64% | 样本少 |
| stairs_down | ~40% | 样本最少 |
| stand | ~66% | 和 shuffle 重叠 |
| sit | ~98% | thigh_z 非常明显 |
| lying | ~44% | 和 shuffle/stand 重叠 |

整体约 65%。准确率瓶颈在静止类之间的特征重叠。软标签的价值在于保留概率分布（不确定性），不追求 100% 准确。

## 训练使用

### run_distill.py

```bash
cd thesis/scripts

# 使用全量软标签
python run_distill.py --dataset harth

# 使用仅正确版本
python run_distill.py --dataset harth --soft_file harth_soft_correct_only.npy
```

### 训练代码中的对应关系

```python
# run_distill.py 中的标签映射（必须与 gen.py 一致）
label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
CLASS_NAMES = ['walk', 'run', 'shuffle', 'stairs_up', 'stairs_down', 'stand', 'sit', 'lying']
```

### 软标签加载

```python
import numpy as np
soft = np.load('thesis/results/soft_labels/harth_soft.npy')
print(soft.shape)  # (59596, 8)
print(soft[0])     # [0.02, 0.03, 0.55, 0.05, 0.03, 0.15, 0.12, 0.05]  概率分布
```

## 已知问题

1. **静止类混淆**: stand/sit/shuffle/lying 之间特征重叠严重，LLM 难以区分
2. **楼梯类样本少**: stairs_up(766) 和 stairs_down(682) 远少于其他类
3. **骑行数据被过滤**: 原始数据中骑行占 ~40%，但不属于研究范围
