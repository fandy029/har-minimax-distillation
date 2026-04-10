# 脚本参考文档

**更新时间**: 2026-04-10  
**注意**: 每次更新脚本后请同步更新本文档

---

## 目录

1. [数据准备](#1-数据准备)
2. [蒸馏脚本](#2-蒸馏脚本)
3. [工具脚本](#3-工具脚本)
4. [实验结果](#4-实验结果)

---

## 1. 数据准备

### step1_pure_cnn.py

**用途**: 为新数据集训练Pure CNN基线（Stage 1）

**使用方法**:
```bash
cd /home/fandy/workplace/thesis
python3 step1_pure_cnn.py
```

**说明**:
- 内置支持: `uci_har_new`, `motion_sense_dm`
- 训练Pure CNN模型，保存到`checkpoints/`
- 适用新加入的数据集

---

## 2. 蒸馏脚本

### 2.1 gen_soft_labels.py

**用途**: 调用MiniMax API生成软标签

**使用方法**:
```bash
python3 gen_soft_labels.py
```

**说明**:
- 每类采样150个样本
- 输出: `checkpoints/*_soft_labels.npy`
- 需要配置API_KEY和API_URL
- 超参: `SAMPLES_PER_CLASS`, `cn`类别名列表

**注意**: 此脚本针对HARTH，如需其他数据集需要修改数据加载和类别配置

---

### 2.2 蒸馏脚本总览

| 脚本 | 数据集 | 类别数 | 推荐方法 | 准确率 |
|------|--------|--------|---------|--------|
| `distill_kuhar_v3.py` | KuHar | 18 | v2课程蒸馏 | 85.02% |
| `distill_pamap2_v3.py` | PAMAP2 | 5 | v3极致蒸馏 | 95.93% |
| `distill_pamap2_v2.py` | PAMAP2 | 5 | v2课程蒸馏 | 95.02% |
| `distill_uci_new_v2.py` | UCI-HAR-New | 12 | v2课程蒸馏 | 93.99% |
| `distill_harth_v2.py` | HARTH | 6 | v2课程蒸馏 | 96.10% |
| `distill_uci_har_v2.py` | UCI-HAR | 6 | v1标准蒸馏 | 96.50% |
| `distill_gait_v2.py` | Gait | 4 | v1标准蒸馏 | 98.37% |
| `distill_motion_sense_v2.py` | MotionSense | 6 | v1标准蒸馏 | 99.40% |
| `distill_saturated_v2.py` | WISDM / MotionSense-DM | 6 | 直接用Pure CNN | 99.6% |

---

### 2.3 各脚本使用方法

#### KuHar (18类) — distill_kuhar_v3.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_kuhar_v3.py > new_results_v2/kuhar_v3_log.txt 2>&1 &
```
**特点**:
- Stage 1: Pure CNN 40 epochs
- Stage 2: v2课程蒸馏 80 epochs (T=1.5, α=0.85)
- 输出: `new_results_v2/kuhar_v3.json`

---

#### PAMAP2 (5类) — distill_pamap2_v3.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_pamap2_v3.py > new_results_v2/pamap2_v3_log.txt 2>&1 &
```
**特点**:
- Stage 1: Pure CNN 40 epochs
- Stage 2: v3极致蒸馏 80 epochs (T=1.5, α=0.85)
- 输出: `new_results_v2/pamap2_v3.json`

**备选**: `distill_pamap2_v2.py` (v2课程蒸馏, T=2.5, α=0.8)

---

#### UCI-HAR-New (12类) — distill_uci_new_v2.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_uci_new_v2.py > new_results_v2/uci_har_new_v2_log.txt 2>&1 &
```
**特点**:
- 含6个过渡态动作 (STAND_TO_SIT, SIT_TO_STAND等)
- v2课程蒸馏 (T=2.5, α=0.8)
- 输出: `new_results_v2/uci_har_new_v2.json`

---

#### HARTH (6类) — distill_harth_v2.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_harth_v2.py > new_results_v2/harth_v2_log.txt 2>&1 &
```
**特点**:
- 中文类别名: 左立/走路/上楼/下楼/右立/站立
- v2课程蒸馏 (T=2.5, α=0.8)
- 输出: `new_results_v2/harth_v2.json`

---

#### UCI-HAR (6类) — distill_uci_har_v2.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_uci_har_v2.py > new_results_v2/uci_har_v2_log.txt 2>&1 &
```
**特点**:
- v1标准蒸馏 (T=3.0, α=0.6)
- 输出: `new_results_v2/uci_har_v2.json`

---

#### Gait (4类) — distill_gait_v2.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_gait_v2.py > new_results_v2/gait_v2_log.txt 2>&1 &
```
**特点**:
- 样本最少 (588训练)
- v1标准蒸馏 (T=3.0, α=0.6)
- 输出: `new_results_v2/gait_v2.json`

---

#### MotionSense (6类) — distill_motion_sense_v2.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_motion_sense_v2.py > new_results_v2/motion_sense_v2_log.txt 2>&1 &
```
**特点**:
- v1标准蒸馏 (T=3.0, α=0.6)
- 输出: `new_results_v2/motion_sense_v2.json`

---

#### WISDM / MotionSense-DM — distill_saturated_v2.py
```bash
cd /home/fandy/workplace/thesis
python3 -u distill_saturated_v2.py > new_results_v2/saturated_v2_log.txt 2>&1 &
```
**特点**:
- 已极度饱和，无需蒸馏
- 直接用Pure CNN (99.6%)
- 输出: `new_results_v2/saturated_v2.json`

---

## 3. 工具脚本

### all_dataset_results.json

**用途**: 汇总所有数据集的最终结果

**格式**:
```json
{
  "known": {
    "pamap2": {"pure_cnn": 92.7, "cnn_minimax": 93.1},
    ...
  },
  "new": {...}
}
```

---

## 4. 实验结果

### 输出位置

所有实验结果保存在 `new_results_v2/` 目录：

```
new_results_v2/
├── kuhar_v3.json          # KuHar v3结果
├── kuhar_v3_log.txt       # KuHar训练日志
├── pamap2_v3.json         # PAMAP2 v3结果
├── pamap2_v3_log.txt      # PAMAP2训练日志
├── harth_v2.json          # HARTH结果
├── gait_v2.json           # Gait结果
├── motion_sense_v2.json    # MotionSense结果
├── uci_har_v2.json        # UCI-HAR结果
├── uci_har_new_v2.json    # UCI-HAR-New结果
└── saturated_v2.json      # WISDM/MotionSense-DM结果
```

### 结果JSON格式

```json
{
  "dataset": "KuHar",
  "num_classes": 18,
  "train": 20000,
  "test": 19129,
  "pure_cnn": 78.29,
  "v1_kd": 81.01,
  "v2_kd": 85.02,
  "v3_kd": 82.90,
  "v3_vs_v2": -2.12,
  "kd_class_acc": {
    "Stand": 0.391,
    "Sit": 0.880,
    ...
  }
}
```

---

## 附录: 蒸馏方法说明

### v1 标准蒸馏
- 温度T=3.0，软标签平滑
- α=0.6 (硬标签权重)
- 适用: 简单分类、数据饱和

### v2 课程蒸馏
- 温度T=2.5
- α=0.8 (高硬标签保护)
- 含课程学习，逐步加入混淆样本
- 适用: 复杂多类、含过渡态

### v3 极致蒸馏
- 温度T=1.5，软标签sharp
- α=0.85 (极高硬标签权重)
- 从Stage1继承模型
- 适用: 数据充足、中等难度

### 脚本参数说明

| 参数 | 含义 | v1 | v2 | v3 |
|------|------|-----|-----|-----|
| `TEMP` / `T` | 温度 | 3.0 | 2.5 | 1.5 |
| `ALPHA` / `α` | 硬标签权重 | 0.6 | 0.8 | 0.85 |
| `DISTILL_WEIGHT` | 蒸馏权重 | 0.4 | 0.2 | 0.15 |

---

*最后更新: 2026-04-10*
