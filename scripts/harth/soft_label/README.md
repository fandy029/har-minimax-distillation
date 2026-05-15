# HARTH 软标签生成

## 1. 数据集简介

**来源**: HARTH 数据集，背部 + 大腿双 IMU 传感器，采样率 50Hz

**窗口**: 128步 × 6通道（背部3 + 大腿3），步进64

**类别 (8类)**:

| ID | 类别名 | 训练样本数 |
|----|--------|-----------|
| 0 | walk | 11,950 |
| 1 | run | 2,912 |
| 2 | shuffle | 2,533 |
| 3 | stairs_up | 766 |
| 4 | stairs_down | 682 |
| 5 | stand | 7,449 |
| 6 | sit | 29,017 |
| 7 | lying | 4,287 |

**训练样本总数**: 59,596

---

## 2. 软标签生成详情（训练相关）

### 输入格式
- 窗口 shape: `(128, 6)` = back(3) + thigh(3)
- 预处理: 10Hz 低通滤波提取重力方向

### 提供给模型的特征（5维关键特征）
```
thigh_std_mag — 大腿运动强度（核心分群指标）
thigh_mag     — 大腿加速度幅值
thigh_gz      — 大腿重力 Z 轴（区分 stand/sit/lying）
back_std_mag  — 背部运动强度
back_mag      — 背部加速度幅值
```

### 决策层次（按 thigh_std 分群）
- **GROUP A** (thigh_std < 0.1): 静态姿势 — stand/sit/lying
- **GROUP B** (0.1 ≤ thigh_std < 0.5): 低强度 — shuffle/stairs_up/stairs_down
- **GROUP C** (0.5 ≤ thigh_std < 0.9): 中等强度 — walk
- **GROUP D** (thigh_std ≥ 0.9): 高强度 — run

### 关键区分点
- **stand vs sit vs lying**: thigh_gz 差异显著（sit 的 thigh_z ≈ +0.90 最独特）
- **walk vs stairs_up vs stairs_down**: thigh_std 和 back_std 逐级递增
- **run**: thigh_std > 1.128，所有指标最高

### 软标签生成参数
- **模型**: MiniMax-M2.7 (API)
- **Temperature**: 0.3
- **API 失败重试**: ≤2 次
- **预测不匹配重试**: 1 次
- **输出**: `harth_soft.npy` (59596, 8) + `harth_soft_correct_only.npy`

### 输出文件
- `results/soft_labels/harth_soft.npy` — 全量软标签
- `results/soft_labels/harth_soft_correct_only.npy` — 仅 pred=true 的软标签
- `results/logs/gen_harth.log` — 运行日志
- `results/logs/gen_harth_errors.log` — 错误日志

### 训练提示
- 类别严重不平衡（sit 29,017 vs stairs_down 682）
- stairs_up 和 stairs_down 样本最少，训练时需关注召回率
- 决策树结构清晰，适合 CNN 层级学习
- 双传感器位置（背+腿）对静态姿势区分最有效
