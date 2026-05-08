# UCI-HAR 软标签生成

## 1. 数据集简介

**来源**: UCI Human Activity Recognition 数据集，三星 Galaxy S2 腰部传感器，50Hz

**窗口**: 128步 × 561维特征（预处理后），滑动窗口

**类别 (6类)**:

| ID | 类别名 | 训练样本数 |
|----|--------|-----------|
| 0 | WALKING | 981 |
| 1 | WALKING_UP | 858 |
| 2 | WALKING_DOWN | 789 |
| 3 | SITTING | 1,029 |
| 4 | STANDING | 1,099 |
| 5 | LAYING | 1,125 |

**训练样本总数**: 5,881

---

## 2. 软标签生成详情（训练相关）

### 输入格式
- 窗口: 128步 × 561维特征（已预计算的时域+频域特征）
- 关键特征索引（0-based）:
  - body_mad_z (index 50): 身体加速度 Z 轴中位数绝对值
  - grav_mean_x/y/z (indices 117-119): 重力加速度均值
  - facc_mean_x/y/z: FFT 后身体加速度均值
  - facc_mag: FFT 身体加速度幅值均值
  - angle_y: Y 轴与重力夹角

### 提供给模型的特征（6维关键特征）
```
body_mad_z      — Z轴中位数绝对值（>-0.6动态，≤-0.6静态）
grav_mean_x     — X轴重力均值（LAYING<0.5; 其他>0.5）
grav_mean_y     — Y轴重力均值（SIT>0; STAND<0）
grav_mean_z     — Z轴重力均值
facc_avg        — FFT 身体加速度均值（动态~-0.3; 静态~-0.98）
facc_mag        — FFT 身体加速度幅值（DOWN>+0.16; WALK~-0.30; UP~-0.24）
angle_y         — Y轴与重力夹角（SIT>0; STAND>0 有重叠）
```

### 决策树（3步）
**Step 1**: body_mad_z > -0.6 → DYNAMIC; ≤ -0.6 → STATIC

**Step 2A STATIC**:
- grav_mean_x < 0.5 → LAYING
- grav_mean_y > 0.0 → SITTING
- grav_mean_y ≤ 0.0 → STANDING

**Step 2B DYNAMIC**:
- facc_mag > 0 → WALKING_DOWN（幅值最高）
- grav_mean_y < -0.25 → WALKING_UP（Y轴更负）
- 否则 → WALKING

### 关键区分点
- **LAYING**: grav_mean_x 独特（<0.5，躺下时X轴重力方向反转）
- **SITTING vs STANDING**: grav_mean_y 分界（SIT>0，STAND<0）
- **WALKING_DOWN**: facc_mag 唯一为正（>0.16），其他动态类为负
- **WALKING_UP vs WALKING**: grav_mean_y 更负（<-0.25）

### 软标签生成参数
- **模型**: MiniMax-M2.7 (API)
- **Temperature**: 0.3
- **API 失败重试**: ≤2 次
- **预测不匹配重试**: 1 次
- **输出**: `uci_har_soft.npy` (5881, 6) + `uci_har_soft_correct_only.npy`

### 输出文件
- `results/soft_labels/uci_har_soft.npy` — 全量软标签
- `results/soft_labels/uci_har_soft_correct_only.npy` — 仅 pred=true 的软标签
- `results/logs/gen_uci_har.log` — 运行日志
- `results/logs/gen_uci_har_errors.log` — 错误日志

### 训练提示
- 561维特征已预计算好，软标签生成时模型直接看到关键6维测量值
- 类别较平衡，但 WALKING_DOWN(789) 样本最少
- 动态类（WALK/UP/DOWN）间区分依赖 facc_mag 和 grav_mean_y，特征边界清晰
- STATIC 类间区分主要靠重力方向（grav_mean_x/y），边界明显
- 软标签中 WALKING_UP 的 grav_mean_y 阈值 (-0.25) 是关键分界线
