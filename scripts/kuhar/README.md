# KuHar 软标签生成

## 1. 数据集简介

**来源**: KuHar (Korean Human Activity Recognition) 数据集，腰部传感器，100Hz

**窗口**: 128步 × 8通道（加速度3 + 陀螺仪3 + 2额外通道），步进64

**类别 (18类)**:

| ID | 类别名 | 样本数 | ID | 类别名 | 样本数 |
|----|--------|--------|----|--------|--------|
| 0 | Stand | 5,191 | 9 | Push-up | 1,389 |
| 1 | Sit | 5,152 | 10 | Sit-up | 2,869 |
| 2 | Talk-sit | 4,971 | 11 | Walk | 2,328 |
| 3 | Talk-stand | 5,155 | 12 | Walk-backwards | 845 |
| 4 | Stand-sit | 5,881 | 13 | Walk-circle | 701 |
| 5 | Lay | 5,058 | 14 | Run | 1,643 |
| 6 | Lay-stand | 4,807 | 15 | Stair-up | 2,321 |
| 7 | Pick | 3,664 | 16 | Stair-down | 2,259 |
| 8 | Jump | 1,903 | 17 | Table-tennis | 1,247 |

**训练样本总数**: 57,384

---

## 2. 软标签生成详情（训练相关）

### 输入格式
- 窗口 shape: `(128, 8)` — 8通道原始时域数据
- cols 1-3: 加速度 X/Y/Z，cols 5-7: 陀螺仪 X/Y/Z

### 提供给模型的特征（核心判别特征）
```
energy_acc    — 加速度总能量（按强度分层：<0.1静态 / 0.1-5过渡 / 5-20轻量 / >100高强度）
acc_y_std     — Y轴变化幅度（行走/跑步/过渡动作）
gyro_x_std    — X轴旋转（乒乓/楼梯关键）
gyro_y_std    — Y轴旋转（站-坐 vs 躺-站过渡区分）
jerk          — 加速度变化率（Jump高，Push-up低）
n_peaks_acc   — 峰值数量（周期性步态检测）
impulsiveness — 峰值/rms比（Jump>3，Push-up<2.1）
z_grav        — 垂直方向重力（>0.65直立，<0.55水平）
kurt_az       — Z轴峰度（Jump>3，Run中等）
```

### 决策流程（4步）
1. **能量分层**: 按 energy_acc 划分动静和强度等级
2. **方向判断**: z_grav 区分直立/水平/坐姿
3. **过渡动作**: gyro_y_std + acc_y_std 区分 Stand-sit/Lay-stand/Sit-up/Talk-stand
4. **动态活动**: impulsiveness + n_peaks + dom_freq 区分 Walk/Run/Stairs/Push-up/Ping-pong

### 关键区分点
- **Push-up vs Jump**: impulsiveness < 2.1 唯一标识 Push-up
- **Stand vs Sit vs Lay**: z_grav 分界清晰（>0.78 Sit, 0.65-0.78 Stand, <0.55 Lay）
- **Ping-pong vs Stairs-down**: gyro_x（乒乓≈1.0，楼梯≈0.53）
- **Jump vs Run**: energy（Jump≈177，Run≈323，但 jerk 差异更大）

### 软标签生成参数
- **模型**: MiniMax-M2.7 (API)
- **Temperature**: 0.3
- **API 失败重试**: ≤3 次
- **预测不匹配重试**: ❌ 已禁用（每次只调1次API）
- **输出**: `kuhar_soft.npy` (57384, 18) + `kuhar_soft_correct_only.npy`

### 输出文件
- `results/soft_labels/kuhar_soft.npy` — 全量软标签
- `results/soft_labels/kuhar_soft_correct_only.npy` — 仅 pred=true 的软标签
- `results/logs/gen_kuhar.log` — 运行日志
- `results/logs/gen_kuhar_errors.log` — 错误日志

### 训练提示
- 18类中最复杂的数据集，动作间特征重叠多
- Walk-circle 和 Walk-backwards 样本最少（701/845），注意召回率
- Talk-sit/Talk-stand 与 Sit/Stand 极为相似，依赖微小的周期性说话动作特征
- Ping-pong（乒乓球）与其他动态活动区分依赖陀螺仪 X 轴高旋转特征
