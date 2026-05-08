# PAMAP2 软标签生成

## 1. 数据集简介

**来源**: PAMAP2 (Physical Activity Monitoring for Aging People) 数据集，手部 IMU，100Hz

**窗口**: 128步 × 6通道（加速度3 + 陀螺仪3），步进64

**类别 (5类)**:

| ID | 类别名 | 训练样本数 |
|----|--------|-----------|
| 0 | lying | 1,783 |
| 1 | sitting | 1,697 |
| 2 | standing | 1,700 |
| 3 | walking | 803 |
| 4 | jogging | 515 |

**训练样本总数**: 6,498

---

## 2. 软标签生成详情（训练相关）

### 输入格式
- 窗口 shape: `(128, 6)` — 手部 IMU (acc_x/y/z + gyro_x/y/z)
- 注意: PAMAP2 原始数据有 52 列，本脚本只提取手部 IMU 的 6 通道

### 提供给模型的特征
```
jerk      — 加速度变化率均值（静态/动态划分：≤0.10静态，>0.10动态）
dom_freq  — 主导频率 Hz（走路~0.8Hz，跑步>1.5Hz）
gyro_std  — 陀螺仪标准差
acc_mean_x — X轴加速度均值（站≈2.4，躺≈7.3，坐≈6.9）
gyro_y    — Y轴陀螺仪均值
acc_range — 加速度值域范围
```

### 决策树（优先级）
1. **jerk > 0.10**: 动态（walk/jog）→ dom_freq≤1.5Hz → walking; dom_freq>1.5Hz → jogging
2. **jerk ≤ 0.10**: 静态 → acc_mean_x<4.0 → standing; acc_mean_x>5.0 → lying/sitting
3. **静态 + gyro_std < 16 + acc_mean_x < 4**: standing
4. **静态 + gyro_std > 16 + acc_mean_x > 5**: lying/sitting

### 关键区分点
- **lying vs sitting vs standing**: acc_mean_x 最关键（躺≈7.3，坐≈6.9，站≈2.4）
- **walking vs jogging**: dom_freq 和 jerk（跑步频率>1.5Hz，冲击更大）
- **静态姿势区分**: 手部位置差异导致重力在各轴分布不同

### 软标签生成参数
- **模型**: MiniMax-M2.7 (API)
- **Temperature**: 0.3
- **API 失败重试**: ≤2 次
- **预测不匹配重试**: 1 次
- **输出**: `pamap2_soft.npy` (6498, 5) + `pamap2_soft_correct_only.npy`

### 输出文件
- `results/soft_labels/pamap2_soft.npy` — 全量软标签
- `results/soft_labels/pamap2_soft_correct_only.npy` — 仅 pred=true 的软标签
- `results/logs/gen_pamap2.log` — 运行日志
- `results/logs/gen_pamap2_errors.log` — 错误日志

### 训练提示
- 类别相对平衡，但 walking(803) 和 jogging(515) 样本较少
- 手部 IMU 对静态姿势区分最有效（acc_mean_x 在三轴分布差异大）
- 动态活动区分依赖频率特征，CNN 可学习步态周期模式
- jog 和 walk 的 jerk 阈值 0.25 可作为软标签学习的参考边界
