# MotionSense 软标签生成

## 1. 数据集简介

**来源**: MotionSense 数据集，iPhone 腰部加速度计 (userAcceleration)，20Hz

**窗口**: 128步 × 3通道 (x/y/z 加速度)，步进64

**类别 (6类)**:

| ID | 类别名 | 训练样本数 |
|----|--------|-----------|
| 0 | downstairs | 1,196 |
| 1 | jogging | 1,237 |
| 2 | sitting | 3,173 |
| 3 | standing | 2,869 |
| 4 | upstairs | 1,436 |
| 5 | walking | 3,208 |

**训练样本总数**: 13,119

---

## 2. 软标签生成详情（训练相关）

### 输入格式
- 窗口 shape: `(128, 3)` — x/y/z 加速度 (userAcceleration)
- 注意: userAcceleration 已去除重力，是纯运动加速度

### 提供给模型的特征
```
magnitude_mean — 加速度幅值均值
mean_x/y/z     — 各轴均值
std_x/y/z      — 各轴标准差
xz_norm        — 重力在x/z轴的合成幅度（standing~0, sitting~1）
peak_count     — 峰值数量（步态周期）
dom_freq       — 主导频率 Hz（上楼<下楼<走路）
jerk_mean      — 加速度变化率均值（平滑度：上楼<0.18，下楼>0.26，走路>0.25）
```

### 决策规则（按优先级）
1. **std 全 < 0.05**: 静态 → y_mean>0.7 & xz_norm<0.25 → standing; y_mean<0.5 & xz_norm>0.2 → sitting
2. **std_x>0.7 & std_y>1.0 & magnitude>1.3**: jogging
3. **jerk_mean<0.18 & std_y<0.45**: upstairs（平滑上升）
4. **jerk_mean>0.26 & std_y>0.55**: walking（高变异性）
5. **jerk_mean 0.18-0.25 & std_y 0.30-0.55 & dom_freq 0.7-1.5Hz**: downstairs

### 关键区分点
- **upstairs vs downstairs**: jerk_mean 最关键（上楼更平滑 jerk<0.18，下楼冲击更大 jerk>0.26）
- **standing vs sitting**: xz_norm 差异大（站着重力在y轴 xz_norm≈0，坐着重力分散 xz_norm≈1）
- **jogging**: std 和 magnitude 均为最高，运动强度最大

### 软标签生成参数
- **模型**: MiniMax-M2.7 (API)
- **Temperature**: 0.3
- **API 失败重试**: ≤2 次
- **预测不匹配重试**: 1 次
- **输出**: `motionsense_soft.npy` (13119, 6) + `motionsense_soft_correct_only.npy`

### 输出文件
- `results/soft_labels/motionsense_soft.npy` — 全量软标签
- `results/soft_labels/motionsense_soft_correct_only.npy` — 仅 pred=true 的软标签
- `results/logs/gen_motionsense.log` — 运行日志
- `results/logs/gen_motionsense_errors.log` — 错误日志

### 训练提示
- userAcceleration（去重力）使静态姿势区分较难，主要依赖 mean_y 和 xz_norm
- downstairs/upstairs 样本较少，楼梯类活动区分是难点
- jogging 与 walking 的区别主要在运动强度（std 和 magnitude）
- dom_freq 和 jerk_mean 是上楼/下楼/走路区分的关键特征对
