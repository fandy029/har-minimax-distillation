# Gait 软标签生成

## 1. 数据集简介

**来源**: Gait_Classification 数据集，3轴加速度计，采样率 40Hz

**窗口**: 128步 × 3通道，步进64，重叠率 50%

**类别 (4类)**:

| ID | 类别名 | 训练样本数 |
|----|--------|-----------|
| 0 | sit_on_bed | 161 |
| 1 | sit_on_chair | 38 |
| 2 | lying | 436 |
| 3 | ambulating | 38 |

**训练样本总数**: 673

---

## 2. 软标签生成详情（训练相关）

### 输入格式
- 窗口 shape: `(128, 3)` — 3通道加速度（frontal/vertical/lateral）
- 预处理: 10Hz 低通滤波提取重力方向，计算去重力后的自由加速度幅值

### 提供给模型的特征
```
gfr (gravity frontal)      — 身体前后倾角
gve (gravity vertical)     — 身体上下方向（躺卧区分关键）
gla (gravity lateral)      — 身体左右倾角
free_mag_mean             — 运动强度（去重力后幅值均值）
free_mag_std              — 运动变异性
acc_mag_std               — 总信号变异性
peaks                     — 步态周期峰值数
```

### 决策关键
- **lying vs sit**: gve 差异大（躺着 gve≈0.14，坐着 gve≈0.69-0.77）
- **sit_on_bed vs sit_on_chair**: gfr 不同（床更靠后 gfr≈0.47，椅子更前倾 gfr≈0.61）
- **ambulating**: 动态特征，free_mag_std 最高，peaks 最多

### 软标签生成参数
- **模型**: MiniMax-M2.7 (API)
- **Temperature**: 0.3
- **API 失败重试**: ≤2 次
- **预测不匹配重试**: 1 次（pred ≠ true_label 时再调一次）
- **输出**: `gait_soft.npy` (673, 4) + `gait_soft_correct_only.npy`

### 输出文件
- `results/soft_labels/gait_soft.npy` — 全量软标签
- `results/soft_labels/gait_soft_correct_only.npy` — 仅 pred=true 的软标签
- `results/logs/gen_gait.log` — 运行日志
- `results/logs/gen_gait_errors.log` — 错误日志

### 训练提示
- 类别极不平衡（lying 436 vs sit_on_chair/ambulating 各38）
- 建议关注 ambulating 的召回率
- 静态姿势类（sit_on_bed/sit_on_chair/lying）区分依赖重力方向，动态类区分依赖运动强度和峰值数
