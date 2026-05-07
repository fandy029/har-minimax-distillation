# Gait 数据集说明

## 数据来源

- `datasets/Gait_Classification/S1_Dataset/`: 60 名受试者 (d1p01M ~ d1p60F)
- `datasets/Gait_Classification/S2_Dataset/`: 10 名受试者 (d2p01F ~ d2p10F)

## 原始数据格式

```
时间(s), acc_frontal(G), acc_vertical(G), acc_lateral(G), antenna_id, RSSI, phase, frequency, label
```

- 采样率: **40Hz** (每 0.025s 一个点)
- 使用列: col[1:4] = 3 轴加速度 (frontal/vertical/lateral)
- 原始标签: 1=sit_on_bed, 2=sit_on_chair, 3=lying, 4=ambulating

## 类别

| class_id | 名称 | 映射 |
|----------|------|------|
| 0 | sit_on_bed | raw=1 |
| 1 | sit_on_chair | raw=2 |
| 2 | lying | raw=3 |
| 3 | ambulating | raw=4 |

## 窗口化

- 窗口长度: **128 步** (3.2 秒 @ 40Hz)
- 步长: **64 步** (1.6 秒, 50% 重叠)
- 仅保留 `shape[0]==128` 的完整窗口

## 软标签生成

- 脚本: `gait_gen.py`
- 输出文件（统一放在 `results/soft_labels/`）：
  - `gait_soft.npy` — 全量软标签
  - `gait_soft_correct_only.npy` — 仅正确预测的软标签
  - `.gen_gait.lock` — 单例锁
- 断点（放在 `results/logs/`）：
  - `gen_gait_checkpoint.json` — 断点续传记录
  - `gen_gait.log` — 主日志
  - `gen_gait_errors.log` — 错误日志
  - `gen_gait_correct.log` — 正确样本日志
- 每类上限: `MAX_PER_CLASS = 3000`

---

## 训练注意事项

### 类别不平衡
注：比例为原始时间点分布（51,520/16,406/4,911/2,291）。窗口化后实际可用窗口：

| class_id | 名称 | train 窗口数 | val 窗口数 | 原始点比例 |
|----------|------|-------------|----------|----------|
| 0 | sit_on_bed | 161 | 40 | ~21.8% |
| 1 | sit_on_chair | 38 | 10 | ~6.5% |
| 2 | lying | 436 | 109 | ~68.6% |
| 3 | ambulating | 38 | 10 | ~3.0% |

**注意**：数据集很小（train 仅 673 窗口），易过拟合。

**建议**: 考虑加权 loss 或过采样少数类。

### 输入形状
- `(batch, 128, 3)` = (样本数, 窗口长度, 通道数)
- 3 通道: frontal / vertical / lateral

### 建议的训练配置
```python
{
    "name": "Gait",
    "channels": 3,
    "window_len": 128,
    "num_classes": 4,
    "class_names": ["sit_on_bed", "sit_on_chair", "lying", "ambulating"],
    "label_map": {1: 0, 2: 1, 3: 2, 4: 3},
    "train_split": 0.6,   # 可用 train_test_split 得到
    "val_split": 0.2,
    "test_split": 0.2,
    "soft_label_file": "results/soft_labels/gait_soft.npy",
    "correct_only_file": "results/soft_labels/gait_soft_correct_only.npy",
    "class_weights": [1.0, 4.2, 0.37, 4.2]   # 按窗口数倒数调整（sit_on_bed=161为基准）
}
```

### 特征工程参考（用于 prompt 设计）
- 重力方向向量 (gfr/gve/gla): 反映身体朝向
- `free_mag_mean/std`: 去除重力后的运动强度
- `acc_mag_std`: 总信号变异性
- `peaks`: 步态周期峰值数
