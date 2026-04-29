# 基于MiniMax大模型知识蒸馏的人体活动识别

利用 MiniMax-M2.7 大模型作为教师模型，基于 IMU 传感器窗口的物理特征生成软标签（Soft Labels），通过知识蒸馏提升 CNN 学生模型的 HAR 分类准确率。

---

## 📁 数据集

| 数据集 | 路径 | 类别数 | 训练样本 | 说明 |
|--------|------|--------|---------|------|
| PAMAP2 | `datasets/PAMAP2/` | 5 | 2200 | 6轴IMU，max_train限制 |
| KuHar | `datasets/KuHar/` | 18 | 20000 | 8轴IMU |
| UCI-HAR | `datasets/UCI_HAR/` | 6 | ~4700 | 9轴，train_test_split 80/20 |
| HARTH | `datasets/HARTH/harth/` | 6 | 20000 | 3轴（背部下传感器） |
| UCI-HAR-New | `datasets/UCI_HAR_New/` | 12 | 5000 | 561维特征（MLP） |
| MotionSense | `datasets/MotionSense/` | 6 | 11000 | 3轴，max_train限制 |
| MotionSense-DM | `datasets/MotionSense_DeviceMotion/` | 6 | 10000 | DeviceMotion多轴 |
| WISDM | `datasets/WISDM/` | 6 | ~8700 | 3轴滑动窗口128步 |
| Gait | `datasets/Gait_Classification/` | 4 | 400 | 6轴（加速度+陀螺仪） |

> 所有数据集划分：训练 80% / 验证 16% / 测试 20%（嵌套划分）。验证集用于 Early Stopping，测试集用于最终评估。

---

## 🔧 脚本说明

所有核心脚本位于 `scripts/` 目录。根目录的 `.sh` 脚本是后台运行包装器。

---

### scripts/run_distill.py —— 单数据集训练（核心脚本）

**作用**：单个数据集的 Stage1（纯CNN）+ Stage2（蒸馏）训练，保存 Checkpoint 和结果。

**用法**：
```bash
python3 scripts/run_distill.py <dataset> <version> [--resume]
```

**参数说明**：
- `<dataset>`：`pamap2`, `kuhar`, `uci_har`, `harth`, `uci_har_new`, `motionsense`, `motionsense_dm`, `gait`, `wisdm`
- `<version>`：`pure_cnn`（纯CNN基线）/ `v1` / `v2` / `v3`（蒸馏版本）
- `--resume`：从 Checkpoint 断点续传

**示例**：
```bash
python3 scripts/run_distill.py pamap2 v2              # 训练 PAMAP2 蒸馏 v2
python3 scripts/run_distill.py wisdm pure_cnn         # 训练 WISDM 纯CNN基线
python3 scripts/run_distill.py kuhar v3 --resume       # 断点续传
```

**训练流程**：
1. **Stage 1（纯CNN）**：focal loss 训练 300 epochs，保存最佳 Val 模型
2. **Stage 2（蒸馏）**：v1/v2/v3 在 Stage1 权重基础上用软标签微调 300 epochs
   - 损失：`α × Focal + (1-α) × KL_div × T²`

**输出文件**：
| 文件 | 路径 |
|------|------|
| Checkpoint | `results/checkpoints/{dataset}_{version}_best.pt` |
| 训练日志 | `results/logs/{dataset}_{version}_train.log` |
| 历史记录 | `results/history/{dataset}_{version}_history.json` |
| 实验结果 | `results/{dataset}_{version}.json` |
| 混淆矩阵 | `results/history/{dataset}_{version}_cm.npy` |
| 每类准确率 | `results/history/{dataset}_{version}_class_acc.json` |

---

### scripts/run_train_all.py —— 批量训练

**作用**：批量训练所有（或指定）数据集的所有版本，支持断点续传。

**用法**：
```bash
python3 scripts/run_train_all.py                          # 训练所有数据集所有版本
python3 scripts/run_train_all.py --datasets pamap2 uci_har  # 只训练指定数据集
python3 scripts/run_train_all.py --version v2             # 只训练 v2 版本（调试用）
python3 scripts/run_train_all.py --resume                  # 断点续传（跳过已完成）
python3 scripts/run_train_all.py --force                   # 强制重新训练（忽略已有结果）
python3 scripts/run_train_all.py --check-only              # 仅查看当前训练状态
```

**输出**：表格展示各数据集 `pure_cnn / v1 / v2 / v3` 完成状态和准确率。

---

### scripts/run_train.py —— 单数据集训练（封装）

**作用**：调用 `run_distill.py` 执行单数据集训练（与直接调用 `run_distill.py` 等效）。

**用法**：
```bash
python3 scripts/run_train.py <dataset> <version> [--resume]
```

---

### scripts/run_gen_soft.py —— 软标签生成（单数据集）

**作用**：调用 MiniMax API 为单个数据集生成软标签。

**用法**：
```bash
python3 scripts/run_gen_soft.py <dataset> [samples_per_class] [--force]
```

**参数说明**：
- `<dataset>`：同 run_distill.py
- `[samples_per_class]`：手动指定每类生成数量（覆盖默认逻辑）
- `--force`：强制从头开始（忽略已有进度）

**默认逻辑**：每类软标签数 = 该类训练样本数 × 35%，上限 400。

**示例**：
```bash
python3 scripts/run_gen_soft.py wisdm              # 自动计算每类软标签数
python3 scripts/run_gen_soft.py pamap2 200         # 强制每类200个
python3 scripts/run_gen_soft.py kuhar --force      # 强制从头开始
```

---

### scripts/run_gen_all.py —— 批量软标签生成

**作用**：并行启动所有数据集的软标签生成。

**用法**：
```bash
python3 scripts/run_gen_all.py                            # 后台运行，每30秒启动一个
python3 scripts/run_gen_all.py --sequential             # 前台顺序执行（调试用）
python3 scripts/run_gen_all.py --ratio 0.40             # 调整采样率为 40%
python3 scripts/run_gen_all.py --limit 500               # 调整每类上限为 500
python3 scripts/run_gen_all.py --datasets pamap2 uci_har  # 只生成指定数据集
```

**注意**：KuHar 使用专门的并行脚本 `gen_kuhar_parallel.py`（18类，样本量大）。

---

### scripts/gen_soft_labels_unified.py —— 核心软标签生成

**作用**：调用 MiniMax-M2.7 API 生成软标签（被 `run_gen_soft.py` 和 `run_gen_all.py` 调用）。

**用法**（通常不直接调用）：
```bash
python scripts/gen_soft_labels_unified.py <dataset> [--ratio 0.35] [--limit 400] [--force]
```

**软标签生成逻辑**：
1. 对每个样本窗口提取物理特征（均值、标准差、峰值、FFT能量、协方差等）
2. 组装 prompt 发送给 MiniMax-M2.7 API
3. 解析返回的类别概率向量，保存为 `.npy`
4. 过滤 one-hot（max > 0.95）和全零的样本

**输出**：`results/soft_labels/{dataset}_soft.npy`

---

### scripts/gen_kuhar_parallel.py —— KuHar 并行软标签生成

**作用**：KuHar 数据集专用并行生成（18 类，样本量大）。

**用法**：
```bash
python3 scripts/gen_kuhar_parallel.py start --ratio 0.35 --limit 400
python3 scripts/gen_kuhar_parallel.py status     # 查看生成状态
python3 scripts/gen_kuhar_parallel.py merge      # 合并所有进程 npy
```

---

### scripts/soft_label_progress.py —— 进度汇报

**作用**：检查所有数据集软标签生成和训练的详细进度。

```bash
python3 scripts/soft_label_progress.py
```

**输出**：各数据集软标签进度条、训练完成状态和准确率、正在运行的进程。

---

### scripts/api_test.py —— MiniMax API 测试

```bash
python3 scripts/api_test.py
```

**输出**：连续5次请求成功率、平均响应时间、最快/最慢响应。

---

## 📊 训练版本参数

| 版本 | Temperature T | Alpha α | 说明 |
|------|-------------|--------|------|
| `pure_cnn` | — | — | 纯 CNN 基线，focal loss，不使用软标签 |
| `v1` | 3.0 | 0.6 | 高温度 + 低 alpha，信任软标签更多 |
| `v2` | 2.5 | 0.8 | 中等温度，平衡蒸馏与硬标签 |
| `v3` | 1.5 | 0.85 | 低温度 + 高 alpha，更信任硬标签 |

**蒸馏损失函数**：
```
Loss = α × Focal + (1-α) × KL_div(softmax(logits/T), softmax(soft_labels/T)) × T²
```

**Early Stopping**：基于验证集准确率保存最优模型，最终报告测试集准确率。

---

## 🔑 典型工作流

**1. 生成单数据集软标签**：
```bash
python3 scripts/run_gen_soft.py pamap2
```

**2. 训练单数据集**：
```bash
python3 scripts/run_distill.py pamap2 v2
```

**3. 批量生成所有软标签（后台）**：
```bash
nohup python3 scripts/run_gen_all.py --ratio 0.35 --limit 400 > run_gen_all.log 2>&1 &
```

**4. 批量训练所有数据集所有版本**：
```bash
nohup python3 scripts/run_train_all.py --resume > train_all.log 2>&1 &
```

**5. 查看进度**：
```bash
python3 scripts/soft_label_progress.py
```

**6. 查看训练日志**：
```bash
tail -f results/logs/<dataset>_<version>_train.log
```

**7. 终止所有训练进程**：
```bash
./kill_all.sh
```

---

## 📁 目录结构

```
thesis/
│
├── 根目录启动脚本（后台运行包装）
│   ├── train.sh              # 批量训练（后台 nohup）
│   ├── train_single.sh       # 单数据集训练（后台）
│   ├── gen_soft.sh           # 批量软标签生成（后台）
│   └── kill_all.sh           # 终止所有训练/生成进程
│
├── scripts/                   # 核心脚本（所有Python脚本在此）
│   ├── run_distill.py         # 单数据集训练（核心）
│   ├── run_train_all.py       # 批量训练
│   ├── run_train.py           # 单数据集训练（封装）
│   ├── run_gen_soft.py        # 单数据集软标签生成
│   ├── run_gen_all.py         # 批量软标签生成
│   ├── gen_soft_labels_unified.py  # 核心软标签API调用
│   ├── gen_kuhar_parallel.py  # KuHar并行生成
│   ├── soft_label_progress.py # 进度汇报
│   └── api_test.py            # API测试
│
├── datasets/                  # 9个数据集原始数据
│
├── results/
│   ├── checkpoints/           # 模型检查点 .pt
│   ├── history/               # 训练历史、混淆矩阵、每类准确率
│   ├── logs/                 # 训练/生成日志
│   ├── soft_labels/           # 软标签 .npy 文件
│   └── *.json                # 各版本实验结果
│
└── docs/                     # 论文相关文档
```
