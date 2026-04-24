# 基于MiniMax大模型知识蒸馏的人体活动识别

利用 MiniMax-M2.7 大模型作为教师模型，基于 IMU 传感器窗口的物理特征生成软标签（Soft Labels），通过知识蒸馏提升 CNN 学生模型的 HAR 分类准确率。

---

## 📁 数据集

| 数据集 | 路径 | 类别数 | 训练样本 | 形状 |
|--------|------|--------|---------|------|
| PAMAP2 | `datasets/PAMAP2/` | 5 | ~2137 | (128, 6) |
| KuHar | `datasets/KuHar/1.Raw_time_domian_data/` | 18 | 20000 | (128, 8) |
| UCI-HAR | `datasets/UCI_HAR/` | 6 | ~7352 | (561, 1) |
| HARTH | `datasets/HARTH/harth/` | 6 | 20000 | (128, 3) |
| UCI-HAR-New | `datasets/UCI_HAR_New/` | 12 | ~6213 | (561, 1) |
| MotionSense | `datasets/MotionSense/` | 6 | ~17492 | (128, 3) |
| MotionSense-DM | `datasets/MotionSense_DeviceMotion/` | 6 | ~13777 | (128, 3) |
| WISDM | `datasets/WISDM/` | 6 | ~13365 | (128, 3) |
| Gait | `datasets/Gait_Classification/` | 4 | ~588 | (128, 6) |

---

## 🔧 脚本说明

本项目脚本分为两级：**根目录启动器**（交互式参数校验 + 调用） + **`scripts/` 核心脚本**（实际执行）。

---

### run_train.py —— 训练启动器（交互式）

**作用**：训练参数校验 + 交互确认，然后调用 `scripts/run_distill.py` 执行实际训练。

**用法**：
```bash
python run_train.py <dataset> <version> [--resume]
```

**参数说明**：
- `<dataset>`：数据集名称
  - `pamap2`, `kuhar`, `uci_har`, `harth`, `uci_har_new`, `motionsense`, `motionsense_dm`, `gait`, `wisdm`
- `<version>`：训练版本
  - `pure_cnn` — 纯 CNN 基线（不使用软标签）
  - `v1` — 蒸馏 v1（T=3.0, alpha=0.6）
  - `v2` — 蒸馏 v2（T=2.5, alpha=0.8）
  - `v3` — 蒸馏 v3（T=1.5, alpha=0.85）
- `--resume`：从断点继续训练

**示例**：
```bash
python run_train.py pamap2 v2              # 训练 PAMAP2 蒸馏 v2
python run_train.py wisdm pure_cnn          # 训练 WISDM 纯CNN基线
python run_train.py kuhar v3 --resume      # 断点续传
```

**工作流程**：
1. 校验数据集和版本参数
2. 检查软标签文件是否存在
3. 显示完整配置信息（交互确认）
4. 调用 `scripts/run_distill.py` 执行训练
5. 日志输出到 `/tmp/train_{dataset}_{version}.log`

---

### run_train_all.py —— 批量训练脚本

**作用**：批量训练所有数据集的所有版本，支持断点续传和状态检查。

**用法**：
```bash
python3 run_train_all.py                          # 训练所有数据集所有版本
python3 run_train_all.py --datasets pamap2 uci_har  # 只训练指定数据集
python3 run_train_all.py --version v2             # 只训练 v2 版本（调试用）
python3 run_train_all.py --resume                  # 断点续传（跳过已完成）
python3 run_train_all.py --force                    # 强制重新训练（忽略已有结果）
python3 run_train_all.py --check-only               # 仅查看当前训练状态，不实际训练
```

**输出内容**：
- 表格展示各数据集 `pure_cnn / v1 / v2 / v3` 的完成状态和准确率
- 标注正在生成软标签的数据集（跳过）
- 每轮训练后显示测试准确率

---

### run_gen_soft.py —— 软标签生成启动器（交互式）

**作用**：生成指定数据集的软标签，参数校验后调用 `scripts/gen_soft_labels_unified.py`。

**用法**：
```bash
python run_gen_soft.py <dataset> [samples_per_class] [--force]
```

**参数说明**：
- `<dataset>`：同 run_train.py
- `[samples_per_class]`：手动指定每类生成数量（覆盖默认逻辑，慎用）
- `--force`：强制从头开始（忽略已有进度）

**默认逻辑**：每类软标签数量 = 该类训练样本数 × 35%，上限 400。

**示例**：
```bash
python run_gen_soft.py wisdm              # 自动计算每类软标签数
python run_gen_soft.py pamap2 200         # 强制每类200个
python run_gen_soft.py kuhar --force       # 强制从头开始
```

---

### run_gen_all.py —— 批量软标签生成脚本

**作用**：并行启动所有 9 个数据集的软标签生成（后台守护进程模式）。

**用法**：
```bash
python3 run_gen_all.py                            # 后台运行（默认），每30秒启动一个
python3 run_gen_all.py --sequential              # 前台运行，顺序执行（调试用）
python3 run_gen_all.py --ratio 0.40              # 调整采样率为 40%
python3 run_gen_all.py --limit 500                # 调整每类上限为 500
python3 run_gen_all.py --datasets pamap2 uci_har  # 只生成指定数据集
```

**注意事项**：
- 默认每 30 秒启动一个新数据集，避免 API 限流
- KuHar 使用专门的并行脚本 `scripts/gen_kuhar_parallel.py`
- 后台运行时日志写入 `run_gen_all.log`

---

### soft_label_progress.py —— 进度汇报脚本

**作用**：检查所有数据集软标签生成 + 训练的详细进度。

**用法**：
```bash
python soft_label_progress.py
```

**输出内容**：
- 各数据集软标签：真软标签数 / one-hot 数 / 未填充数 / 进度条 / 最新日志
- 各数据集训练进度：pure_cnn / v1 / v2 / v3 完成状态 + 准确率
- 正在运行的软标签生成进程
- 正在运行的训练进程

---

### api_test.py —— MiniMax API 测试

**作用**：测试 MiniMax API 连通性和响应速度。

**用法**：
```bash
python api_test.py
```

**输出**：连续 5 次请求的成功率、平均响应时间、最快/最慢响应。

---

### scripts/run_distill.py —— 核心蒸馏训练脚本

**作用**：实际执行模型训练（被 `run_train.py` 和 `run_train_all.py` 调用）。

**用法**（通常不直接调用）：
```bash
python scripts/run_distill.py <dataset> <version> [--resume]
```

**训练流程**：
1. **Stage 1（纯CNN）**：所有版本共用 focal loss 训练 300 epochs，保存 `*_pure_cnn_best.pt`
2. **Stage 2（蒸馏）**：v1/v2/v3 在 Stage 1 权重基础上用软标签继续训练
   - 损失函数：`ALPHA × Focal + (1-ALPHA) × KL_div`
   - `pure_cnn` 模式跳过 Stage 2

**输出文件**：
- Checkpoint: `results/checkpoints/{dataset}_{version}_best.pt`
- 训练日志: `results/logs/{dataset}_{version}_train.log`
- 历史记录: `results/history/{dataset}_{version}_history.json`
- 结果 JSON: `results/{dataset}_{version}.json`

---

### scripts/gen_soft_labels_unified.py —— 核心软标签生成脚本

**作用**：调用 MiniMax API 为数据集生成软标签（被 `run_gen_soft.py` 和 `run_gen_all.py` 调用）。

**用法**（通常不直接调用）：
```bash
python scripts/gen_soft_labels_unified.py <dataset> [--ratio 0.35] [--limit 400] [--force]
```

**软标签生成逻辑**：
- 对每个样本窗口提取物理特征（均值、标准差、峰值、FFT 能量等）
- 组装 prompt 发送给 MiniMax-M2.7 API
- 解析返回的类别概率向量，保存为 `.npy`

**输出文件**：`results/soft_labels/{dataset}_soft.npy`

**过滤规则**：one-hot（max > 0.95）或全零的样本被过滤，只保留真实软标签。

---

### scripts/gen_kuhar_parallel.py —— KuHar 并行软标签生成

**作用**：KuHar 数据集专用并行生成脚本（18 类，样本量大）。

**用法**：
```bash
python scripts/gen_kuhar_parallel.py start --ratio 0.35 --limit 400
python scripts/gen_kuhar_parallel.py status     # 查看生成状态
python scripts/gen_kuhar_parallel.py merge      # 合并所有 proc 的 npy
```

**特点**：多进程并行生成，最终合并为 `kuhar_soft.npy`。

---

## 📊 训练版本参数

| 版本 | Temperature T | Alpha α | 说明 |
|------|-------------|--------|------|
| `pure_cnn` | — | — | 纯 CNN 基线，不使用软标签 |
| `v1` | 3.0 | 0.6 | 高温度 + 低 alpha，信任软标签更多 |
| `v2` | 2.5 | 0.8 | 中等温度 |
| `v3` | 1.5 | 0.85 | 低温度 + 高 alpha，更信任硬标签 |

蒸馏损失函数：
```
Loss = α × Focal + (1-α) × KL_div(softmax(logits/T), softmax(soft_labels/T)) × T²
```

---

## 🔑 典型工作流

**1. 生成软标签（单数据集）**：
```bash
python run_gen_soft.py pamap2
```

**2. 训练单数据集**：
```bash
python run_train.py pamap2 v2
```

**3. 批量生成所有软标签**：
```bash
python3 run_gen_all.py --ratio 0.35 --limit 400
```

**4. 批量训练所有数据集所有版本**：
```bash
python3 run_train_all.py --resume
```

**5. 查看当前进度**：
```bash
python soft_label_progress.py
```

---

## 📁 目录结构

```
thesis/
├── run_train.py              # 训练启动器（交互式）
├── run_train_all.py          # 批量训练
├── run_gen_soft.py           # 软标签生成启动器（交互式）
├── run_gen_all.py            # 批量软标签生成
├── soft_label_progress.py    # 进度汇报
├── api_test.py               # API 测试
│
├── scripts/
│   ├── run_distill.py        # 核心训练脚本
│   ├── gen_soft_labels_unified.py  # 核心软标签生成脚本
│   └── gen_kuhar_parallel.py      # KuHar 并行生成
│
├── datasets/                 # 9 个数据集
├── results/
│   ├── checkpoints/           # 模型检查点
│   ├── history/              # 训练历史 JSON
│   ├── logs/                 # 训练/生成日志
│   ├── soft_labels/          # 软标签 .npy 文件
│   └── *.json                # 各版本实验结果
└── docs/                     # 论文文档
```
