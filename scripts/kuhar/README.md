# KuHar 软标签生成

## 📝 数据集概览

KuHar (Kansas University Human Activity Recognition) 数据集，包含 18 类日常活动，来自 90 位参与者（75 男 15 女），使用手机加速度计 + 陀螺仪采集，采样率 100Hz。

**⚠️ 重要：数据集中加速度计的重力分量已被去除（body acceleration only），因此 z_grav 特征无意义。**

## 📊 数据统计

### 原始文件

| 类 ID | 类名 | 原始文件数 | 可提取窗口数 | 每文件平均窗口 |
|:-----:|------|:----------:|:----------:|:------------:|
| 0 | Stand | 91 | 8,653 | 95 |
| 1 | Sit | 90 | 8,587 | 95 |
| 2 | Talk-sit | 86 | 8,285 | 96 |
| 3 | Talk-stand | 88 | 8,593 | 97 |
| 4 | Stand-sit | 339 | 9,803 | 28 |
| 5 | Lay | 87 | 8,430 | 96 |
| 6 | Lay-stand | 148 | 8,011 | 54 |
| 7 | Pick | 105 | 6,108 | 58 |
| 8 | Jump | 130 | 3,171 | 24 |
| 9 | Push-up | 111 | 2,315 | 20 |
| 10 | Sit-up | 121 | 4,781 | 39 |
| 11 | Walk | 188 | 3,880 | 20 |
| 12 | Walk-backwards | 50 | 1,408 | 28 |
| 13 | Walk-circle | 35 | 1,168 | 33 |
| 14 | Run | 146 | 2,738 | 18 |
| 15 | Stair-up | 53 | 3,868 | 72 |
| 16 | Stair-down | 57 | 3,765 | 66 |
| 17 | Table-tennis | 20 | 2,078 | 103 |
| | **总计** | **1,945** | **95,642** | |

### 训练/验证/测试划分

80% / 10% / 10%，分层采样确保各类比例一致。

### 输入形状

每样本：**(128, 8)** = 128 时间步 × 8 通道
- col 0: Acc 时间戳 (ms，不用)
- col 1-3: Acc X/Y/Z (m/s²)
- col 4: Gyro 时间戳 (ms，不用)
- col 5-7: Gyro X/Y/Z (rad/s)

滑动窗口：步长 64，窗口 128 步（1.28 秒 @ 100Hz）。

---

## 🎯 软标签生成方法

### 生成脚本

`kuhar_gen.py`

### 配置

- **API**: Mimo v2.5 Pro (Mimo API)
- **Temperature**: 0.3（保持低确定性，不做随机探索）
- **MAX_TOKENS**: 10000（给推理文字留足空间）
- **每类采样上限**: 3000 样本

### Prompt 设计（v84）

**核心策略**：能量分层 + 强制推理 + 禁止模板值

1. **能量分层**：将 18 类按加速度能量分为 4 组：
   - **A 组 — 静态** (energy<0.1): Stand, Sit, Lay, Talk-sit
   - **B 组 — 低-中** (0.1-12): Talk-stand, Stand-sit, Lay-stand, Pick, Push-up, Sit-up
   - **C 组 — 中** (10-80): Walk, Walk-back, Walk-circle, Stair-up, Stair-down, Table-tennis
   - **D 组 — 高** (>80): Jump, Run
2. **3 步推理**：先定能量组 → 组内比较特征 → 下结论
3. **2 个具体示例**：一个静态活动、一个中等活动
4. **禁止 4 个模板值模式**：0.556/0.152、0.529/0.144、0.514/0.280、0.660/0.142

### 选择使用的特征（来自 compute_features）

| 特征 | 作用 |
|------|------|
| energy | 主强度区分器（0.002~258，跨 5 个量级）|
| jerk | 运动平滑度 |
| n_peaks | 静态 vs 动态 |
| gyro_mag | 旋转强度（Push-up 最低 0.51）|
| gyro_x | Walk-circle 唯一定位特征（|gyro_x|>0.15）|
| impulsiveness | Jump(>3.0) vs Run(<3.0) |
| dom_freq | 步频分析 |
| zcr_acc | zero-crossing rate |
| acc_auto1 | 周期性 |

**未使用的特征**：z_grav（重力已去除，无效）

---

## 💡 软标签训练参考

### 输出文件

| 文件 | 内容 |
|------|------|
| `results/soft_labels/kuhar_soft.npy` | 全量软标签 (95642, 18)，未采样样本为全零 |
| `results/soft_labels/kuhar_soft_correct_only.npy` | 仅 LLM 预测正确的软标签 |
| `results/logs/gen_kuhar.log` | 主日志，含每样本预测详情 |
| `results/logs/gen_kuhar_errors.log` | 失败/异常日志 |
| `results/logs/gen_kuhar_correct.log` | 仅正确预测日志 |
| `results/logs/gen_kuhar_checkpoint.json` | 断点续传文件 |

### 训练代码编写参考

#### 数据加载

```python
X, y = load_kuhar_data()  # 同 gen.py 的 load_kuhar_data
soft = np.load('results/soft_labels/kuhar_soft.npy')
# soft.shape = (95642, 18), soft[i] 是第 i 个训练样本的软标签
```

#### 软标签格式

- 每行是一个 18 维概率分布，和为 1
- 未生成的样本为全零向量（需要过滤）
- 软标签质量：尽力避免 one-hot（max<0.95 才保留），但部分置信度过高样本可能接近 one-hot

#### 训练时需要注意

1. **只选择 soft.sum(axis=1) > 0 的样本训练**（未生成的保持零向量）
2. **蒸馏损失函数**：推荐 KL 散度（软标签 vs CNN softmax）或 MSE
3. **CNN 架构**：与软标签生成脚本中的滑动窗口参数一致（128 步，步长 64）
4. **采样平衡**：每类采样上限 3000，但 Sit 原始样本 29,017 远多于 Jump 的 1,903，训练时建议做类平衡

#### 已知局限

- **18 类上 LLM 准确率约 40-55%**，因此软标签质量有限
- **静态类（Stand/Sit/Lay）** 区分度极低（重力已去除），软标签在这些类上倾向于均匀分布
- **Walk-circle 的 gyro_x 方向**：94% 为正、6% 为负，模型在负方向样本上可能误判
- **软标签的用途**是在蒸馏中提供"不确定性"信息，而非替代真实标签。蒸馏时建议 soft_label_loss_weight < 1.0

---

## 🔧 运行方法

```bash
# 首次运行（断点续传）
python scripts/kuhar/kuhar_gen.py

# 重新生成（清空旧数据）
python scripts/kuhar/kuhar_gen.py --force
```

运行时可以通过查看日志监控进度：
```bash
tail -f results/logs/gen_kuhar.log
```
