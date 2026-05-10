# KuHar 软标签生成

## 文件结构

```
kuhar/
├── README.md                    ← 本文件
├── DATASET_INFO.md              ← 数据集详细介绍 (18类特征、分布)
├── api_config.py                ← API 配置 (Mimo-v2.5-pro, T=0.8)
│
├── prepare_per_class_data.py    ← ① 预存每类数据 (一次性, 减少内存)
├── kuhar_gen_per_class.py       ← ② 按类并行生成 (加载预存数据)
├── run_kuhar_all.sh             ← ③ 一键启动/停止 18进程
├── merge_kuhar_soft.py          ← ④ 合并 → A/B/C三版
├── cleanup.sh                   ← 删除所有生成内容
│
├── kuhar_gen_v2.py              ← 旧版: 全类串行 (已废弃)
│
└── output/                      ← 所有输出
    ├── train_labels.npy              全局标签 (57k样本)
    ├── soft_labels/
    │   ├── kuhar_soft_all.npy           (A版: 全部软标签)
    │   ├── kuhar_soft_filtered.npy      (B版: 质量筛选)
    │   └── kuhar_soft_correct_only.npy  (C版: 仅正确)
    ├── logs/
    │   ├── all.log / filtered.log / correct.log
    │   └── stdout_class_*.log           (每个进程的标准输出)
    ├── checkpoints/              ← 断点续传 (每个类独立)
    └── per_class/                ← 18类中间产物 (永久保留)
        └── class_N/
            ├── windows.npy            预存的窗口数据 (~13MB)
            ├── indices.npy            全局索引映射
            ├── soft_all.npy           本类软标签
            ├── log_all.txt            本类全部日志
            ├── log_filtered.txt       本类通过筛选的日志
            └── log_correct.txt        本类预测正确的日志
```

## 快速开始

### ① 准备数据 (一次性, ~30秒)
```bash
python3 prepare_per_class_data.py
```
按类预存窗口数据到 `output/per_class/class_N/windows.npy`。
之后每个生成进程只加载自己的类 (~13MB), 18进程总计 ~250MB 内存。

### ② 测试 (每类50样本, ~10分钟)
```bash
bash run_kuhar_all.sh --quick
python3 merge_kuhar_soft.py --quick
```

### ③ 全量生成 (18进程并行, ~25小时)
```bash
bash run_kuhar_all.sh
# 等全部完成...
python3 merge_kuhar_soft.py
```

### 停止
```bash
bash run_kuhar_all.sh --stop
```

### 强制重新开始
```bash
# 先清理
bash cleanup.sh
# 再重新准备
python3 prepare_per_class_data.py
# 再启动
bash run_kuhar_all.sh
```

### 查看进度
```bash
# 某类进度
tail -f output/per_class/class_0/log_all.txt

# 所有类完成数
wc -l output/per_class/class_*/log_all.txt

# 所有进程状态
pgrep -a kuhar_gen_per_class
```

## API 配置

- 模型: **Mimo-v2.5-pro**
- 地址: `https://token-plan-cn.xiaomimimo.com/v1`
- 温度: **0.8** (从 `api_config.py` 读取, 0.8 为最优)
- 所有生成脚本通过 `import api_config` 读取配置

## 输出说明

| 版本 | 文件 | 说明 | 软标签比例 | 用途 |
|------|------|------|-----------|------|
| A | `kuhar_soft_all.npy` | 全部软标签 | 100% | 蒸馏主实验 |
| B | `kuhar_soft_filtered.npy` | 质量筛选 | ~87% | 去除噪声的蒸馏对照 |
| C | `kuhar_soft_correct_only.npy` | 仅正确 | ~45% | 蒸馏效果上界 |

筛选规则 (B版): 满足全部条件保留, 否则 → one-hot
- `entropy < 1.5` (不过于均匀)
- `gap > 0.05` (有区分度)
- `confidence > 0.5` (有一定置信度)

## 资源占用

| 模式 | 内存 | 说明 |
|------|------|------|
| 有预存数据 | ~250MB (18进程总计) | 每进程 ~13MB |
| 无预存数据 | ~4.5GB (18进程总计) | 每进程加载全量 235MB |

## 断点续传

每个类独立 checkpoint (`output/checkpoints/ckpt_class_N.json`)。
中断后重新运行 `bash run_kuhar_all.sh` 即可从断点继续。
用 `--force` 参数可清除断点重来。
