#!/usr/bin/env python3
"""
预存 KuHar 每类窗口数据 → 各生成进程只需加载自己的类
运行一次即可: python3 prepare_per_class_data.py
"""
import os, sys, numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
# fixed
BASE_DIR   = THESIS_DIR

CLASS_NAMES = ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand',
    'Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle',
    'Run','Stair-up','Stair-down','Table-tennis']
N_CLS = len(CLASS_NAMES)

PER_CLASS_DIR = os.path.join(SCRIPT_DIR, 'output', 'per_class')
os.makedirs(PER_CLASS_DIR, exist_ok=True)

base = os.path.join(BASE_DIR, 'datasets', 'KuHar', '1.Raw_time_domian_data')

# 1. 加载全部
print("加载全部 KuHar 数据...")
X_all, y_all = [], []
for folder in sorted(glob(os.path.join(base, '*'))):
    if not os.path.isdir(folder): continue
    cid = int(os.path.basename(folder).split('.')[0])
    if cid < 0 or cid >= N_CLS: continue
    for f in sorted(glob(os.path.join(folder, '*.csv'))):
        try:
            df = pd.read_csv(f, header=None); data = df.values.astype(np.float32)
            for s in range(0, len(data) - 127, 64):
                w = data[s:s+128]
                if w.shape[0] == 128 and not np.any(np.isnan(w)):
                    X_all.append(w); y_all.append(cid)
        except: continue
X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int64)
print(f"  全量: {len(X_all)} 窗口")

# 2. 划分 train
X, X_te, y, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.1765, random_state=42, stratify=y)
print(f"  训练集: {len(X)}, 验证集: {len(X_vl)}, 测试集: {len(X_te)}")

# 3. 每类存一个文件 (索引映射 + 窗口数据)
idx_map = np.zeros(len(X_all), dtype=np.int32) - 1  # global→train 索引映射
for i, gidx in enumerate(X_all.tolist() if False else []):
    pass  # can't easily map

# 简化: 直接按类存储train集的窗口
for c in range(N_CLS):
    cidx = np.where(y == c)[0]
    out_path = os.path.join(PER_CLASS_DIR, f'class_{c}', 'windows.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, X[cidx])
    np.save(os.path.join(PER_CLASS_DIR, f'class_{c}', 'indices.npy'), cidx)
    print(f"  class {c} ({CLASS_NAMES[c]:<16s}): {len(cidx)} 窗口 → class_{c}/windows.npy ({X[cidx].nbytes/1024/1024:.1f}MB)")

# 4. 存全局 train 索引映射
np.save(os.path.join(SCRIPT_DIR, 'output', 'train_labels.npy'), y)
print(f"\n全局标签: output/train_labels.npy ({len(y)} 样本)")
print("准备完成!")
