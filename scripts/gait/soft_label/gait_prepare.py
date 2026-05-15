#!/usr/bin/env python3
"""预存 Gait 每类窗口数据"""
import os, sys, numpy as np, pandas as pd, glob
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
BASE_DIR = THESIS_DIR  # datasets/ 根目录

CLASS_NAMES = ['sit_on_bed','sit_on_chair','lying','ambulating']
LABEL_MAP = {1:0, 2:1, 3:2, 4:3}
N_CLS = 4
PER_CLASS_DIR = os.path.join(GAIT_DIR, 'output', 'per_class')

base = os.path.join(BASE_DIR, 'datasets', 'Gait_Classification')
os.makedirs(PER_CLASS_DIR, exist_ok=True)

print("加载 Gait 数据...")
X_all, y_all = [], []
for folder in ['S1_Dataset', 'S2_Dataset']:
    for f in glob.glob(os.path.join(base, folder, '*')):
        if f.endswith('.txt') or 'README' in f: continue
        try:
            df = pd.read_csv(f, header=None)
            acc = df.iloc[:, 1:4].values.astype(np.float32)
            labels = df.iloc[:, 8].astype(int)
            for i in range(0, len(df) - 127, 64):
                w = acc[i:i+128]
                label = int(labels.iloc[i])
                if w.shape[0] == 128 and not np.any(np.isnan(w)) and label in LABEL_MAP:
                    X_all.append(w); y_all.append(LABEL_MAP[label])
        except: continue
X_all = np.array(X_all, dtype=np.float32); y_all = np.array(y_all, dtype=np.int64)
print(f"  全量: {len(X_all)} 窗口")

X, X_te, y, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.1765, random_state=42, stratify=y)
print(f"  train={len(X)}, val={len(X_vl)}, test={len(X_te)}")

for c in range(N_CLS):
    cidx = np.where(y == c)[0]
    d = os.path.join(PER_CLASS_DIR, f'class_{c}')
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'windows.npy'), X[cidx])
    np.save(os.path.join(d, 'indices.npy'), cidx)
    print(f"  class {c} {CLASS_NAMES[c]:<15s}: {len(cidx)} 窗口 ({X[cidx].nbytes/1024:.0f}KB)")

np.save(os.path.join(GAIT_DIR, 'output', 'train_labels.npy'), y)
print(f"完成!")
