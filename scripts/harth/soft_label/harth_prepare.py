#!/usr/bin/env python3
"""HARTH 自包含准备"""
import os, sys, numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, ".."))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
# fixed
BASE_DIR = THESIS_DIR  # datasets/
N_CLS = 8; CLASS_NAMES = ['walk','run','shuffle','stairs_up','stairs_down','stand','sit','lying']
label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}  # CSV integer labels

base = os.path.join(BASE_DIR,'datasets','HARTH')
d,l = [],[]
for f in sorted(glob(os.path.join(base,'**','*.csv'), recursive=True)):
    try:
        df = pd.read_csv(f)
        # some CSVs have extra Unnamed col; use column names
        sensor_cols = ['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
        sensor = df[sensor_cols].values.astype(np.float32)
        labels_raw = df['label'].astype(int)
        for s in range(0,len(df)-127,64):
            w = sensor[s:s+128]
            if w.shape[0]==128 and not np.any(np.isnan(w)):
                label = int(labels_raw.iloc[s])
                if label in label_map:
                    d.append(w); l.append(label_map[label])
    except: continue

X = np.array(d,dtype=np.float32); y = np.array(l,dtype=np.int64)
X,X_te,y,y_te = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)
X,X_vl,y,y_vl = train_test_split(X,y,test_size=0.1765,random_state=42,stratify=y)

out = os.path.join(GAIT_DIR,'output'); os.makedirs(out,exist_ok=True)
for c in range(N_CLS):
    cidx = np.where(y==c)[0]; dd = os.path.join(out,'per_class',f'class_{c}'); os.makedirs(dd,exist_ok=True)
    np.save(os.path.join(dd,'windows.npy'),X[cidx]); np.save(os.path.join(dd,'indices.npy'),cidx)
    print(f"  class {c} {CLASS_NAMES[c]:<12s}: {len(cidx)}")
np.save(os.path.join(out,'train_labels.npy'),y)
print(f"Done: {len(y)}")
