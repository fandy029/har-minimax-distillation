#!/usr/bin/env python
"""Test data loading for PAMAP2"""
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

print("Starting test...")
base = '/home/fandy/workplace/thesis/datasets/PAMAP2/PAMAP2_Dataset'
d, l = [], []
PAMAP_MAP = {9:0, 2:1, 3:2, 4:3, 5:4}
for folder in ['Protocol', 'Optional']:
    for f in sorted(glob(f"{base}/{folder}/*.dat")):
        try:
            print(f"Reading {f}...")
            df = pd.read_csv(f, sep=' ', header=None).iloc[::2].reset_index(drop=True)
            imu = df.iloc[:,9:15].values.astype(np.float32)
            acts = df.iloc[:,1].values
            for aid, unlabel in PAMAP_MAP.items():
                mask = acts == aid
                idx = np.where(mask)[0]
                for s in range(0, len(idx)-127, 64):
                    w = imu[idx[s:s+128]]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(unlabel)
        except Exception as e:
            print(f"Error: {e}")
            pass
X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
print(f"X shape: {X.shape}, y shape: {y.shape}")
X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_tr)}, Val: {len(X_vl)}, Test: {len(X_te)}")
print("Done.")
