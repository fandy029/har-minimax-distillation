#!/usr/bin/env python
"""Cache PAMAP2 dataset to numpy format for faster loading"""
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

print("Starting PAMAP2 caching...")
base = '/home/fandy/workplace/thesis/datasets/PAMAP2/PAMAP2_Dataset'
d, l = [], []
PAMAP_MAP = {9:0, 2:1, 3:2, 4:3, 5:4}

total_files = len(list(glob(f"{base}/Protocol/*.dat"))) + len(list(glob(f"{base}/Optional/*.dat")))
processed = 0

for folder in ['Protocol', 'Optional']:
    for f in sorted(glob(f"{base}/{folder}/*.dat")):
        print(f"Processing {f} ({processed + 1}/{total_files})...")
        try:
            df = pd.read_csv(f, sep=' ', header=None).iloc[::2].reset_index(drop=True)
            imu = df.iloc[:,9:15].values.astype(np.float32)
            acts = df.iloc[:,1].values
            for aid, unlabel in PAMAP_MAP.items():
                mask = acts == aid
                idx = np.where(mask)[0]
                for s in range(0, len(idx)-127, 64):
                    w = imu[idx[s:s+128]]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w)
                        l.append(unlabel)
        except Exception as e:
            print(f"  Error processing {f}: {e}")
        processed += 1

X = np.array(d, dtype=np.float32)
y = np.array(l, dtype=np.int64)
print(f"\nTotal data created: X={X.shape}, y={y.shape}")

print("Creating train/val/test split...")
X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Saving cached data...")
np.savez('/home/fandy/workplace/thesis/cache/pamap2_full.npz', 
         X_tr=X_tr, y_tr=y_tr, X_vl=X_vl, y_vl=y_vl, X_te=X_te, y_te=y_te)

print(f"\n✅ Caching complete!")
print(f"  Train: {len(X_tr)} samples")
print(f"  Val: {len(X_vl)} samples")
print(f"  Test: {len(X_te)} samples")
