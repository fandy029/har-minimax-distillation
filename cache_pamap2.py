#!/usr/bin/env python
"""Cache PAMAP2 data to numpy format for faster loading"""
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

print("Caching PAMAP2 data...")
base = '/home/fandy/workplace/thesis/datasets/PAMAP2/PAMAP2_Dataset'
d, l = [], []
PAMAP_MAP = {9:0, 2:1, 3:2, 4:3, 5:4}
for folder in ['Protocol', 'Optional']:
    for f in sorted(glob(f"{base}/{folder}/*.dat")):
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
X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
print(f"Total data: X={X.shape}, y={y.shape}")
print("Saving cached data...")
np.savez('/home/fandy/workplace/thesis/cache/pamap2_data.npz', X=X, y=y)
print("Done!")
