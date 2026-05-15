#!/usr/bin/env python3
"""PAMAP2 准备 — 直接加载 .dat 文件"""
import os, numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
BASE_DIR = THESIS_DIR  # datasets/ 根目录
N_CLS=5; TARGET_IDS = {1,2,3,4,5}  # lying=1,sitting=2,standing=3,walking=4,jogging=5
ID_MAP = {1:0,2:1,3:2,4:3,5:4}
CLASS_NAMES = ['lying','sitting','standing','walking','jogging']

base = os.path.join(BASE_DIR,'datasets','PAMAP2','PAMAP2_Dataset')
d,l = [],[]
for folder in ['Protocol','Optional']:
    for f in sorted(glob(os.path.join(base,folder,'*.dat'))):
        try:
            data = np.loadtxt(f)
            if data.ndim != 2: continue
            # col1=activityID, col9-14=IMU(acc3+gyro3)
            ids = data[:,1].astype(int)
            imu = data[:,9:15].astype(np.float32)
            for s in range(0,len(imu)-127,64):
                w = imu[s:s+128]
                aid = ids[s]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and aid in TARGET_IDS:
                    d.append(w); l.append(ID_MAP[aid])
        except Exception as e: continue

X = np.array(d,dtype=np.float32); y = np.array(l,dtype=np.int64)
print(f"Loaded: {len(X)} windows")
X,X_te,y,y_te = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)
X,X_vl,y,y_vl = train_test_split(X,y,test_size=0.1765,random_state=42,stratify=y)

out = os.path.join(GAIT_DIR,'output'); os.makedirs(out,exist_ok=True)
for c in range(N_CLS):
    cidx = np.where(y==c)[0]; dd = os.path.join(out,'per_class',f'class_{c}'); os.makedirs(dd,exist_ok=True)
    np.save(os.path.join(dd,'windows.npy'),X[cidx]); np.save(os.path.join(dd,'indices.npy'),cidx)
    print(f"  class {c} {CLASS_NAMES[c]:<10s}: {len(cidx)}")
np.save(os.path.join(out,'train_labels.npy'),y)
print(f"Done: {len(y)}")
