#!/usr/bin/env python3
"""MotionSense 自包含准备"""
import os, sys, numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR = THESIS_DIR
N_CLS = 6; CLASS_NAMES = ['downstairs','jogging','sitting','standing','upstairs','walking']

dir_map = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
base = os.path.join(BASE_DIR,'datasets','MotionSense')

d,l = [],[]
for folder in sorted(glob(os.path.join(base,'*'))):
    if not os.path.isdir(folder): continue
    name = os.path.basename(folder)
    prefix = name.split('_')[0]
    if prefix not in dir_map: continue
    cid = dir_map[prefix]
    for f in sorted(glob(os.path.join(folder,'*.csv'))):
        try:
            df = pd.read_csv(f)
            acc = df[['x','y','z']].values.astype(np.float32)
            for s in range(0,len(acc)-127,64):
                w = acc[s:s+128]
                if w.shape[0]==128 and not np.any(np.isnan(w)):
                    d.append(w); l.append(cid)
        except: continue

X = np.array(d,dtype=np.float32); y = np.array(l,dtype=np.int64)
X,X_te,y,y_te = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)
X,X_vl,y,y_vl = train_test_split(X,y,test_size=0.1765,random_state=42,stratify=y)

out = os.path.join(SCRIPT_DIR,'output'); os.makedirs(out,exist_ok=True)
for c in range(N_CLS):
    cidx = np.where(y==c)[0]; dd = os.path.join(out,'per_class',f'class_{c}'); os.makedirs(dd,exist_ok=True)
    np.save(os.path.join(dd,'windows.npy'),X[cidx]); np.save(os.path.join(dd,'indices.npy'),cidx)
    print(f"  class {c} {CLASS_NAMES[c]:<12s}: {len(cidx)}")
np.save(os.path.join(out,'train_labels.npy'),y)
print(f"Done: {len(y)}")
