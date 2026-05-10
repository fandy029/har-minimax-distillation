#!/usr/bin/env python3
"""UCI-HAR 自包含准备"""
import os, numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR = THESIS_DIR
N_CLS = 6; CLASS_NAMES = ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING']

base = os.path.join(BASE_DIR,'datasets','UCI_HAR')
X_tr = np.loadtxt(os.path.join(base,'train','X_train.txt')).astype(np.float32)
y_tr = np.loadtxt(os.path.join(base,'train','y_train.txt')).astype(np.int64) - 1
X_te = np.loadtxt(os.path.join(base,'test','X_test.txt')).astype(np.float32)
y_te = np.loadtxt(os.path.join(base,'test','y_test.txt')).astype(np.int64) - 1
X = np.vstack([X_tr,X_te]); y = np.concatenate([y_tr,y_te])
X,X_te2,y,y_te2 = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)
X,X_vl,y,y_vl = train_test_split(X,y,test_size=0.1765,random_state=42,stratify=y)

out = os.path.join(SCRIPT_DIR,'output'); os.makedirs(out,exist_ok=True)
for c in range(N_CLS):
    cidx = np.where(y==c)[0]; dd = os.path.join(out,'per_class',f'class_{c}'); os.makedirs(dd,exist_ok=True)
    np.save(os.path.join(dd,'features.npy'),X[cidx]); np.save(os.path.join(dd,'indices.npy'),cidx)
    print(f"  class {c} {CLASS_NAMES[c]:<15s}: {len(cidx)}")
np.save(os.path.join(out,'train_labels.npy'),y)
print(f"Done: {len(y)}")
