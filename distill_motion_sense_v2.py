"""
distill_motion_sense_v2.py
"""
import os, sys, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cpu")
cn = ['downstairs','jogging','sitting','standing','upstairs','walking']

def load():
    base = '/home/fandy/workplace/thesis/datasets/MotionSense'
    d, l = [], []
    folders = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
    for fd, label in folders.items():
        for f in sorted(glob(f"{base}/{fd}*/*.csv")):
            try:
                df = pd.read_csv(f)
                data = df[['userAcceleration.x','userAcceleration.y','userAcceleration.z','gravity.x','gravity.y','gravity.z']].values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label)
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

class CNN(nn.Module):
    def __init__(self, c=6, n=6):
        super().__init__()
        self.conv1 = nn.Conv1d(c, 64, 7, 2, 3); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 5, 2, 2); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, 2, 1); self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, 3, 1, 1); self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 128); self.fc2 = nn.Linear(128, 64); self.fc3 = nn.Linear(64, n)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(F.relu(self.fc1(x))); x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

def fl(logits, targets):
    ce = F.cross_entropy(logits, targets, reduction='none'); pt = torch.exp(-ce)
    return ((1-pt)**2.0 * ce).mean()

def eval_model(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).cpu().numpy()
    y_np = y.numpy() if hasattr(y, 'numpy') else y
    acc = float((preds == y_np).mean())
    ca = {}
    for c in range(len(cn)):
        m = y_np == c
        if m.sum() > 0: ca[cn[c]] = float((preds[m] == y_np[m]).mean())
    return acc, ca

if __name__ == "__main__":
    print("\n=== MotionSense v2 ===")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load()
    print(f"Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}")
    
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    y_soft = np.load("/home/fandy/workplace/thesis/soft_labels_motionsense.npy")
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n)
    ys = torch.FloatTensor(y_soft)
    
    # Stage 1
    print("Stage 1 Pure CNN (15 epochs)...")
    model = CNN(6, 6).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_state = None; best_val = 0; t0 = time.time()
    
    for ep in range(1, 16):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]
            bx = Xt[idx].to(DEVICE) + torch.randn_like(Xt[idx])*0.02
            bh = yt[idx].to(DEVICE)
            out = model(bx); loss = fl(out, bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if va > best_val: best_val = va; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1: print(f"  ep{ep}: {va*100:.1f}%")
    
    model.load_state_dict(best_state)
    pure_acc, _ = eval_model(model, Xte.to(DEVICE), yte)
    print(f"  Pure test: {pure_acc*100:.2f}%")
    
    # Stage 2
    print("Stage 2 Distillation (15 epochs)...")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_ft_state = None; best_ft_val = 0; t2 = time.time()
    T, ALPHA = 2.5, 0.8
    
    for ep in range(1, 16):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]
            bx = Xt[idx].to(DEVICE) + (torch.randn_like(Xt[idx])*0.02 if ep >= 8 else torch.zeros_like(Xt[idx]))
            bh = yt[idx].to(DEVICE); bs = ys[idx].to(DEVICE)
            out = model(bx)
            ce = F.cross_entropy(out, bh, reduction='none'); pt = torch.exp(-ce)
            focal = ((1-pt)**2.0 * ce).mean()
            kl = F.kl_div(F.log_softmax(out/T, dim=1), F.softmax(bs/T, dim=1), reduction='batchmean') * (T**2)
            loss = ALPHA * focal + (1-ALPHA) * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if va > best_ft_val: best_ft_val = va; best_ft_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            ft_acc, _ = eval_model(model, Xte.to(DEVICE), yte)
            print(f"  ep{ep}: val={va*100:.1f}% test={ft_acc*100:.1f}%")
    
    model.load_state_dict(best_ft_state)
    ft_acc, ft_ca = eval_model(model, Xte.to(DEVICE), yte)
    
    v1 = 99.40
    print(f"\nPure: {pure_acc*100:.2f}% | v1: {v1:.2f}% | v2: {ft_acc*100:.2f}% | vs v1: {ft_acc*100-v1:+.2f}%")
    
    result = {"dataset": "MotionSense", "num_classes": 6, "train": len(X_tr), "test": len(X_te),
              "pure_cnn": round(pure_acc*100, 2), "v1_kd": v1,
              "v2_kd": round(ft_acc*100, 2), "v2_vs_v1": round(ft_acc*100-v1, 2),
              "kd_class_acc": ft_ca}
    with open("/home/fandy/workplace/thesis/new_results_v2/motion_sense_v2.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Done!")
