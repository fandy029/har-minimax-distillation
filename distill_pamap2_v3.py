"""
distill_pamap2_v3.py - PAMAP2 极致优化版
=====================================
"""
import os, sys, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cpu")
BATCH = 64
cn = ['downstairs','sitting','standing','walking','jogging']

def load():
    base = '/home/fandy/workplace/thesis/datasets/PAMAP2/PAMAP2_Dataset'
    d, l = [], []
    PAMAP_MAP = {9:0, 2:1, 3:2, 4:3, 5:4}
    for folder in ['Protocol','Optional']:
        for f in sorted(glob(f"{base}/{folder}/*.dat")):
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
                            d.append(w); l.append(unlabel)
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

class CNN(nn.Module):
    def __init__(self, c=6, n=5):
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

def focal_loss(logits, targets):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1-pt)**2.0 * ce).mean()

def evaluate(model, X, y, cn):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).cpu().numpy()
    yn = y.numpy() if hasattr(y, 'numpy') else y
    acc = float((preds == yn).mean())
    ca = {}
    for c in range(len(cn)):
        m = yn == c
        if m.sum() > 0: ca[cn[c]] = float((preds[m] == yn[m]).mean())
    return acc, ca

if __name__ == "__main__":
    t0 = time.time()
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load()
    print(f"\nPAMAP2 v3: Train={len(X_tr)} Val={len(X_vl)} Test={len(X_te)}")
    
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    # soft labels: one-hot fallback
    n_cls = 5
    y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    for i, label in enumerate(y_tr): y_soft[i, label] = 1.0
    
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n); ys = torch.FloatTensor(y_soft)
    
    # Stage 1
    print("\n[Stage 1] Pure CNN (40 epochs)...")
    model = CNN(6, 5).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_state = None; best_val = 0; t1 = time.time()
    
    for ep in range(1, 41):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE) + torch.randn_like(Xt[idx])*0.02
            bh = yt[idx].to(DEVICE)
            out = model(bx); loss = focal_loss(out, bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if va > best_val: best_val = va; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 10 == 0: print(f"  ep{ep:>3}: {va*100:.1f}% ({time.time()-t1:.0f}s)")
    
    model.load_state_dict(best_state)
    pure_acc, pure_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
    print(f"  Pure: {pure_acc*100:.2f}%")
    
    # Stage 2
    print("\n[Stage 2] Distillation (80 epochs, T=1.5, alpha=0.85)...")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    best_ft_state = None; best_ft_val = 0; t2 = time.time()
    T, ALPHA = 1.5, 0.85
    
    for ep in range(1, 81):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0; n_batches = 0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE) + (torch.randn_like(Xt[idx])*0.02 if ep >= 50 else torch.zeros_like(Xt[idx]))
            bh = yt[idx].to(DEVICE); bs = ys[idx].to(DEVICE)
            out = model(bx)
            ce = F.cross_entropy(out, bh, reduction='none'); pt = torch.exp(-ce)
            fl = ((1-pt)**2.0 * ce).mean()
            kl = F.kl_div(F.log_softmax(out/T, dim=1), F.softmax(bs/T, dim=1), reduction='batchmean') * (T**2)
            loss = ALPHA * fl + (1-ALPHA) * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        sch.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if va > best_ft_val: best_ft_val = va; best_ft_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
            print(f"  ep{ep:>3}: val={va*100:.1f}% best={best_ft_val*100:.1f}% | test={ft_acc*100:.1f}%")
    
    model.load_state_dict(best_ft_state)
    ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
    
    print(f"\n=== RESULTS ===")
    print(f"  Pure: {pure_acc*100:.2f}%")
    print(f"  v1:   93.10%")
    print(f"  v2:   95.02%")
    print(f"  v3:   {ft_acc*100:.2f}%")
    print(f"  vs v2: {ft_acc*100-95.02:+.2f}%")
    print(f"\n  Classes:")
    for c in range(5):
        name = cn[c]
        p = pure_ca.get(name, 0); k = ft_ca.get(name, 0)
        diff = k - p
        m = "✅" if diff > 0.02 else "❌" if diff < -0.02 else "  "
        print(f"  {m} {name:<15s}: Pure={p*100:5.1f}% v3={k*100:5.1f}% ({diff:+.1f}%)")
    
    result = {"dataset": "PAMAP2", "num_classes": 5, "train": len(X_tr), "test": len(X_te),
              "pure_cnn": round(pure_acc*100, 2), "v1_kd": 93.10, "v2_kd": 95.02,
              "v3_kd": round(ft_acc*100, 2), "v3_vs_v2": round(ft_acc*100-95.02, 2),
              "kd_class_acc": ft_ca}
    with open("/home/fandy/workplace/thesis/new_results_v2/pamap2_v3.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  DONE! {(time.time()-t0)/60:.1f}min")
