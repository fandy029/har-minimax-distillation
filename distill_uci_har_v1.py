"""
distill_uci_har_v1.py - UCI-HAR 标准蒸馏 v1
标准蒸馏: T=3.0, α=0.6
"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cpu")
EPOCHS_PURE = 25
EPOCHS_KD = 50
BATCH = 64
T = 3.0  # v1标准蒸馏
ALPHA = 0.6  # v1标准蒸馏

cn = ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING']

def load():
    base = '/home/fandy/workplace/thesis/datasets/UCI_HAR/UCI HAR Dataset'
    X_tr = np.loadtxt(f"{base}/train/X_train.txt").astype(np.float32)
    y_tr = (np.loadtxt(f"{base}/train/y_train.txt")-1).astype(np.int64)
    X_te = np.loadtxt(f"{base}/test/X_test.txt").astype(np.float32)
    y_te = (np.loadtxt(f"{base}/test/y_test.txt")-1).astype(np.int64)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

class MLP(nn.Module):
    def __init__(self, in_dim=561, n_cls=6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128); self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64); self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, n_cls)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none'); pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

def evaluate(model, X, y, cn):
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
    t0 = time.time()
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load()
    print(f"\nUCI-HAR v1: Train={len(X_tr)} Val={len(X_vl)} Test={len(X_te)}")
    
    mean = X_tr.mean(axis=0, keepdims=True); std = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    # One-hot soft labels
    n_cls = 6
    soft_file = "/home/fandy/workplace/thesis/results/soft_labels/uci_har_soft.npy"
    if os.path.exists(soft_file):
        y_soft = np.load(soft_file)
        if len(y_soft) > len(X_tr): y_soft = y_soft[:len(X_tr)]
        elif len(y_soft) < len(X_tr):
            pad = np.zeros((len(X_tr) - len(y_soft), n_cls), dtype=np.float32)
            y_soft = np.vstack([y_soft, pad])
    else:
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        for i, label in enumerate(y_tr): y_soft[i, label] = 1.0
        print(f"  [WARN] Soft labels not found, using one-hot fallback")
    
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n); ys = torch.FloatTensor(y_soft)
    
    # Stage 1: Pure CNN
    print(f"\n[Stage 1] Pure CNN ({EPOCHS_PURE} epochs)...")
    model = MLP(561, 6).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)
    crit = FocalLoss(gamma=2.0)
    best_state = None; best_val_acc = 0; t1 = time.time()
    
    for ep in range(1, EPOCHS_PURE+1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE); bh = yt[idx].to(DEVICE)
            out = model(bx); loss = crit(out, bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            val_acc = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0: print(f"  ep{ep:>3}: val={val_acc*100:.1f}%")
    
    model.load_state_dict(best_state)
    pure_acc, pure_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
    print(f"  Pure CNN: {pure_acc*100:.2f}%")
    
    # Stage 2: v1标准蒸馏
    print(f"\n[Stage 2] v1 Standard KD ({EPOCHS_KD} epochs, T={T}, alpha={ALPHA})...")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_ft_state = None; best_ft_acc = 0; t2 = time.time()
    
    for ep in range(1, EPOCHS_KD+1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE); bh = yt[idx].to(DEVICE); bs = ys[idx].to(DEVICE)
            out = model(bx)
            ce = F.cross_entropy(out, bh, reduction='none'); pt = torch.exp(-ce)
            fl = ((1-pt)**2.0 * ce).mean()
            kl = F.kl_div(F.log_softmax(out/T, dim=1), F.softmax(bs/T, dim=1), reduction='batchmean') * (T**2)
            loss = ALPHA * fl + (1-ALPHA) * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            val_acc = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if val_acc > best_ft_acc:
            best_ft_acc = val_acc; best_ft_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 10 == 0:
            ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
            print(f"  ep{ep:>3}: val={val_acc*100:.1f}% best_ft={best_ft_acc*100:.1f}%")
    
    model.load_state_dict(best_ft_state)
    ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
    
    print(f"\n=== RESULTS ===")
    print(f"  Pure CNN: {pure_acc*100:.2f}%")
    print(f"  v1 KD:   {ft_acc*100:.2f}%")
    
    result = {"dataset": "UCI-HAR", "num_classes": 6, "train": len(X_tr), "test": len(X_te),
              "pure_cnn": round(pure_acc*100, 2), "v1_kd": round(ft_acc*100, 2),
              "kd_class_acc": ft_ca}
    with open("/home/fandy/workplace/thesis/results/uci_har_v1.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  DONE! {(time.time()-t0)/60:.1f}min")
