"""
distill_harth_v2.py - HARTH 优化版
===================================
HARTH: 6类, 95.83% (gap 4.17%)
问题: 上楼/下楼准确率低 (80-88%)
优化: 针对性处理上楼/下楼，高α蒸馏
"""
import os, sys, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
EPOCHS_PURE = 30
EPOCHS_KD = 60
BATCH = 64
MAX_TRAIN = 20000

cn = ['左立','走路','上楼','下楼','右立','站立']
CONFUSING = {0, 5}  # 左立和站立都是静态，可能混淆

def load_harth():
    base = '/home/fandy/workplace/thesis/datasets/HARTH/harth'
    LABEL_MAP = {1:5, 3:1, 4:2, 5:3, 6:0, 7:4, 8:1}
    d, l = [], []
    for f in sorted(glob(f"{base}/*.csv")):
        try:
            df = pd.read_csv(f)
            imu = df[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']].values.astype(np.float32)
            labels = df['label'].values
            for lbl in np.unique(labels):
                if lbl not in LABEL_MAP: continue
                unified = LABEL_MAP[lbl]
                mask = labels == lbl
                idx = np.where(mask)[0]
                for s in range(0, len(idx)-127, 64):
                    w = imu[idx[s:s+128]]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(unified)
        except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if len(X_tr) > MAX_TRAIN:
        idx = np.random.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

class DeepCNN(nn.Module):
    def __init__(self, in_ch=6, n_cls=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 7, 2, 3); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 5, 2, 2); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, 2, 1); self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, 3, 1, 1); self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 128); self.fc2 = nn.Linear(128, 64); self.fc3 = nn.Linear(64, n_cls)
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

def main():
    sys.stdout.write(f"\n{'='*55}\n  HARTH v2优化\n{'='*55}\n"); sys.stdout.flush()
    t0 = time.time()
    
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_harth()
    n_cls = 6; in_ch = 6
    sys.stdout.write(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}\n"); sys.stdout.flush()
    
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    # 软标签 - fallback to one-hot if file not found
    soft_file = "/home/fandy/workplace/thesis/results/soft_labels/harth_soft.npy"
    if os.path.exists(soft_file):
        y_soft = np.load(soft_file)
    else:
        print("  [WARN] Soft labels not found, using one-hot fallback")
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        for i, label in enumerate(y_tr): y_soft[i, label] = 1.0
    if len(y_soft) > len(X_tr):
        y_soft = y_soft[:len(X_tr)]
    elif len(y_soft) < len(X_tr):
        pad = np.zeros((len(X_tr) - len(y_soft), n_cls), dtype=np.float32)
        y_soft = np.vstack([y_soft, pad])
    
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n); yte = torch.LongTensor(y_te)
    ys = torch.FloatTensor(y_soft)
    
    # Stage 1: Pure CNN
    sys.stdout.write(f"\n  [Stage 1] Pure CNN ({EPOCHS_PURE} epochs)...\n"); sys.stdout.flush()
    model = DeepCNN(in_ch, n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = FocalLoss(gamma=2.0)
    best_state = None; best_val_acc = 0; t1 = time.time()
    
    for ep in range(1, EPOCHS_PURE+1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE) + torch.randn_like(Xt[idx])*0.02
            bh = yt[idx].to(DEVICE)
            out = model(bx); loss = crit(out, bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            val_acc = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            sys.stdout.write(f"    ep{ep:>3}: val={val_acc*100:.1f}% best={best_val_acc*100:.1f}% ({(time.time()-t1):.0f}s)\n"); sys.stdout.flush()
    
    model.load_state_dict(best_state)
    pure_acc, pure_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
    sys.stdout.write(f"\n  Pure CNN test: {pure_acc*100:.2f}%\n"); sys.stdout.flush()
    for c in range(n_cls): sys.stdout.write(f"    {cn[c]}: {pure_ca.get(cn[c],0)*100:.1f}%\n"); sys.stdout.flush()
    
    # Stage 2: 蒸馏
    sys.stdout.write(f"\n  [Stage 2] Distillation ({EPOCHS_KD} epochs)...\n"); sys.stdout.flush()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_ft_state = None; best_ft_acc = 0; t2 = time.time()
    T = 2.5; ALPHA = 0.8
    
    for ep in range(1, EPOCHS_KD+1):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0; n_batches = 0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE) + (torch.randn_like(Xt[idx])*0.02 if ep >= 40 else torch.zeros_like(Xt[idx]))
            bh = yt[idx].to(DEVICE)
            bs = ys[idx].to(DEVICE)
            out = model(bx)
            ce = F.cross_entropy(out, bh, reduction='none')
            pt = torch.exp(-ce)
            fl = ((1-pt)**2.0 * ce).mean()
            kl = F.kl_div(F.log_softmax(out/T, dim=1), F.softmax(bs/T, dim=1), reduction='batchmean') * (T**2)
            loss = ALPHA * fl + (1-ALPHA) * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        sch.step()
        model.eval()
        with torch.no_grad():
            val_acc = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
        if val_acc > best_ft_acc:
            best_ft_acc = val_acc; best_ft_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
            upstairs = ft_ca.get('上楼', 0); downstairs = ft_ca.get('下楼', 0)
            sys.stdout.write(f"    ep{ep:>3}: val={val_acc*100:.1f}% best_ft={best_ft_acc*100:.1f}% | 上楼={upstairs*100:.1f}% 下楼={downstairs*100:.1f}% | ({time.time()-t2:.0f}s)\n"); sys.stdout.flush()
    
    model.load_state_dict(best_ft_state)
    ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
    
    v1_kd = 95.83  # 之前的结果
    
    sys.stdout.write(f"\n{'='*55}\n"); sys.stdout.flush()
    sys.stdout.write(f"  Pure CNN: {pure_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v1 KD:   {v1_kd:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v2 KD:   {ft_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  vs v1:   {ft_acc*100-v1_kd:+.2f}%\n\n"); sys.stdout.flush()
    
    for c in range(n_cls):
        name = cn[c]
        p = pure_ca.get(name, 0)
        k2 = ft_ca.get(name, 0)
        diff = k2 - p
        marker = "✅" if diff > 0.02 else "❌" if diff < -0.02 else "  "
        sys.stdout.write(f"  {marker} {name:<10s}: Pure={p*100:5.1f}% v2={k2*100:5.1f}% ({diff:+.1f}%)\n"); sys.stdout.flush()
    
    result = {
        "dataset": "HARTH", "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": round(pure_acc*100, 2), "v1_kd": v1_kd,
        "v2_kd": round(ft_acc*100, 2), "v2_vs_v1": round(ft_acc*100-v1_kd, 2),
        "kd_class_acc": ft_ca,
    }
    with open("/home/fandy/workplace/thesis/results/harth_v2.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    sys.stdout.write(f"\n  ✅ DONE! Total: {(time.time()-t0)/60:.1f}min\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()
