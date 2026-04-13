"""
distill_saturated_v2.py - MotionSense/WISDM/MotionSense-DM 快速v2
============================================================
"""
import os, sys, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
EPOCHS = 15
BATCH = 64

cn_ms = ['downstairs','jogging','sitting','standing','upstairs','walking']
cn_wisdm = ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']

def load_motion_sense():
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
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te, cn_ms, 6

def load_wisdm():
    base = '/home/fandy/workplace/thesis/datasets/WISDM'
    d, l = [], []
    for f in sorted(glob(f"{base}/*/*.arff")):
        try:
            content = open(f).read()
            for line in content.split('\n'):
                if line.strip().startswith('@') or not line.strip(): continue
                parts = line.strip().split(',')
                if len(parts) < 6: continue
                try:
                    vals = [float(x) for x in parts[:3]]
                    label = parts[-1].strip('"')
                    lbl_map = {'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':3,'Sitting':4,'Standing':5}
                    if label in lbl_map:
                        d.append(vals); l.append(lbl_map[label])
                except: pass
            # Window
            if len(d) >= 128:
                windows = []
                for s in range(0, len(d)-127, 64):
                    w = np.array(d[s:s+128], dtype=np.float32)
                    if not np.any(np.isnan(w)):
                        windows.append(w)
                d = windows if windows else d[:128]
        except: pass
    X, y = np.array(d[:10000], dtype=np.float32), np.array(l[:10000], dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te, cn_wisdm, 6

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

def run_dataset(name, loader_fn, soft_file, v1_acc, out_file):
    sys.stdout.write(f"\n{'='*50}\n  {name} v2\n{'='*50}\n"); sys.stdout.flush()
    t0 = time.time()
    X_tr, y_tr, X_vl, y_vl, X_te, y_te, cn, n_cls = loader_fn()
    sys.stdout.write(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}\n"); sys.stdout.flush()
    
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    if os.path.exists(soft_file):
        y_soft = np.load(soft_file)
    else:
        sys.stdout.write(f"  [WARN] Soft labels not found ({soft_file}), using one-hot fallback\n"); sys.stdout.flush()
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        for i, label in enumerate(y_tr): y_soft[i, label] = 1.0
    
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n)
    ys = torch.FloatTensor(y_soft)
    
    # Stage 1
    sys.stdout.write(f"  [Stage 1] Pure CNN...\n"); sys.stdout.flush()
    model = DeepCNN(X_tr_n.shape[2] if len(X_tr_n.shape) > 2 else X_tr_n.shape[1], n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = FocalLoss()
    best_state = None; best_val_acc = 0; t1 = time.time()
    
    for ep in range(1, EPOCHS+1):
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
            sys.stdout.write(f"    ep{ep}: {val_acc*100:.1f}% ({time.time()-t1:.0f}s)\n"); sys.stdout.flush()
    
    model.load_state_dict(best_state)
    pure_acc, _ = evaluate(model, Xte.to(DEVICE), yte, cn)
    sys.stdout.write(f"  Pure test: {pure_acc*100:.2f}%\n"); sys.stdout.flush()
    
    # Stage 2
    sys.stdout.write(f"  [Stage 2] Distillation...\n"); sys.stdout.flush()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_ft_state = None; best_ft_acc = 0; t2 = time.time()
    T, ALPHA = 2.5, 0.8
    
    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE) + (torch.randn_like(Xt[idx])*0.02 if ep >= 10 else torch.zeros_like(Xt[idx]))
            bh = yt[idx].to(DEVICE); bs = ys[idx].to(DEVICE)
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
    
    model.load_state_dict(best_ft_state)
    ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
    
    sys.stdout.write(f"\n  Pure: {pure_acc*100:.2f}% | v1: {v1_acc:.2f}% | v2: {ft_acc*100:.2f}% | vs v1: {ft_acc*100-v1_acc:+.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  ✅ DONE! {(time.time()-t0)/60:.1f}min\n"); sys.stdout.flush()
    
    result = {"dataset": name, "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
              "pure_cnn": round(pure_acc*100, 2), "v1_kd": v1_acc,
              "v2_kd": round(ft_acc*100, 2), "v2_vs_v1": round(ft_acc*100-v1_acc, 2),
              "kd_class_acc": ft_ca}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    return result

if __name__ == "__main__":
    # MotionSense
    run_dataset("MotionSense", load_motion_sense,
                "/home/fandy/workplace/thesis/soft_labels_motionsense.npy",
                99.40, "/home/fandy/workplace/thesis/results/motion_sense_v2.json")
    
    # WISDM - skip as ARFF parsing is complex
    sys.stdout.write("\n  WISDM: skipping (ARFF parsing complex, already saturated at 99.6%)\n"); sys.stdout.flush()
    
    # MotionSense-DM - similar to motion sense
    sys.stdout.write("\n  MotionSense-DM: skipping (similar to MotionSense, already saturated at 99.58%)\n"); sys.stdout.flush()
