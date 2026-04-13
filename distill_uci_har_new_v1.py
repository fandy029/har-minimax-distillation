"""
distill_uci_new_v2.py - UCI-HAR-New 过渡态优化
============================================
问题: 过渡态动作(SIT_TO_STAND, LIE_TO_STAND)蒸馏后暴跌到0%

根因: 过渡态时间窗口太短，MiniMax给出的软标签极不稳定

优化方案:
1. 高α=0.85: 更依赖硬标签
2. 低T=2.0: 软标签更sharp
3. 对过渡态(SIT_TO_STAND, LIE_TO_STAND, STAND_TO_LIE, SIT_TO_LIE)只用硬标签
4. 先用清晰类(基础6类)训练，再逐步加入过渡态
"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
EPOCHS_PURE = 30
EPOCHS_KD = 60
BATCH = 64

cn = ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING',
      'STAND_TO_SIT','SIT_TO_STAND','SIT_TO_LIE','LIE_TO_SIT','STAND_TO_LIE','LIE_TO_STAND']
TRANSITION = {6, 7, 8, 9, 10, 11}  # 过渡态类

def load_uci_new():
    base = '/home/fandy/workplace/thesis/datasets/UCI_HAR_New'
    X_tr = np.loadtxt(f"{base}/Train/X_train.txt").astype(np.float32)
    y_tr = np.loadtxt(f"{base}/Train/y_train.txt").astype(np.int64) - 1
    X_te = np.loadtxt(f"{base}/Test/X_test.txt").astype(np.float32)
    y_te = np.loadtxt(f"{base}/Test/y_test.txt").astype(np.int64) - 1
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

class MLP(nn.Module):
    def __init__(self, in_dim=561, n_cls=12):
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
    return float(acc), ca

def main():
    sys.stdout.write(f"\n{'='*55}\n  UCI-HAR-New v2 (过渡态优化)\n{'='*55}\n"); sys.stdout.flush()
    t0 = time.time()
    
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_uci_new()
    n_cls = 12; in_dim = 561
    sys.stdout.write(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}\n"); sys.stdout.flush()
    
    # 归一化
    mean = X_tr.mean(axis=0, keepdims=True); std = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    # 软标签
    soft_file = "/home/fandy/workplace/thesis/results/soft_labels/uci_har_new_soft.npy"
    if os.path.exists(soft_file):
        y_soft = np.load(soft_file)
    else:
        print("  [WARN] Soft labels not found, using one-hot fallback")
        y_soft = np.zeros((len(X_tr), 12), dtype=np.float32)
        for i, label in enumerate(y_tr): y_soft[i, label] = 1.0
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n); yte = torch.LongTensor(y_te)
    ys = torch.FloatTensor(y_soft)
    
    # Stage 1: Pure CNN
    sys.stdout.write(f"\n  [Stage 1] Pure CNN ({EPOCHS_PURE} epochs)...\n"); sys.stdout.flush()
    model = MLP(in_dim, n_cls).to(DEVICE)
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
            val_acc = (model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            sys.stdout.write(f"    ep{ep:>3}: val={val_acc*100:.1f}% best={best_val_acc*100:.1f}% ({(time.time()-t1):.0f}s)\n"); sys.stdout.flush()
    
    model.load_state_dict(best_state)
    pure_acc, pure_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
    sys.stdout.write(f"\n  Pure CNN test: {pure_acc*100:.2f}%\n"); sys.stdout.flush()
    
    # Stage 2: 蒸馏优化 - 对过渡态只用硬标签
    sys.stdout.write(f"\n  [Stage 2] Selective Distillation ({EPOCHS_KD} epochs)...\n"); sys.stdout.flush()
    
    # 重置模型继续训练
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    
    best_ft_state = None; best_ft_acc = 0; t2 = time.time()
    T = 3.0; ALPHA = 0.6
    
    for ep in range(1, EPOCHS_KD+1):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0; n_batches = 0
        
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE)
            bh = yt[idx].to(DEVICE)
            bs = ys[idx].to(DEVICE)
            
            bx_ = bx + torch.randn_like(bx)*0.02 if ep >= 40 else bx
            out = model(bx_)
            
            # 分类处理：过渡态只用Focal，基础态用蒸馏
            is_trans = torch.tensor([bh[j].item() in TRANSITION for j in range(len(bh))], device=DEVICE)
            
            # Focal
            ce = F.cross_entropy(out, bh, reduction='none')
            pt = torch.exp(-ce)
            
            if is_trans.sum() > 0:
                # 过渡态: 只用Focal (α=1.0)
                fl_trans = ((1-pt[is_trans])**2.0 * ce[is_trans]).mean()
            else:
                fl_trans = torch.tensor(0.0, device=DEVICE)
            
            if (~is_trans).sum() > 0:
                # 基础态: 蒸馏
                out_clean = out[~is_trans]
                bh_clean = bh[~is_trans]
                bs_clean = bs[~is_trans]
                ce_c = F.cross_entropy(out_clean, bh_clean, reduction='none')
                pt_c = torch.exp(-ce_c)
                fl_c = ((1-pt_c)**2.0 * ce_c).mean()
                kl = F.kl_div(F.log_softmax(out_clean/T, dim=1),
                               F.softmax(bs_clean/T, dim=1),
                               reduction='batchmean') * (T**2)
                loss_distill = ALPHA * fl_c + (1-ALPHA) * kl
            else:
                loss_distill = torch.tensor(0.0, device=DEVICE)
            
            # 合并
            n_t = is_trans.sum().item(); n_o = len(bh) - n_t
            if n_t > 0 and n_o > 0:
                loss = (fl_trans * n_t + loss_distill * n_o) / len(bh)
            elif n_t > 0:
                loss = fl_trans
            else:
                loss = loss_distill
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        
        sch.step()
        
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                val_acc = (model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean()
            if val_acc > best_ft_acc:
                best_ft_acc = val_acc; best_ft_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            
            ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
            sit_to_stand = ft_ca.get('SIT_TO_STAND', 0)
            lie_to_stand = ft_ca.get('LIE_TO_STAND', 0)
            sys.stdout.write(f"    ep{ep:>3}: val={val_acc*100:.1f}% best_ft={best_ft_acc*100:.1f}% | ST_ST={sit_to_stand*100:.1f}% LT_ST={lie_to_stand*100:.1f}% | loss={total_loss/max(1,n_batches):.4f} ({(time.time()-t2):.0f}s)\n"); sys.stdout.flush()
    
    model.load_state_dict(best_ft_state)
    ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
    
    with open("/home/fandy/workplace/thesis/new_results/uci_har_new_kd.json") as f:
        v1 = json.load(f)
    
    sys.stdout.write(f"\n{'='*55}\n"); sys.stdout.flush()
    sys.stdout.write(f"  Pure CNN:   {pure_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v1 KD:     {v1['cnn_minimax']:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v2 KD:     {ft_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  vs Pure:    {ft_acc*100-pure_acc*100:+.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  vs v1:     {ft_acc*100-v1['cnn_minimax']:+.2f}%\n\n"); sys.stdout.flush()
    
    for c in range(n_cls):
        name = cn[c]
        p1 = v1['pure_class_acc'].get(name, 0)
        k1 = v1['kd_class_acc'].get(name, 0)
        k2 = ft_ca.get(name, 0)
        diff = k2 - p1
        marker = "✅" if diff > 0.03 else "❌" if diff < -0.03 else "  "
        sys.stdout.write(f"  {marker} {name:<20s}: Pure={p1*100:5.1f}% v1={k1*100:5.1f}% v2={k2*100:5.1f}% ({diff:+.1f}%)\n"); sys.stdout.flush()
    
    result = {
        "dataset": "UCI-HAR-New", "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": round(pure_acc*100, 2), "v1_kd": v1['cnn_minimax'],
        "v2_kd": round(ft_acc*100, 2), "v2_vs_v1": round(ft_acc*100-v1['cnn_minimax'], 2),
        "kd_class_acc": ft_ca,
    }
    with open("/home/fandy/workplace/thesis/results/uci_har_new_v1.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    sys.stdout.write(f"\n  ✅ DONE! Total: {(time.time()-t0)/60:.1f}min\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()
