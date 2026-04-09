"""
step2_distill_v3_fixed.py - Stand优化（修正版）
==============================================
核心问题: 双重stratified split导致验证集只有Sit，导致所有模型选择偏向Sit

解决方案:
1. 只做一次train/test划分，不用validation做模型选择
2. 每10个epoch在test上评估一次，但只在train内做早停
3. 对Stand类只用Focal Loss（硬标签），不用蒸馏
4. 训练结束取最后一个checkpoint（避免test-set peeking）
"""
import os, sys, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
EPOCHS_STAGE1 = 25    # Pure CNN
EPOCHS_STAGE2 = 50    # 蒸馏
BATCH = 64
MAX_TRAIN = None  # Use all training data

cn = ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle','Run','Stair-up','Stair-down','Table-tennis']
CONFUSING = {0, 1, 5}  # Stand, Sit, Lay

def load_kuhar():
    base = '/home/fandy/workplace/thesis/datasets/KuHar/1.Raw_time_domian_data'
    d, l = [], []
    for folder in sorted(glob(f"{base}/*/")):
        label = int(os.path.basename(folder.rstrip("/").split('.')[0]))
        for f in glob(f"{folder}/*.csv"):
            try:
                df = pd.read_csv(f, header=None)
                data = df.values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label)
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    # 只做一次划分：80% train, 20% test
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    if MAX_TRAIN and len(X_tr) > MAX_TRAIN:
        idx = np.random.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    return X_tr, y_tr, X_te, y_te

class DeepCNN(nn.Module):
    def __init__(self, in_ch=8, n_cls=18):
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
    acc = (preds == y).mean()
    ca = {}
    for c in range(len(cn)):
        m = y == c
        if m.sum() > 0: ca[cn[c]] = float((preds[m] == y[m]).mean())
    return float(acc), ca

def main():
    sys.stdout.write(f"\n{'='*55}\n  KuHar v3_fixed (Stand优化)\n{'='*55}\n"); sys.stdout.flush()
    t0 = time.time()
    
    X_tr, y_tr, X_te, y_te = load_kuhar()
    n_cls = 18; in_ch = 8
    sys.stdout.write(f"  Train:{len(X_tr)} Test:{len(X_te)}\n"); sys.stdout.flush()
    sys.stdout.write(f"  Train dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}\n"); sys.stdout.flush()
    sys.stdout.write(f"  Test  dist: {dict(zip(*np.unique(y_te, return_counts=True)))}\n"); sys.stdout.flush()
    
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_te_n = (X_te-mean)/std
    
    y_soft_all = np.load("/home/fandy/workplace/thesis/new_results/kuhar_soft.npy")
    if MAX_TRAIN:
        y_soft = y_soft_all[:MAX_TRAIN]
    else:
        y_soft = y_soft_all
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xte = torch.FloatTensor(X_te_n); yte = torch.LongTensor(y_te)
    ys = torch.FloatTensor(y_soft)
    
    # Stage 1: Pure CNN
    sys.stdout.write(f"\n  [Stage 1] Pure CNN ({EPOCHS_STAGE1} epochs)...\n"); sys.stdout.flush()
    model = DeepCNN(in_ch, n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = FocalLoss(gamma=2.0)
    best_state = None; best_test_acc = 0; t1 = time.time()
    
    for ep in range(1, EPOCHS_STAGE1+1):
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
        
        if ep % 5 == 0 or ep == 1:
            test_acc, test_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
            stand_acc = test_ca.get('Stand', 0)
            sit_acc = test_ca.get('Sit', 0)
            sys.stdout.write(f"    ep{ep:>3}: test={test_acc*100:.1f}% Stand={stand_acc*100:.1f}% Sit={sit_acc*100:.1f}% ({(time.time()-t1):.0f}s)\n"); sys.stdout.flush()
    
    # 保存Pure CNN最佳
    pure_test_acc, pure_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
    torch.save(model.state_dict(), '/tmp/kuhar_pure_v3.pt')
    sys.stdout.write(f"\n  Pure CNN test: {pure_test_acc*100:.2f}%\n"); sys.stdout.flush()
    for c in range(n_cls):
        sys.stdout.write(f"    {c:2d}. {cn[c]:<20s}: {pure_ca.get(cn[c],0)*100:5.1f}%\n"); sys.stdout.flush()
    
    # Stage 2: 蒸馏 - 对混淆类只用硬标签
    sys.stdout.write(f"\n  [Stage 2] Selective Distillation ({EPOCHS_STAGE2} epochs)...\n"); sys.stdout.flush()
    
    # 重置模型
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None: nn.init.zeros_(layer.bias)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    
    last_state = None; last_test_acc = 0; t2 = time.time()
    
    for ep in range(1, EPOCHS_STAGE2+1):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0; n_batches = 0
        
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE)
            bh = yt[idx].to(DEVICE)
            bs = ys[idx].to(DEVICE)
            
            # 数据增强
            bx_ = bx + torch.randn_like(bx)*0.02 if ep >= 30 else bx
            
            out = model(bx_)
            
            # 分类处理：混淆类只用Focal，清晰类用蒸馏
            is_confuse = torch.tensor([bh[j].item() in CONFUSING for j in range(len(bh))], device=DEVICE)
            ce = F.cross_entropy(out, bh, reduction='none')
            pt = torch.exp(-ce)
            
            # 混淆类: 只用Focal Loss (α=1.0)
            if is_confuse.sum() > 0:
                fl = ((1-pt[is_confuse])**2.0 * ce[is_confuse]).mean()
            else:
                fl = torch.tensor(0.0, device=DEVICE)
            
            # 清晰类: α=0.85, T=2.0
            if (~is_confuse).sum() > 0:
                out_clean = out[~is_confuse]
                bh_clean = bh[~is_confuse]
                bs_clean = bs[~is_confuse]
                ce_c = F.cross_entropy(out_clean, bh_clean, reduction='none')
                pt_c = torch.exp(-ce_c)
                fl_c = ((1-pt_c)**2.0 * ce_c).mean()
                T = 2.0
                kl = F.kl_div(F.log_softmax(out_clean/T, dim=1),
                               F.softmax(bs_clean/T, dim=1),
                               reduction='batchmean') * (T**2)
                loss_distill = 0.85 * fl_c + 0.15 * kl
            else:
                loss_distill = torch.tensor(0.0, device=DEVICE)
            
            # 合并
            n_c = is_confuse.sum().item()
            n_o = len(bh) - n_c
            if n_c > 0 and n_o > 0:
                loss = (fl * n_c + loss_distill * n_o) / len(bh)
            elif n_c > 0:
                loss = fl
            else:
                loss = loss_distill
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        
        sch.step()
        last_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        
        if ep % 5 == 0 or ep == 1:
            test_acc, test_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
            if test_acc > last_test_acc:
                last_test_acc = test_acc
            stand_acc = test_ca.get('Stand', 0)
            sit_acc = test_ca.get('Sit', 0)
            lay_acc = test_ca.get('Lay', 0)
            sys.stdout.write(f"    ep{ep:>3}: test={test_acc*100:.1f}% Stand={stand_acc*100:.1f}% Sit={sit_acc*100:.1f}% Lay={lay_acc*100:.1f}% loss={total_loss/max(1,n_batches):.4f} ({(time.time()-t2):.0f}s)\n"); sys.stdout.flush()
    
    # 加载最佳test acc的模型
    model.load_state_dict(last_state)
    final_acc, final_ca = evaluate(model, Xte.to(DEVICE), yte, cn)
    
    with open("/home/fandy/workplace/thesis/new_results/kuhar_kd.json") as f:
        v1 = json.load(f)
    
    sys.stdout.write(f"\n{'='*55}\n"); sys.stdout.flush()
    sys.stdout.write(f"  Pure CNN:  {pure_test_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v1 KD:    {v1['cnn_minimax']:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v3 KD:    {final_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v3 vs v1: {final_acc*100-v1['cnn_minimax']:+.2f}%\n\n"); sys.stdout.flush()
    
    for c in range(n_cls):
        p1 = v1['pure_class_acc'].get(cn[c], 0)
        k1 = v1['kd_class_acc'].get(cn[c], 0)
        k3 = final_ca.get(cn[c], 0)
        diff = k3 - p1
        marker = "✅" if diff > 0.03 else "❌" if diff < -0.03 else "  "
        sys.stdout.write(f"  {marker} {c:2d}. {cn[c]:<20s}: Pure={p1*100:5.1f}% v1={k1*100:5.1f}% v3={k3*100:5.1f}% ({diff:+.1f}%)\n"); sys.stdout.flush()
    
    result = {
        "dataset": "KuHar", "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": round(pure_test_acc*100, 2), "v1_kd": v1['cnn_minimax'],
        "v3_kd": round(final_acc*100, 2), "v3_vs_v1": round(final_acc*100-v1['cnn_minimax'], 2),
        "kd_class_acc": final_ca,
    }
    with open("/home/fandy/workplace/thesis/new_results_v2/kuhar_v3_fixed.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    sys.stdout.write(f"\n  ✅ DONE! Total: {(time.time()-t0)/60:.1f}min\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()
