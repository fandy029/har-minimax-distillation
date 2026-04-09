"""
step2_distill_ft.py - 从Pure CNN微调而非重置
============================================
核心改进: Stage 2从Stage 1最佳模型继续训练，而非随机初始化
这样可以保留Stage 1学到的Stand知识，同时用蒸馏微调
"""
import os, sys, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
EPOCHS_PURE = 25
EPOCHS_FT = 50
BATCH = 64
MAX_TRAIN = 20000

cn = ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle','Run','Stair-up','Stair-down','Table-tennis']

def load_kuhar():
    base = '/home/fandy/workplace/thesis/datasets/KuHar/1.Raw_time_domian_data'
    d, l = [], []
    for folder in sorted(glob(f"{base}/*/")):
        label = int(os.path.basename(folder.rstrip("/")).split(".")[0])
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
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if len(X_tr) > MAX_TRAIN:
        idx = np.random.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

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
    sys.stdout.write(f"\n{'='*55}\n  KuHar Fine-Tune (从Pure CNN继续)\n{'='*55}\n"); sys.stdout.flush()
    t0 = time.time()
    
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_kuhar()
    n_cls = 18; in_ch = 8
    sys.stdout.write(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}\n"); sys.stdout.flush()
    
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    y_soft = np.load("/home/fandy/workplace/thesis/new_results/kuhar_soft.npy")[:MAX_TRAIN]
    
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n)
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
            val_acc = (model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            sys.stdout.write(f"    ep{ep:>3}: val={val_acc*100:.1f}% best={best_val_acc*100:.1f}% ({(time.time()-t1):.0f}s)\n"); sys.stdout.flush()
    
    model.load_state_dict(best_state)
    pure_acc, pure_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
    torch.save(best_state, '/tmp/kuhar_pure_best.pt')
    sys.stdout.write(f"\n  Pure CNN test: {pure_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  Stand: {pure_ca.get('Stand',0)*100:.1f}% | Sit: {pure_ca.get('Sit',0)*100:.1f}%\n"); sys.stdout.flush()
    
    # 保存Pure CNN各类结果
    with open('/tmp/kuhar_pure_ft.json', 'w') as f:
        json.dump({'pure_cnn': round(pure_acc*100,2), 'class_acc': pure_ca}, f)
    
    # Stage 2: Fine-tune from best Pure CNN (不重置!)
    sys.stdout.write(f"\n  [Stage 2] Fine-tune from Pure CNN ({EPOCHS_FT} epochs)...\n"); sys.stdout.flush()
    sys.stdout.write(f"  NOT resetting model - continuing from best Pure CNN\n"); sys.stdout.flush()
    
    # 降低学习率微调
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 更低学习率
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    
    # CNN logits用于一致性判断
    model.eval()
    with torch.no_grad():
        cnn_logits = model(Xt.to(DEVICE)).cpu()
    cnn_probs = F.softmax(cnn_logits, dim=1)
    cnn_pred = cnn_probs.argmax(dim=1)
    mm_pred = torch.from_numpy(y_soft.argmax(axis=1))
    
    # 分析一致性
    agree = (cnn_pred == mm_pred)
    sys.stdout.write(f"\n  CNN-MM agreement: {agree.float().mean()*100:.1f}%\n"); sys.stdout.flush()
    for c in range(n_cls):
        m = yt == c
        if m.sum() > 0:
            a = agree[m].float().mean()
            sys.stdout.write(f"    {c:2d}. {cn[c]:<20s}: agree={a*100:.1f}%\n"); sys.stdout.flush()
    
    # 微调：蒸馏，但高一致性用强化，低一致性用软化
    T_HIGH = 4.0; T_LOW = 1.5; ALPHA = 0.8  # 高α减少蒸馏影响
    
    best_ft_state = None; best_ft_acc = 0; t2 = time.time()
    
    for ep in range(1, EPOCHS_FT+1):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0; n_batches = 0
        
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            bx = Xt[idx].to(DEVICE)
            bh = yt[idx].to(DEVICE)
            bs = ys[idx].to(DEVICE)
            bc = cnn_probs[idx].to(DEVICE)
            
            # 数据增强
            bx_ = bx + torch.randn_like(bx)*0.02 if ep >= 30 else bx
            
            out = model(bx_)
            
            # CNN一致性判断
            cnn_max, cnn_p = bc.max(dim=1)
            mm_p = bs.argmax(dim=1)
            agree_b = (cnn_p == mm_p)
            
            # Focal Loss
            ce = F.cross_entropy(out, bh, reduction='none')
            pt = torch.exp(-ce)
            fl = ((1-pt)**2.0 * ce).mean()
            
            # 对每个样本决定蒸馏强度
            # 高一致性+cnn高置信: T低强化
            # 低一致性: T高软化
            T_batch = torch.where(agree_b & (cnn_max > 0.7),
                        torch.full((len(bx),), T_LOW, device=DEVICE),
                        torch.full((len(bx),), T_HIGH, device=DEVICE))
            
            # KL散度(用组平均)
            n_agree = agree_b.sum().item()
            if n_agree > 0 and n_agree < len(bx):
                T_agree = T_LOW; T_dis = T_HIGH
                kl_agree = F.kl_div(F.log_softmax(out[agree_b]/T_agree, dim=1),
                                     F.softmax(bs[agree_b]/T_agree, dim=1),
                                     reduction='batchmean') * (T_agree**2)
                kl_dis = F.kl_div(F.log_softmax(out[~agree_b]/T_dis, dim=1),
                                   F.softmax(bs[~agree_b]/T_dis, dim=1),
                                   reduction='batchmean') * (T_dis**2)
                kl = ((kl_agree * n_agree) + (kl_dis * (len(bx)-n_agree))) / len(bx)
            else:
                T_avg = T_batch.mean().item()
                kl = F.kl_div(F.log_softmax(out/T_avg, dim=1),
                               F.softmax(bs/T_avg, dim=1),
                               reduction='batchmean') * (T_avg**2)
            
            loss = ALPHA * fl + (1-ALPHA) * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        
        sch.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean()
        
        if val_acc > best_ft_acc:
            best_ft_acc = val_acc; best_ft_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        
        if ep % 5 == 0 or ep == 1:
            ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
            stand = ft_ca.get('Stand', 0); sit = ft_ca.get('Sit', 0)
            sys.stdout.write(f"    ep{ep:>3}: val={val_acc*100:.1f}% best_ft={best_ft_acc*100:.1f}% | Stand={stand*100:.1f}% Sit={sit*100:.1f}% | loss={total_loss/max(1,n_batches):.4f} ({(time.time()-t2):.0f}s)\n"); sys.stdout.flush()
    
    # 最终评估 - 用最佳val模型
    model.load_state_dict(best_ft_state)
    ft_acc, ft_ca = evaluate(model, Xte.to(DEVICE), y_te, cn)
    
    with open("/home/fandy/workplace/thesis/new_results/kuhar_kd.json") as f:
        v1 = json.load(f)
    
    sys.stdout.write(f"\n{'='*55}\n"); sys.stdout.flush()
    sys.stdout.write(f"  Pure CNN:    {pure_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v1 KD:      {v1['cnn_minimax']:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  Fine-tune:  {ft_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  vs Pure:     {ft_acc*100-pure_acc*100:+.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  vs v1:      {ft_acc*100-v1['cnn_minimax']:+.2f}%\n\n"); sys.stdout.flush()
    
    for c in range(n_cls):
        p1 = v1['pure_class_acc'].get(cn[c], 0)
        k1 = v1['kd_class_acc'].get(cn[c], 0)
        kf = ft_ca.get(cn[c], 0)
        diff = kf - p1
        marker = "✅" if diff > 0.03 else "❌" if diff < -0.03 else "  "
        sys.stdout.write(f"  {marker} {c:2d}. {cn[c]:<20s}: Pure={p1*100:5.1f}% v1={k1*100:5.1f}% FT={kf*100:5.1f}% ({diff:+.1f}%)\n"); sys.stdout.flush()
    
    result = {
        "dataset": "KuHar", "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": round(pure_acc*100, 2), "v1_kd": v1['cnn_minimax'],
        "ft_kd": round(ft_acc*100, 2), "ft_vs_v1": round(ft_acc*100-v1['cnn_minimax'], 2),
        "kd_class_acc": ft_ca,
    }
    with open("/home/fandy/workplace/thesis/new_results_v2/kuhar_ft.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    sys.stdout.write(f"\n  ✅ DONE! Total: {(time.time()-t0)/60:.1f}min\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()
