"""
step2_distill_v2_quick.py - 快速测试版 v2蒸馏
==============================================
Stage 1: 20 epochs (原40)
Stage 2: 40 epochs (原80)
"""
import os, sys, json, time, re
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

API_KEY  = "sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc"
API_URL = "https://api.minimaxi.com/v1"
MODEL   = "MiniMax-M2.7-highspeed"
DEVICE  = torch.device("cpu")
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 40
BATCH   = 64
MAX_TRAIN = 20000
DS_DIR = "/home/fandy/workplace/thesis/datasets"
OUT_DIR = "/home/fandy/workplace/thesis/new_results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

T_HIGH = 4.0; T_LOW = 1.5; ALPHA = 0.75; WRONG_KD_WEIGHT = 0.15

DATASETS = {
    "kuhar": {
        "name": "KuHar", "path": f"{DS_DIR}/KuHar/1.Raw_time_domian_data",
        "num_classes": 18, "channels": 8, "model": "cnn",
        "cn": ["Stand","Sit","Talk-sit","Talk-stand","Stand-sit","Lay","Lay-stand","Pick","Jump","Push-up","Sit-up","Walk","Walk-backwards","Walk-circle","Run","Stair-up","Stair-down","Table-tennis"],
    },
}

def load_kuhar():
    base = DATASETS["kuhar"]["path"]
    d, l = [], []
    folders = sorted(glob(f"{base}/*/"))
    for folder in folders:
        label = int(os.path.basename(folder.rstrip("/")).split(".")[0])
        for f in glob(f"{folder}/*.csv"):
            try:
                df = pd.read_csv(f, header=None)
                data = df.values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0] == 128 and not np.any(np.isnan(w)):
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

class CNNAwareLoss(nn.Module):
    """CNN一致性蒸馏损失 - 简化版（固定T+高α）
    关键改进：alpha=0.75（原0.6），T=2.5（原3.0）
    课程学习在主循环通过掩码控制"""
    def __init__(self, T=2.5, alpha=0.75):
        super().__init__(); self.T = T; self.alpha = alpha
    
    def forward(self, logits, hard, soft, cnn_probs):
        # Focal: 向量化
        ce = F.cross_entropy(logits, hard, reduction='none')
        pt = torch.exp(-ce)
        fl = ((1-pt)**2.0 * ce).mean()
        # KL: 向量化
        kl = F.kl_div(F.log_softmax(logits/self.T, dim=1),
                      F.softmax(soft/self.T, dim=1),
                      reduction='batchmean') * (self.T**2)
        return self.alpha * fl + (1-self.alpha) * kl

def main():
    ds_key = "kuhar"
    cfg = DATASETS[ds_key]
    sys.stdout.write(f"\n{'='*55}\n  [{ds_key}] KuHar v2蒸馏快速版\n{'='*55}\n"); sys.stdout.flush()
    
    t0 = time.time()
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_kuhar()
    n_cls = cfg["num_classes"]; cn = cfg["cn"]; in_ch = cfg["channels"]
    sys.stdout.write(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}\n"); sys.stdout.flush()
    
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    y_soft = np.load("/home/fandy/workplace/thesis/new_results/kuhar_soft.npy")[:MAX_TRAIN]
    mm_pred = y_soft.argmax(axis=1)
    max_probs = y_soft.max(axis=1)
    sys.stdout.write(f"\n  Soft: max_prob>0.85: {(max_probs>0.85).sum()}, teacher_correct: {(mm_pred==y_tr).sum()} ({(mm_pred==y_tr).mean()*100:.1f}%)\n"); sys.stdout.flush()
    
    Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n); yv = torch.LongTensor(y_vl)
    ys = torch.FloatTensor(y_soft)
    
    model = DeepCNN(in_ch, n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = FocalLoss()
    
    # Stage 1: Pure CNN
    sys.stdout.write(f"\n  [Stage 1] Pure CNN ({EPOCHS_STAGE1} epochs)...\n"); sys.stdout.flush()
    t1 = time.time(); best_acc1 = 0; best_state1 = None
    
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
        model.eval()
        with torch.no_grad():
            acc = (model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean()
        if acc > best_acc1:
            best_acc1 = acc; best_state1 = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            sys.stdout.write(f"    ep{ep:>3}: {acc*100:.1f}% best={best_acc1*100:.1f}% ({(time.time()-t1):.0f}s)\n"); sys.stdout.flush()
    
    model.load_state_dict(best_state1)
    model.eval()
    with torch.no_grad():
        cnn_probs = F.softmax(model(Xt.to(DEVICE)), dim=1).cpu()
    cnn_pred = cnn_probs.argmax(dim=1)
    mm_pred_t = torch.from_numpy(y_soft.argmax(axis=1))
    agree = (cnn_pred == mm_pred_t).float()
    sys.stdout.write(f"  Pure CNN: {best_acc1*100:.2f}% | CNN-MM agree: {agree.mean()*100:.1f}%\n"); sys.stdout.flush()
    
    # Stage 2: CNN感知蒸馏
    sys.stdout.write(f"\n  [Stage 2] CNN-Aware Distillation ({EPOCHS_STAGE2} epochs)...\n"); sys.stdout.flush()
    
    # 重置模型
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None: nn.init.zeros_(layer.bias)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    distill_crit = CNNAwareLoss(T=2.5, alpha=0.75)
    
    best_acc2 = 0; best_state2 = None; t2 = time.time()
    milestones = [EPOCHS_STAGE2//2, EPOCHS_STAGE2//4*3]  # 20, 30
    
    for ep in range(1, EPOCHS_STAGE2+1):
        # 课程学习掩码
        if ep < milestones[0]:
            mask = agree >= 0.5  # 高一致性
        elif ep < milestones[1]:
            mask = cnn_probs.max(dim=1)[0] >= 0.4  # 中等
        else:
            mask = torch.ones(len(Xt), dtype=torch.bool)
        
        active = torch.where(mask)[0]
        if len(active) == 0: active = torch.arange(len(Xt))
        
        perm = torch.randperm(len(active))
        total_loss = 0.0; n_batches = 0
        
        for i in range(0, len(active), BATCH):
            idx = perm[i:i+BATCH]
            si = active[idx]
            bx = Xt[si].to(DEVICE)
            bh = yt[si].to(DEVICE)
            bs = ys[si].to(DEVICE)
            bc = cnn_probs[si].to(DEVICE)
            bx_ = bx + torch.randn_like(bx)*0.02 if ep >= 30 else bx
            out = model(bx_)
            loss = distill_crit(out, bh, bs, bc)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        
        sch.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean()
        if acc > best_acc2:
            best_acc2 = acc; best_state2 = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        
        if ep % 5 == 0 or ep == 1:
            n_active = mask.sum().item()
            sys.stdout.write(f"    ep{ep:>3}: {acc*100:.1f}% best={best_acc2*100:.1f}% | active={n_active} | loss={total_loss/max(1,n_batches):.4f} ({(time.time()-t2):.0f}s)\n"); sys.stdout.flush()
    
    # Final eval
    model.load_state_dict(best_state2)
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te_n).to(DEVICE)).argmax(1).cpu().numpy()
    test_acc = (preds == y_te).mean()
    ca = {}
    for c in range(n_cls):
        m = y_te == c
        if m.sum() > 0: ca[cn[c]] = float((preds[m] == y_te[m]).mean())
    
    # 对比v1
    with open("/home/fandy/workplace/thesis/new_results/kuhar_kd.json") as f:
        v1 = json.load(f)
    v1_kd = v1["cnn_minimax"]
    v1_pure = v1["pure_cnn"]
    
    sys.stdout.write(f"\n  ===== RESULTS =====\n"); sys.stdout.flush()
    sys.stdout.write(f"  Pure CNN:     {best_acc1*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v1 KD:       {v1_kd:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v2 KD:       {test_acc*100:.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v2 vs Pure:  {test_acc*100-best_acc1*100:+.2f}%\n"); sys.stdout.flush()
    sys.stdout.write(f"  v2 vs v1:    {test_acc*100-v1_kd:+.2f}%\n"); sys.stdout.flush()
    
    # 每个类的对比
    sys.stdout.write(f"\n  Class comparison (Pure vs v1 KD vs v2 KD):\n"); sys.stdout.flush()
    for c in range(n_cls):
        p1 = v1["pure_class_acc"].get(cn[c], 0)
        k1 = v1["kd_class_acc"].get(cn[c], 0)
        k2 = ca.get(cn[c], 0)
        diff1 = k2 - p1
        sys.stdout.write(f"  {c:2d}. {cn[c]:<20s}: Pure={p1*100:5.1f}% v1={k1*100:5.1f}% v2={k2*100:5.1f}% ({diff1:+.1f}%)\n"); sys.stdout.flush()
    
    # 保存
    result = {
        "dataset": "KuHar", "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": round(best_acc1*100, 2), "v1_kd": v1_kd, "v2_kd": round(test_acc*100, 2),
        "v2_vs_pure": round(test_acc*100-best_acc1*100, 2), "v2_vs_v1": round(test_acc*100-v1_kd, 2),
        "kd_class_acc": ca,
    }
    with open(f"{OUT_DIR}/kuhar_v2_quick.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    sys.stdout.write(f"\n  ✅ DONE! Total time: {(time.time()-t0)/60:.1f}min\n"); sys.stdout.flush()
    sys.stdout.write(f"  Saved: {OUT_DIR}/kuhar_v2_quick.json\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()
