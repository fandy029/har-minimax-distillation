#!/usr/bin/env python3
"""PAMAP2 PureCNN 蒸馏参数速搜 — 高α + 低T"""
import os, sys, json, copy, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
SCRIPTS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))
from models import PureCNN

DATASET = 'pamap2'
CLASS_NAMES = ['lying', 'sitting', 'standing', 'walking', 'jogging']
N_CLS = 5; TARGET_IDS = {1,2,3,4,5}; ID_MAP = {1:0,2:1,3:2,4:3,5:4}
SEED = 42; DEVICE = torch.device('cpu')
OUT_DIR = os.path.join(GAIT_DIR, 'output')
RESULTS_DIR = os.path.join(OUT_DIR, 'results')
SOFT_DIR = os.path.join(OUT_DIR, 'soft_labels')
CLIP_GRAD = 5.0; PATIENCE = 15; BS = 64; MAX_EP = 300

class FocalLoss(torch.nn.Module):
    def __init__(self, g=2.0): super().__init__(); self.g = g
    def forward(self, l, t):
        ce = F.cross_entropy(l, t, reduction='none')
        return ((1 - torch.exp(-ce))**self.g * ce).mean()

def distill_loss(l, s, y, a, T):
    focal = FocalLoss(2.0)(l, y)
    logp = F.log_softmax(l/T, dim=1)
    kl = (s.clamp(1e-8,1) * (s.clamp(1e-8,1).log() - logp)).sum(1).mean()
    return a*focal + (1-a)*T*T*kl

@torch.no_grad()
def evaluate(model, X, y):
    model.eval(); dl = DataLoader(TensorDataset(X, y), batch_size=256)
    correct, total = 0, 0
    for xb, yb in dl: correct += (model(xb.to(DEVICE)).argmax(1)==yb).sum().item(); total += len(yb)
    return correct/total

def train_epoch(model, loader, opt, soft, a, T):
    model.train(); total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, sb, yb in loader:
        xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
        loss = distill_loss(model(xb), sb, yb, a, T)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        opt.step()
        acc = float((model(xb).argmax(1)==yb).float().mean())
        bs = len(xb); total_loss += loss.item()*bs; total_acc += acc*bs; n += bs
    return total_loss/n, total_acc/n

# ---- 加载数据 ----
base = os.path.join(THESIS_DIR, 'datasets', 'PAMAP2', 'PAMAP2_Dataset')
X_all, y_all = [], []
for folder in ['Protocol', 'Optional']:
    for f in sorted(glob(os.path.join(base, folder, '*.dat'))):
        try:
            data = np.loadtxt(f)
            if data.ndim != 2: continue
            ids = data[:,1].astype(int); imu = data[:,9:15].astype(np.float32)
            for s in range(0, len(imu)-127, 64):
                w = imu[s:s+128]; aid = ids[s]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and aid in TARGET_IDS:
                    X_all.append(w); y_all.append(ID_MAP[aid])
        except: continue
X_all = np.array(X_all, dtype=np.float32); y_all = np.array(y_all, dtype=np.int64)
X, X_te, y, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.1765, random_state=SEED, stratify=y)
mean = X.mean(axis=(0,1), keepdims=True); std = X.std(axis=(0,1), keepdims=True)+1e-6
X = (X-mean)/std; X_vl = (X_vl-mean)/std; X_te = (X_te-mean)/std
X_t = torch.FloatTensor(X).permute(0,2,1); X_v = torch.FloatTensor(X_vl).permute(0,2,1)
X_te_t = torch.FloatTensor(X_te).permute(0,2,1)
y_t = torch.LongTensor(y); y_v = torch.LongTensor(y_vl); y_te_t = torch.LongTensor(y_te)
C = X_t.shape[1]

# 软标签
soft_all = torch.FloatTensor(np.load(os.path.join(SOFT_DIR, 'pamap2_soft_all.npy')))
soft_correct = torch.FloatTensor(np.load(os.path.join(SOFT_DIR, 'pamap2_soft_correct_only.npy')))
soft_filtered = torch.FloatTensor(np.load(os.path.join(SOFT_DIR, 'pamap2_soft_filtered.npy')))

# 基线
p1 = json.load(open(os.path.join(RESULTS_DIR, 'pamap2_purecnn_phase1.json')))
baseline = p1['test_acc']
pt_path = os.path.join(RESULTS_DIR, 'pamap2_purecnn_phase1.pt')

# ---- 搜索 ----
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CosWR

SEARCH = [
    # (strategy_name, soft_tensor, alpha, T)
    ('correct_a0.9_T1.5', soft_correct, 0.9, 1.5),
    ('correct_a0.8_T1.5', soft_correct, 0.8, 1.5),
    ('correct_a0.9_T2.0', soft_correct, 0.9, 2.0),
    ('correct_a0.8_T2.0', soft_correct, 0.8, 2.0),
    ('filtered_a0.9_T1.5', soft_filtered, 0.9, 1.5),
    ('filtered_a0.8_T1.5', soft_filtered, 0.8, 1.5),
    ('all_a0.9_T1.5', soft_all, 0.9, 1.5),
    ('all_a0.8_T1.5', soft_all, 0.8, 1.5),
]

# 加载已有结果
p2_path = os.path.join(RESULTS_DIR, 'phase2_all.json')
all_p2 = json.load(open(p2_path)) if os.path.exists(p2_path) else {}

torch.manual_seed(SEED)
print(f"PureCNN 基线: {baseline:.4f}\n")

for label, soft, alpha, T in SEARCH:
    tag = f"purecnn_{label}"
    if tag in all_p2:
        r = all_p2[tag]
        print(f"SKIP {tag}: test={r['test_acc']:.4f} gain={r['gain']*100:+.2f}pp")
        continue
    
    print(f"RUN {tag} α={alpha} T={T} ...", end=' ', flush=True)
    model = PureCNN(in_channels=C, n_cls=N_CLS).to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = CosWR(opt, T_0=20, T_mult=2)
    loader = DataLoader(TensorDataset(X_t, soft, y_t), batch_size=BS, shuffle=True)
    
    best_val, best_st, best_ep, ct = 0.0, None, 0, 0
    for ep in range(MAX_EP):
        tl, ta = train_epoch(model, loader, opt, soft, alpha, T)
        va = evaluate(model, X_v, y_v); sched.step()
        if va > best_val: best_val, best_ep, best_st, ct = va, ep+1, copy.deepcopy(model.state_dict()), 0
        else: ct += 1
        if ct >= PATIENCE: break
    
    model.load_state_dict(best_st)
    te = evaluate(model, X_te_t, y_te_t)
    gain = te - baseline
    
    all_p2[tag] = {'model':'purecnn','strategy':label.split('_')[0],'alpha':alpha,'T':T,
                    'val_acc':best_val,'test_acc':te,'gain':gain,'best_epoch':best_ep}
    json.dump(all_p2, open(p2_path, 'w'), indent=2)
    print(f"test={te:.4f} gain={gain*100:+.2f}pp @E{best_ep}")

# Top results
print(f"\n{'='*60}")
sorted_r = sorted([(k,v) for k,v in all_p2.items() if v['model']=='purecnn' and v['gain']>0],
                  key=lambda x: x[1]['gain'], reverse=True)
if sorted_r:
    print(f"正增益实验 ({len(sorted_r)}组):")
    for tag, r in sorted_r:
        print(f"  {tag}: test={r['test_acc']:.4f} gain={r['gain']*100:+.2f}pp α={r['alpha']} T={r['T']}")
else:
    print("无正增益实验")

best_overall = max(all_p2.items(), key=lambda x: x[1]['test_acc'])
print(f"\n🏆 最佳: {best_overall[0]} test={best_overall[1]['test_acc']:.4f} (baseline={baseline:.4f})")
