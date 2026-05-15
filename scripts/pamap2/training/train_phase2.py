#!/usr/bin/env python3
"""
PAMAP2 — 阶段2: 知识蒸馏微调

最佳参数 (参考GAIT实验):
  PureCNN:     α=0.5  T=3.5  strategy=all
  CNN-Res:     α=0.7  T=1.5  strategy=filtered
  Transformer: α=0.7  T=1.5  strategy=all
"""
import os, sys, json, time, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CosWR
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
SCRIPTS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))  # 导入 models, trainer

from models import build_model

BEST_PARAMS = {
    'purecnn':     {'alpha': 0.5, 'T': 3.5, 'strategy': 'all',       'label': 'all_a0.5_T3.5'},
    'cnnres':      {'alpha': 0.7, 'T': 1.5, 'strategy': 'filtered',  'label': 'filtered_a0.7_T1.5'},
    'transformer': {'alpha': 0.7, 'T': 1.5, 'strategy': 'all',       'label': 'all_a0.7_T1.5'},
}

DATASET = 'pamap2'
CLASS_NAMES = ['lying', 'sitting', 'standing', 'walking', 'jogging']
N_CLS = len(CLASS_NAMES)
TARGET_IDS = {1, 2, 3, 4, 5}
ID_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

BATCH_SIZE = 64
PHASE2_LR = 1e-4
PATIENCE = 15
MAX_EPOCHS = 300
CLIP_GRAD = 5.0
SEED = 42
DEVICE_STR = 'cpu'

OUT_DIR = os.path.join(GAIT_DIR, 'output')
RESULTS_DIR = os.path.join(OUT_DIR, 'results')
SOFT_DIR = os.path.join(OUT_DIR, 'soft_labels')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()

def distill_loss_fn(logits, soft_labels, y_hard, alpha, T):
    focal = FocalLoss(gamma=2.0)(logits, y_hard)
    logp_T = F.log_softmax(logits / T, dim=1)
    soft = soft_labels.clamp(1e-8, 1.0)
    kl = (soft * (soft.log() - logp_T)).sum(dim=1).mean()
    return alpha * focal + (1 - alpha) * (T ** 2) * kl

@torch.no_grad()
def full_evaluate(model, X, y, device, batch_size=256):
    model.eval(); ds = TensorDataset(X, y); dl = DataLoader(ds, batch_size=batch_size)
    all_preds, all_labels = [], []
    for xb, yb in dl:
        all_preds.append(model(xb.to(device)).argmax(1).cpu()); all_labels.append(yb)
    preds = torch.cat(all_preds).numpy(); labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    per_class = {}
    for c in range(N_CLS):
        m = labels == c
        per_class[CLASS_NAMES[c]] = float((preds[m] == c).mean()) if m.sum() > 0 else 0.0
    cm = confusion_matrix(labels, preds, labels=list(range(N_CLS)))
    return float(acc), per_class, cm

def train_epoch_distill(model, loader, optimizer, device, soft_labels, alpha, T, clip_grad=5.0):
    model.train(); total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, soft_b, yb in loader:
        xb, soft_b, yb = xb.to(device), soft_b.to(device), yb.to(device)
        loss = distill_loss_fn(model(xb), soft_b, yb, alpha, T)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        acc = (model(xb).argmax(1) == yb).float().mean().item()
        bs = len(xb); total_loss += loss.item() * bs; total_acc += acc * bs; n += bs
    return total_loss / n, total_acc / n

def train_phase2(model_name, X_t, y_t, X_v, y_v, X_te, y_te, C, soft_t_all, soft_t_filtered, soft_t_correct):
    params = BEST_PARAMS[model_name]; alpha, T_val, strategy = params['alpha'], params['T'], params['strategy']
    soft_map = {'all': soft_t_all, 'filtered': soft_t_filtered, 'correct_only': soft_t_correct}
    soft_t = soft_map[strategy]; param_label = params['label']
    tag = f"{DATASET}_{model_name}_phase2_{param_label}"
    log_path = os.path.join(LOG_DIR, f'{tag}.log')

    logger = logging.getLogger(tag); logger.setLevel(logging.DEBUG); logger.handlers.clear()
    fh = logging.FileHandler(log_path); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in [fh, ch]: h.setFormatter(fmt); logger.addHandler(h)

    device = torch.device(DEVICE_STR); torch.manual_seed(SEED)
    pt_path = os.path.join(RESULTS_DIR, f'{DATASET}_{model_name}_phase1.pt')
    p1 = json.load(open(os.path.join(RESULTS_DIR, f'{DATASET}_{model_name}_phase1.json')))
    baseline_test = p1['test_acc']

    model = build_model(model_name, in_channels=C, n_cls=N_CLS).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    n_params = sum(p.numel() for p in model.parameters())

    logger.info(f"{'='*60}")
    logger.info(f"阶段2: {model_name.upper()} 蒸馏 | PAMAP2 | {strategy} α={alpha} T={T_val}")
    logger.info(f"硬标签 {alpha*100:.0f}% | 软标签 {(1-alpha)*100:.0f}% | 基线 test={baseline_test:.4f}")
    logger.info(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)
    scheduler = CosWR(optimizer, T_0=20, T_mult=2)
    train_loader = DataLoader(TensorDataset(X_t, soft_t, y_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    best_val_acc, best_state, best_epoch, best_metrics, counter = 0.0, None, 0, None, 0
    t0 = time.time(); epoch_log = []

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch_distill(model, train_loader, optimizer, device, soft_t, alpha, T_val, CLIP_GRAD)
        val_acc, val_per_cls, val_cm = full_evaluate(model, X_v, y_v, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        epoch_log.append({'epoch': epoch+1, 'train_loss': float(train_loss), 'train_acc': float(train_acc),
                          'val_acc': float(val_acc), 'val_per_class': {k: float(v) for k, v in val_per_cls.items()}, 'lr': float(lr_now)})
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f"E{epoch+1:03d} | loss={train_loss:.4f} train={train_acc:.4f} val={val_acc:.4f} lr={lr_now:.2e}")
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch+1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {'val_acc': float(val_acc), 'val_per_class': {k: float(v) for k, v in val_per_cls.items()}, 'val_cm': val_cm.tolist()}
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                logger.info(f"早停 @ E{epoch+1} (best val={best_val_acc:.4f} @ E{best_epoch})"); break

    model.load_state_dict(best_state)
    test_acc, test_per_cls, test_cm = full_evaluate(model, X_te, y_te, device)
    train_acc_f, train_per_cls, _ = full_evaluate(model, X_t, y_t, device)
    gain = test_acc - baseline_test; elapsed = (time.time() - t0) / 60

    logger.info(f"\n{'='*60}")
    logger.info(f"阶段2 完成!  train={train_acc_f:.4f}  val={best_val_acc:.4f} @ E{best_epoch}")
    logger.info(f"test={test_acc:.4f} (基线 {baseline_test:.4f}, {'+' if gain>0 else ''}{gain*100:.2f}pp)  {elapsed:.1f}min")
    logger.info(f"每类准确率 (test):")
    for n in CLASS_NAMES: logger.info(f"  {n:<12s}: {test_per_cls[n]:.4f}")
    logger.info(f"混淆矩阵 (test):")
    logger.info("        " + " ".join(f"{n[:8]:>8s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        logger.info(f"  {n:<8s} " + " ".join(f"{test_cm[i][j]:>8d}" for j in range(N_CLS)))
    logger.info(f"{'='*60}")

    results = {'model': model_name, 'phase': 2, 'n_params': n_params,
        'strategy': strategy, 'alpha': alpha, 'T': T_val,
        'hard_label_pct': f"{alpha*100:.0f}%", 'soft_label_pct': f"{(1-alpha)*100:.0f}%",
        'baseline_test_acc': float(baseline_test), 'train_acc': float(train_acc_f),
        'val_acc': float(best_val_acc), 'test_acc': float(test_acc),
        'gain': float(gain), 'gain_pct': f"{gain*100:+.2f}pp", 'best_epoch': best_epoch, 'time_min': elapsed,
        'test_per_class': {k: float(v) for k, v in test_per_cls.items()},
        'test_confusion_matrix': test_cm.tolist(),
        'train_per_class': {k: float(v) for k, v in train_per_cls.items()},
        'config': {'loss': 'α·Focal(γ=2.0)+(1-α)·T²·KL', 'alpha': alpha, 'T': T_val,
                   'optimizer': 'AdamW(lr=1e-4,wd=1e-4)', 'scheduler': 'CosineAnnealingWarmRestarts(T0=20,Tmult=2)',
                   'batch_size': BATCH_SIZE, 'patience': PATIENCE, 'max_epochs': MAX_EPOCHS, 'grad_clip': CLIP_GRAD, 'seed': SEED},
        'epoch_log': epoch_log}
    torch.save(best_state, os.path.join(RESULTS_DIR, f'{tag}.pt'))
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f: json.dump(results, f, indent=2)
    logger.info(f"模型: {tag}.pt  结果: {tag}.json  日志: {tag}.log")
    for h in logger.handlers[:]: logger.removeHandler(h)
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--models', type=str, default='purecnn,cnnres,transformer')
    args_ = ap.parse_args(); MODELS = [m.strip() for m in args_.models.split(',')]

    print(f"{'='*60}\n加载 PAMAP2 数据...")
    base = os.path.join(THESIS_DIR, 'datasets', 'PAMAP2', 'PAMAP2_Dataset')
    X_all, y_all = [], []
    for folder in ['Protocol', 'Optional']:
        for f in sorted(glob(os.path.join(base, folder, '*.dat'))):
            try:
                data = np.loadtxt(f)
                if data.ndim != 2: continue
                ids = data[:, 1].astype(int); imu = data[:, 9:15].astype(np.float32)
                for s in range(0, len(imu)-127, 64):
                    w = imu[s:s+128]; aid = ids[s]
                    if w.shape[0] == 128 and not np.any(np.isnan(w)) and aid in TARGET_IDS:
                        X_all.append(w); y_all.append(ID_MAP[aid])
            except: continue
    X_all = np.array(X_all, dtype=np.float32); y_all = np.array(y_all, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.1765, random_state=SEED, stratify=y)
    print(f"train={len(X)} val={len(X_vl)} test={len(X_te)}")
    mean = X.mean(axis=(0, 1), keepdims=True); std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X-mean)/std; X_vl = (X_vl-mean)/std; X_te = (X_te-mean)/std
    X_t = torch.FloatTensor(X).permute(0, 2, 1); X_v = torch.FloatTensor(X_vl).permute(0, 2, 1)
    X_te_t = torch.FloatTensor(X_te).permute(0, 2, 1)
    y_t = torch.LongTensor(y); y_v = torch.LongTensor(y_vl); y_te_t = torch.LongTensor(y_te)
    C = X_t.shape[1]

    soft_all = torch.FloatTensor(np.load(os.path.join(SOFT_DIR, 'pamap2_soft_all.npy')))
    soft_filtered = torch.FloatTensor(np.load(os.path.join(SOFT_DIR, 'pamap2_soft_filtered.npy')))
    soft_correct = torch.FloatTensor(np.load(os.path.join(SOFT_DIR, 'pamap2_soft_correct_only.npy')))

    all_results = {}
    for model_name in MODELS:
        phase1_pt = os.path.join(RESULTS_DIR, f'{DATASET}_{model_name}_phase1.pt')
        if not os.path.exists(phase1_pt):
            print(f"⚠️  {model_name}: 阶段1权重不存在, 跳过"); continue
        print(f"\n{'#'*60}\n# 蒸馏: {model_name.upper()} ({BEST_PARAMS[model_name]['label']})\n{'#'*60}")
        all_results[model_name] = train_phase2(model_name, X_t, y_t, X_v, y_v, X_te_t, y_te_t, C, soft_all, soft_filtered, soft_correct)

    summary = {'dataset': 'PAMAP2', 'phase': 2, 'timestamp': datetime.now().isoformat(),
               'best_params_per_model': BEST_PARAMS, 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase2_summary.json'), 'w') as f: json.dump(summary, f, indent=2, default=str)
    print(f"\n{'='*60}\n阶段2 全部完成!")
    for m, r in all_results.items():
        print(f"  {m:<14s}: test={r['test_acc']:.4f} gain={r['gain_pct']} time={r['time_min']:.1f}min")
    print(f"{'='*60}")
