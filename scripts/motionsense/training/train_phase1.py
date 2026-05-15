#!/usr/bin/env python3
"""MOTIONSENSE — 阶段1: Focal Loss 纯硬标签预训练"""
import os, sys, json, time, copy, logging
from datetime import datetime
import numpy as np
import pandas as pd
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
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

from models import build_model

DATASET = 'motionsense'
CLASS_NAMES = ['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']
N_CLS = len(CLASS_NAMES)

DIR_MAP = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}

BATCH_SIZE, PHASE1_LR, PATIENCE, MAX_EPOCHS, CLIP_GRAD, SEED = 64, 5e-4, 15, 300, 5.0, 42
DEVICE_STR = 'cpu'
OUT_DIR = os.path.join(GAIT_DIR, 'output')
RESULTS_DIR = os.path.join(OUT_DIR, 'results')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()

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

def train_epoch(model, loader, optimizer, criterion, device, clip_grad=5.0):
    model.train(); total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb); loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        acc = (logits.argmax(1) == yb).float().mean().item()
        bs = len(xb); total_loss += loss.item() * bs; total_acc += acc * bs; n += bs
    return total_loss / n, total_acc / n

def train_phase1(model_name, X_t, y_t, X_v, y_v, X_te, y_te, C):
    tag = f"{DATASET}_{model_name}_phase1"
    log_path = os.path.join(LOG_DIR, f'{tag}.log')
    logger = logging.getLogger(tag); logger.setLevel(logging.DEBUG); logger.handlers.clear()
    fh = logging.FileHandler(log_path); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in [fh, ch]: h.setFormatter(fmt); logger.addHandler(h)
    device = torch.device(DEVICE_STR); torch.manual_seed(SEED)
    model = build_model(model_name, in_channels=C, n_cls=N_CLS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{'='*60}\n阶段1: {model_name.upper()} | MOTIONSENSE\nparams={n_params:,} Focal γ=2.0 lr={PHASE1_LR}\ntrain={len(y_t)} val={len(y_v)} test={len(y_te)}\n{'='*60}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE1_LR, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0); scheduler = CosWR(optimizer, T_0=20, T_mult=2)
    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    best_val_acc, best_state, best_epoch, best_metrics, counter = 0.0, None, 0, None, 0
    t0 = time.time(); epoch_log = []
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, CLIP_GRAD)
        val_acc, val_per_cls, val_cm = full_evaluate(model, X_v, y_v, device); scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        epoch_log.append({'epoch': epoch+1, 'train_loss': float(train_loss), 'train_acc': float(train_acc),
            'val_acc': float(val_acc), 'val_per_class': {k: float(v) for k, v in val_per_cls.items()}, 'lr': float(lr_now)})
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f"E{epoch+1:03d} | loss={train_loss:.4f} train={train_acc:.4f} val={val_acc:.4f} lr={lr_now:.2e}")
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch+1; best_state = copy.deepcopy(model.state_dict())
            best_metrics = {'val_acc': float(val_acc), 'val_per_class': {k: float(v) for k, v in val_per_cls.items()}, 'val_cm': val_cm.tolist()}; counter = 0
        else:
            counter += 1
            if counter >= PATIENCE: logger.info(f"早停 @ E{epoch+1} (best val={best_val_acc:.4f} @ E{best_epoch})"); break
    model.load_state_dict(best_state)
    test_acc, test_per_cls, test_cm = full_evaluate(model, X_te, y_te, device)
    train_acc_f, train_per_cls, _ = full_evaluate(model, X_t, y_t, device)
    elapsed = (time.time() - t0) / 60
    logger.info(f"\n{'='*60}\n完成! train={train_acc_f:.4f} val={best_val_acc:.4f} @E{best_epoch} test={test_acc:.4f} {elapsed:.1f}min")
    logger.info(f"每类准确率 (test):")
    for n in CLASS_NAMES: logger.info(f"  {n}: {test_per_cls[n]:.4f}")
    logger.info(f"混淆矩阵:\n        " + " ".join(f"{n[:8]:>8s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES): logger.info(f"  {n:<8s} " + " ".join(f"{test_cm[i][j]:>8d}" for j in range(N_CLS)))
    logger.info(f"{'='*60}")
    results = {'model': model_name, 'phase': 1, 'n_params': n_params,
        'train_acc': float(train_acc_f), 'val_acc': float(best_val_acc), 'test_acc': float(test_acc),
        'best_epoch': best_epoch, 'time_min': elapsed,
        'test_per_class': {k: float(v) for k, v in test_per_cls.items()},
        'test_confusion_matrix': test_cm.tolist(),
        'train_per_class': {k: float(v) for k, v in train_per_cls.items()},
        'config': {'loss': 'FocalLoss(γ=2.0)', 'optimizer': 'AdamW(lr=5e-4,wd=1e-4)', 'scheduler': 'CosWR(T0=20)',
            'batch_size': BATCH_SIZE, 'patience': PATIENCE, 'grad_clip': CLIP_GRAD, 'seed': SEED}, 'epoch_log': epoch_log}
    torch.save(best_state, os.path.join(RESULTS_DIR, f'{tag}.pt'))
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f: json.dump(results, f, indent=2)
    logger.info(f"模型: {tag}.pt"); [logger.removeHandler(h) for h in logger.handlers[:]]
    return results

if __name__ == '__main__':
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument('--models', type=str, default='purecnn,cnnres,transformer')
    args_ = ap.parse_args(); MODELS = [m.strip() for m in args_.models.split(',')]
    print(f"{'='*60}\n加载 MOTIONSENSE 数据...")
        base = os.path.join(THESIS_DIR, 'datasets', 'MotionSense')
    X_all, y_all = [], []
    for folder in sorted(glob(os.path.join(base, '*'))):
        if not os.path.isdir(folder): continue
        prefix = os.path.basename(folder).split('_')[0]
        if prefix not in DIR_MAP: continue
        cid = DIR_MAP[prefix]
        for f in sorted(glob(os.path.join(folder, '*.csv'))):
            try:
                df = pd.read_csv(f)
                acc = df[['x','y','z']].values.astype(np.float32)
                for s in range(0, len(acc)-127, 64):
                    w = acc[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)): X_all.append(w); y_all.append(cid)
            except: continue
    X_all = np.array(X_all, dtype=np.float32); y_all = np.array(y_all, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.1765, random_state=SEED, stratify=y)
    print(f"train={len(X)} val={len(X_vl)} test={len(X_te)}")
    for c in range(N_CLS): print(f"  {CLASS_NAMES[c]:<16s}: train={int((y==c).sum())} val={int((y_vl==c).sum())} test={int((y_te==c).sum())}")
    mean = X.mean(axis=(0,1), keepdims=True); std = X.std(axis=(0,1), keepdims=True) + 1e-6
    X=(X-mean)/std; X_vl=(X_vl-mean)/std; X_te=(X_te-mean)/std
    X_t = torch.FloatTensor(X).permute(0,2,1); X_v = torch.FloatTensor(X_vl).permute(0,2,1)
    X_te_t = torch.FloatTensor(X_te).permute(0,2,1)
    y_t = torch.LongTensor(y); y_v = torch.LongTensor(y_vl); y_te_t = torch.LongTensor(y_te)
    C = X_t.shape[1]
    all_results = {}
    for model_name in MODELS:
        print(f"\n{'#'*60}\n# 训练: {model_name.upper()}\n{'#'*60}")
        all_results[model_name] = train_phase1(model_name, X_t, y_t, X_v, y_v, X_te_t, y_te_t, C)
    summary = {'dataset': 'MOTIONSENSE', 'phase': 1, 'timestamp': datetime.now().isoformat(),
        'n_train': int(len(X)), 'n_val': int(len(X_vl)), 'n_test': int(len(X_te)),
        'n_classes': N_CLS, 'class_names': CLASS_NAMES, 'results': all_results}
    with open(os.path.join(RESULTS_DIR, 'phase1_summary.json'), 'w') as f: json.dump(summary, f, indent=2, default=str)
    print(f"\n{'='*60}\n阶段1 全部完成!")
    for m, r in all_results.items(): print(f"  {m:<14s}: val={r['val_acc']:.4f} test={r['test_acc']:.4f} params={r['n_params']:,} time={r['time_min']:.1f}min")
