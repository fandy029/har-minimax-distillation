#!/usr/bin/env python3
"""
GAIT 数据集 — 论文方案完整训练脚本 (阶段1: 纯硬标签预训练)
================================================================
- 三种架构: PureCNN / CNN-Residual / Transformer
- Focal Loss (γ=2.0), AdamW, CosineAnnealingWarmRestarts
- 每 epoch 记录: loss, train/val/test acc, 每类准确率
- 训练结束后保存: 混淆矩阵, 最佳模型, 完整日志
"""
import os, sys, json, time, copy, logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CosWR
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
SCRIPTS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))  # 导入 models, trainer

from models import build_model, PureCNN, CNNResidual, TransformerModel

# ============================================================
# 配置
# ============================================================
DATASET = 'gait'
CLASS_NAMES = ['sit_on_bed', 'sit_on_chair', 'lying', 'ambulating']
N_CLS = len(CLASS_NAMES)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

BATCH_SIZE = 64
PHASE1_LR = 5e-4
PATIENCE = 15
MAX_EPOCHS = 300
CLIP_GRAD = 5.0
SEED = 42
DEVICE_STR = 'cpu'

OUT_DIR = os.path.join(GAIT_DIR, 'output')
RESULTS_DIR = os.path.join(OUT_DIR, 'results')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ============================================================
# 评估工具
# ============================================================
@torch.no_grad()
def full_evaluate(model, X, y, device, batch_size=256):
    """返回: acc, per_class_acc (dict), cm (np.ndarray)"""
    model.eval()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size)
    all_preds, all_labels = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(yb)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    per_class = {}
    for c in range(N_CLS):
        mask = labels == c
        if mask.sum() > 0:
            per_class[CLASS_NAMES[c]] = (preds[mask] == c).mean()
        else:
            per_class[CLASS_NAMES[c]] = 0.0
    cm = confusion_matrix(labels, preds, labels=list(range(N_CLS)))
    return float(acc), per_class, cm

# ============================================================
# 训练一个 epoch
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, clip_grad=5.0):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        acc = (logits.argmax(1) == yb).float().mean().item()
        bs = len(xb)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs
    return total_loss / n, total_acc / n

# ============================================================
# 主训练函数
# ============================================================
def train_phase1(model_name, X_t, y_t, X_v, y_v, X_te, y_te, C):
    tag = f"{DATASET}_{model_name}_phase1"
    log_path = os.path.join(LOG_DIR, f'{tag}.log')
    
    # 日志
    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

    device = torch.device(DEVICE_STR)
    torch.manual_seed(SEED)

    model = build_model(model_name, in_channels=C, n_cls=N_CLS).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    logger.info(f"{'='*60}")
    logger.info(f"阶段1: {model_name.upper()} 预训练")
    logger.info(f"参数量: {n_params:,}  |  Focal Loss γ=2.0  |  lr={PHASE1_LR}")
    logger.info(f"train={len(y_t)}  val={len(y_v)}  test={len(y_te)}")
    logger.info(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE1_LR, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0)
    scheduler = CosWR(optimizer, T_0=20, T_mult=2)
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    best_metrics = None
    counter = 0
    t0 = time.time()
    epoch_log = []

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, CLIP_GRAD)
        val_acc, val_per_cls, val_cm = full_evaluate(model, X_v, y_v, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        entry = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'val_per_class': {k: float(v) for k, v in val_per_cls.items()},
            'lr': float(lr_now),
        }
        epoch_log.append(entry)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"E{epoch+1:03d} | loss={train_loss:.4f} "
                       f"train={train_acc:.4f} val={val_acc:.4f} lr={lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                'val_acc': float(val_acc),
                'val_per_class': {k: float(v) for k, v in val_per_cls.items()},
                'val_cm': val_cm.tolist(),
            }
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                logger.info(f"早停 @ E{epoch+1} (best val={best_val_acc:.4f} @ E{best_epoch})")
                break

    # 加载最佳 → 最终评估
    model.load_state_dict(best_state)
    test_acc, test_per_cls, test_cm = full_evaluate(model, X_te, y_te, device)
    train_acc, train_per_cls, train_cm = full_evaluate(model, X_t, y_t, device)
    elapsed = (time.time() - t0) / 60

    logger.info(f"\n{'='*60}")
    logger.info(f"阶段1 完成!")
    logger.info(f"  train_acc: {train_acc:.4f}")
    logger.info(f"  val_acc:   {best_val_acc:.4f} @ E{best_epoch}")
    logger.info(f"  test_acc:  {test_acc:.4f}")
    logger.info(f"  耗时: {elapsed:.1f} min")
    logger.info(f"\n每类准确率 (test):")
    for cname in CLASS_NAMES:
        logger.info(f"  {cname:<15s}: {test_per_cls[cname]:.4f}")
    logger.info(f"\n混淆矩阵 (test):")
    header = "        " + " ".join(f"{n[:8]:>8s}" for n in CLASS_NAMES)
    logger.info(header)
    for i, cname in enumerate(CLASS_NAMES):
        row = " ".join(f"{test_cm[i][j]:>8d}" for j in range(N_CLS))
        logger.info(f"  {cname:<8s} {row}")
    logger.info(f"{'='*60}")

    # 保存
    results = {
        'model': model_name,
        'phase': 1,
        'n_params': n_params,
        'train_acc': float(train_acc),
        'val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'best_epoch': best_epoch,
        'time_min': elapsed,
        'test_per_class': {k: float(v) for k, v in test_per_cls.items()},
        'test_confusion_matrix': test_cm.tolist(),
        'train_per_class': {k: float(v) for k, v in train_per_cls.items()},
        'config': {
            'loss': 'FocalLoss(γ=2.0)',
            'optimizer': 'AdamW(lr=5e-4, wd=1e-4)',
            'scheduler': 'CosineAnnealingWarmRestarts(T0=20,Tmult=2)',
            'batch_size': BATCH_SIZE,
            'patience': PATIENCE,
            'max_epochs': MAX_EPOCHS,
            'grad_clip': CLIP_GRAD,
            'seed': SEED,
        },
        'epoch_log': epoch_log,
    }
    
    pt_path = os.path.join(RESULTS_DIR, f'{tag}.pt')
    json_path = os.path.join(RESULTS_DIR, f'{tag}.json')
    torch.save(best_state, pt_path)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"模型: {pt_path}")
    logger.info(f"结果: {json_path}")
    logger.info(f"日志: {log_path}")

    # 清理 logger handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=str, default='purecnn,cnnres,transformer')
    args_ = ap.parse_args()
    MODELS = [m.strip() for m in args_.models.split(',')]

    # ===== 加载数据 =====
    print(f"{'='*60}")
    print(f"加载 GAIT 数据...")
    base = os.path.join(THESIS_DIR, 'datasets', 'Gait_Classification')
    X_all, y_all = [], []
    for folder in ['S1_Dataset', 'S2_Dataset']:
        for f in sorted(glob(os.path.join(base, folder, '*'))):
            if f.endswith('.txt') or 'README' in f:
                continue
            try:
                df = pd.read_csv(f, header=None)
                acc = df.iloc[:, 1:4].values.astype(np.float32)
                labels = df.iloc[:, 8].astype(int)
                for i in range(0, len(df) - 127, 64):
                    w = acc[i:i + 128]
                    label = int(labels.iloc[i])
                    if w.shape[0] == 128 and not np.any(np.isnan(w)) and label in LABEL_MAP:
                        X_all.append(w)
                        y_all.append(LABEL_MAP[label])
            except:
                continue
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    X, X_te, y, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.1765, random_state=SEED, stratify=y)
    print(f"train={len(X)}  val={len(X_vl)}  test={len(X_te)}")
    for c in range(N_CLS):
        print(f"  {CLASS_NAMES[c]:<15s}: train={int((y==c).sum())}  val={int((y_vl==c).sum())}  test={int((y_te==c).sum())}")

    # 归一化
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std
    X_vl = (X_vl - mean) / std
    X_te = (X_te - mean) / std

    X_t = torch.FloatTensor(X).permute(0, 2, 1)
    X_v = torch.FloatTensor(X_vl).permute(0, 2, 1)
    X_te_t = torch.FloatTensor(X_te).permute(0, 2, 1)
    y_t = torch.LongTensor(y)
    y_v = torch.LongTensor(y_vl)
    y_te_t = torch.LongTensor(y_te)
    C = X_t.shape[1]

    # ===== 逐架构训练 =====
    all_results = {}
    for model_name in MODELS:
        print(f"\n{'#'*60}")
        print(f"# 训练: {model_name.upper()}")
        print(f"{'#'*60}")
        r = train_phase1(model_name, X_t, y_t, X_v, y_v, X_te_t, y_te_t, C)
        all_results[model_name] = r

    # 保存汇总
    summary = {
        'dataset': 'GAIT',
        'phase': 1,
        'timestamp': datetime.now().isoformat(),
        'n_train': int(len(X)), 'n_val': int(len(X_vl)), 'n_test': int(len(X_te)),
        'n_classes': N_CLS,
        'class_names': CLASS_NAMES,
        'results': all_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase1_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"阶段1 全部完成!")
    for m, r in all_results.items():
        print(f"  {m:<14s}: val={r['val_acc']:.4f}  test={r['test_acc']:.4f}  "
              f"params={r['n_params']:,}  time={r['time_min']:.1f}min")
    print(f"{'='*60}")
