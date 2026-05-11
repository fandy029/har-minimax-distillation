#!/usr/bin/env python3
"""Gait Pure CNN 训练"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# ===== 超参 =====
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

EPOCHS = args.epochs; PATIENCE = args.patience; LR = args.lr; BS = args.batch_size
SEED = args.seed

np.random.seed(SEED); torch.manual_seed(SEED)

OUT_BASE = os.path.join(SCRIPT_DIR, 'output')
PER_CLASS = os.path.join(OUT_BASE, 'per_class')
CLASS_NAMES = ['sit_on_bed','sit_on_chair','lying','ambulating']; N_CLS = 4

# ===== 加载数据 =====
base = os.path.join(THESIS_DIR, 'datasets', 'Gait_Classification')
X_all, y_all = [], []
from glob import glob
LABEL_MAP = {1:0, 2:1, 3:2, 4:3}
import pandas as pd
for folder in ['S1_Dataset','S2_Dataset']:
    for f in sorted(glob(os.path.join(base, folder, '*'))):
        if f.endswith('.txt') or 'README' in f: continue
        try:
            df = pd.read_csv(f, header=None)
            acc = df.iloc[:,1:4].values.astype(np.float32)
            labels = df.iloc[:,8].astype(int)
            for i in range(0, len(df)-127, 64):
                w = acc[i:i+128]
                label = int(labels.iloc[i])
                if w.shape[0]==128 and not np.any(np.isnan(w)) and label in LABEL_MAP:
                    X_all.append(w); y_all.append(LABEL_MAP[label])
        except: continue
X_all = np.array(X_all, dtype=np.float32); y_all = np.array(y_all, dtype=np.int64)
print(f"总窗口: {len(X_all)}")

X, X_te, y, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.1765, random_state=SEED, stratify=y)
print(f"train={len(X)} val={len(X_vl)} test={len(X_te)}")

# normalize per-channel
def normalize(X_tr, X_vl, X_te):
    mean = X_tr.mean(axis=(0,1), keepdims=True)
    std  = X_tr.std(axis=(0,1), keepdims=True) + 1e-6
    for x in [X_tr, X_vl, X_te]: x -= mean; x /= std
    return X_tr, X_vl, X_te

X_t = torch.FloatTensor(X).permute(0,2,1)   # (N,3,128)
X_v = torch.FloatTensor(X_vl).permute(0,2,1)
X_te= torch.FloatTensor(X_te).permute(0,2,1)
y_t = torch.LongTensor(y); y_v = torch.LongTensor(y_vl); y_te_t = torch.LongTensor(y_te)

# ===== CNN 模型 =====
class GaitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, N_CLS)
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GaitCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

train_ds = TensorDataset(X_t, y_t)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)

best_val_acc = 0.0; patience_counter = 0; best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(xb)
    train_loss = total_loss / len(X_t)

    model.eval()
    with torch.no_grad():
        vl_logits = model(X_v.to(device)); vl_loss = criterion(vl_logits, y_v.to(device)).item()
        vl_acc = (vl_logits.argmax(1)==y_v.to(device)).float().mean().item()
        te_logits = model(X_te.to(device)); te_acc = (te_logits.argmax(1)==y_te_t.to(device)).float().mean().item()

    scheduler.step(vl_loss)
    print(f"E{epoch:03d} loss={train_loss:.4f} val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} test_acc={te_acc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_val_acc_epoch = epoch
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"早停 @ epoch {epoch} (最佳 val_acc={best_val_acc:.4f} @ E{best_val_acc_epoch})")
            break

# 保存
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_acc = (model(X_te.to(device)).argmax(1)==y_te_t.to(device)).float().mean().item()
    train_acc = (model(X_t.to(device)).argmax(1)==y_t.to(device)).float().mean().item()

print(f"\n=== Pure CNN 结果 ===")
print(f"train_acc: {train_acc:.4f}")
print(f"best_val_acc: {best_val_acc:.4f}")
print(f"test_acc: {test_acc:.4f}")

out = {
    'test_acc': float(test_acc),
    'best_val_acc': float(best_val_acc),
    'train_acc': float(train_acc),
    'best_epoch': best_val_acc_epoch,
    'config': {'epochs':EPOCHS,'patience':PATIENCE,'lr':LR,'batch_size':BS,'seed':SEED}
}
torch.save(best_state, os.path.join(OUT_BASE, 'gait_pure_cnn.pt'))
with open(os.path.join(OUT_BASE, 'gait_pure_cnn.json'), 'w') as f: json.dump(out, f, indent=2)
print(f"模型已保存: gait_pure_cnn.pt")
