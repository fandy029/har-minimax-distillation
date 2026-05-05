#!/usr/bin/env python3
"""
HARTH CNN软标签生成：
  Stage 1: 用硬标签训练 HARTH CNN（~10-20分钟）
  Stage 2: 用训练好的 CNN 生成软标签（秒级）

用法:
  python gen_harth_cnn_soft_labels.py [--stage 1] [--stage 2]
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# ============ 路径 ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from glob import glob

def load_harth():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    base = BASE_DIR + '/datasets/HARTH/harth'
    files = sorted(glob(f"{base}/*.csv"))
    d, l = [], []
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
    for f in files:
        try:
            df = pd.read_csv(f)
            back = df.iloc[:, 1:4].values.astype(np.float32)
            thigh = df.iloc[:, 4:7].values.astype(np.float32)
            x = np.concatenate([back, thigh], axis=1)
            y_ = df.iloc[:, 7].values.astype(int)
            for i in range(0, len(x)-127, 64):
                w = x[i:i+128]
                label = y_[i]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                    d.append(w); l.append(label_map[label])
        except: pass
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y

# ============ 配置 ============
SOFT_LABEL_FILE = f"{BASE_DIR}/results/soft_labels/harth_soft.npy"
CHECKPOINT_FILE = f"{BASE_DIR}/results/checkpoints/harth_cnn_best.pt"
CKPT_DIR = f"{BASE_DIR}/results/checkpoints"
N_CLS = 6
IN_CH = 3
TEMPERATURE = 3.0  # 蒸馏温度

# ============ 数据加载 ============
def load_data():
    X_tr, y_tr = load_harth()
    # 划分训练/验证 (90/10)
    n = len(X_tr)
    perm = np.random.RandomState(42).permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]
    X_train, y_train = X_tr[train_idx], y_tr[train_idx]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    return X_train, y_train, X_val, y_val

# ============ 模型 ============
class CNN1D(nn.Module):
    def __init__(self, in_ch=3, n_cls=6):
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

# ============ 训练 ============
def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    X_train, y_train, X_val, y_val = load_data()

    # 转换为 tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    model = CNN1D(in_ch=IN_CH, n_cls=N_CLS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    epochs = 80
    batch_size = 256

    print(f"训练 HARTH CNN...")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: 1e-3")

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(X_train))
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_FILE)
            marker = " *"
        else:
            marker = ""
        if (epoch + 1) % 10 == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={total_loss:.4f}, val_acc={val_acc:.4f}{marker}")

    print(f"\n✅ HARTH CNN 训练完成! 最佳验证准确率: {best_acc:.4f}")
    return model

# ============ 生成软标签 ============
def generate_soft_labels(model=None):
    if model is None:
        if not os.path.exists(CHECKPOINT_FILE):
            print(f"❌ 没有找到 checkpoint: {CHECKPOINT_FILE}")
            print("   请先运行 --stage 1 训练 CNN")
            return
        model = CNN1D(in_ch=IN_CH, n_cls=N_CLS)
        model.load_state_dict(torch.load(CHECKPOINT_FILE, weights_only=True))
        print(f"加载 checkpoint: {CHECKPOINT_FILE}")

    model.eval()
    X_all, y_all = load_harth()
    X_t = torch.FloatTensor(X_all)

    os.makedirs(os.path.dirname(SOFT_LABEL_FILE), exist_ok=True)

    print(f"\n生成 HARTH 软标签...")
    print(f"  样本数: {len(X_all)}, 温度: {TEMPERATURE}")

    soft_labels = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size]
            logits = model(xb)
            probs = F.softmax(logits / TEMPERATURE, dim=1)
            soft_labels.append(probs.cpu().numpy())

    soft_labels = np.concatenate(soft_labels, axis=0).astype(np.float32)

    # 质量报告
    preds = soft_labels.argmax(axis=1)
    acc = (preds == y_all).mean()
    print(f"\n--- CNN 软标签质量报告 ---")
    print(f"  样本数: {len(soft_labels)}")
    print(f"  软标签形状: {soft_labels.shape}")
    print(f"  CNN 预测准确率: {acc:.4f}")
    cn = ['left_stand', 'walk', 'stair_up', 'stair_down', 'right_stand', 'stand_still']
    for c, name in enumerate(cn):
        mask = y_all == c
        if mask.sum() > 0:
            c_acc = (preds[mask] == y_all[mask]).mean()
            print(f"  Class {c} ({name}): n={mask.sum()}, acc={c_acc:.4f}")

    # 保存
    np.save(SOFT_LABEL_FILE, soft_labels)
    print(f"\n💾 已保存: {SOFT_LABEL_FILE}")
    print(f"   软标签形状: {soft_labels.shape}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                        help='1=训练CNN, 2=生成软标签, 默认=1')
    args = parser.parse_args()

    if args.stage == 1:
        model = train()
        generate_soft_labels(model)
    elif args.stage == 2:
        generate_soft_labels()

if __name__ == '__main__':
    main()
