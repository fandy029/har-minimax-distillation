"""
WISDM MLP训练脚本 - 使用ARFF 43个统计特征
支持Pure CNN（baseline）、蒸馏v1/v2/v3
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cn = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
n_cls = 6
WISDM_DATA = '/home/fandy/workplace/thesis/datasets/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff'
SOFT_LABEL_FILE = "/home/fandy/workplace/thesis/results/soft_labels/wisdm_soft.npy"


def load_wisdm():
    """加载WISDM ARFF格式数据"""
    d, l = [], []
    label_map = {
        'Walking': 0, 'Jogging': 1,
        'Upstairs': 2, 'Downstairs': 3,
        'Sitting': 4, 'Standing': 5
    }
    with open(WISDM_DATA, 'r') as f:
        content = f.read()

    in_data = False
    for line in content.split('\n'):
        line = line.strip()
        if line == '@data':
            in_data = True
            continue
        if not in_data or not line:
            continue
        parts = line.split(',')
        if len(parts) < 46:
            continue
        try:
            features = [float(parts[i]) for i in range(2, 45)]
            label_str = parts[45].strip().strip('"')
            if label_str in label_map:
                d.append(features)
                l.append(label_map[label_str])
        except (ValueError, IndexError):
            continue

    X = np.array(d, dtype=np.float32)
    y = np.array(l, dtype=np.int64)
    print(f"  WISDM: {len(X)} samples, {X.shape[1]} features, classes: {n_cls}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Class dist: {dict(zip(unique, counts))}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te


class MLP(nn.Module):
    """WISDM专用MLP: 43 -> 256 -> 128 -> 64 -> 6"""
    def __init__(self, in_dim=43, n_cls=6, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, n_cls)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


class DistillMLPv1(nn.Module):
    """蒸馏v1: 硬标签CE + 软标签KL(单独) + 类别加权"""
    def __init__(self, in_dim=43, n_cls=6):
        super().__init__()
        self.mlp = MLP(in_dim, n_cls, dropout=0.4)
        # teacher projection
        self.proj = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_cls)
        )

    def forward(self, x, soft_labels=None, alpha=0.7, temperature=4.0):
        feat = F.relu(self.mlp.bn1(self.mlp.fc1(x)))
        out = self.mlp.fc4(
            self.mlp.drop(F.relu(self.mlp.bn2(self.mlp.fc2(feat)))) +
            self.mlp.drop(F.relu(self.mlp.bn3(self.mlp.fc3(feat))))
        )
        return out, feat


class DistillMLPv2(nn.Module):
    """蒸馏v2: 硬+软联合训练 + MLP隐藏层蒸馏"""
    def __init__(self, in_dim=43, n_cls=6):
        super().__init__()
        self.mlp = MLP(in_dim, n_cls, dropout=0.35)

    def forward(self, x, soft_labels=None, alpha=0.5, temperature=3.0):
        # 特征提取
        h1 = F.relu(self.mlp.bn1(self.mlp.fc1(x)))
        h2 = F.relu(self.mlp.bn2(self.mlp.fc2(h1)))
        h3 = F.relu(self.mlp.bn3(self.mlp.fc3(h2)))
        logits = self.mlp.fc4(self.mlp.drop(h3))
        return logits, h3


class DistillMLPv3(nn.Module):
    """蒸馏v3: 双向 KL 散度 + Focal Loss"""
    def __init__(self, in_dim=43, n_cls=6):
        super().__init__()
        self.mlp = MLP(in_dim, n_cls, dropout=0.3)
        self.focal_gamma = 2.0

    def focal_loss(self, logits, targets, weight=None):
        ce = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.focal_gamma) * ce
        return focal.mean()

    def forward(self, x, soft_labels=None, alpha=0.5, temperature=3.0):
        h1 = F.relu(self.mlp.bn1(self.mlp.fc1(x)))
        h2 = F.relu(self.mlp.bn2(self.mlp.fc2(h1)))
        h3 = F.relu(self.mlp.bn3(self.mlp.fc3(h2)))
        logits = self.mlp.fc4(self.mlp.drop(h3))
        return logits, h3


def compute_class_weights(y, n_cls):
    """计算类别权重用于处理不平衡"""
    counts = np.bincount(y, minlength=n_cls)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_cls
    return torch.FloatTensor(weights)


def train_pure_mlp(X_tr, y_tr, X_vl, y_vl, X_te, y_te, epochs=80):
    """Pure MLP baseline"""
    print("\n=== Pure MLP ===")
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_vl_n = scaler.transform(X_vl)
    X_te_n = scaler.transform(X_te)

    X_tr_t = torch.FloatTensor(X_tr_n).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    X_vl_t = torch.FloatTensor(X_vl_n).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te_n).to(DEVICE)

    model = MLP(43, n_cls).to(DEVICE)
    weight = compute_class_weights(y_tr, n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = nn.CrossEntropyLoss(weight=weight)

    best_state, best_acc = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 128):
            idx = perm[i:i+128]
            out = model(X_tr_t[idx])
            loss = crit(out, y_tr_t[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            vl_acc = (model(X_vl_t).argmax(1) == y_tr[y_vl]).float().mean() if len(y_vl) > 0 else 0
            # compute val acc properly
            val_pred = model(X_vl_t).argmax(1).cpu().numpy()
            vl_acc = (val_pred == y_vl).mean()
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep}: val_acc={vl_acc*100:.1f}%")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(X_te_t).argmax(1).cpu().numpy()
        te_acc = (te_pred == y_te).mean()
    print(f"  Pure MLP test: {te_acc*100:.2f}%")
    return round(te_acc * 100, 2)


def train_distill_v1(X_tr, y_tr, X_vl, y_vl, X_te, y_te, soft_labels, epochs=80):
    """蒸馏v1: KL(soft) + CE(hard) + 类别加权"""
    print("\n=== Distill MLP v1 ===")
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_vl_n = scaler.transform(X_vl)
    X_te_n = scaler.transform(X_te)

    X_tr_t = torch.FloatTensor(X_tr_n).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    soft_t = torch.FloatTensor(soft_labels).to(DEVICE)
    X_vl_t = torch.FloatTensor(X_vl_n).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te_n).to(DEVICE)

    model = MLP(43, n_cls).to(DEVICE)
    weight = compute_class_weights(y_tr, n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    best_state, best_acc = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 64):
            idx = perm[i:i+64]
            logits = model(X_tr_t[idx])

            # Hard label CE
            ce_loss = F.cross_entropy(logits, y_tr_t[idx], weight=weight)

            # Soft label KL (separate, no hard label mixing)
            soft_out = logits / 4.0
            soft_target = soft_t[idx] / soft_t[idx].sum(dim=1, keepdim=True).clamp(min=1e-6)
            # Simple KL without complex teacher projection
            kd_loss = F.kl_div(
                F.log_softmax(soft_out, dim=1),
                soft_target,
                reduction='batchmean'
            ) * (4.0 ** 2)

            loss = (1 - 0.7) * ce_loss + 0.7 * kd_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl_t).argmax(1).cpu().numpy()
            vl_acc = (val_pred == y_vl).mean()
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep}: val_acc={vl_acc*100:.1f}%")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(X_te_t).argmax(1).cpu().numpy()
        te_acc = (te_pred == y_te).mean()
    print(f"  Distill v1 test: {te_acc*100:.2f}%")
    return round(te_acc * 100, 2)


def train_distill_v2(X_tr, y_tr, X_vl, y_vl, X_te, y_te, soft_labels, epochs=100):
    """蒸馏v2: 联合训练 + 隐藏层匹配"""
    print("\n=== Distill MLP v2 ===")
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_vl_n = scaler.transform(X_vl)
    X_te_n = scaler.transform(X_te)

    X_tr_t = torch.FloatTensor(X_tr_n).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    soft_t = torch.FloatTensor(soft_labels).to(DEVICE)
    X_vl_t = torch.FloatTensor(X_vl_n).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te_n).to(DEVICE)

    model = MLP(43, n_cls, dropout=0.35).to(DEVICE)
    weight = compute_class_weights(y_tr, n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)

    best_state, best_acc = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 64):
            idx = perm[i:i+64]

            # Forward
            h2 = F.relu(model.bn2(model.fc2(F.relu(model.bn1(model.fc1(X_tr_t[idx]))))))
            h3 = F.relu(model.bn3(model.fc3(h2)))
            logits = model.fc4(model.drop(h3))

            ce_loss = F.cross_entropy(logits, y_tr_t[idx], weight=weight)
            kd_loss = F.kl_div(
                F.log_softmax(logits / 3.0, dim=1),
                soft_t[idx].clamp(min=1e-6),
                reduction='batchmean'
            ) * (3.0 ** 2)
            # Hidden layer alignment: encourage h3 to have class-separating structure
            with torch.no_grad():
                # Create target from soft labels: each row is a soft target for corresponding sample
                soft_target_h3 = soft_t[idx] @ soft_t[idx].T  # (batch, batch)
                soft_target_h3 = soft_target_h3 / soft_target_h3.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # Align sample-wise similarity matrices
            h3_norm = F.normalize(h3, dim=1)
            sim_matrix = h3_norm @ h3_norm.T
            hid_loss = F.mse_loss(sim_matrix, soft_target_h3.detach()) * 0.1

            loss = 0.4 * ce_loss + 0.5 * kd_loss + 0.1 * hid_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl_t).argmax(1).cpu().numpy()
            vl_acc = (val_pred == y_vl).mean()
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep}: val_acc={vl_acc*100:.1f}%")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(X_te_t).argmax(1).cpu().numpy()
        te_acc = (te_pred == y_te).mean()
    print(f"  Distill v2 test: {te_acc*100:.2f}%")
    return round(te_acc * 100, 2)


def train_distill_v3(X_tr, y_tr, X_vl, y_vl, X_te, y_te, soft_labels, epochs=100):
    """蒸馏v3: Focal Loss + 双向KL + 类别加权"""
    print("\n=== Distill MLP v3 ===")
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_vl_n = scaler.transform(X_vl)
    X_te_n = scaler.transform(X_te)

    X_tr_t = torch.FloatTensor(X_tr_n).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    soft_t = torch.FloatTensor(soft_labels).to(DEVICE)
    X_vl_t = torch.FloatTensor(X_vl_n).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te_n).to(DEVICE)

    model = MLP(43, n_cls, dropout=0.3).to(DEVICE)
    weight = compute_class_weights(y_tr, n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)
    gamma = 2.0

    best_state, best_acc = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 64):
            idx = perm[i:i+64]
            logits = model(X_tr_t[idx])

            # Focal CE
            ce = F.cross_entropy(logits, y_tr_t[idx], weight=weight, reduction='none')
            pt = torch.exp(-ce)
            focal_loss = ((1 - pt) ** gamma * ce).mean()

            # Bidirectional KL
            T = 3.0
            soft_stu = F.log_softmax(logits / T, dim=1)
            soft_tea = soft_t[idx].clamp(min=1e-6)
            kd_forward = F.kl_div(soft_stu, soft_tea, reduction='batchmean') * (T ** 2)
            kd_backward = F.kl_div(soft_tea.log(), F.softmax(logits.detach() / T, dim=1), reduction='batchmean') * (T ** 2)
            kd_loss = (kd_forward + kd_backward) / 2

            loss = 0.4 * focal_loss + 0.6 * kd_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl_t).argmax(1).cpu().numpy()
            vl_acc = (val_pred == y_vl).mean()
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep}: val_acc={vl_acc*100:.1f}%")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(X_te_t).argmax(1).cpu().numpy()
        te_acc = (te_pred == y_te).mean()
    print(f"  Distill v3 test: {te_acc*100:.2f}%")
    return round(te_acc * 100, 2)


if __name__ == "__main__":
    print("=" * 60)
    print("  WISDM MLP 训练 (43统计特征)")
    print("=" * 60)

    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_wisdm()

    # 检查软标签是否存在
    has_soft = os.path.exists(SOFT_LABEL_FILE)
    if has_soft:
        soft_labels = np.load(SOFT_LABEL_FILE)
        print(f"  软标签已加载: {soft_labels.shape}, mean={soft_labels.mean():.3f}")
    else:
        print("  ⚠️ 软标签不存在，先运行gen_soft_labels_wisdm.py")
        soft_labels = None

    results = {}

    # Pure MLP
    results['pure_mlp'] = train_pure_mlp(X_tr, y_tr, X_vl, y_vl, X_te, y_te)

    if soft_labels is not None:
        # 确保软标签和训练数据对应
        if len(soft_labels) >= len(X_tr):
            soft_for_train = soft_labels[:len(X_tr)]
        else:
            soft_for_train = soft_labels

        results['v1'] = train_distill_v1(X_tr, y_tr, X_vl, y_vl, X_te, y_te, soft_for_train)
        results['v2'] = train_distill_v2(X_tr, y_tr, X_vl, y_vl, X_te, y_te, soft_for_train)
        results['v3'] = train_distill_v3(X_tr, y_tr, X_vl, y_vl, X_te, y_te, soft_for_train)

    print("\n" + "=" * 60)
    print("  WISDM MLP 结果汇总")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}%")

    # 保存结果
    out = {
        "dataset": "WISDM",
        "num_classes": n_cls,
        "features": 43,
        "train": len(X_tr),
        "val": len(X_vl),
        "test": len(X_te),
        **results
    }
    out_path = "/home/fandy/workplace/thesis/results/wisdm_mlp_results.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ 结果已保存: {out_path}")
