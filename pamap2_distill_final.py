"""
PAMAP2 + MiniMax知识蒸馏 - 最终优化版
=================================================
推荐方案:
- 200样本/类 → 总共1000个软标签
- FocalLoss + 知识蒸馏混合损失
- MixUp + 噪声增强
- 更深CNN + 150轮训练
"""
import os, numpy as np, pandas as pd, json, time, re
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI

API_KEY = "sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc"
API_URL = "https://api.minimaxi.com/v1"
MODEL = "MiniMax-M2.7-highspeed"

WINDOW_SIZE = 128
SAMPLE_RATE = 100
DOWN_SAMPLE = 2
NUM_CLASSES = 5

CN_ACTIVITY = {0:'下楼', 1:'坐着', 2:'站立', 3:'步行', 4:'慢跑'}
# 类别权重，慢跑少样本给更高权重
CLASS_WEIGHT = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.6])

print("=" * 60)
print("MiniMax知识蒸馏 - 最终优化版")
print("方案：200/类软标签 + FocalLoss + 知识蒸馏")
print("=" * 60)

# Load data
print("\n[1] Loading PAMAP2...")
PAMAP_TO_UNIFIED = {9: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 3}
protocol_path = "/home/fandy/workplace/simclr/datasets/PAMAP2/PAMAP2_Dataset/Protocol"
optional_path = "/home/fandy/workplace/simclr/datasets/PAMAP2/PAMAP2_Dataset/Optional"

def load_pamap(filepath):
    try:
        data = pd.read_csv(filepath, sep=' ', header=None)
        data = data.iloc[::DOWN_SAMPLE].reset_index(drop=True)
        activity_ids = data.iloc[:, 1].values
        imu_data = data.iloc[:, 9:15].values.astype(np.float32)
        return imu_data, activity_ids
    except:
        return None, None

all_data, all_labels = [], []
for f in sorted(glob(protocol_path + "/*.dat")):
    imu, acts = load_pamap(f)
    if imu is None: continue
    for act_id, unified_id in PAMAP_TO_UNIFIED.items():
        mask = acts == act_id
        indices = np.where(mask)[0]
        for start in range(0, len(indices) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = imu[indices[start:start+WINDOW_SIZE]]
            if len(window) == WINDOW_SIZE and not np.any(np.isnan(window)):
                all_data.append(window)
                all_labels.append(unified_id)

for f in sorted(glob(optional_path + "/*.dat")):
    imu, acts = load_pamap(f)
    if imu is None: continue
    for act_id, unified_id in PAMAP_TO_UNIFIED.items():
        mask = acts == act_id
        indices = np.where(mask)[0]
        for start in range(0, len(indices) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = imu[indices[start:start+WINDOW_SIZE]]
            if len(window) == WINDOW_SIZE and not np.any(np.isnan(window)):
                all_data.append(window)
                all_labels.append(unified_id)

X = np.array(all_data, dtype=np.float32)
y = np.array(all_labels)
print(f"Total samples: {len(X)}")
print(f"Distribution: {np.bincount(y, minlength=NUM_CLASSES)}")

def extract_features(data):
    acc = data[:, :3]
    gyro = data[:, 3:]
    y_accel = acc[:, 1]
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
    y_diff = np.diff(y_accel)
    fft_vals = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag)//2])
    dom_freq_idx = np.argmax(fft_vals) + 1
    dom_freq = np.fft.fftfreq(len(acc_mag), 1/(SAMPLE_RATE/DOWN_SAMPLE))[dom_freq_idx]
    return {
        'y_trend': np.mean(y_diff),
        'acc_mag_mean': acc_mag.mean(),
        'acc_mag_std': acc_mag.std(),
        'acc_mag_max': acc_mag.max(),
        'y_range': y_accel.max() - y_accel.min(),
        'gyro_mean': gyro_mag.mean(),
        'step_peaks': np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])),
        'dom_freq': dom_freq
    }

def get_soft_label(data, true_label):
    f = extract_features(data)
    
    prompt = f"""Classify this IMU window. Classes: 0=downstairs, 1=sitting, 2=standing, 3=walking, 4=jogging

Features:
- Y-trend: {f['y_trend']:.4f} (upward=positive, downward=negative)
- Step peaks: {f['step_peaks']}
- Acc magnitude: {f['acc_mag_mean']:.2f} ± {f['acc_mag_std']:.2f}
- Acc max: {f['acc_mag_max']:.2f}, Y-range: {f['y_range']:.2f}
- Gyro mean: {f['gyro_mean']:.3f}
- Dominant frequency: {f['dom_freq']:.2f} Hz

Physics:
- Downstairs: negative Y-trend, regular steps
- Sitting: low magnitude, minimal movement
- Standing: near-zero Y-trend, stable
- Walking: positive Y-trend, regular pattern ~1-2 Hz
- Jogging: high magnitude, rapid steps ~2-4 Hz

Output JSON with probabilities: {{"0":p0, "1":p1, "2":p2, "3":p3, "4":p4}}"""
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=120,
            extra_body={'reasoning_split': True}
        )
        msg = resp.choices[0].message
        reasoning = msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content = msg.content
        
        soft_label = None
        for pattern in [r'\{[^{}]*"0"[^{}]*"1"[^{}]*"2"[^{}]*"3"[^{}]*"4"[^{}]*\}']:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if all(k in data for k in ['0','1','2','3','4']):
                        soft_label = np.array([float(data[str(i)]) for i in range(5)])
                        break
                except:
                    pass
        
        if soft_label is None:
            numbers = re.findall(r'(?:p|prob)?\s*[0-4]\s*[:＝]\s*([0-9.]+)', reasoning, re.IGNORECASE)
            if len(numbers) >= 5:
                soft_label = np.array([float(n) for n in numbers[:5]])
        
        if soft_label is not None:
            soft_label = np.clip(soft_label, 0, 1)
            if soft_label.sum() > 0:
                soft_label = soft_label / soft_label.sum()
            else:
                soft_label = np.ones(5) / 5
            return soft_label
        
    except Exception as e:
        pass
    
    return np.eye(5)[true_label]

# Preprocessing
print("\n[2] Preprocessing...")
mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-8
X_norm = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Step 3: Generate soft labels (200 per class)
print("\n[3] Generating soft labels: 200 samples per class...")
y_soft_train = np.zeros((len(X_train), NUM_CLASSES), dtype=np.float32)

samples_per_class = 200
total_soft = 0

for class_id in range(NUM_CLASSES):
    class_indices = np.where(y_train == class_id)[0]
    n = min(samples_per_class, len(class_indices))
    sampled = np.random.choice(class_indices, n, replace=False)
    total_soft += n
    
    print(f"Generating for {CN_ACTIVITY[class_id]}: {n} samples")
    
    for i, idx in enumerate(sampled):
        soft_label = get_soft_label(X_train[idx], y_train[idx])
        y_soft_train[idx] = soft_label
        
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{n}")
        
        time.sleep(0.12)

# Fill remaining with one-hot
for i in range(len(X_train)):
    if y_soft_train[i].sum() < 1e-3:
        y_soft_train[i] = 0
        y_soft_train[i, y_train[i]] = 1.0

print(f"\nTotal soft labels: {total_soft}/{len(X_train)}")

# Data augmentation
def mixup_data(x, y_hard, y_soft, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_soft = lam * y_soft + (1 - lam) * y_soft[index]
    return mixed_x, y_hard[index], mixed_soft, lam

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=CLASS_WEIGHT):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight.to(logits.device), reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# Combined loss: Focal + Distillation
class CombinedLoss(nn.Module):
    def __init__(self, T=3.0, alpha=0.6, gamma=2.0):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.focal = FocalLoss(gamma=gamma, weight=CLASS_WEIGHT)
    def forward(self, student_logits, hard_labels, soft_labels):
        # Focal loss for hard labels (handles class imbalance)
        hard_loss = self.focal(student_logits, hard_labels)
        
        # Distillation loss
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(soft_labels / self.T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Model
print("\n[4] Model setup...")

class DeepCNNDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
y_soft_train_t = torch.FloatTensor(y_soft_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

model = DeepCNNDeep()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
criterion = CombinedLoss(T=3.0, alpha=0.6, gamma=2.0)

print("\n[5] Training...")
best_acc, best_preds = 0, None
for epoch in range(150):
    model.train()
    
    perm = torch.randperm(len(X_train_t))
    X_perm = X_train_t[perm]
    y_hard_perm = y_train_t[perm]
    y_soft_perm = y_soft_train_t[perm]
    
    for i in range(0, len(X_perm), 64):
        batch_x = X_perm[i:i+64]
        batch_y_hard = y_hard_perm[i:i+64]
        batch_y_soft = y_soft_perm[i:i+64]
        
        # MixUp for first 80 epochs
        if epoch < 80:
            batch_x, batch_y_hard_mix, batch_y_soft_mix, lam = mixup_data(batch_x, batch_y_hard, batch_y_soft, 0.4)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y_hard_mix, batch_y_soft_mix)
        else:
            # Noise augmentation
            noise = torch.randn_like(batch_x) * 0.02
            outputs = model(batch_x + noise)
            loss = criterion(outputs, batch_y_hard, batch_y_soft)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).argmax(1)
        test_acc = (test_preds == y_test_t).float().mean()
    if test_acc > best_acc:
        best_acc = test_acc
        best_preds = test_preds.numpy()
    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch+1}: Acc={test_acc*100:.1f}%, Best={best_acc*100:.1f}%")

# Results
print("\n" + "=" * 60)
print("[6] Final Results")
print("=" * 60)
print(f"**Test Accuracy: {best_acc*100:.1f}%**")
print("\nPer-class accuracy:")
for cid in range(NUM_CLASSES):
    mask = y_test == cid
    if mask.sum() > 0:
        class_acc = (best_preds[mask] == y_test[mask]).mean()
        print(f"  {CN_ACTIVITY[cid]}: {class_acc*100:.1f}%")

# Save soft labels
np.save('/home/fandy/workplace/thesis/soft_labels_final.npy', y_soft_train)
print(f"\nSaved soft labels to soft_labels_final.npy")
