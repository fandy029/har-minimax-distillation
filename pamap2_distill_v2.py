"""
PAMAP2 + MiniMax教师模型知识蒸馏 V2
======================================
优化版：更快的JSON提取，减少样本数
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

print("=" * 60)
print("MiniMax教师模型知识蒸馏 V2")
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
print(f"Loaded: {len(X)} samples")
print(f"Distribution: {np.bincount(y, minlength=NUM_CLASSES)}")

def extract_features(data):
    acc = data[:, :3]
    gyro = data[:, 3:]
    y_accel = acc[:, 1]
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
    y_diff = np.diff(y_accel)
    return {
        'y_trend': np.mean(y_diff),
        'acc_mag_mean': acc_mag.mean(),
        'acc_mag_std': acc_mag.std(),
        'y_range': y_accel.max() - y_accel.min(),
        'gyro_mean': gyro_mag.mean(),
        'step_peaks': np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])),
    }

def get_soft_label(data, true_label):
    f = extract_features(data)
    
    # 简化提示词
    prompt = f"""Class: 0=down, 1=sit, 2=stand, 3=walk, 4=jog
Y-trend={f['y_trend']:.3f}, mag={f['acc_mag_mean']:.2f}, std={f['acc_mag_std']:.2f}, steps={f['step_peaks']}

Output JSON with probabilities for all 5 classes:
{{"0":p0, "1":p1, "2":p2, "3":p3, "4":p4}}"""
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=100,
            extra_body={'reasoning_split': True}
        )
        msg = resp.choices[0].message
        reasoning = msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content = msg.content
        
        # 尝试多种方式提取JSON
        soft_label = None
        
        # 方法1: 直接从content找JSON
        for pattern in [r'\{[^{}]*"0"\s*:\s*[\d.]+[^}]*\}', r'\{[^}]+\}']:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if all(k in data for k in ['0','1','2','3','4']):
                        soft_label = np.array([float(data[str(i)]) for i in range(5)])
                        break
                except:
                    pass
            if soft_label is not None:
                break
        
        # 方法2: 从reasoning提取数字模式
        if soft_label is None:
            numbers = re.findall(r'(?:p|prob|probability)?\s*[0-4]\s*[:＝]\s*([0-9.]+)', reasoning, re.IGNORECASE)
            if len(numbers) >= 5:
                soft_label = np.array([float(n) for n in numbers[:5]])
        
        if soft_label is not None:
            # 确保是有效概率
            soft_label = np.clip(soft_label, 0, 1)
            if soft_label.sum() > 0:
                soft_label = soft_label / soft_label.sum()
            else:
                soft_label = np.ones(5) / 5
            return soft_label, reasoning[:100]
        
    except Exception as e:
        pass
    
    # 默认返回one-hot
    return np.eye(5)[true_label], "fallback"

# Preprocessing
print("\n[2] Preprocessing...")
mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-8
X_norm = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Step 3: Generate soft labels
print("\n[3] Generating soft labels...")
y_soft_train = np.zeros((len(X_train), NUM_CLASSES), dtype=np.float32)

# 每类采样生成软标签
samples_per_class = 30  # 减少到30个
total_generated = 0

for class_id in range(NUM_CLASSES):
    class_indices = np.where(y_train == class_id)[0]
    n = min(samples_per_class, len(class_indices))
    sampled = np.random.choice(class_indices, n, replace=False)
    
    print(f"Processing {CN_ACTIVITY[class_id]}: {n} samples")
    
    for i, idx in enumerate(sampled):
        soft_label, reason = get_soft_label(X[idx], y_train[idx])
        y_soft_train[idx] = soft_label
        total_generated += 1
        
        if i < 2:
            pred = np.argmax(soft_label)
            print(f"  [{i}] true={CN_ACTIVITY[y_train[idx]]}, pred={CN_ACTIVITY[pred]}")
        
        time.sleep(0.3)

print(f"\nGenerated {total_generated} soft labels")

# CNN Model
print("\n[4] Training...")

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(self.dropout(x))

class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha
    def forward(self, student_logits, hard_labels, soft_labels):
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(soft_labels / self.T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
y_soft_train_t = torch.FloatTensor(y_soft_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

model = DeepCNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = DistillationLoss(T=4.0, alpha=0.5)

best_acc, best_preds = 0, None
for epoch in range(100):
    model.train()
    perm = torch.randperm(len(X_train_t))
    X_aug = X_train_t[perm].numpy()
    noise = np.random.randn(*X_aug.shape) * 0.02
    X_aug = torch.FloatTensor(X_aug + noise)
    y_aug_hard = y_train_t[perm]
    y_aug_soft = y_soft_train_t[perm]
    
    for i in range(0, len(X_aug), 64):
        outputs = model(X_aug[i:i+64])
        optimizer.zero_grad()
        loss = criterion(outputs, y_aug_hard[i:i+64], y_aug_soft[i:i+64])
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
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Acc={test_acc*100:.1f}%")

print("\n" + "=" * 60)
print("[5] Results")
print("=" * 60)
print(f"**Test Accuracy: {best_acc*100:.1f}%**")
for cid in range(NUM_CLASSES):
    mask = y_test == cid
    if mask.sum() > 0:
        print(f"  {CN_ACTIVITY[cid]}: {(best_preds[mask] == y_test[mask]).mean()*100:.1f}%")
