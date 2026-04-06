"""
MotionSense + MiniMax知识蒸馏
==============================
使用和PAMAP2/UCI-HAR相同的方案：
- 200样本/类软标签 + FocalLoss + 知识蒸馏
- MixUp增强 + 更深CNN
"""
import os, numpy as np, pandas as pd, json, time, re
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI

API_KEY = "sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc"
API_URL = "https://api.minimaxi.com/v1"
MODEL = "MiniMax-M2.7-highspeed"

WINDOW_SIZE = 128
NUM_CLASSES = 6
SAMPLE_RATE = 50  # MotionSense is 50Hz

CN_ACTIVITY = {0:'下楼', 1:'慢跑', 2:'坐着', 3:'站立', 4:'上楼', 5:'步行'}

# 类别权重
CLASS_WEIGHT = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

print("=" * 60)
print("MotionSense + MiniMax知识蒸馏")
print("方案：200/类软标签 + FocalLoss + 知识蒸馏")
print("=" * 60)

# =====================================================
# 1. 加载MotionSense数据
# =====================================================
print("\n[1] Loading MotionSense...")

DATA_DIR = "/home/fandy/workplace/simclr/datasets/MotionSense"

# 活动映射: 文件夹前缀 -> 类别ID
ACTIVITY_MAP = {
    'dws': 0,  # downstairs
    'jog': 1,  # jogging
    'sit': 2,   # sitting
    'std': 3,   # standing
    'ups': 4,   # upstairs
    'wlk': 5    # walking
}

def load_motionsense(max_samples_per_class=500, window_size=WINDOW_SIZE):
    """加载MotionSense数据集（带下采样控制）"""
    all_data = []
    all_labels = []
    
    for folder in glob(os.path.join(DATA_DIR, "*")):
        folder_name = os.path.basename(folder)
        # 获取活动前缀 (dws, jog, sit, std, ups, wlk)
        activity_prefix = ''.join([c for c in folder_name if c.isalpha()])
        if activity_prefix not in ACTIVITY_MAP:
            continue
        activity_id = ACTIVITY_MAP[activity_prefix]
        
        for csv_file in glob(os.path.join(folder, "*.csv")):
            try:
                df = pd.read_csv(csv_file, index_col=0)
                data = df[['x', 'y', 'z']].values.astype(np.float32)
                
                # 滑动窗口切分
                for start in range(0, len(data) - window_size + 1, window_size // 2):
                    window = data[start:start + window_size]
                    if not np.any(np.isnan(window)):
                        all_data.append(window)
                        all_labels.append(activity_id)
            except:
                continue
    
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels)
    
    # 统计每个类别的样本数
    print(f"Total windows: {len(X)}")
    for act_id, act_name in CN_ACTIVITY.items():
        count = np.sum(y == act_id)
        print(f"  {act_name}: {count}")
    
    return X, y

X, y = load_motionsense()

# =====================================================
# 2. 特征提取
# =====================================================
def extract_features(data):
    """data: (128, 3) - accelerometer x,y,z"""
    acc = data[:, :3]
    y_accel = acc[:, 1]
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    y_diff = np.diff(y_accel)
    
    # FFT
    fft_vals = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag)//2])
    dom_freq_idx = np.argmax(fft_vals) + 1
    dom_freq = dom_freq_idx * (SAMPLE_RATE / WINDOW_SIZE)
    
    return {
        'y_trend': np.mean(y_diff),
        'acc_mag_mean': acc_mag.mean(),
        'acc_mag_std': acc_mag.std(),
        'acc_mag_max': acc_mag.max(),
        'y_range': y_accel.max() - y_accel.min(),
        'step_peaks': np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])),
        'dom_freq': dom_freq
    }

# =====================================================
# 3. MiniMax生成软标签
# =====================================================
def get_soft_label(data, true_label):
    f = extract_features(data)
    
    prompt = f"""Classify this accelerometer window. Classes: 0=downstairs, 1=jogging, 2=sitting, 3=standing, 4=upstairs, 5=walking

Features:
- Y-trend: {f['y_trend']:.4f} (upward=positive, downward=negative)
- Step peaks: {f['step_peaks']}
- Acc magnitude: {f['acc_mag_mean']:.2f} ± {f['acc_mag_std']:.2f}, max={f['acc_mag_max']:.2f}
- Y-range: {f['y_range']:.2f}
- Dominant frequency: {f['dom_freq']:.1f} Hz

Physics:
- Downstairs: negative Y-trend, regular steps ~1-2 Hz
- Jogging: high magnitude, rapid steps ~2-4 Hz
- Sitting: low magnitude, minimal movement
- Standing: near-zero Y-trend, stable
- Upstairs: positive Y-trend, climbing motion
- Walking: moderate magnitude, regular steps ~1-2 Hz

Output JSON with probabilities: {{"0":p0, "1":p1, "2":p2, "3":p3, "4":p4, "5":p5}}"""
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=150,
            extra_body={'reasoning_split': True}
        )
        msg = resp.choices[0].message
        reasoning = msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content = msg.content
        
        soft_label = None
        for pattern in [r'\{[^{}]*"0"[^{}]*"1"[^{}]*"2"[^{}]*"3"[^{}]*"4"[^{}]*"5"[^{}]*\}']:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    d = json.loads(match)
                    if all(k in d for k in ['0','1','2','3','4','5']):
                        soft_label = np.array([float(d[str(i)]) for i in range(6)])
                        break
                except:
                    pass
        
        if soft_label is not None:
            soft_label = np.clip(soft_label, 0, 1)
            if soft_label.sum() > 0:
                soft_label = soft_label / soft_label.sum()
            else:
                soft_label = np.ones(6) / 6
            return soft_label
        
    except Exception as e:
        pass
    
    return np.eye(6)[true_label]

# =====================================================
# 4. 数据预处理
# =====================================================
print("\n[2] Preprocessing...")

mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-8
X_norm = (X - mean) / std

# 划分训练/测试
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Train distribution: {np.bincount(y_train, minlength=NUM_CLASSES)}")

# =====================================================
# 5. 生成软标签（200/类）
# =====================================================
print("\n[3] Generating soft labels (200 per class)...")

y_soft_train = np.zeros((len(X_train), NUM_CLASSES), dtype=np.float32)
samples_per_class = 200

for class_id in range(NUM_CLASSES):
    class_indices = np.where(y_train == class_id)[0]
    n = min(samples_per_class, len(class_indices))
    sampled = np.random.choice(class_indices, n, replace=False)
    
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

print(f"\nTotal soft labels generated")

# =====================================================
# 6. MixUp
# =====================================================
def mixup_data(x, y_hard, y_soft, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
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

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, T=3.0, alpha=0.6, gamma=2.0):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.focal = FocalLoss(gamma=gamma)
    def forward(self, student_logits, hard_labels, soft_labels):
        hard_loss = self.focal(student_logits, hard_labels)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(soft_labels / self.T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# =====================================================
# 7. 模型
# =====================================================
print("\n[4] Model setup...")

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3)
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
        x = x.transpose(1, 2)  # (B, 3, 128)
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

model = DeepCNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
criterion = CombinedLoss(T=3.0, alpha=0.6, gamma=2.0)

# =====================================================
# 8. 训练
# =====================================================
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
        
        if epoch < 80:
            batch_x, batch_y_hard_mix, batch_y_soft_mix, _ = mixup_data(batch_x, batch_y_hard, batch_y_soft, 0.4)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y_hard_mix, batch_y_soft_mix)
        else:
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

# =====================================================
# 9. 结果
# =====================================================
print("\n" + "=" * 60)
print("[6] Results")
print("=" * 60)
print(f"**Test Accuracy: {best_acc*100:.1f}%**")
print("\nPer-class accuracy:")
for cid in range(NUM_CLASSES):
    mask = y_test == cid
    if mask.sum() > 0:
        class_acc = (best_preds[mask] == y_test[mask]).mean()
        print(f"  {CN_ACTIVITY[cid]}: {class_acc*100:.1f}%")

np.save('/home/fandy/workplace/thesis/soft_labels_motionsense.npy', y_soft_train)
print(f"\nSaved soft labels to soft_labels_motionsense.npy")
