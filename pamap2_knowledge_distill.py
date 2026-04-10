"""
PAMAP2 + MiniMax教师模型知识蒸馏
===================================
MiniMax分析原始数据 → 生成软标签(概率分布) → CNN学习
"""
import os, numpy as np, pandas as pd, json, time, re
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI

# MiniMax API配置
API_KEY = "sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc"
API_URL = "https://api.minimaxi.com/v1"
MODEL = "MiniMax-M2.7-highspeed"

WINDOW_SIZE = 128
SAMPLE_RATE = 100
DOWN_SAMPLE = 2
NUM_CLASSES = 5

CN_ACTIVITY = {0:'下楼', 1:'坐着', 2:'站立', 3:'步行', 4:'慢跑'}

print("=" * 60)
print("MiniMax教师模型知识蒸馏")
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

# Extract features for LLM
def extract_features(data):
    acc = data[:, :3]
    gyro = data[:, 3:]
    y_accel = acc[:, 1]
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
    y_diff = np.diff(y_accel)
    
    # 计算各周期的特征变化
    n_segments = 4
    segment_len = len(data) // n_segments
    y_trends = [np.mean(y_diff[i*segment_len:(i+1)*segment_len]) for i in range(n_segments)]
    
    return {
        'y_trend': np.mean(y_diff),
        'y_trends': y_trends,
        'acc_mag_mean': acc_mag.mean(),
        'acc_mag_std': acc_mag.std(),
        'acc_mag_max': acc_mag.max(),
        'acc_mag_min': acc_mag.min(),
        'y_max': y_accel.max(),
        'y_min': y_accel.min(),
        'y_range': y_accel.max() - y_accel.min(),
        'gyro_mean': gyro_mag.mean(),
        'gyro_std': gyro_mag.std(),
        'step_peaks': np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])),
        'zero_crossings': np.sum(np.diff(np.sign(y_accel)) != 0),
    }

# MiniMax生成软标签
def get_soft_labels_from_minimax(data, true_label, batch_size=20, max_retries=2):
    """
    让MiniMax分析多个候选活动，给出概率分布
    """
    f = extract_features(data)
    
    # 构造详细的分析请求
    prompt = f"""Analyze this IMU sensor window and give probability distribution over activity classes.

Activity Classes:
- 0: downstairs (going down stairs)
- 1: sitting (seated)
- 2: standing (upright, no movement)
- 3: walking (level ground walking)
- 4: jogging (running)

Sensor Features:
- Y-acceleration trend: {f['y_trend']:.4f} (positive=upward, negative=downward)
- Y-acceleration range: {f['y_range']:.3f} (max-min)
- Step/peaks count: {f['step_peaks']}
- Zero crossings: {f['zero_crossings']}
- Acceleration magnitude: mean={f['acc_mag_mean']:.3f}, std={f['acc_mag_std']:.3f}
- Gyroscope magnitude: mean={f['gyro_mean']:.3f}, std={f['gyro_std']:.3f}
- Y-trend per quarter: {[f'{x:.4f}' for x in f['y_trends']]}

Think step by step about which activities match these features.
Then output your probability distribution as JSON with all 5 classes:
{{"0": probability, "1": probability, "2": probability, "3": probability, "4": probability}}

The probabilities should sum to 1.0. Be careful to give reasonable probabilities for ALL classes, not just the obvious ones."""
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=120.0)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=300,
            extra_body={'reasoning_split': True}
        )
        msg = resp.choices[0].message
        reasoning = msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content = msg.content
        
        # 尝试从content提取JSON
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                probs = json.loads(json_str)
                # 转换为概率向量
                soft_label = np.zeros(NUM_CLASSES)
                for k, v in probs.items():
                    if k.isdigit() and int(k) < NUM_CLASSES:
                        soft_label[int(k)] = max(0, min(1, float(v)))
                # 归一化
                soft_label = soft_label / (soft_label.sum() + 1e-8)
                return soft_label, reasoning
        except:
            pass
        
        # 如果JSON解析失败，从reasoning提取
        # 查找类似 "0: 0.1", "1: 0.2" 等模式
        soft_label = np.ones(NUM_CLASSES) / NUM_CLASSES  # 默认均匀分布
        for i in range(NUM_CLASSES):
            pattern = rf'(?<![0-9]){i}(?![0-9])\s*[:：]\s*([0-9.]+)'
            match = re.search(pattern, reasoning)
            if match:
                soft_label[i] = max(0, min(1, float(match.group(1))))
        soft_label = soft_label / (soft_label.sum() + 1e-8)
        return soft_label, reasoning
        
    except Exception as e:
        return None, str(e)

# Preprocessing
print("\n[2] Preprocessing...")
mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-8
X_norm = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Step 3: Generate soft labels from MiniMax
print("\n[3] Generating soft labels from MiniMax...")
y_soft_train = np.zeros((len(X_train), NUM_CLASSES), dtype=np.float32)
y_hard_train = y_train.copy()

# 对训练集的每个样本生成软标签
# 为了效率，每个类采样一定数量
samples_per_class = 50  # 每类50个样本生成软标签
total_generated = 0

for class_id in range(NUM_CLASSES):
    class_indices = np.where(y_train == class_id)[0]
    n_samples = min(samples_per_class, len(class_indices))
    sampled = np.random.choice(class_indices, n_samples, replace=False)
    
    print(f"Generating soft labels for {CN_ACTIVITY[class_id]}: {n_samples} samples")
    
    for i, idx in enumerate(sampled):
        if i % 10 == 0:
            print(f"  {i}/{n_samples}", flush=True)
        
        soft_label, reasoning = get_soft_labels_from_minimax(X[idx], y_train[idx])
        
        if soft_label is not None:
            y_soft_train[idx] = soft_label
            total_generated += 1
            
            if i < 3:  # 打印前几个样本的分布
                pred_class = np.argmax(soft_label)
                print(f"    Sample {idx}: true={CN_ACTIVITY[y_train[idx]]}, pred={CN_ACTIVITY[pred_class]}")
                print(f"    Soft: {[f'{x:.2f}' for x in soft_label]}")
        else:
            # 如果失败，使用硬标签的one-hot
            y_soft_train[idx] = 0
            y_soft_train[idx, y_train[idx]] = 1.0
        
        time.sleep(0.2)  # 避免API过载

print(f"\nTotal soft labels generated: {total_generated}/{len(X_train)}")

# 对没有生成软标签的样本使用硬标签的one-hot
for i in range(len(X_train)):
    if y_soft_train[i].sum() == 0:
        y_soft_train[i] = 0
        y_soft_train[i, y_train[i]] = 1.0

# Step 4: CNN Training with Knowledge Distillation
print("\n[4] Training CNN with Knowledge Distillation...")

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
    """知识蒸馏损失 = 硬标签交叉熵 + 软标签KL散度"""
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_logits, hard_labels, soft_labels):
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        
        # 软标签损失 (KL散度)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(soft_labels / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
y_soft_train_t = torch.FloatTensor(y_soft_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

model = DeepCNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = DistillationLoss(temperature=4.0, alpha=0.5)

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

# Step 5: Results
print("\n" + "=" * 60)
print("[5] Results")
print("=" * 60)
print(f"**Test Accuracy: {best_acc*100:.1f}%**")
print("\nPer-class accuracy:")
for cid in range(NUM_CLASSES):
    mask = y_test == cid
    if mask.sum() > 0:
        class_acc = (best_preds[mask] == y_test[mask]).mean()
        print(f"  {CN_ACTIVITY[cid]}: {class_acc*100:.1f}%")
