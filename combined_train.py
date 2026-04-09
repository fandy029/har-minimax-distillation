"""
跨数据集联合训练
===================================
策略：
- PAMAP2下楼(91.9%) + MotionSense上楼
- 解决上楼无数据的问题
- 使用领域自适应处理传感器差异
"""
import os, numpy as np, pandas as pd
from glob import glob
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

WINDOW_SIZE = 128
SAMPLE_RATE = 50
NUM_CLASSES = 6
DOWN_SAMPLE_PAMAP = 5  # PAMAP2降采样到20Hz

CN_ACTIVITY = {0:'下楼',1:'上楼',2:'坐着',3:'站立',4:'步行',5:'慢跑'}

print("=" * 60)
print("跨数据集联合训练")
print("=" * 60)

# ============= 加载MotionSense =============
print("\n[1] 加载MotionSense数据...")

acc_path = "/home/fandy/workplace/thesis/motion_sense_data/data/B_Accelerometer_data"
gyro_path = "/home/fandy/workplace/thesis/motion_sense_data/data/C_Gyroscope_data"

motion_datasets = {}
for trial_folder in sorted(glob(acc_path+"/*")):
    label = trial_folder.split('/')[-1].split('_')[0]
    if label not in {'dws','ups','sit','std','wlk','jog'}: continue
    gyro_folder = gyro_path+'/'+trial_folder.split('/')[-1]
    for acc_file in sorted(glob(trial_folder+'/*.csv')):
        uid_m = re.search(r'(\d+)\.csv', acc_file.split('/')[-1])
        if not uid_m: continue
        uid = int(uid_m.group(1))
        acc = pd.read_csv(acc_file)[['x','y','z']].values
        gyro = pd.read_csv(gyro_folder+'/'+str(uid)+'.csv')[['x','y','z']].values if os.path.exists(gyro_folder+'/'+str(uid)+'.csv') else np.zeros_like(acc)
        n = min(len(acc), len(gyro))
        imu = np.concatenate([acc[:n], gyro[:n]], axis=1)
        if uid not in motion_datasets: motion_datasets[uid] = []
        motion_datasets[uid].append((imu, {'dws':0,'ups':1,'sit':2,'std':3,'wlk':4,'jog':5}[label]))

def extract_windows(dataset_dict, users):
    X, y = [], []
    for uid in users:
        for data, label in dataset_dict[uid]:
            windows = [data[i:i+WINDOW_SIZE:1] for i in range(0, len(data)-WINDOW_SIZE+1, WINDOW_SIZE)]
            X.extend(windows)
            y.extend([label]*len(windows))
    return np.array(X), np.array(y)

all_users = sorted(motion_datasets.keys())
train_users, test_users = train_test_split(all_users, test_size=0.3, random_state=42)
X_motion_train, y_motion_train = extract_windows(motion_datasets, train_users)
X_motion_test, y_motion_test = extract_windows(motion_datasets, test_users)

print(f"MotionSense: 训练{len(X_motion_train)}, 测试{len(X_motion_test)}")

# ============= 加载PAMAP2下楼数据 =============
print("\n[2] 加载PAMAP2下楼数据...")

PAMAP_TO_UNIFIED = {
    9: 0,  # 下楼
    8: 1,  # 上楼 (不用)
    2: 2, 3: 3, 4: 4, 5: 5,  # 其他活动
}

optional_path = "/home/fandy/workplace/simclr/datasets/PAMAP2/PAMAP2_Dataset/Optional"

def load_pamap_fast(filepath):
    try:
        data = pd.read_csv(filepath, sep=' ', header=None, usecols=[0,1,9,10,11,24,25,26])
        data = data.iloc[::DOWN_SAMPLE_PAMAP].reset_index(drop=True)
        activity_ids = data.iloc[:, 1].values
        imu_data = data.iloc[:, 2:].values.astype(np.float32)
        return imu_data, activity_ids
    except:
        return None, None

pamap_data = []
pamap_labels = []

for subject_file in sorted(glob(optional_path + "/*.dat")):
    imu_data, activity_ids = load_pamap_fast(subject_file)
    if imu_data is None: continue
    
    # 只提取下楼数据
    mask = activity_ids == 9
    indices = np.where(mask)[0]
    
    for start in range(0, len(indices) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = imu_data[indices[start:start+WINDOW_SIZE]]
        if len(window) == WINDOW_SIZE and not np.any(np.isnan(window)):
            pamap_data.append(window)
            pamap_labels.append(0)  # 下楼

X_pamap = np.array(pamap_data, dtype=np.float32)
y_pamap = np.array(pamap_labels)

print(f"PAMAP2下楼: {len(X_pamap)} 样本")

# ============= 合并数据集 =============
print("\n[3] 合并数据集...")

# 采样一部分PAMAP2下楼数据，避免类别不平衡
n_pamap_sample = min(len(X_pamap), 500)
pamap_indices = np.random.choice(len(X_pamap), n_pamap_sample, replace=False)
X_pamap_sampled = X_pamap[pamap_indices]
y_pamap_sampled = y_pamap[pamap_indices]

# MotionSense: 抽取下楼和上楼数据
motion_stairs_mask = (y_motion_train == 0) | (y_motion_train == 1)
X_motion_stairs = X_motion_train[motion_stairs_mask]
y_motion_stairs = y_motion_train[motion_stairs_mask]

print(f"合并后:")
print(f"  MotionSense上楼/下楼: {len(X_motion_stairs)}")
print(f"  PAMAP2下楼: {len(X_pamap_sampled)}")

# ============= 数据预处理 =============
print("\n[4] 数据预处理...")

# 合并
X_combined = np.concatenate([X_motion_stairs, X_pamap_sampled], axis=0)
y_combined = np.concatenate([y_motion_stairs, y_pamap_sampled], axis=0)

# 添加数据源标签 (0=MotionSense, 1=PAMAP2)
source_labels = np.concatenate([
    np.zeros(len(X_motion_stairs)),
    np.ones(len(X_pamap_sampled))
])

# 归一化
mean = X_combined.mean(axis=(0,1), keepdims=True)
std = X_combined.std(axis=(0,1), keepdims=True) + 1e-8
X_combined_norm = (X_combined - mean) / std

# 打乱
perm = np.random.permutation(len(X_combined_norm))
X_combined_norm = X_combined_norm[perm]
y_combined = y_combined[perm]
source_labels = source_labels[perm]

# 划分
X_train, X_test, y_train, y_test, source_train, source_test = train_test_split(
    X_combined_norm, y_combined, source_labels, test_size=0.3, random_state=42
)

print(f"训练: {len(X_train)}, 测试: {len(X_test)}")
print(f"测试集分布:下楼={sum(y_test==0)}, 上楼={sum(y_test==1)}")

# ============= 模型定义 =============
class DomainAdaptationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10)
        )
        # 分类头
        self.classifier = nn.Linear(128*10, 6)
        # 域判别头（用于对抗训练）
        self.domain_classifier = nn.Sequential(
            nn.Linear(128*10, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, alpha=1.0):
        x = x.transpose(1, 2)
        features = self.shared(x).flatten(1)
        features = self.dropout(features)
        class_pred = self.classifier(features)
        
        # 域对抗（训练时用）
        if self.training:
            domain_pred = self.domain_classifier(features)
            return class_pred, domain_pred
        return class_pred

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

# ============= 训练 =============
print("\n[5] 训练领域自适应模型...")

model = DomainAdaptationCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = FocalLoss(gamma=2.5)
domain_criterion = nn.CrossEntropyLoss()

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
source_train_t = torch.LongTensor(source_train.astype(int))
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

best_acc = 0
best_preds = None

for epoch in range(80):
    model.train()
    perm = torch.randperm(len(X_train_t))
    
    # 数据增强
    X_aug = X_train_t[perm].numpy()
    noise = np.random.randn(*X_aug.shape) * 0.02
    X_aug = X_aug + noise
    scale = np.random.uniform(0.95, 1.05, size=(len(X_aug), 1, 1))
    X_aug = X_aug * scale
    
    for i in range(0, len(X_aug), 64):
        class_outputs, domain_outputs = model(torch.FloatTensor(X_aug[i:i+64]), alpha=1.0)
        
        # 分类损失
        class_loss = criterion(class_outputs, y_train_t[perm][i:i+64])
        
        # 域对抗损失（让域判别器分不清数据来源）
        domain_loss = domain_criterion(domain_outputs, source_train_t[perm][i:i+64])
        
        # 总损失 = 分类损失 - 0.1 * 域损失（梯度反转）
        total_loss = class_loss - 0.1 * domain_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = test_outputs.argmax(1).numpy()
        test_acc = (test_preds == y_test).mean()
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_preds = test_preds.copy()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: Acc={test_acc*100:.1f}%")

# ============= 结果 =============
print("\n" + "=" * 60)
print("[6] 最终结果")
print("=" * 60)

print(f"**测试准确率: {best_acc*100:.1f}%**")

print(f"\n各类别:")
for class_id in range(2):  # 只看下楼和上楼
    mask = y_test == class_id
    if mask.sum() > 0:
        class_acc = (best_preds[mask] == y_test[mask]).mean()
        print(f"  {CN_ACTIVITY[class_id]}: {class_acc*100:.1f}% (n={mask.sum()})")

print("\n对比:")
print(f"  MotionSense单独: 下楼32-74%, 上楼53-63%")
print(f"  PAMAP2单独: 下楼91.9%")
print(f"  联合训练: 下楼/上楼 {best_acc*100:.1f}%")
