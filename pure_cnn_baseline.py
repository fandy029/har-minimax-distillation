"""
纯CNN基线实验 - 四个数据集对比（精简版）
==========================================
使用相同的CNN架构，但减少训练轮数加快速度
"""
import os, numpy as np, pandas as pd, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from glob import glob

WINDOW_SIZE = 128
EPOCHS = 100  # 减少轮数加快速度

print("=" * 60)
print("纯CNN基线实验 - 四个数据集 (精简版)")
print("=" * 60)

# =====================================================
# 1. 数据集加载
# =====================================================

def load_pamap2():
    """PAMAP2数据集"""
    DATA_DIR = '/home/fandy/workplace/simclr/datasets/PAMAP2/PAMAP2_Dataset/Protocol'
    
    activity_map = {
        1: 0,  # downstairs -> 0
        2: 1,  # sitting -> 1
        3: 2,  # standing -> 2
        4: 3,  # walking -> 3
        5: 4   # jogging -> 4
    }
    
    all_data = []
    all_labels = []
    
    for fname in glob(os.path.join(DATA_DIR, 'subject*.dat')):
        try:
            df = pd.read_csv(fname, sep=' ', header=None)
            # column 1 is activity ID, columns 4-6 is IMU1 accelerometer
            for _, row in df.iterrows():
                activity_id = int(row[1])
                if activity_id not in activity_map:
                    continue
                try:
                    acc = row[[4, 5, 6]].values.astype(np.float32)
                    if np.any(np.isnan(acc)):
                        continue
                    all_data.append(acc)
                    all_labels.append(activity_map[activity_id])
                except:
                    continue
        except:
            continue
    
    # 按时间顺序切窗口
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    # 重排成窗口
    windows = []
    labels = []
    step = WINDOW_SIZE // 2
    for i in range(0, len(X) - WINDOW_SIZE + 1, step):
        window = X[i:i+WINDOW_SIZE]
        if not np.any(np.isnan(window)):
            windows.append(window)
            labels.append(y[i])  # 使用窗口起始的标签
    
    X_windows = np.array(windows, dtype=np.float32)
    y_windows = np.array(labels, dtype=np.int64)
    
    print(f"PAMAP2: {len(X_windows)} windows")
    return X_windows, y_windows, 5

def load_uci_har():
    """UCI-HAR数据集"""
    DATA_DIR = '/home/fandy/workplace/simclr/datasets/UCI_HAR/'
    
    activity_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                      'SITTING', 'STANDING', 'LAYING']
    activity_to_idx = {a: i for i, a in enumerate(activity_names)}
    
    all_data = {'train': [], 'test': []}
    all_labels = {'train': [], 'test': []}
    
    for subset in ['train', 'test']:
        base = os.path.join(DATA_DIR, subset, 'Inertial Signals')
        if not os.path.exists(base):
            continue
        
        signals = ['total_acc_x', 'total_acc_y', 'total_acc_z']
        
        # 加载所有信号
        signal_data = []
        for sig in signals:
            files = sorted(glob(os.path.join(base, f'{sig}_*.txt')))
            if not files:
                continue
            data = []
            for f in files:
                data.append(np.loadtxt(f))
            signal_data.append(np.vstack(data))
        
        if not signal_data:
            continue
        
        # 合并成 (N, 3, 128)
        combined = np.stack([signal_data[0], signal_data[1], signal_data[2]], axis=1)
        
        # 加载标签
        label_file = os.path.join(DATA_DIR, subset, 'y_' + subset + '.txt')
        labels = np.loadtxt(label_file, dtype=int) - 1  # 转0-indexed
        
        for i in range(len(combined)):
            all_data[subset].append(combined[i])
            all_labels[subset].append(labels[i])
    
    X = np.array(all_data['train'] + all_data['test'], dtype=np.float32)
    y = np.array(all_labels['train'] + all_labels['test'], dtype=np.int64)
    
    print(f"UCI-HAR: {len(X)} windows")
    return X, y, 6

def load_motionsense():
    """MotionSense数据集"""
    DATA_DIR = '/home/fandy/workplace/simclr/datasets/MotionSense'
    
    ACTIVITY_MAP = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}
    
    all_data = []
    all_labels = []
    
    for folder in glob(os.path.join(DATA_DIR, '*')):
        folder_name = os.path.basename(folder)
        prefix = ''.join([c for c in folder_name if c.isalpha()])
        if prefix not in ACTIVITY_MAP:
            continue
        act_id = ACTIVITY_MAP[prefix]
        
        for csv_file in glob(os.path.join(folder, '*.csv')):
            try:
                df = pd.read_csv(csv_file, index_col=0)
                data = df[['x', 'y', 'z']].values.astype(np.float32)
                
                for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE // 2):
                    window = data[start:start + WINDOW_SIZE]
                    if not np.any(np.isnan(window)):
                        all_data.append(window)
                        all_labels.append(act_id)
            except:
                continue
    
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"MotionSense: {len(X)} windows")
    return X, y, 6

def load_wisdm():
    """WISDM数据集"""
    DATA_PATH = '/home/fandy/workplace/simclr/datasets/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    
    ACTIVITY_MAP = {'Walking': 0, 'Jogging': 1, 'Upstairs': 2,
                    'Downstairs': 3, 'Sitting': 4, 'Standing': 5}
    
    cleaned_lines = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.endswith(';') is False:
                continue
            parts = line[:-1].split(',')
            if len(parts) != 6:
                continue
            try:
                int(parts[0])
                float(parts[3])
                cleaned_lines.append(parts)
            except:
                continue
    
    df = pd.DataFrame(cleaned_lines, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    
    all_X = []
    all_y = []
    
    for user in df['user'].unique():
        user_df = df[df['user'] == user].sort_index()
        for activity in user_df['activity'].unique():
            if activity not in ACTIVITY_MAP:
                continue
            act_df = user_df[user_df['activity'] == activity]
            if len(act_df) < WINDOW_SIZE:
                continue
            values = act_df[['x', 'y', 'z']].values.astype(np.float32)
            
            for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_SIZE // 2):
                window = values[start:start + WINDOW_SIZE]
                all_X.append(window)
                all_y.append(ACTIVITY_MAP[activity])
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    
    print(f"WISDM: {len(X)} windows")
    return X, y, 6

# =====================================================
# 2. CNN模型
# =====================================================

class DeepCNN(nn.Module):
    def __init__(self, num_classes):
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
        self.fc3 = nn.Linear(64, num_classes)
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

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y[index], lam

# =====================================================
# 3. 训练
# =====================================================

def train_model(X, y, dataset_name, num_classes):
    print(f"\n--- {dataset_name} ---")
    print(f"Train: {len(X)}, Test: ~{len(X)//5}")
    
    # 预处理
    mean = X.mean(axis=(0,1), keepdims=True)
    std = X.std(axis=(0,1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    model = DeepCNN(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train_t))
        X_perm = X_train_t[perm]
        y_perm = y_train_t[perm]
        
        for i in range(0, len(X_perm), 64):
            batch_x = X_perm[i:i+64]
            batch_y = y_perm[i:i+64]
            
            if epoch < 80:
                batch_x, batch_y_mix, _ = mixup_data(batch_x, batch_y, 0.4)
                loss = criterion(model(batch_x), batch_y_mix)
            else:
                noise = torch.randn_like(batch_x) * 0.02
                loss = criterion(model(batch_x + noise), batch_y)
            
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
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: {test_acc*100:.1f}%, Best: {best_acc*100:.1f}%")
    
    print(f"  >>> Final Best: {best_acc*100:.1f}%")
    return best_acc

# =====================================================
# 4. 主程序
# =====================================================

if __name__ == '__main__':
    results = {}
    
    # PAMAP2
    print("\n[1/4] PAMAP2")
    try:
        X, y, nc = load_pamap2()
        results['PAMAP2'] = train_model(X, y, 'PAMAP2', nc)
    except Exception as e:
        print(f"  Error: {e}")
        results['PAMAP2'] = None
    
    # UCI-HAR
    print("\n[2/4] UCI-HAR")
    try:
        X, y, nc = load_uci_har()
        results['UCI-HAR'] = train_model(X, y, 'UCI-HAR', nc)
    except Exception as e:
        print(f"  Error: {e}")
        results['UCI-HAR'] = None
    
    # MotionSense
    print("\n[3/4] MotionSense")
    try:
        X, y, nc = load_motionsense()
        results['MotionSense'] = train_model(X, y, 'MotionSense', nc)
    except Exception as e:
        print(f"  Error: {e}")
        results['MotionSense'] = None
    
    # WISDM
    print("\n[4/4] WISDM")
    try:
        X, y, nc = load_wisdm()
        results['WISDM'] = train_model(X, y, 'WISDM', nc)
    except Exception as e:
        print(f"  Error: {e}")
        results['WISDM'] = None
    
    # 汇总
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Pure CNN':<12} {'+MiniMax KD':<12} {'Delta':<10}")
    print("-" * 50)
    
    kd_results = {
        'PAMAP2': 0.931,
        'UCI-HAR': 0.965,
        'MotionSense': 0.994,
        'WISDM': 0.996
    }
    
    for name, pure_acc in results.items():
        if pure_acc is not None:
            kd_acc = kd_results.get(name, 0)
            delta = (kd_acc - pure_acc) * 100
            print(f"{name:<15} {pure_acc*100:>6.1f}%      {kd_acc*100:>6.1f}%      +{delta:.1f}%")
        else:
            print(f"{name:<15} {'Error':<12}")
