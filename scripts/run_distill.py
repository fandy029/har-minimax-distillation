#!/usr/bin/env python3
"""
统一蒸馏训练脚本
用法: python run_distill.py <dataset> <version>
dataset: pamap2, kuhar, uci_har, harth, uci_har_new, motionsense, gait, wisdm, motionsense_dm
version: pure_cnn, v1, v2, v3
示例: python run_distill.py pamap2 v3
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import confusion_matrix
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64
EPOCHS_PURE = 300
EPOCHS_KD = 300
SOFT_DIR = BASE_DIR + "/results/soft_labels"
CHECKPOINT_DIR = BASE_DIR + "/results/checkpoints"
LOG_DIR = BASE_DIR + "/results/logs"
HISTORY_DIR = BASE_DIR + "/results/history"

# 创建目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# 蒸馏参数
VERSION_PARAMS = {
    'v1': {'T': 3.0, 'ALPHA': 0.6, 'epochs': 300},
    'v2': {'T': 2.5, 'ALPHA': 0.8, 'epochs': 300},
    'v3': {'T': 1.5, 'ALPHA': 0.85, 'epochs': 300},
}

# ============ 数据增强 ============
def augment_mav(x, kernel_size=5):
    """Moving Average Filter - 平滑信号"""
    if len(x.shape) == 2:  # (T, C)
        kernel = np.ones(kernel_size) / kernel_size
        result = np.zeros_like(x)
        for c in range(x.shape[1]):
            result[:, c] = np.convolve(x[:, c], kernel, mode='same')
        return result
    return x

def augment_jitter(x, sigma=0.03):
    """Jitter - 加高斯噪声"""
    noise = np.random.normal(0, sigma, x.shape)
    return x + noise.astype(x.dtype)

def augment_scale(x, sigma=0.1):
    """Scale - 随机缩放"""
    scale = np.random.normal(1.0, sigma, (1, x.shape[1]) if len(x.shape)==2 else (1,))
    return x * scale

def augment_time_shift(x, shift_max=5):
    """Time Shift - 时间轴微移"""
    if len(x.shape) != 2:
        return x
    T, C = x.shape
    shift = np.random.randint(-shift_max, shift_max + 1)
    if shift == 0:
        return x
    result = np.zeros_like(x)
    if shift > 0:
        result[shift:, :] = x[:-shift, :]
        result[:shift, :] = x[0, :]
    else:
        result[:shift, :] = x[-shift:, :]
        result[shift:, :] = x[0, :]
    return result

def apply_augmentation(x, prob=0.5):
    """对单个样本应用随机数据增强"""
    if np.random.rand() > prob:
        return x
    augs = [lambda x: x,  # no augmentation
            augment_jitter,
            lambda x: augment_scale(x, sigma=0.1),
            lambda x: augment_mav(x, kernel_size=3),
            lambda x: augment_mav(x, kernel_size=5),
            lambda x: augment_time_shift(x, shift_max=3),
    ]
    aug = augs[np.random.randint(0, len(augs))]
    return aug(x)

# 过拟合数据集配置（需要更多增强）
AUG_DATASETS = {'harth', 'gait', 'pamap2'}
AUG_PROB = {'harth': 0.6, 'gait': 0.7, 'pamap2': 0.4}  # 增强概率
WEIGHT_DECAY_OVERRIDE = {'harth': 5e-4, 'gait': 5e-4, 'pamap2': 2e-4}  # 增大学习率

# 数据集配置
DATASET_CONFIG = {
    'pamap2': {
        'classes': 5,
        'cn': ['downstairs', 'sitting', 'standing', 'walking', 'jogging'],
        'loader': 'load_pamap2',
        'channels': 6,
        'max_train': 2200,
    },
    'kuhar': {
        'classes': 18,
        'cn': ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle','Run','Stair-up','Stair-down','Table-tennis'],
        'loader': 'load_kuhar',
        'channels': 8,
        'max_train': 20000,
    },
    'uci_har': {
        'classes': 6,
        'cn': ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING'],
        'loader': 'load_uci_har',
        'channels': 9,
        'max_train': 6000,
    },
    'harth': {
        'classes': 6,
        'cn': ['左立','走路','上楼','下楼','右立','站立'],
        'loader': 'load_harth',
        'channels': 3,
        'max_train': 20000,
    },
    'uci_har_new': {
        'classes': 12,
        'cn': ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING',
               'SIT_TO_STAND','STAND_TO_SIT','SIT_TO_LIE','LIE_TO_SIT','STAND_TO_LIE','LIE_TO_STAND'],
        'loader': 'load_uci_har_new',
        'channels': 9,
        'max_train': 5000,
    },
    'motionsense': {
        'classes': 6,
        'cn': ['downstairs','jogging','sitting','standing','upstairs','walking'],
        'loader': 'load_motionsense',
        'channels': 3,
        'max_train': 11000,
    },
    'gait': {
        'classes': 4,
        'cn': ['type_01','type_02','type_03','type_04'],
        'loader': 'load_gait',
        'channels': 6,
        'max_train': 400,
    },
    'wisdm': {
        'classes': 6,
        'cn': ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing'],
        'loader': 'load_wisdm',
        'channels': 3,
        'max_train': 100000,
    },
    'motionsense_dm': {
        'classes': 6,
        'cn': ['downstairs','jogging','sitting','standing','upstairs','walking'],
        'loader': 'load_motionsense_dm',
        'channels': 3,
        'max_train': 10000,
    },
}

# ============ 数据加载 ============
def load_pamap2():
    base = BASE_DIR + '/datasets/PAMAP2/PAMAP2_Dataset'
    d, l = [], []
    PAMAP_MAP = {9:0, 2:1, 3:2, 4:3, 5:4}
    for folder in ['Protocol', 'Optional']:
        for f in sorted(glob(f"{base}/{folder}/*.dat")):
            try:
                df = pd.read_csv(f, sep=' ', header=None).iloc[::2].reset_index(drop=True)
                imu = df.iloc[:,9:15].values.astype(np.float32)
                acts = df.iloc[:,1].values
                for aid, unlabel in PAMAP_MAP.items():
                    mask = acts == aid
                    idx = np.where(mask)[0]
                    for s in range(0, len(idx)-127, 64):
                        w = imu[idx[s:s+128]]
                        if w.shape[0]==128 and not np.any(np.isnan(w)):
                            d.append(w); l.append(unlabel)
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_kuhar():
    base = BASE_DIR + '/datasets/KuHar/1.Raw_time_domian_data'
    d, l = [], []
    for folder in sorted(glob(f"{base}/*/")):
        label = int(os.path.basename(folder.rstrip("/")).split(".")[0])
        for f in glob(f"{folder}/*.csv"):
            try:
                df = pd.read_csv(f, header=None)
                data = df.values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label)
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_uci_har():
    base = BASE_DIR + '/datasets/UCI_HAR/UCI HAR Dataset'
    X_tr = np.loadtxt(f"{base}/train/X_train.txt").astype(np.float32)
    y_tr = (np.loadtxt(f"{base}/train/y_train.txt")-1).astype(np.int64)
    X_te = np.loadtxt(f"{base}/test/X_test.txt").astype(np.float32)
    y_te = (np.loadtxt(f"{base}/test/y_test.txt")-1).astype(np.int64)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_harth():
    base = BASE_DIR + '/datasets/HARTH/harth'
    files = sorted(glob(f"{base}/*.csv"))
    d, l = [], []
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
    for f in files:
        try:
            df = pd.read_csv(f)
            # Columns: timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
            # Use only back sensor (3 channels) to match config
            x = df.iloc[:, 1:4].values.astype(np.float32)  # back_x, back_y, back_z
            y_ = df.iloc[:, 7].values.astype(int)
            for i in range(0, len(x)-127, 64):
                w = x[i:i+128]
                label = y_[i]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                    d.append(w); l.append(label_map[label])
        except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    # X is (N, 128, 3) = (samples, timesteps, channels) - CNN will transpose to (N, 3, 128)
    if len(X) == 0:
        print("WARNING: HARTH loader found 0 samples!")
        return np.zeros((0,128,3)), np.zeros((0,),dtype=np.int64), np.zeros((0,128,3)), np.zeros((0,),dtype=np.int64), np.zeros((0,128,3)), np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_uci_har_new():
    # UCI_HAR_New has pre-extracted features (561 dims), not time series
    # Use 2D features for MLP
    base = BASE_DIR + '/datasets/UCI_HAR_New'
    X_tr = np.loadtxt(f"{base}/Train/X_train.txt").astype(np.float32)
    y_tr = (np.loadtxt(f"{base}/Train/y_train.txt")-1).astype(np.int64)
    X_te = np.loadtxt(f"{base}/Test/X_test.txt").astype(np.float32)
    y_te = (np.loadtxt(f"{base}/Test/y_test.txt")-1).astype(np.int64)
    # X_tr is (N, 561) - 2D data for MLP
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_motionsense():
    base = BASE_DIR + '/datasets/MotionSense'
    d, l = [], []
    label_map = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
    for folder in sorted(glob(f"{base}/*/")):
        folder_name = folder.rstrip("/").split("/")[-1]
        label_prefix = folder_name.split('_')[0]
        if label_prefix not in label_map: continue
        for f in sorted(glob(f"{folder}/*.csv")):
            try:
                df = pd.read_csv(f)
                # Columns: (index), x, y, z
                data = df.iloc[:, 1:4].values.astype(np.float32)  # x, y, z columns
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label_map[label_prefix])
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    if len(X) == 0:
        print("WARNING: MotionSense loader found 0 samples!")
        return np.zeros((0,3,128)), np.zeros((0,),dtype=np.int64), np.zeros((0,3,128)), np.zeros((0,),dtype=np.int64), np.zeros((0,3,128)), np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_gait():
    base = BASE_DIR + '/datasets/Gait_Classification'
    d, l = [], []
    # label_map: activity label in last column -> class index
    label_map = {1:0, 2:1, 3:2, 4:3}  # Activities 1-4
    for folder in sorted(glob(f"{base}/*/")):
        for f in sorted(glob(f"{folder}/*")):
            if f.endswith('.txt') or 'README' in f: continue
            try:
                df = pd.read_csv(f, header=None)
                # Columns: time, acc_x, acc_y, acc_z, ?, gyro_x, gyro_y, gyro_z, activity
                data = df.values.astype(np.float32)
                # Use columns 1-3 (acc) and 5-7 (gyro) = 6 channels
                acc_gyro = data[:, [1,2,3,5,6,7]]
                labels = data[:, 8].astype(int)
                for i in range(0, len(data)-127, 64):
                    w = acc_gyro[i:i+128]
                    label = labels[i]
                    if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                        d.append(w); l.append(label_map[label])
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    if len(X) == 0:
        print("WARNING: Gait data loader found 0 samples!")
        return np.zeros((0,6,128)), np.zeros((0,),dtype=np.int64), np.zeros((0,6,128)), np.zeros((0,),dtype=np.int64), np.zeros((0,6,128)), np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_wisdm():
    # Load raw time series data (128 time steps, 3 channels) - CNN format
    raw_path = BASE_DIR + '/datasets/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    d, l = [], []
    label_map = {'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':3,'Sitting':4,'Standing':5}
    with open(raw_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.rstrip(';')
            parts = line.split(',')
            if len(parts) != 6:
                continue
            try:
                activity = parts[1].strip()
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                if activity in label_map:
                    d.append([x, y, z])
                    l.append(label_map[activity])
            except:
                continue
    # Sliding window: 128 steps, step 64 (same as gen_soft_labels_unified.py)
    window_size = 128
    step = 64
    X = []
    y_window = []
    n_samples = len(d)
    for start in range(0, n_samples - window_size + 1, step):
        window = d[start:start+window_size]
        window_arr = np.array(window, dtype=np.float32)  # (128, 3)
        X.append(window_arr)
        y_window.append(l[start + window_size // 2])
    X = np.array(X, dtype=np.float32)
    y = np.array(y_window, dtype=np.int64)
    if len(X) > 0:
        print(f"  [WISDM] Loaded {len(X)} windows, shape {X.shape}")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    else:
        print("WARNING: WISDM loader found 0 samples!")
        X_tr, X_vl, y_tr, y_vl, X_te, y_te = X[:0], X[:0], y[:0], y[:0], X[:0], y[:0]
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_motionsense_dm():
    base = BASE_DIR + '/datasets/MotionSense_DeviceMotion/A_DeviceMotion_data'
    d, l = [], []
    label_map = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
    for folder in sorted(glob(f"{base}/*/")):
        folder_name = folder.rstrip("/").split("/")[-1]
        # folder_name like 'dws_1' -> label 'dws'
        label_prefix = folder_name.split('_')[0]
        if label_prefix not in label_map: continue
        for f in sorted(glob(f"{folder}/*.csv")):
            try:
                df = pd.read_csv(f)
                cols = [c for c in df.columns if 'userAcceleration' in c or c in ['x','y','z']]
                if len(cols) >= 3:
                    data = df[cols].values.astype(np.float32)
                    for s in range(0, len(data)-127, 64):
                        w = data[s:s+128]
                        if w.shape[0]==128 and not np.any(np.isnan(w)):
                            d.append(w); l.append(label_map[label_prefix])
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    if len(X) == 0:
        print("WARNING: MotionSense-DM loader found 0 samples!")
        return np.zeros((0,6,128)), np.zeros((0,),dtype=np.int64), np.zeros((0,6,128)), np.zeros((0,),dtype=np.int64), np.zeros((0,6,128)), np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

LOADERS = {
    'pamap2': load_pamap2,
    'kuhar': load_kuhar,
    'uci_har': load_uci_har,
    'harth': load_harth,
    'uci_har_new': load_uci_har_new,
    'motionsense': load_motionsense,
    'gait': load_gait,
    'wisdm': load_wisdm,
    'motionsense_dm': load_motionsense_dm,
}

# ============ 模型定义 ============
class CNN1D(nn.Module):
    def __init__(self, in_ch=6, n_cls=6):
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

class MLP(nn.Module):
    def __init__(self, in_dim=561, n_cls=6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128); self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64); self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, n_cls)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

def focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1-pt)**gamma * ce).mean()

def evaluate(model, X, y, cn):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).cpu().numpy()
    yn = y.numpy() if hasattr(y, 'numpy') else y
    acc = float((preds == yn).mean())
    ca = {}
    for c in range(len(cn)):
        m = yn == c
        if m.sum() > 0: ca[cn[c]] = float((preds[m] == yn[m]).mean())
    cm = confusion_matrix(yn, preds, labels=range(len(cn)))
    return acc, ca, cm

# ============ 主训练流程 ============
def train(dataset, version, resume=False):
    cfg = DATASET_CONFIG[dataset]
    n_cls = cfg['classes']
    cn = cfg['cn']
    in_ch = cfg['channels']
    max_train = cfg.get('max_train', 20000)
    
    # 输出文件路径
    checkpoint_file = f"{CHECKPOINT_DIR}/{dataset}_{version}_best.pt"
    log_file = f"{LOG_DIR}/{dataset}_{version}_train.log"
    history_file = f"{HISTORY_DIR}/{dataset}_{version}_history.json"
    
    print(f"\n{'='*50}")
    print(f"  {dataset.upper()} - {version.upper()}")
    print(f"{'='*50}")
    print(f"  Checkpoint: {checkpoint_file}")
    print(f"  Log: {log_file}")
    print(f"  History: {history_file}")
    
    # 打开日志文件（追加模式 if resume）
    log_f = open(log_file, 'a' if resume else 'w')
    log_f.write(f"\n=== {dataset.upper()} - {version.upper()} ===\n")
    log_f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if resume:
        log_f.write(f"[RESUME MODE]\n")
    log_f.write(f"Dataset config: {cfg}\n\n")
    log_f.flush()
    
    # 加载数据
    loader = LOADERS[dataset]
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = loader()
    print(f"  Train: {len(X_tr)} Val: {len(X_vl)} Test: {len(X_te)}")
    log_f.write(f"Data: Train={len(X_tr)}, Val={len(X_vl)}, Test={len(X_te)}\n")
    
    # 限制训练数据量
    if len(X_tr) > max_train:
        idx = np.random.choice(len(X_tr), max_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"  [Limit] Max train: {max_train}")
        log_f.write(f"[Limit] Max train: {max_train}\n")
    
    # 标准化
    if len(X_tr.shape) == 3:  # 3D数据 (N, T, C)
        mean = X_tr.mean(axis=(0,1), keepdims=True)
        std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
        X_tr_n = (X_tr-mean)/std
        X_vl_n = (X_vl-mean)/std
        X_te_n = (X_te-mean)/std
    else:  # 2D数据 (N, F)
        mean = X_tr.mean(axis=0, keepdims=True)
        std = X_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr_n = (X_tr-mean)/std
        X_vl_n = (X_vl-mean)/std
        X_te_n = (X_te-mean)/std
    
    # 转为tensor
    Xt = torch.FloatTensor(X_tr_n)
    yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl_n)
    yv = torch.LongTensor(y_vl)
    Xte = torch.FloatTensor(X_te_n)
    
    # 选择模型
    if len(X_tr.shape) == 3:
        model = CNN1D(in_ch, n_cls).to(DEVICE)
    else:
        model = MLP(X_tr.shape[1], n_cls).to(DEVICE)
    
    log_f.write(f"Model: {model.__class__.__name__}, in_dim={X_tr.shape[1]}, n_cls={n_cls}\n")
    log_f.flush()
    
    # 软标签
    soft_file = f"{SOFT_DIR}/{dataset}_soft.npy"
    if version == 'pure_cnn':
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        for i, label in enumerate(y_tr): y_soft[i, label] = 1.0
        log_f.write(f"Soft labels: one-hot (pure_cnn mode)\n")
    elif os.path.exists(soft_file):
        y_soft = np.load(soft_file)
        if len(y_soft) > len(X_tr):
            y_soft = y_soft[:len(X_tr)]
            print(f"  [Soft] WARNING: 软标签样本数({np.load(soft_file).shape[0]}) > 训练样本数({len(X_tr)}), 已截断")
            log_f.write(f"Soft labels truncated: {np.load(soft_file).shape[0]} -> {len(X_tr)}\n")
        elif len(y_soft) < len(X_tr):
            print(f"  [Soft] WARNING: 软标签样本数({len(y_soft)}) < 训练样本数({len(X_tr)}), 训练将使用部分软标签")
            log_f.write(f"Soft labels shortage: {len(y_soft)} < {len(X_tr)}\n")
        else:
            print(f"  [Soft] Loaded: {soft_file}")
        log_f.write(f"Soft labels: {soft_file}, shape={y_soft.shape}\n")
    else:
        print(f"  [WARN] Soft labels not found, using one-hot fallback")
        log_f.write(f"Soft labels: NOT FOUND, using one-hot fallback\n")
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        for i, label in enumerate(y_tr): y_soft[i, label] = 1.0
    ys = torch.FloatTensor(y_soft)
    log_f.flush()
    
    # ============ 断点续传恢复 ============
    resume_state = None
    resume_stage = None  # 'stage1', 'stage2', or None
    if resume:
        ckpt_file = f"{CHECKPOINT_DIR}/{dataset}_{version}_best.pt"
        pure_ckpt = f"{CHECKPOINT_DIR}/{dataset}_{version}_pure_cnn_best.pt"
        # 优先加载纯净CNN的断点（包含完整Stage1历史）
        src = pure_ckpt if os.path.exists(pure_ckpt) else (ckpt_file if os.path.exists(ckpt_file) else None)
        if src:
            print(f"  [RESUME] Loading checkpoint: {src}")
            log_f.write(f"[RESUME] Loading checkpoint: {src}\n")
            ckpt = torch.load(src, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            best_val = ckpt.get('val_acc', 0)
            resume_state = ckpt
            resume_stage = ckpt.get('stage', 'stage1')
            print(f"  [RESUME] stage={resume_stage}, best_val={best_val*100:.2f}%")
            log_f.write(f"[RESUME] stage={resume_stage}, best_val={best_val*100:.2f}%\n")
    
    # ============ Stage 1: 纯CNN训练（无蒸馏）============
    do_train_stage1 = True
    if resume_stage == 'stage2':
        print(f"  [SKIP] Stage 1 (already completed, resuming Stage 2)")
        log_f.write("[SKIP] Stage 1 (resuming from Stage 2)\n")
        do_train_stage1 = False
    elif version != 'pure_cnn':
        # v1/v2/v3: 检查是否已有 pure_cnn checkpoint，有则跳过 Stage 1
        candidates = [
            f"{CHECKPOINT_DIR}/{dataset}_{version}_pure_cnn_best.pt",
            f"{CHECKPOINT_DIR}/{dataset}_pure_cnn_best.pt",
        ]
        pure_ckpt = None
        for c in candidates:
            if os.path.exists(c):
                pure_ckpt = c
                break
        if pure_ckpt:
            ckpt = torch.load(pure_ckpt, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            best_val = ckpt.get('val_acc', 0)
            print(f"  [SKIP Stage1] 跳过纯CNN训练，直接用已有权重 (val_acc={best_val*100:.2f}%)")
            print(f"  [LOAD] {os.path.basename(pure_ckpt)} → model.load_state_dict()")
            log_f.write(f"[SKIP Stage1] 跳过纯CNN训练，直接用已有权重 (val_acc={best_val*100:.2f}%)\n")
            log_f.write(f"[LOAD] {os.path.basename(pure_ckpt)}\n")
            do_train_stage1 = False
        else:
            print(f"  [WARN] 未找到pure_cnn checkpoint，将从头训练Stage1")
    else:
        print(f"\n[Stage 1] 纯CNN训练 ({EPOCHS_PURE} epochs, 无蒸馏)...")
        print(f"         用focal_loss训练，不使用软标签...")
        log_f.write(f"\n=== Stage 1: 纯CNN训练 ({EPOCHS_PURE} epochs) ===\n")
    
    # 使用增强weight_decay防止过拟合
    wd = WEIGHT_DECAY_OVERRIDE.get(dataset, 1e-4)
    use_aug = dataset in AUG_DATASETS
    aug_prob = AUG_PROB.get(dataset, 0.0)
    if use_aug:
        print(f"  [Aug] 启用数据增强 (prob={aug_prob}), weight_decay={wd}")
        log_f.write(f"[Aug] 启用数据增强 (prob={aug_prob}), weight_decay={wd}\n")
    
    t1 = time.time()  # 总计时开始（无论Stage1是否跳过）
    if do_train_stage1:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
        best_state = None; best_val = 0.0; best_test = 0
        
        pure_history = resume_state.get('stage1_history', []) if resume_state and resume_stage == 'stage1' else []
        start_ep_pure = resume_state.get('epoch', 0) + 1 if resume_state and resume_stage == 'stage1' else 1
        
        if resume_stage != 'stage2':
            # 恢复optimizer和scheduler状态（如有）
            if resume_state and resume_stage == 'stage1':
                opt.load_state_dict(resume_state['optimizer_state_dict'])
                sch.load_state_dict(resume_state['scheduler_state_dict'])
        
        for ep in range(start_ep_pure, EPOCHS_PURE+1):
            model.train()
            perm = torch.randperm(len(Xt))
            epoch_loss = 0.0; n_batches = 0
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i+BATCH]
                bx = Xt[idx].to(DEVICE)
                bh = yt[idx].to(DEVICE)
                # 对HARTH/Gait/PAMAP2应用数据增强
                if use_aug:
                    bx_np = bx.cpu().numpy()
                    bx_aug = np.array([apply_augmentation(bx_np[j], prob=aug_prob) for j in range(len(bx_np))])
                    bx = torch.FloatTensor(bx_aug).to(DEVICE)
                out = model(bx)
                loss = focal_loss(out, bh)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                epoch_loss += loss.item(); n_batches += 1
            
            sch.step()
            model.eval()
            with torch.no_grad():
                va = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
                train_acc = float((model(Xt[:len(Xv)].to(DEVICE)).argmax(1).cpu().numpy() == yt[:len(Xv)].numpy()).mean())
            
            if va > best_val:
                best_val = va
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                best_test, _, _ = evaluate(model, Xte.to(DEVICE), y_te, cn)
            
            pure_history.append({
                'epoch': ep,
                'train_loss': epoch_loss / n_batches,
                'train_acc': train_acc,
                'val_acc': va,
            })
            
            if ep % 10 == 0:
                print(f"  [S1] ep{ep:>3}: loss={epoch_loss/n_batches:.4f} val={va*100:.1f}%")
                log_f.write(f"[S1] ep{ep:>3}: loss={epoch_loss/n_batches:.4f} val={va*100:.1f}% best_val={best_val*100:.1f}%\n")
                log_f.flush()
        
        # 保存 Pure CNN checkpoint
        model.load_state_dict(best_state)
        torch.save({
            'epoch': EPOCHS_PURE,
            'model_state_dict': best_state,
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sch.state_dict(),
            'val_acc': best_val,
            'test_acc': best_test,
            'stage': 'stage1',
            'stage1_history': pure_history,
        }, f"{CHECKPOINT_DIR}/{dataset}_pure_cnn_best.pt")
        pure_acc, pure_ca, _ = evaluate(model, Xte.to(DEVICE), y_te, cn)
        print(f"  [S1] Final: val_acc={best_val*100:.2f}% | test_acc={pure_acc*100:.2f}%")
        log_f.write(f"\n[S1] Final: val={best_val*100:.2f}% test={pure_acc*100:.2f}%\n")
        
        # Stage 1: 保存每类准确率和混淆矩阵
        _, pure_ca, pure_cm = evaluate(model, Xte.to(DEVICE), y_te, cn)
        np.save(f"{HISTORY_DIR}/{dataset}_pure_cnn_cm.npy", pure_cm)
        with open(f"{HISTORY_DIR}/{dataset}_pure_cnn_class_acc.json", 'w') as fjson:
            json.dump(pure_ca, fjson, indent=2, ensure_ascii=False)
        print(f"  [S1] Confusion matrix saved ({pure_cm.shape})")
        print(f"  [S1] Per-class accuracy saved:")
        for cls_name, cls_acc in pure_ca.items():
            print(f"    {cls_name}: {cls_acc*100:.1f}%")
        log_f.write(f"[S1] Per-class acc: {pure_ca}\n")
    else:
        # Stage 1 已跳过：加载已有checkpoint
        pure_ckpt_file = f"{CHECKPOINT_DIR}/{dataset}_pure_cnn_best.pt"
        if os.path.exists(pure_ckpt_file):
            ckpt = torch.load(pure_ckpt_file, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            best_val = ckpt.get('val_acc', 0)
            best_state = ckpt['model_state_dict']
            pure_history = ckpt.get('stage1_history', [])
        pure_acc, pure_ca, _ = evaluate(model, Xte.to(DEVICE), y_te, cn)
        print(f"  [S1] Loaded: val_acc={best_val*100:.2f}% | test_acc={pure_acc*100:.2f}%")
        log_f.write(f"\n[S1] Loaded: val={best_val*100:.2f}% test={pure_acc*100:.2f}%\n")
        
        # Stage 1 (loaded): 保存每类准确率和混淆矩阵
        _, pure_ca, pure_cm = evaluate(model, Xte.to(DEVICE), y_te, cn)
        np.save(f"{HISTORY_DIR}/{dataset}_pure_cnn_cm.npy", pure_cm)
        with open(f"{HISTORY_DIR}/{dataset}_pure_cnn_class_acc.json", 'w') as fjson:
            json.dump(pure_ca, fjson, indent=2, ensure_ascii=False)
        print(f"  [S1] Confusion matrix saved ({pure_cm.shape})")
        print(f"  [S1] Per-class accuracy saved:")
        for cls_name, cls_acc in pure_ca.items():
            print(f"    {cls_name}: {cls_acc*100:.1f}%")
        log_f.write(f"[S1] Per-class acc: {pure_ca}\n")
    log_f.flush()
    
    result = {
        'dataset': dataset,
        'version': version,
        'num_classes': n_cls,
        'train': len(X_tr),
        'test': len(X_te),
        'pure_cnn': round(pure_acc*100, 2),
        'stage1_history': pure_history,
    }
    
    # ============ Stage 2: Distillation ============
    if version != 'pure_cnn':
        params = VERSION_PARAMS[version]
        T, ALPHA, EPOCHS_KD = params['T'], params['ALPHA'], params['epochs']
        print(f"\n[Stage 2] {version.upper()} 蒸馏训练 ({EPOCHS_KD} epochs, T={T}, alpha={ALPHA})")
        print(f"         用Stage1的权重初始化，用软标签训练...")
        log_f.write(f"\n=== Stage 2: {version.upper()} Distillation (T={T}, alpha={ALPHA}) ===\n")
        
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
        best_ft_state = None; best_ft_val = 0.0; best_ft_test = 0
        t2 = time.time()
        
        kd_history = []
        start_ep_kd = 1
        
        # 从Stage2断点恢复
        if resume_stage == 'stage2' and resume_state:
            model.load_state_dict(resume_state['model_state_dict'])
            opt.load_state_dict(resume_state['optimizer_state_dict'])
            sch.load_state_dict(resume_state['scheduler_state_dict'])
            best_ft_val = resume_state.get('val_acc', 0)
            best_ft_state = resume_state['model_state_dict']
            kd_history = resume_state.get('stage2_history', [])
            start_ep_kd = resume_state.get('epoch', 0) + 1
            print(f"  [RESUME Stage2] from epoch {start_ep_kd}, best_val={best_ft_val*100:.2f}%")
            log_f.write(f"[RESUME Stage2] from epoch {start_ep_kd}\n")
        elif resume_stage == 'stage1':
            # Stage1刚完成，从Stage2从头开始
            best_ft_val = 0.0
        else:
            # Stage1被跳过（checkpoint存在但resume_stage未设置），从头开始Stage2
            best_ft_val = 0.0
        
        for ep in range(start_ep_kd, EPOCHS_KD+1):
            model.train()
            perm = torch.randperm(len(Xt))
            epoch_loss = 0.0; n_batches = 0
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i+BATCH]
                bx = Xt[idx].to(DEVICE); bh = yt[idx].to(DEVICE); bs = ys[idx].to(DEVICE)
                # 对HARTH/Gait/PAMAP2应用数据增强
                if use_aug:
                    bx_np = bx.cpu().numpy()
                    bx_aug = np.array([apply_augmentation(bx_np[j], prob=aug_prob) for j in range(len(bx_np))])
                    bx = torch.FloatTensor(bx_aug).to(DEVICE)
                out = model(bx)
                ce = F.cross_entropy(out, bh, reduction='none'); pt = torch.exp(-ce)
                fl = ((1-pt)**2.0 * ce).mean()
                kl = F.kl_div(F.log_softmax(out/T, dim=1), F.softmax(bs/T, dim=1), reduction='batchmean') * (T**2)
                loss = ALPHA * fl + (1-ALPHA) * kl
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                epoch_loss += loss.item(); n_batches += 1
            
            sch.step()
            model.eval()
            with torch.no_grad():
                va = float((model(Xv.to(DEVICE)).argmax(1).cpu().numpy() == yv.numpy()).mean())
            
            if va > best_ft_val:
                best_ft_val = va
                best_ft_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                best_ft_test, _, _ = evaluate(model, Xte.to(DEVICE), y_te, cn)
            
            kd_history.append({
                'epoch': ep,
                'train_loss': epoch_loss / n_batches,
                'val_acc': va,
            })
            
            if ep % 10 == 0:
                ft_acc, _, _ = evaluate(model, Xte.to(DEVICE), y_te, cn)
                print(f"  [S2] ep{ep:>3}: loss={epoch_loss/n_batches:.4f} val={va*100:.1f}% best={best_ft_val*100:.1f}%")
                log_f.write(f"[S2] ep{ep:>3}: loss={epoch_loss/n_batches:.4f} val={va*100:.1f}% best_val={best_ft_val*100:.1f}%\n")
                log_f.flush()
        
        # 保存蒸馏 checkpoint（含stage标识）
        torch.save({
            'epoch': EPOCHS_KD,
            'model_state_dict': best_ft_state,
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sch.state_dict(),
            'val_acc': best_ft_val,
            'test_acc': best_ft_test,
            'T': T,
            'ALPHA': ALPHA,
            'stage': 'stage2',
            'stage2_history': kd_history,
        }, checkpoint_file)
        print(f"  [S2] {version.upper()} 保存: val_acc={best_ft_val*100:.2f}%")
        
        model.load_state_dict(best_ft_state)
        ft_acc, ft_ca, _ = evaluate(model, Xte.to(DEVICE), y_te, cn)
        print(f"  {version.upper()}: {ft_acc*100:.2f}%")
        log_f.write(f"\n{version.upper()} Final: val={best_ft_val*100:.2f}% test={ft_acc*100:.2f}%\n")
        log_f.flush()
        
        # Stage 2: 保存每类准确率和混淆矩阵
        _, ft_ca, ft_cm = evaluate(model, Xte.to(DEVICE), y_te, cn)
        np.save(f"{HISTORY_DIR}/{dataset}_{version}_cm.npy", ft_cm)
        with open(f"{HISTORY_DIR}/{dataset}_{version}_class_acc.json", 'w') as f:
            json.dump(ft_ca, f, indent=2, ensure_ascii=False)
        print(f"  [{version.upper()}] Confusion matrix saved ({ft_cm.shape})")
        print(f"  [{version.upper()}] Per-class accuracy saved:")
        for cls_name, cls_acc in ft_ca.items():
            print(f"    {cls_name}: {cls_acc*100:.1f}%")
        log_f.write(f"[{version.upper()}] Per-class acc: {ft_ca}\n")
        
        result[f'{version}_kd'] = round(ft_acc*100, 2)
        result['kd_class_acc'] = ft_ca
        result['stage2_history'] = kd_history
    
    # 保存最终结果JSON
    result_file = BASE_DIR + "/results/" + dataset + "_" + version + ".json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Result saved: {result_file}")
    
    # 保存历史JSON
    with open(history_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  ✅ History saved: {history_file}")
    
    total_time = (time.time()-t1)/60
    log_f.write(f"\nTotal time: {total_time:.1f}min\n")
    log_f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.close()
    
    print(f"  Total time: {total_time:.1f}min")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HAR Distillation Training')
    parser.add_argument('dataset', help='Dataset name: pamap2, kuhar, uci_har, harth, uci_har_new, motionsense, gait, wisdm, motionsense_dm')
    parser.add_argument('version', help='Version: pure_cnn, v1, v2, v3')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if exists')
    args = parser.parse_args()
    
    if args.dataset not in DATASET_CONFIG:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)
    if args.version not in ['pure_cnn', 'v1', 'v2', 'v3']:
        print(f"Unknown version: {args.version}")
        print(f"Available: pure_cnn, v1, v2, v3")
        sys.exit(1)
    
    train(args.dataset, args.version, resume=args.resume)
