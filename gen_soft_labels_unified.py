#!/usr/bin/env python3
"""
统一软标签生成脚本
用法: python gen_soft_labels_unified.py <dataset> [samples_per_class]
dataset: pamap2, kuhar, uci_har, harth, uci_har_new, motionsense, gait, wisdm, motionsense_dm
示例: python gen_soft_labels_unified.py uci_har 200
"""
import os, sys, json, time, re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

# ============ API 配置 (已修复) ============
API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
TEMPERATURE = 0.7
MAX_TOKENS = 5000
SLEEP_SEC = 0.12

# ============ 软标签有效性判断 ============
def is_valid_soft_label(row):
    """判断这一行是否是有效的真软标签（不是one-hot，不是全零）"""
    s = row.sum()
    if s < 0.99:
        return False
    max_val = row.max()
    second_val = np.sort(row)[-2]
    # one-hot: 最大值接近1.0，且第二大值很小（说明是argmax独大）
    if max_val > 0.90 and second_val < 0.20:
        return False
    return True


# ============ JSON 解析 (已修复) ============
def extract_json_probs(text, n_cls):
    """从任意文本中 robust 提取概率向量，支持THOUGHT标签和嵌套结构"""
    if not text:
        return None
    # 去掉 <THOUGHT>...</THOUGHT> 包裹层
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    # 去掉所有 XML/HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    
    # 找所有 {...} 块（支持简单嵌套）
    for m in re.finditer(r'\{[^}]+\}', text):
        try:
            block = m.group()
            d = json.loads(block)
            if all(str(k) in d for k in range(n_cls)):
                vals = [float(d[str(k)]) for k in range(n_cls)]
                s = np.clip(np.array(vals), 0, 1)
                if s.sum() > 0:
                    return s / s.sum()
        except:
            pass
    return None

# ============ 数据加载函数 ============
def load_pamap2():
    base = '/home/fandy/workplace/thesis/datasets/PAMAP2/PAMAP2_Dataset'
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
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

def load_kuhar():
    base = '/home/fandy/workplace/thesis/datasets/KuHar/1.Raw_time_domian_data'
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
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

def load_uci_har():
    base = '/home/fandy/workplace/thesis/datasets/UCI_HAR/UCI HAR Dataset'
    X_tr = np.loadtxt(f"{base}/train/X_train.txt").astype(np.float32)
    y_tr = (np.loadtxt(f"{base}/train/y_train.txt")-1).astype(np.int64)
    X_te = np.loadtxt(f"{base}/test/X_test.txt").astype(np.float32)
    y_te = (np.loadtxt(f"{base}/test/y_test.txt")-1).astype(np.int64)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_harth():
    base = '/home/fandy/workplace/thesis/datasets/HARTH/harth'
    files = sorted(glob(f"{base}/*.csv"))
    d, l = [], []
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
    for f in files:
        try:
            df = pd.read_csv(f)
            x = df.iloc[:, 1:4].values.astype(np.float32)
            y_ = df.iloc[:, 7].values.astype(int)
            for i in range(0, len(x)-127, 64):
                w = x[i:i+128]
                label = y_[i]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                    d.append(w); l.append(label_map[label])
        except: pass
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    if len(X)==0: return np.zeros((0,128,3)),np.zeros((0,),dtype=np.int64),np.zeros((0,128,3)),np.zeros((0,),dtype=np.int64),np.zeros((0,128,3)),np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

def load_uci_har_new():
    base = '/home/fandy/workplace/thesis/datasets/UCI_HAR_New'
    X_tr = np.loadtxt(f"{base}/Train/X_train.txt").astype(np.float32)
    y_tr = (np.loadtxt(f"{base}/Train/y_train.txt")-1).astype(np.int64)
    X_te = np.loadtxt(f"{base}/Test/X_test.txt").astype(np.float32)
    y_te = (np.loadtxt(f"{base}/Test/y_test.txt")-1).astype(np.int64)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_motionsense():
    base = '/home/fandy/workplace/thesis/datasets/MotionSense'
    d, l = [], []
    label_map = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
    for folder in sorted(glob(f"{base}/*/")):
        folder_name = folder.rstrip("/").split("/")[-1]
        label_prefix = folder_name.split('_')[0]
        if label_prefix not in label_map: continue
        for f in sorted(glob(f"{folder}/*.csv")):
            try:
                df = pd.read_csv(f)
                data = df.iloc[:, 1:4].values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label_map[label_prefix])
            except: pass
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    if len(X)==0: return np.zeros((0,3,128)),np.zeros((0,),dtype=np.int64),np.zeros((0,3,128)),np.zeros((0,),dtype=np.int64),np.zeros((0,3,128)),np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

def load_gait():
    base = '/home/fandy/workplace/thesis/datasets/Gait_Classification'
    d, l = [], []
    label_map = {1:0, 2:1, 3:2, 4:3}
    for folder in sorted(glob(f"{base}/*/")):
        for f in sorted(glob(f"{folder}/*")):
            if f.endswith('.txt') or 'README' in f: continue
            try:
                df = pd.read_csv(f, header=None)
                acc_gyro = df.iloc[:, [1,2,3,5,6,7]].values.astype(np.float32)
                labels = df.iloc[:, 8].astype(int)
                for i in range(0, len(df)-127, 64):
                    w = acc_gyro[i:i+128]
                    label = labels[i]
                    if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                        d.append(w); l.append(label_map[label])
            except: pass
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    if len(X)==0: return np.zeros((0,6,128)),np.zeros((0,),dtype=np.int64),np.zeros((0,6,128)),np.zeros((0,),dtype=np.int64),np.zeros((0,6,128)),np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

def load_wisdm():
    # Load raw time series data instead of pre-extracted features
    # Raw format: [user],[activity],[timestamp],[x-acceleration],[y-accel],[z-accel];
    raw_path = '/home/fandy/workplace/thesis/datasets/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    d, l = [], []
    label_map = {'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':3,'Sitting':4,'Standing':5}
    
    with open(raw_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove trailing semicolon
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
    
    # Now slide window over sequences
    # Same approach as other datasets: window=128, step=64
    window_size = 128
    step = 64
    X = []
    y = np.array(l, dtype=np.int64)
    
    n_samples = len(d)
    for start in range(0, n_samples - window_size + 1, step):
        window = d[start:start+window_size]
        window_arr = np.array(window, dtype=np.float32)  # (128, 3)
        X.append(window_arr)
    
    X = np.array(X, dtype=np.float32)
    y_window = np.array([y[start + window_size//2] for start in range(0, n_samples - window_size + 1, step)], dtype=np.int64)
    
    if len(X) > 0:
        X, X_te, y, y_te = train_test_split(X, y_window, test_size=0.2, random_state=42, stratify=y_window)
        X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_vl, y_vl, X_te, y_te = X[:0], y[:0], X[:0], y[:0]
    
    print(f'  WISDM raw: {len(d)} time points, {len(X)} train windows, {len(X_vl)} val, {len(X_te)} test')
    return X, y, X_vl, y_vl, X_te, y_te

def load_motionsense_dm():
    base = '/home/fandy/workplace/thesis/datasets/MotionSense_DeviceMotion/A_DeviceMotion_data'
    d, l = [], []
    label_map = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
    for folder in sorted(glob(f"{base}/*/")):
        folder_name = folder.rstrip("/").split("/")[-1]
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
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    if len(X)==0: return np.zeros((0,6,128)),np.zeros((0,),dtype=np.int64),np.zeros((0,6,128)),np.zeros((0,),dtype=np.int64),np.zeros((0,6,128)),np.zeros((0,),dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

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

# ============ Prompt 模板函数 ============
def build_prompt_pamap2(data, cn):
    acc = data[:, :3]; gyro = data[:, 3:6] if data.shape[1] >= 6 else np.zeros_like(acc)
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from IMU sensor data (accelerometer + gyroscope).
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f}±{acc_mag.std():.3f}, acc_mean={[f"{v:.3f}" for v in acc.mean(axis=0)]}, gyro_mean={[f"{v:.3f}" for v in gyro.mean(axis=0)]}
Physics: walking/jogging=periodic, sitting/standing=minimal motion, downstairs=negative Y pattern
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,"3":0.03,"4":0.02}}}}'''

def build_prompt_kuhar(data, cn):
    acc = data[:, :3]; gyro = data[:, 3:6] if data.shape[1] >= 6 else np.zeros_like(acc)
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from IMU sensor data (accelerometer + gyroscope).
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f}±{acc_mag.std():.3f}, acc_mean={[f"{v:.3f}" for v in acc.mean(axis=0)[:3]]}, gyro_mean={[f"{v:.3f}" for v in gyro.mean(axis=0)[:3]]}
Physics: Stand/Sit/Lay=stationary, Walk/Run/Jump=periodic motion, Stair-up/down=vertical pattern
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,...}}}}'''

def build_prompt_uci_har(data, cn):
    vals = np.array(data)
    try:
        means = vals[:9] if len(vals) >= 9 else np.zeros(9)
        stds = vals[40:49] if len(vals) >= 49 else np.zeros(9)
    except:
        means = np.zeros(9); stds = np.zeros(9)
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify HAR from 561-dim pre-extracted sensor features.
Classes: {", ".join(descs)}
Key features: acc_means={[f"{v:.3f}" for v in means[:3]]}, acc_stds={[f"{v:.3f}" for v in stds[:3]]}
Physics: WALKING~periodic, WALKING_UP~posY, WALKING_DOWN~negY, SITTING/STANDING~static, LAYING~supine
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,"3":0.02,"4":0.02,"5":0.01}}}}'''

def build_prompt_harth(data, cn, sr=50):
    acc = data[:, :3]; acc_m = np.sqrt((acc**2).sum(axis=1))
    y_a = acc[:, 1]
    fft_v = np.abs(np.fft.fft(acc_m)[1:len(acc_m)//2])
    dom_f = np.fft.fftfreq(len(acc_m), 1/sr)[np.argmax(fft_v)+1] if len(fft_v) > 0 else 0
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify IMU window. Classes: {", ".join(descs)}
Features: acc_mag={acc_m.mean():.2f}±{acc_m.std():.2f}, y_mean={y_a.mean():.4f}, peaks={np.sum((acc_m[1:-1]>acc_m[:-2])&(acc_m[1:-1]>acc_m[2:]))}, freq={dom_f:.1f}Hz
Physics: upstairs=posY~1Hz, downstairs=negY~1Hz, walk=posY~1-2Hz, jog=high~2-4Hz, sit/stand=low~0Hz
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,...}}}}'''

def build_prompt_uci_har_new(data, cn):
    vals = np.array(data)
    try:
        means = vals[:9] if len(vals) >= 9 else np.zeros(9)
        stds = vals[40:49] if len(vals) >= 49 else np.zeros(9)
    except:
        means = np.zeros(9); stds = np.zeros(9)
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify HAR from 561-dim pre-extracted sensor features (12 classes including transitions).
Classes: {", ".join(descs)}
Key features: acc_means={[f"{v:.3f}" for v in means[:3]]}, acc_stds={[f"{v:.3f}" for v in stds[:3]]}
Static: WALKING/WALKING_UP/WALKING_DOWN/SITTING/STANDING/LAYING
Transitions: STAND_TO_SIT/SIT_TO_STAND/SIT_TO_LIE/LIE_TO_SIT/STAND_TO_LIE/LIE_TO_STAND
Output JSON with probability distribution: {{"0":0.8,"1":0.05,...}}}}'''

def build_prompt_motionsense(data, cn):
    acc = data[:, :3]; acc_mag = np.sqrt((acc**2).sum(axis=1))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from accelerometer data.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f}±{acc_mag.std():.3f}, acc_mean={[f"{v:.3f}" for v in acc.mean(axis=0)]}
Physics: downstairs=negative Y, upstairs=positive Y, walking=jogging=moderate periodic, sitting/standing=stationary
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,...}}}}'''

def build_prompt_gait(data, cn):
    acc = data[:, :3]; acc_mag = np.sqrt((acc**2).sum(axis=1)); acc_mean = acc.mean(axis=0)
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify gait type from accelerometer data.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f}±{acc_mag.std():.3f}, acc_mean={[f"{v:.3f}" for v in acc_mean]}
Physics: slow_walk=lower freq, normal_walk=regular, standing=minimal motion, activity=diverse
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,...}}}}'''

def build_prompt_wisdm(data, cn):
    # Data is (128, 3) time series: x, y, z accelerometer
    acc = data
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    y_a = acc[:, 1]
    # Compute dominant frequency
    fft_v = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag)//2])
    dom_f = np.fft.fftfreq(len(acc_mag), 1/20)[np.argmax(fft_v)+1] if len(fft_v) > 0 else 0
    peaks = np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:]))
    
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from 128-step (6.4 second) 20Hz accelerometer window.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.2f}±{acc_mag.std():.2f}, y_mean={y_a.mean():.4f}, peaks={peaks}, freq={dom_f:.1f}Hz
Physics: jogging~periodic 2-4Hz, walking~periodic 1-2Hz, up/downstairs~Y-axis vertical bias, sitting/standing~low movement, near 0Hz
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,"3":0.03,"4":0.01,"5":0.01}}}}'''

def build_prompt_motionsense_dm(data, cn):
    acc = data[:, :3] if len(data.shape) == 3 else data[:, :3]
    acc_mag = np.sqrt((acc**2).sum(axis=1)) if len(acc.shape) == 2 else np.zeros(len(acc))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from accelerometer data.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f}±{acc_mag.std():.3f} if len(acc_mag) > 0 else "N/A"
Physics: walking/jogging/downstairs/upstairs=periodic, sitting/standing=stationary
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,...}}}}'''

PROMPT_BUILDERS = {
    'pamap2': build_prompt_pamap2,
    'kuhar': build_prompt_kuhar,
    'uci_har': build_prompt_uci_har,
    'harth': build_prompt_harth,
    'uci_har_new': build_prompt_uci_har_new,
    'motionsense': build_prompt_motionsense,
    'gait': build_prompt_gait,
    'wisdm': build_prompt_wisdm,
    'motionsense_dm': build_prompt_motionsense_dm,
}

# ============ 数据集配置 ============
DATASET_CONFIG = {
    'pamap2': {
        'cn': ['downstairs', 'sitting', 'standing', 'walking', 'jogging'],
        'samples_per_class': 30,
    },
    'kuhar': {
        'cn': ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle','Run','Stair-up','Stair-down','Table-tennis'],
        'samples_per_class': 200,
    },
    'uci_har': {
        'cn': ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING'],
        'samples_per_class': 200,
    },
    'harth': {
        'cn': ['左立','走路','上楼','下楼','右立','站立'],
        'samples_per_class': 200,
    },
    'uci_har_new': {
        'cn': ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING','SIT_TO_STAND','STAND_TO_SIT','SIT_TO_LIE','LIE_TO_SIT','STAND_TO_LIE','LIE_TO_STAND'],
        'samples_per_class': 129,
    },
    'motionsense': {
        'cn': ['downstairs','jogging','sitting','standing','upstairs','walking'],
        'samples_per_class': 200,
    },
    'gait': {
        'cn': ['慢速走','正常走','站立','活动'],
        'samples_per_class': 29,
    },
    'wisdm': {
        'cn': ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing'],
        'samples_per_class': 200,
    },
    'motionsense_dm': {
        'cn': ['downstairs','jogging','sitting','standing','upstairs','walking'],
        'samples_per_class': 200,
    },
}

# ============ API 调用（带重试） ============
def call_api(prompt, n_cls, max_retries=1, retry_sleep=10):
    """
    调用 MiniMax API，失败时自动重试
    - 限流/网络错误/超时: 等待60秒后重试
    - 返回非JSON/空: 等待30秒后重试
    - 最多重试 max_retries 次
    """
    import time as time_module
    from openai import OpenAI
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            c = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=120.0)
            r = c.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                extra_body={'reasoning_split': True}
            )
            msg = r.choices[0].message
            content = msg.content or ''
            reasoning = getattr(msg, 'reasoning_content', None) or ''
            
            # 优先从 content 提取 JSON
            result = extract_json_probs(content, n_cls)
            
            # 备用：从 reasoning_content 提取（当 content 为空时）
            if result is None and reasoning:
                result = extract_json_probs(reasoning, n_cls)
            
            # 备用：从 reasoning_details 提取
            if result is None and hasattr(msg, 'reasoning_details') and msg.reasoning_details:
                for rd in msg.reasoning_details:
                    result = extract_json_probs(rd.get('text', ''), n_cls)
                    if result is not None:
                        break
            
            if result is not None:
                return result, None  # (result, error_msg)
            
            # JSON 解析失败
            last_error = f"JSON解析失败, content长度={len(content)}"
            if attempt < max_retries:
                wait = retry_sleep  # 使用参数化等待时间
                print(f"\n    ⚠️  [{attempt+1}/{max_retries+1}] {last_error}, 等待{wait}秒后重试...")
                time_module.sleep(wait)
                continue
            else:
                return None, last_error
                
        except Exception as e:
            last_error = str(e)
            err_type = type(e).__name__
            wait = retry_sleep  # 使用参数化等待时间
            print(f"\n    ⚠️  [{attempt+1}/{max_retries+1}] {err_type}: {last_error[:80]}, 等待{wait}秒后重试...")
            if attempt < max_retries:
                time_module.sleep(wait)
                continue
            else:
                return None, last_error
    
    return None, last_error

# ============ 主程序 ============
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python gen_soft_labels_unified.py <dataset> [samples_per_class] [--force]")
        print(f"可用数据集: {list(DATASET_CONFIG.keys())}")
        print("  --force: 从头开始（忽略已有进度）")
        sys.exit(1)
    
    dataset = sys.argv[1]
    if dataset not in DATASET_CONFIG:
        print(f"未知数据集: {dataset}")
        print(f"可用: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)
    
    force_restart = '--force' in sys.argv
    
    cfg = DATASET_CONFIG[dataset]
    n_cls = len(cfg['cn'])
    cn = cfg['cn']
    
    out_file = f"/home/fandy/workplace/thesis/results/soft_labels/{dataset}_soft.npy"
    err_log = f"/home/fandy/workplace/thesis/results/logs/gen_{dataset}_errors.log"
    import datetime
    err_f = open(err_log, 'a')
    err_f.write(f"\n=== {dataset} 软标签生成开始 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    err_f.flush()
    
    loader = LOADERS[dataset]
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = loader()
    print(f"  训练数据: {len(X_tr)} 样本")
    
    # ============ 动态计算每类软标签数量 ============
    # 逻辑：每类样本数 × 25%，上限200
    per_class_counts = {}
    for c in range(n_cls):
        count = int(np.sum(y_tr == c))
        per_class_counts[c] = count
    
    if len(sys.argv) > 2 and sys.argv[2] != '--force':
        # 用户手动指定每类数量
        spc = int(sys.argv[2])
        print(f"  手动指定每类: {spc} (忽略动态计算)")
    else:
        # 动态计算：每类样本数 × 25%，上限 200
        dynamic_spc = {}
        for c, cnt in per_class_counts.items():
            dynamic_spc[c] = min(200, max(1, int(cnt * 0.25)))
        print(f"  各class样本数: {per_class_counts}")
        print(f"  各class软标签(25%上限200): {dynamic_spc}")
        total_planned = sum(dynamic_spc.values())
        print(f"  计划总软标签: {total_planned}")
        
        # 打印每类的详情
        print(f"  各class详情:")
        for c in range(n_cls):
            cnt = per_class_counts[c]
            sp = dynamic_spc[c]
            print(f"    Class {c} ({cn[c]}): {cnt}样本 → {sp}软标签")
    
    print("=" * 60)
    print(f"  {dataset.upper()} 软标签生成 (T={TEMPERATURE}, max_tokens={MAX_TOKENS})")
    print("=" * 60)
    print(f"  类别数: {n_cls}, 每类软标签: {dynamic_spc}")
    
    # ============ 断点续传逻辑 ============
    # 固定随机种子确保每次采样顺序一致
    np.random.seed(42)
    
    if not force_restart and os.path.exists(out_file):
        y_soft = np.load(out_file)
        if y_soft.shape[0] != len(X_tr):
            print(f"  ❌ 形状不匹配！")
            print(f"     已有文件: {y_soft.shape}")
            print(f"     当前数据: ({len(X_tr)}, {n_cls})")
            print(f"     数据集结构已变（旧loader和新loader不一致），请用 --force 重新开始")
            print(f"     正确做法: python gen_soft_labels_unified.py {dataset} --force")
            sys.exit(1)
        print(f"  📂 发现已有进度: {out_file}")
        print(f"     已填充: {np.sum(y_soft.sum(axis=1) > 0)}/{len(X_tr)}")
    else:
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        if force_restart:
            deleted = []
            if os.path.exists(out_file):
                os.remove(out_file)
                deleted.append(os.path.basename(out_file))
            err_log = f"/home/fandy/workplace/thesis/results/logs/gen_{dataset}_errors.log"
            if os.path.exists(err_log):
                os.remove(err_log)
                deleted.append(os.path.basename(err_log))
            if deleted:
                print(f"  🔄 --force 模式：已删除 {', '.join(deleted)}，从头开始")
            else:
                print("  🔄 --force 模式：从头开始")
    
    prompt_builder = PROMPT_BUILDERS[dataset]
    
    total_target = sum(dynamic_spc.values())
    done = 0  # 成功生成的软标签数（受target限制）
    
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        target_n = dynamic_spc[c]
        
        # 统计本类已有真软标签数量
        already_done = sum(1 for idx in cidx if is_valid_soft_label(y_soft[idx]))
        real_needed = max(0, target_n - already_done)
        print(f"\n  Class {c} ({cn[c]}): 目标{target_n} ({already_done}已有, {real_needed}待生成)", flush=True)
        class_done = 0  # 本类成功生成的软标签数
        
        for idx in cidx:
            # 已有真软标签 → 跳过（不占目标名额）
            if is_valid_soft_label(y_soft[idx]):
                continue
            
            # 本类已达到目标数 → 停止处理该类剩余样本（避免无效遍历one-hot）
            if class_done >= real_needed:
                break
            
            data = X_tr[idx]
            prompt = prompt_builder(data, cn)
            
            # 先调一次API，失败则重试最多3次，每次等15秒
            result, err_msg = call_api(prompt, n_cls)
            if result is None:
                for retry_attempt in range(2):
                    time.sleep(5)
                    result, err_msg = call_api(prompt, n_cls)
                    if result is not None:
                        print(f"    🔄 第{retry_attempt+1}次重试成功 idx={idx}")
                        break
                if result is None:
                    # 3次重试都失败，回退到one-hot
                    y_soft[idx, y_tr[idx]] = 1.0
                    status = f"❌FALLBACK(重试耗尽,{err_msg[:20] if err_msg else 'unknown'})"
                    print(f"    ⚠️ idx={idx} 回退到one-hot: {err_msg}")
                    err_f.write(f"FALLBACK idx={idx} class={y_tr[idx]} err={err_msg}\n")
                    err_f.flush()
                    done += 1  # one-hot 也计为已处理
                else:
                    y_soft[idx] = result
                    is_onehot = (result > 0.99).sum() == 1
                    status = "⚠️ONEHOT" if is_onehot else "✅REAL"
                    if is_onehot:
                        err_f.write(f"ONEHOT  idx={idx} class={y_tr[idx]} vals={result.tolist()}\n")
                        err_f.flush()
                        done += 1  # one-hot计为已处理
                    else:
                        class_done += 1
                        done += 1
            else:
                y_soft[idx] = result
                is_onehot = (result > 0.99).sum() == 1
                status = "⚠️ONEHOT" if is_onehot else "✅REAL"
                if is_onehot:
                    err_f.write(f"ONEHOT  idx={idx} class={y_tr[idx]} vals={result.tolist()}\n")
                    err_f.flush()
                    done += 1  # one-hot计为已处理
                else:
                    class_done += 1
                    done += 1
            
            # 每处理1个就打印状态
            print(f"    [{done}] idx={idx}: {status} | sum={y_soft[idx].sum():.3f} | max={y_soft[idx].max():.3f}", flush=True)
            
            if done % 5 == 0:
                # 强制刷盘
                base = out_file[:-4]
                tmp_file = f"{base}.tmp"
                np.save(tmp_file, y_soft.copy())
                os.replace(f"{base}.tmp.npy", out_file)
                err_f.flush()
                os.fsync(err_f.fileno())
                # 统计
                real_count = int(np.sum([1 for i in range(len(y_soft)) if is_valid_soft_label(y_soft[i])]))
                onehot_count = int(np.sum([1 for i in range(len(y_soft)) if (y_soft[i] > 0.99).sum() == 1 and y_soft[i].sum() > 0.99]))
                print(f"  >>> 进度: +{done}已处理, 累计{real_count}真软标签, one-hot={onehot_count}, 已保存")
            
            time.sleep(SLEEP_SEC)
        print()
    
    # 确保完全未填充的样本回退到 one-hot（兜底）
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0
    
    np.save(out_file, y_soft)
    err_f.write(f"=== 完成 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    err_f.close()
    err_f = open(err_log, 'a')
    err_f.write(f"[FINAL] 完成写入 {out_file}\n")
    err_f.close()
    
    soft_mean = y_soft.mean()
    soft_max = y_soft.max(axis=1).mean()
    entropy = -np.sum(y_soft * np.log(y_soft + 1e-10), axis=1).mean()
    print(f"\n✅ 保存到: {out_file}")
    print(f"   Shape: {y_soft.shape}")
    print(f"   mean={soft_mean:.3f}, mean_max={soft_max:.3f}, mean_entropy={entropy:.3f}")
