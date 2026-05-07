#!/usr/bin/env python3
"""
统一软标签生成脚本
用法: python gen_soft_labels_unified.py <dataset> [samples_per_class]
示例: python gen_soft_labels_unified.py uci_har 200
"""
import os, sys, json, time, re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ API 配置 (已修复) ============
API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
TEMPERATURE = 0.7
MAX_TOKENS = 5000
SLEEP_SEC = 1.0
DEFAULT_RATIO = 0.40
DEFAULT_LIMIT = 400

# ============ 软标签有效性判断 ============
def is_valid_soft_label(row):
    """判断这一行是否是有效的真软标签（不是one-hot，不是全零）"""
    s = row.sum()
    if s < 0.99:
        return False
    max_val = row.max()
    second_val = np.sort(row)[-2]
    # one-hot: 最大值接近1.0，且第二大值很小（说明是argmax独大）
    if max_val > 0.97 and second_val < 0.20:
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
    base = BASE_DIR + '/datasets/PAMAP2/PAMAP2_Dataset'
    d, l = [], []
    PAMAP_MAP = {1:0, 2:1, 3:2, 4:3, 5:4}  # 1=lying,2=sitting,3=standing,4=walking,5=jogging
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
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
    for f in files:
        try:
            df = pd.read_csv(f)
            # Columns: timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
            back  = df.iloc[:, 1:4].values.astype(np.float32)
            thigh = df.iloc[:, 4:7].values.astype(np.float32)
            x = np.concatenate([back, thigh], axis=1)  # 6 channels
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
    base = BASE_DIR + '/datasets/UCI_HAR_New'
    X_tr = np.loadtxt(f"{base}/Train/X_train.txt").astype(np.float32)
    y_tr = (np.loadtxt(f"{base}/Train/y_train.txt")-1).astype(np.int64)
    X_te = np.loadtxt(f"{base}/Test/X_test.txt").astype(np.float32)
    y_te = (np.loadtxt(f"{base}/Test/y_test.txt")-1).astype(np.int64)
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
    base = BASE_DIR + '/datasets/Gait_Classification'
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
    
    window_size = 128
    step = 64
    X = []
    y_window = np.array(l, dtype=np.int64)
    
    n_samples = len(d)
    for start in range(0, n_samples - window_size + 1, step):
        window = d[start:start+window_size]
        window_arr = np.array(window, dtype=np.float32)
        X.append(window_arr)
    
    X = np.array(X, dtype=np.float32)
    y = np.array([y_window[start + window_size//2] for start in range(0, n_samples - window_size + 1, step)], dtype=np.int64)
    
    if len(X) > 0:
        X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_vl, y_vl, X_te, y_te = X[:0], y[:0], X[:0], y[:0]
    
    print(f'  WISDM raw: {len(d)} time points, {len(X)} train windows, {len(X_vl)} val, {len(X_te)} test')
    return X, y, X_vl, y_vl, X_te, y_te

def load_motionsense_dm():
    base = BASE_DIR + '/datasets/MotionSense_DeviceMotion/A_DeviceMotion_data'
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
Features: acc_mag={acc_mag.mean():.3f} acc_mean={[f"{v:.3f}" for v in acc.mean(axis=0)]}, gyro_mean={[f"{v:.3f}" for v in gyro.mean(axis=0)]}
Physics: walking/jogging=periodic, sitting/standing=minimal motion, downstairs=negative Y pattern
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

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
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

def build_prompt_harth(data, cn, sr=50):
    # data: (128, 6) = back(3) + thigh(3)
    back  = data[:, :3]; thigh = data[:, 3:6]
    back_m  = np.sqrt((back**2).sum(axis=1))
    thigh_m = np.sqrt((thigh**2).sum(axis=1))
    # Gravity direction (low-pass) for orientation
    from scipy.ndimage import uniform_filter1d
    back_g  = uniform_filter1d(back,  10, axis=0)[:, 2]  # back gz
    thigh_g = uniform_filter1d(thigh, 10, axis=0)[:, 2]   # thigh gz
    back_std  = back_m.std()
    thigh_std = thigh_m.std()
    peaks = np.sum((thigh_m[1:-1] > thigh_m[:-2]) & (thigh_m[1:-1] > thigh_m[2:]))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify human activity from back IMU + thigh IMU (128 steps @ 50Hz, 6 channels).
Classes: {", ".join(descs)}
Features (back):  gz={back_g.mean():+.3f} std={back_std:.4f}
Features (thigh): gz={thigh_g.mean():+.3f} std={thigh_std:.4f} peaks={peaks}
Discriminative rules:
  - thigh_gz > +0.60: stairs_down (thigh kicks backward-upward during descent)
  - thigh_gz < -0.60: stairs_up (thigh moves forward-upward)
  - back_gz  > +0.60: lie (lying face-up)
  - thigh_std > 0.25: walk (rhythmic leg motion)
  - else: stand/sit/stand_still (minimal motion, indistinguishable)
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

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
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

def build_prompt_motionsense(data, cn):
    acc = data[:, :3]; acc_mag = np.sqrt((acc**2).sum(axis=1))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from accelerometer data.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f} acc_mean={[f"{v:.3f}" for v in acc.mean(axis=0)]}
Physics: downstairs=negative Y, upstairs=positive Y, walking=jogging=moderate periodic, sitting/standing=stationary
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

def build_prompt_gait(data, cn):
    acc = data[:, :3]; acc_mag = np.sqrt((acc**2).sum(axis=1)); acc_mean = acc.mean(axis=0)
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify gait type from accelerometer data.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f} acc_mean={[f"{v:.3f}" for v in acc_mean]}
Physics: slow_walk=lower freq, normal_walk=regular, standing=minimal motion, activity=diverse
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

def build_prompt_wisdm(data, cn):
    acc = data
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    y_a = acc[:, 1]
    fft_v = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag)//2])
    dom_f = np.fft.fftfreq(len(acc_mag), 1/20)[np.argmax(fft_v)+1] if len(fft_v) > 0 else 0
    peaks = np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:]))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from 128-step (6.4 second) 20Hz accelerometer window.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.2f} y_mean={y_a.mean():.4f}, peaks={peaks}, freq={dom_f:.1f}Hz
Physics: jogging~periodic 2-4Hz, walking~periodic 1-2Hz, up/downstairs~Y-axis vertical bias, sitting/standing~low movement
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

def build_prompt_motionsense_dm(data, cn):
    acc = data[:, :3] if len(data.shape) == 3 else data[:, :3]
    acc_mag = np.sqrt((acc**2).sum(axis=1)) if len(acc.shape) == 2 else np.zeros(len(acc))
    descs = [f'{i}={cn[i]}' for i in range(len(cn))]
    return f'''Classify physical activity from accelerometer data.
Classes: {", ".join(descs)}
Features: acc_mag={acc_mag.mean():.3f}
Physics: walking/jogging/downstairs/upstairs=periodic, sitting/standing=stationary
Output ONLY valid JSON with {len(cn)} probabilities that sum to 1: {{"0":p0,"1":p1,...}}}}'''

PROMPT_BUILDERS = {
    'pamap2': build_prompt_pamap2,
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
        'cn': ['lying', 'sitting', 'standing', 'walking', 'jogging'],
        'samples_per_class': 30,
    },
    'uci_har': {
        'cn': ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING'],
        'samples_per_class': 200,
    },
    'harth': {
        'cn': ['stand','stairs_up','sit','lie','walk','stand_still','stairs_down'],
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

# ============ API 调用（带重试 + 429限流特殊处理） ============
def call_api(prompt, n_cls, max_retries=3, retry_sleep=10):
    import time as time_module
    from openai import OpenAI, RateLimitError
    
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
            
            result = extract_json_probs(content, n_cls)
            
            if result is None and reasoning:
                result = extract_json_probs(reasoning, n_cls)
            
            if result is None and hasattr(msg, 'reasoning_details') and msg.reasoning_details:
                for rd in msg.reasoning_details:
                    result = extract_json_probs(rd.get('text', ''), n_cls)
                    if result is not None:
                        break
            
            if result is not None:
                return result, None
            
            last_error = f"JSON解析失败, content长度={len(content)}"
            if attempt < max_retries:
                wait = retry_sleep
                print(f"\n    WARNING [{attempt+1}/{max_retries+1}] {last_error}, 等待{wait}秒后重试...")
                time_module.sleep(wait)
                continue
            else:
                return None, last_error
                
        except RateLimitError as e:
            last_error = f"429限流: {str(e)[:80]}"
            wait = 600 if attempt < 1 else 1200  # 429: 等600秒/1200秒
            print(f"\n    WARNING [RateLimit] {last_error}, 等待{wait}秒后重试...")
            if attempt < max_retries:
                time_module.sleep(wait)
                continue
            else:
                return None, last_error
        except Exception as e:
            last_error = str(e)
            err_type = type(e).__name__
            # 判断是否为限流相关错误
            is_rate_limit = any(x in last_error.lower() for x in ['429', 'rate', 'limit', 'too many', 'throttle'])
            wait = 60 if is_rate_limit else retry_sleep
            print(f"\n    WARNING [{attempt+1}/{max_retries+1}] {err_type}: {last_error[:80]}, 等待{wait}秒后重试...")
            if attempt < max_retries:
                time_module.sleep(wait)
                continue
            else:
                return None, last_error
    
    return None, last_error

class TimestampedWriter:
    def __init__(self, file):
        self.file = file
        self.buf = ""
    def write(self, msg):
        self.buf += msg
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self.file.write("[" + ts + "] " + line + "\n")
            self.file.flush()
    def flush(self):
        if self.buf:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self.file.write("[" + ts + "] " + self.buf + "\n")
            self.file.flush()
            self.buf = ""
    def close(self):
        self.flush()
        self.file.close()

# ============ 主程序 ============
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python gen_soft_labels_unified.py <dataset> [--ratio {DEFAULT_RATIO}] [--limit {DEFAULT_LIMIT}] [--force]")
        print(f"可用数据集: {list(DATASET_CONFIG.keys())}")
        print(f"  --ratio: 每类采样率，默认 {DEFAULT_RATIO}")
        print(f"  --limit: 每类软标签上限，默认 {DEFAULT_LIMIT}")
        print("  --force: 从头开始（忽略已有进度）")
        sys.exit(1)
    
    dataset = sys.argv[1]
    if dataset not in DATASET_CONFIG:
        print(f"未知数据集: {dataset}")
        print(f"可用: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)
    
    force_restart = '--force' in sys.argv
    
    ratio_override = None
    limit_override = None
    for i, arg in enumerate(sys.argv):
        if arg == '--ratio' and i+1 < len(sys.argv):
            ratio_override = float(sys.argv[i+1])
        if arg == '--limit' and i+1 < len(sys.argv):
            limit_override = int(sys.argv[i+1])

    cfg = DATASET_CONFIG[dataset]
    n_cls = len(cfg['cn'])
    cn = cfg['cn']
    
    LOG_DIR = BASE_DIR + '/results/logs'
    os.makedirs(LOG_DIR, exist_ok=True)
    stdout_log = LOG_DIR + '/gen_' + dataset + '.log'
    stdout_file = open(stdout_log, 'a', buffering=1)
    ts_writer = TimestampedWriter(stdout_file)
    sys.stdout = ts_writer
    sys.stderr = ts_writer
    
    out_file = BASE_DIR + '/results/soft_labels/' + dataset + '_soft.npy'
    err_log = LOG_DIR + '/gen_' + dataset + '_errors.log'
    import datetime
    err_f = open(err_log, 'a')
    err_f.write("\n=== " + dataset + " 软标签生成开始 " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " ===\n")
    err_f.flush()
    
    loader = LOADERS[dataset]
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = loader()
    print("  训练数据: " + str(len(X_tr)) + " 样本")
    
    per_class_counts = {}
    for c in range(n_cls):
        count = int(np.sum(y_tr == c))
        per_class_counts[c] = count
    
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        spc = int(sys.argv[2])
        print("  手动指定每类: " + str(spc) + " (忽略动态计算)")
    else:
        dynamic_spc = {}
        for c, cnt in per_class_counts.items():
            dynamic_spc[c] = min(limit_override or DEFAULT_LIMIT, max(1, int(cnt * (ratio_override or DEFAULT_RATIO))))
        print("  各class样本数: " + str(per_class_counts))
        print(f"  各class软标签({int(DEFAULT_RATIO*100)}%上限{DEFAULT_LIMIT}): " + str(dynamic_spc))
        total_planned = sum(dynamic_spc.values())
        print("  计划总软标签: " + str(total_planned))
        
        print("  各class详情:")
        for c in range(n_cls):
            cnt = per_class_counts[c]
            sp = dynamic_spc[c]
            print("    Class " + str(c) + " (" + str(cn[c]) + "): " + str(cnt) + "样本 → " + str(sp) + "软标签")
    
    print("=" * 60)
    print("  " + dataset.upper() + " 软标签生成 (T=" + str(TEMPERATURE) + ", max_tokens=" + str(MAX_TOKENS) + ")")
    print("=" * 60)
    print("  类别数: " + str(n_cls) + ", 每类软标签: " + str(dynamic_spc))
    
    np.random.seed(42)
    
    if not force_restart and os.path.exists(out_file):
        y_soft = np.load(out_file)
        if y_soft.shape[0] != len(X_tr):
            print("  错误: 形状不匹配！")
            print("     已有文件: " + str(y_soft.shape))
            print("     当前数据: (" + str(len(X_tr)) + ", " + str(n_cls) + ")")
            print("     数据集结构已变（旧loader和新loader不一致），请用 --force 重新开始")
            print("     正确做法: python gen_soft_labels_unified.py " + dataset + " --force")
            sys.exit(1)
        print("  发现已有进度: " + out_file)
        print("     已填充: " + str(np.sum(y_soft.sum(axis=1) > 0)) + "/" + str(len(X_tr)))
    else:
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        if force_restart:
            deleted = []
            # 软标签文件（需删除重新生成）
            if os.path.exists(out_file):
                os.remove(out_file)
                deleted.append(os.path.basename(out_file))
            tmp_file = out_file.replace('.npy', '.tmp.npy')
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                deleted.append(os.path.basename(tmp_file))
            # 日志文件：截断不清除 fd（open+close 保留文件描述符）
            for log_name in ['gen_' + dataset + '.log', 'gen_' + dataset + '_errors.log']:
                log_path = BASE_DIR + '/results/logs/' + log_name
                if os.path.exists(log_path):
                    open(log_path, 'w').close()
                    deleted.append(log_name)
            # 并行版本遗留日志
            for base in ['gen_full_' + dataset, 'gen_full_' + dataset + '_errors']:
                for suffix in ['', '.log']:
                    full_log = BASE_DIR + '/results/logs/' + base + suffix + '.log'
                    if os.path.exists(full_log):
                        os.remove(full_log)
                        deleted.append(os.path.basename(full_log))
            print("  --force 模式：已清除 " + (', '.join(deleted) if deleted else '旧文件') + "，从头开始")
    

    total_target = sum(dynamic_spc.values())
    done = 0
    exhausted = set()
    prompt_builder = PROMPT_BUILDERS[dataset]

    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        target_n = dynamic_spc[c]

        already_done = sum(1 for idx in cidx if is_valid_soft_label(y_soft[idx]))
        real_needed = max(0, target_n - already_done)
        print("\n  Class " + str(c) + " (" + str(cn[c]) + "): 目标" + str(target_n) + " (" + str(already_done) + "已有, " + str(real_needed) + "待生成)", flush=True)
        class_done = 0

        for idx in cidx:
            if is_valid_soft_label(y_soft[idx]):
                continue
            if idx in exhausted:
                continue

            if class_done >= real_needed:
                break

            data = X_tr[idx]
            prompt = prompt_builder(data, cn)

            result, err_msg = call_api(prompt, n_cls)
            if result is None:
                for retry_attempt in range(2):
                    time.sleep(5)
                    result, err_msg = call_api(prompt, n_cls)
                    if result is not None:
                        print("    重试成功 idx=" + str(idx))
                        break
                if result is None:
                    exhausted.add(idx)
                    y_soft[idx, y_tr[idx]] = 1.0
                    print("    WARNING idx=" + str(idx) + " 回退到one-hot: " + str(err_msg))
                    err_f.write("FALLBACK idx=" + str(idx) + " class=" + str(y_tr[idx]) + " err=" + str(err_msg) + "\n")
                    err_f.flush()
                    continue

            MAX_ONEHOT_RETRY = 10
            reject_count = 0
            while not is_valid_soft_label(result):
                reject_count += 1
                if reject_count >= MAX_ONEHOT_RETRY:
                    y_soft[idx, y_tr[idx]] = 1.0
                    print("    WARNING idx=" + str(idx) + " 重试" + str(MAX_ONEHOT_RETRY) + "次仍为one-hot，强制接受")
                    err_f.write("ONEHOT_FORCE idx=" + str(idx) + " class=" + str(y_tr[idx]) + " reject=" + str(reject_count) + "\n")
                    err_f.flush()
                    break
                print("    WARNING idx=" + str(idx) + " one-hot被舍弃，重新生成 (第" + str(reject_count) + "次)")
                err_f.write("ONEHOT_REJECT idx=" + str(idx) + " class=" + str(y_tr[idx]) + " reject=" + str(reject_count) + "\n")
                err_f.flush()
                result, err_msg = call_api(prompt, n_cls)
                if result is None:
                    exhausted.add(idx)
                    y_soft[idx, y_tr[idx]] = 1.0
                    print("    WARNING idx=" + str(idx) + " 重试耗尽，标记为耗尽，设置one-hot")
                    err_f.write("FALLBACK idx=" + str(idx) + " class=" + str(y_tr[idx]) + " err=" + str(err_msg) + "\n")
                    err_f.flush()
                    break

            if is_valid_soft_label(result):
                y_soft[idx] = result
                class_done += 1
                done += 1
                extra = " (" + str(reject_count) + "次重试后)" if reject_count > 0 else ""
                pred = int(np.argmax(result))
                true_l = int(y_tr[idx])
                ok = "✓" if pred == true_l else "✗"
                ent = float(-(result * np.log(np.clip(result, 1e-8, 1))).sum())
                top2 = sorted(enumerate(result), key=lambda x: -x[1])[:2]
                print("    [" + str(done) + "] idx=" + str(idx) + " true=" + str(true_l) + "(" + cn[true_l] + ") pred=" + str(pred) + "(" + cn[pred] + ")[" + ok + "] ent=" + str(round(ent,3)) + " top=[" + str(top2[0][0]) + ":" + str(round(top2[0][1],3)) + "," + str(top2[1][0]) + ":" + str(round(top2[1][1],3)) + "]" + extra)

            if done > 0 and done % 5 == 0:
                tmp_file = out_file.replace('.npy', '.tmp.npy')
                np.save(tmp_file, y_soft)
                os.replace(tmp_file, out_file)
                err_f.flush()
                real_count = int(np.sum([1 for i in range(len(y_soft)) if is_valid_soft_label(y_soft[i])]))
                # 计算当前准确率（仅对有软标签的行）
                valid_rows = np.array([i for i in range(len(y_soft)) if is_valid_soft_label(y_soft[i])])
                if len(valid_rows) > 0:
                    valid_labels = y_tr[valid_rows]
                    valid_preds = np.argmax(y_soft[valid_rows], axis=1)
                    correct = int(np.sum(valid_preds == valid_labels))
                    acc_pct = correct / len(valid_rows) * 100
                    print("  进度: +" + str(done) + "已处理, 累计" + str(real_count) + "真软标签, 准确率=" + str(round(acc_pct,1)) + "%(" + str(correct) + "/" + str(len(valid_rows)) + "), 已保存")
                else:
                    print("  进度: +" + str(done) + "已处理, 累计" + str(real_count) + "真软标签, 已保存")

            time.sleep(SLEEP_SEC)

        # 本类完成，输出准确率统计
        c_valid = np.array([i for i in cidx if is_valid_soft_label(y_soft[i])])
        if len(c_valid) > 0:
            c_correct = int(np.sum(np.argmax(y_soft[c_valid], axis=1) == y_tr[c_valid]))
            c_acc = c_correct / len(c_valid) * 100
            print("  >>> Class " + str(c) + " (" + str(cn[c]) + ") 完成: " + str(len(c_valid)) + "个软标签, 准确率=" + str(round(c_acc,1)) + "%(" + str(c_correct) + "/" + str(len(c_valid)) + ")")
        else:
            print("  >>> Class " + str(c) + " (" + str(cn[c]) + ") 完成: 0个软标签")
        print()

    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0

    tmp_file = out_file.replace('.npy', '.tmp.npy')
    np.save(tmp_file, y_soft)
    os.replace(tmp_file, out_file)
    err_f.write("=== 完成 " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " ===\n")
    err_f.close()
    ts_writer.close()

    real_count = int(np.sum([1 for i in range(len(y_soft)) if is_valid_soft_label(y_soft[i])]))
    print("\n保存到: " + out_file)
    print("  Shape: " + str(y_soft.shape))
    print("  有效软标签: " + str(real_count) + "/" + str(len(y_soft)))
