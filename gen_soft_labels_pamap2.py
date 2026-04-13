"""
生成PAMAP2 MiniMax软标签
每类150样本，5类共750次API调用
"""
import os, sys, json, time, re
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
SAMPLES_PER_CLASS = 30
cn = ['downstairs', 'sitting', 'standing', 'walking', 'jogging']
n_cls = 5

def load_pamap2():
    cache_file = '/home/fandy/workplace/thesis/cache/pamap2_full.npz'
    if os.path.exists(cache_file):
        print(f'  Loading from cache: {cache_file}')
        data = np.load(cache_file)
        return data['X_tr'], data['y_tr'], data['X_vl'], data['y_vl'], data['X_te'], data['y_te']
    
    base = '/home/fandy/workplace/thesis/datasets/PAMAP2/PAMAP2_Dataset'
    d, l = [], []
    PAMAP_MAP = {9:0, 2:1, 3:2, 4:3, 5:4}
    for folder in ['Protocol', 'Optional']:
        for f in sorted(glob(f"{base}/{folder}/*.dat")):
            try:
                df = pd.read_csv(f, sep=' ', header=None).iloc[::2].reset_index(drop=True)
                imu = df.iloc[:,9:15].values.astype(np.float32)  # accelerometer + magnetometer
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
    
    # Save cache
    np.savez(cache_file, X_tr=X_tr, y_tr=y_tr, X_vl=X_vl, y_vl=y_vl, X_te=X_te, y_te=y_te)
    print(f'  Saved cache to: {cache_file}')
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def get_soft_label(data, true_label, n_cls, cn):
    acc = data[:, :3]  # accelerometer
    gyro = data[:, 3:6] if data.shape[1] >= 6 else np.zeros_like(acc)
    
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    acc_mean = acc.mean(axis=0)
    acc_std = acc.std(axis=0)
    gyro_mean = gyro.mean(axis=0)
    
    descs = [f'{i}={cn[i]}' for i in range(n_cls)]
    prompt = f'''Classify physical activity from IMU sensor data (accelerometer + gyroscope).
Classes: {', '.join(descs)}
Features:
- acc_magnitude mean={acc_mag.mean():.3f} std={acc_mag.std():.3f}
- acc_mean (x,y,z): {[f"{x:.3f}" for x in acc_mean]}
- acc_std (x,y,z): {[f"{x:.3f}" for x in acc_std]}
- gyro_mean (x,y,z): {[f"{x:.3f}" for x in gyro_mean]}
Physics:
- walking/jogging: periodic motion, different intensity
- sitting/standing: minimal motion, different posture
- downstairs: negative Y acceleration pattern
Output JSON: {{"0":p0,"1":p1,...}}'''
    
    try:
        from openai import OpenAI
        c = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        r = c.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=120,
            extra_body={'reasoning_split': True}
        )
        msg = r.choices[0].message
        reasoning = msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content = msg.content
        
        for m in re.findall(r'\{[^{}]*\}', content, re.DOTALL):
            try:
                d = json.loads(m)
                if all(str(k) in d for k in range(n_cls)):
                    s = np.clip(np.array([float(d[str(k)]) for k in range(n_cls)]), 0, 1)
                    if s.sum() > 0: return s / s.sum()
            except: pass
        
        nums = re.findall(r'(?:p|prob)?\s*[0-9]\s*[:＝]\s*([0-9.]+)', reasoning, re.IGNORECASE)
        if len(nums) >= n_cls:
            s = np.clip(np.array([float(n) for n in nums[:n_cls]]), 0, 1)
            if s.sum() > 0: return s / s.sum()
    except Exception as e:
        print(f"  [ERR] {e}", end='')
    
    s = np.zeros(n_cls); s[true_label] = 1.0
    return s

if __name__ == "__main__":
    print("=" * 50)
    print("  生成PAMAP2软标签")
    print("=" * 50)
    
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_pamap2()
    print(f"  Train: {len(X_tr)} samples, Classes: {n_cls}")
    
    out_file = "/home/fandy/workplace/thesis/results/soft_labels/pamap2_soft.npy"
    y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        print(f"  Class {c}({cn[c]}): {n} samples", end='', flush=True)
        
        for i, idx in enumerate(sampled):
            y_soft[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn)
            time.sleep(10.0)
            if (i + 1) % 50 == 0: print(f" {i+1}", end='', flush=True)
        print()
        np.save(out_file, y_soft)
    
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0
    
    np.save(out_file, y_soft)
    print(f"\n✅ PAMAP2软标签已保存: {out_file}, Shape: {y_soft.shape}")
