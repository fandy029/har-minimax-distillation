"""
生成HARTH MiniMax软标签
每类150样本，6类共900次API调用
"""
import os, sys, json, time, re
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
SAMPLES_PER_CLASS = 200
cn = ['左立','走路','上楼','下楼','右立','站立']
n_cls = 6

def load_harth():
    base = '/home/fandy/workplace/thesis/datasets/HARTH/harth'
    LABEL_MAP = {1:5, 3:1, 4:2, 5:3, 6:0, 7:4, 8:1}
    d, l = [], []
    for f in sorted(glob(f"{base}/*.csv")):
        try:
            df = pd.read_csv(f)
            imu = df[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']].values.astype(np.float32)
            labels = df['label'].values
            for lbl in np.unique(labels):
                if lbl not in LABEL_MAP: continue
                unified = LABEL_MAP[lbl]
                mask = labels == lbl
                idx = np.where(mask)[0]
                for s in range(0, len(idx)-127, 64):
                    w = imu[idx[s:s+128]]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(unified)
        except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def get_soft_label(data, true_label, n_cls, cn, sr=50):
    acc = data[:, :3]
    acc_m = np.sqrt((acc**2).sum(axis=1))
    y_a = acc[:, 1]
    fft_v = np.abs(np.fft.fft(acc_m)[1:len(acc_m)//2])
    dom_f = np.fft.fftfreq(len(acc_m), 1/sr)[np.argmax(fft_v) + 1]
    descs = [f'{i}={cn[i]}' for i in range(n_cls)]
    prompt = f'''Classify IMU window. Classes: {', '.join(descs)}
Features: acc_mag={acc_m.mean():.2f}±{acc_m.std():.2f}, y_mean={y_a.mean():.4f}, peaks={np.sum((acc_m[1:-1]>acc_m[:-2])&(acc_m[1:-1]>acc_m[2:]))}, freq={dom_f:.1f}Hz
Physics: upstairs=posY~1Hz, downstairs=negY~1Hz, walk=posY~1-2Hz, sit/stand=low~0Hz
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
    except:
        pass
    s = np.zeros(n_cls); s[true_label] = 1.0
    return s

if __name__ == "__main__":
    print("=" * 50)
    print("  生成HARTH软标签")
    print("=" * 50)
    
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_harth()
    print(f"  Train: {len(X_tr)} samples")
    
    out_file = "/home/fandy/workplace/thesis/results/soft_labels/harth_soft.npy"
    y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        print(f"  Class {c}({cn[c]}): {n} samples", end='', flush=True)
        for i, idx in enumerate(sampled):
            y_soft[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn, 50)
            time.sleep(0.12)
            if (i + 1) % 50 == 0: print(f" {i+1}", end='', flush=True)
        print()
        np.save(out_file, y_soft)
    
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0
    
    np.save(out_file, y_soft)
    print(f"\n✅ HARTH软标签已保存: {out_file}, Shape: {y_soft.shape}")