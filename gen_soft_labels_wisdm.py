"""
生成WISDM MiniMax软标签
每类200样本（25%上限），6类共1200次API调用
"""
import os, sys, json, time, re
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
SAMPLES_PER_CLASS = 200
cn = ['Walking', 'Jogging', 'Stairs', 'Standing', 'Sitting', 'Lying']
n_cls = 6

def load_wisdm():
    base = '/home/fandy/workplace/thesis/datasets/WISDM'
    d, l = [], []
    label_map = {'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':2,'Standing':3,'Sitting':4}
    for f in sorted(glob(f"{base}/*/*.arff")):
        try:
            content = open(f).read()
            for line in content.split('\n'):
                if not line or line.startswith('@'): continue
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        vals = [float(x) for x in parts[2:8]]
                        label = str(parts[-1]).strip()
                        if label in label_map:
                            d.append(vals[:6]); l.append(label_map[label])
                    except: pass
        except: pass
    X = np.array(d, dtype=np.float32) if d else np.zeros((0,6), dtype=np.float32)
    y = np.array(l, dtype=np.int64)
    if len(X) > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    else:
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = X[:0], y[:0], X[:0], y[:0], X[:0], y[:0]
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def get_soft_label(data, true_label, n_cls, cn):
    acc = np.array(data[:3]) if len(data) >= 3 else np.zeros(3)
    gyro = np.array(data[3:6]) if len(data) >= 6 else np.zeros(3)
    acc_mag = np.sqrt((acc**2).sum())
    descs = [f'{i}={cn[i]}' for i in range(n_cls)]
    prompt = f'''Classify physical activity from accelerometer data.
Classes: {', '.join(descs)}
Features: acc_magnitude={acc_mag:.3f}, acc_x={acc[0]:.3f}, acc_y={acc[1]:.3f}, acc_z={acc[2]:.3f}
Physics: Walking/Jogging=periodic motion, Stairs=vertical movement, Standing/Sitting/Lying=postures
Output JSON: {{"0":p0,"1":p1,...}}'''
    try:
        from openai import OpenAI
        c = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        r = c.chat.completions.create(model=MODEL, messages=[{'role':'user','content':prompt}], max_tokens=120, extra_body={'reasoning_split':True})
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
    print("  生成WISDM软标签")
    print("=" * 50)
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_wisdm()
    print(f"  Train: {len(X_tr)} samples, Classes: {n_cls}")
    out_file = "/home/fandy/workplace/thesis/results/soft_labels/wisdm_soft.npy"
    y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        print(f"  Class {c}({cn[c]}): {n} samples", end='', flush=True)
        for i, idx in enumerate(sampled):
            y_soft[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn)
            time.sleep(0.12)
            if (i + 1) % 50 == 0: print(f" {i+1}", end='', flush=True)
        print()
        np.save(out_file, y_soft)
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3: y_soft[i, y_tr[i]] = 1.0
    np.save(out_file, y_soft)
    print(f"\n✅ WISDM软标签已保存: {out_file}, Shape: {y_soft.shape}")
