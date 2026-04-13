"""
生成MotionSense-DM MiniMax软标签
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
cn = ['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']
n_cls = 6

def load_motionsense_dm():
    base = '/home/fandy/workplace/thesis/datasets/MotionSense_DeviceMotion/A_DeviceMotion_data'
    d, l = [], []
    label_map = {
        'dws':0, 'downstairs':0,
        'jog':1, 'jogging':1,
        'sit':2, 'sitting':2,
        'std':3, 'standing':3,
        'ups':4, 'upstairs':4,
        'wlk':5, 'walking':5
    }
    for folder in sorted(glob(f"{base}/*/")):
        folder_name = folder.rstrip("/").split("/")[-1]
        # folder_name format: 'dws_1', 'jog_16', 'sit_13', etc.
        activity = folder_name.rsplit('_', 1)[0]  # 'dws', 'jog', etc.
        if activity not in label_map: continue
        for f in sorted(glob(f"{folder}/*.csv")):
            try:
                df = pd.read_csv(f)
                cols = [c for c in df.columns if 'userAcceleration' in c]
                if len(cols) >= 3:
                    data = df[cols].values.astype(np.float32)
                    for s in range(0, len(data)-127, 64):
                        w = data[s:s+128]
                        if w.shape[0]==128 and not np.any(np.isnan(w)):
                            d.append(w); l.append(label_map[activity])
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def get_soft_label(data, true_label, n_cls, cn):
    acc = data[:, :3]
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    acc_mean = acc.mean(axis=0)
    descs = [f'{i}={cn[i]}' for i in range(n_cls)]
    prompt = f'''Classify physical activity from accelerometer data.
Classes: {', '.join(descs)}
Features: acc_magnitude mean={acc_mag.mean():.3f}, acc_mean={[f"{x:.3f}" for x in acc_mean]}
Physics: walking/jogging/downstairs/upstairs=periodic, sitting/standing=stationary
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
    print("  生成MotionSense-DM软标签")
    print("=" * 50)
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_motionsense_dm()
    print(f"  Train: {len(X_tr)} samples, Classes: {n_cls}")
    out_file = "/home/fandy/workplace/thesis/results/soft_labels/motionsense_dm_soft.npy"
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
    print(f"\n✅ MotionSense-DM软标签已保存: {out_file}, Shape: {y_soft.shape}")
