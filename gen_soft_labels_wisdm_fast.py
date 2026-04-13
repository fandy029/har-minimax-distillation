"""
快速生成WISDM软标签 - 并行API调用
"""
import os, sys, json, time, re, threading
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
SAMPLES_PER_CLASS = 200
cn = ['Walking', 'Jogging', 'Stairs', 'Standing', 'Sitting', 'Lying']
n_cls = 6
MAX_WORKERS = 5

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
        pass
    s = np.zeros(n_cls); s[true_label] = 1.0
    return s

def process_samples(args):
    indices, X_tr, y_tr, n_cls, cn = args
    results = {}
    for idx in indices:
        results[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn)
        time.sleep(0.05)
    return results

if __name__ == "__main__":
    print("=" * 50)
    print("  快速生成WISDM软标签 (并行模式)")
    print("=" * 50)
    
    out_file = "/home/fandy/workplace/thesis/results/soft_labels/wisdm_soft.npy"
    
    # Load existing file if exists
    if os.path.exists(out_file):
        y_soft = np.load(out_file)
        nonzero = np.where(y_soft.sum(axis=1) > 0.1)[0]
        print(f"  已加载现有文件: {nonzero.shape[0]} valid rows")
    else:
        y_soft = None
    
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_wisdm()
    print(f"  Train: {len(X_tr)} samples, Classes: {n_cls}")
    
    if y_soft is None:
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    
    lock = threading.Lock()
    
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        
        # Check if already done
        rows_to_check = sampled[:5]
        if all(y_soft[idx].sum() > 0.1 for idx in rows_to_check):
            print(f"  Class {c}({cn[c]}): 已完成 (skip)")
            continue
        
        print(f"\n  Class {c}({cn[c]}): {n} samples (并行 {MAX_WORKERS} workers)", flush=True)
        
        # Split into chunks for parallel processing
        chunk_size = max(1, n // MAX_WORKERS)
        chunks = []
        for w in range(MAX_WORKERS):
            start = w * chunk_size
            end = n if w == MAX_WORKERS - 1 else (w + 1) * chunk_size
            chunk_indices = sampled[start:end]
            if len(chunk_indices) > 0:
                chunks.append((list(chunk_indices), X_tr, y_tr, n_cls, cn))
        
        done_count = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_samples, chunk) for chunk in chunks]
            for future in as_completed(futures):
                results = future.result()
                with lock:
                    for idx, soft in results.items():
                        y_soft[idx] = soft
                    done_count += len(results)
                    print(f"    [{done_count}/{n}]", end='', flush=True)
                time.sleep(0.1)
        
        print()
        np.save(out_file, y_soft)
        print(f"  ✅ Class {c} 完成")
    
    # Fill zeros
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0
    
    np.save(out_file, y_soft)
    nonzero = np.where(y_soft.sum(axis=1) > 0.1)[0]
    print(f"\n✅ WISDM软标签完成: {nonzero.shape[0]}/{len(y_soft)} valid rows, Shape: {y_soft.shape}")
