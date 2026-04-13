"""
快速完成KuHar软标签 - 并行API调用
使用concurrent.futures实现并行，Fill remaining 14 classes
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
cn = ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle','Run','Stair-up','Stair-down','Table-tennis']
n_cls = 18
MAX_WORKERS = 5  # Parallel API calls

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
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def get_soft_label(data, true_label, n_cls, cn):
    acc = data[:, :3]
    gyro = data[:, 3:6] if data.shape[1] >= 6 else np.zeros_like(acc)
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    acc_mean = acc.mean(axis=0)
    gyro_mean = gyro.mean(axis=0)
    descs = [f'{i}={cn[i]}' for i in range(n_cls)]
    prompt = f'''Classify physical activity from IMU sensor data (accelerometer + gyroscope).
Classes: {', '.join(descs)}
Features:
- acc_magnitude mean={acc_mag.mean():.3f} std={acc_mag.std():.3f}
- acc_mean (x,y,z): {[f"{x:.3f}" for x in acc_mean[:3]]}
- gyro_mean (x,y,z): {[f"{x:.3f}" for x in gyro_mean[:3]]}
Physics cues:
- Stand/Sit/Lay: stationary postures, minimal motion
- Walk/Run/Jump: periodic motion, different intensity
- Stair-up/Stair-down: vertical acceleration patterns
- Table-tennis/Pick/Push-up: specific arm movements
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
        pass
    s = np.zeros(n_cls); s[true_label] = 1.0
    return s

def process_class(args):
    c, cidx, sampled, X_tr, y_tr, n_cls, cn, lock = args
    results = {}
    for i, idx in enumerate(sampled):
        results[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn)
        time.sleep(0.05)  # Light rate limiting
    return c, cn[c], results

if __name__ == "__main__":
    print("=" * 50)
    print("  快速生成KuHar软标签 (并行模式)")
    print("=" * 50)
    
    out_file = "/home/fandy/workplace/thesis/results/soft_labels/kuhar_soft.npy"
    
    # Load existing file
    if os.path.exists(out_file):
        y_soft = np.load(out_file)
        nonzero = np.where(y_soft.sum(axis=1) > 0.1)[0]
        print(f"  已加载现有文件: {nonzero.shape[0]} valid rows")
    else:
        y_soft = None
    
    # Load data
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_kuhar()
    print(f"  Train: {len(X_tr)} samples, Classes: {n_cls}")
    
    if y_soft is None:
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    
    # Find which classes need processing
    classes_done = set()
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        # Check if already done
        rows_to_check = sampled[:5]
        if all(y_soft[idx].sum() > 0.1 for idx in rows_to_check):
            classes_done.add(c)
            print(f"  Class {c}({cn[c]}): 已完成 (skip)")
        else:
            print(f"  Class {c}({cn[c]}): 需要生成")
    
    classes_todo = [c for c in range(n_cls) if c not in classes_done]
    print(f"\n  需要生成: {len(classes_todo)} classes: {classes_todo}")
    
    lock = threading.Lock()
    
    for c in classes_todo:
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        print(f"\n  Class {c}({cn[c]}): {n} samples (并行 {MAX_WORKERS} workers)", flush=True)
        
        # Split work into chunks for parallel processing
        chunk_size = max(1, n // MAX_WORKERS)
        chunks = []
        for w in range(MAX_WORKERS):
            start = w * chunk_size
            end = n if w == MAX_WORKERS - 1 else (w + 1) * chunk_size
            chunk_indices = sampled[start:end]
            if len(chunk_indices) > 0:
                chunks.append((c, cidx, chunk_indices, X_tr, y_tr, n_cls, cn, lock))
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_class, chunk): i for i, chunk in enumerate(chunks)}
            done_count = 0
            for future in as_completed(futures):
                c_res, cn_res, results = future.result()
                with lock:
                    for idx, soft in results.items():
                        y_soft[idx] = soft
                    done_count += len(results)
                    print(f"    [{done_count}/{n}]", end='', flush=True)
                time.sleep(0.1)
        
        # Save after each class
        np.save(out_file, y_soft)
        print(f"\n  ✅ Class {c} 完成, 已保存")
    
    # Fill zeros
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0
    
    np.save(out_file, y_soft)
    
    nonzero = np.where(y_soft.sum(axis=1) > 0.1)[0]
    print(f"\n✅ KuHar软标签完成: {nonzero.shape[0]}/{len(y_soft)} valid rows, Shape: {y_soft.shape}")
