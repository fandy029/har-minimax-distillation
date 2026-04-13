"""
生成UCI-HAR-New MiniMax软标签
每类150样本，12类共1800次API调用
"""
import os, sys, json, time, re
import numpy as np
from sklearn.model_selection import train_test_split

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
SAMPLES_PER_CLASS = 129
cn = ['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING',
      'STAND_TO_SIT','SIT_TO_STAND','SIT_TO_LIE','LIE_TO_SIT','STAND_TO_LIE','LIE_TO_STAND']
n_cls = 12

def load_uci_new():
    base = '/home/fandy/workplace/thesis/datasets/UCI_HAR_New'
    X_tr = np.loadtxt(f"{base}/Train/X_train.txt").astype(np.float32)
    y_tr = np.loadtxt(f"{base}/Train/y_train.txt").astype(np.int64) - 1
    X_te = np.loadtxt(f"{base}/Test/X_test.txt").astype(np.float32)
    y_te = np.loadtxt(f"{base}/Test/y_test.txt").astype(np.int64) - 1
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def get_soft_label(data, true_label, n_cls, cn):
    vals = np.array(data)
    try:
        means = vals[:9] if len(vals) >= 9 else np.zeros(9)
        stds = vals[128*9:128*9+9] if len(vals) >= 128*9+9 else np.zeros(9)
        energy = vals[256*9:256*9+9] if len(vals) >= 256*9+9 else np.zeros(9)
    except:
        means = np.zeros(9)
        stds = np.zeros(9)
        energy = np.zeros(9)
    
    descs = [f'{i}={cn[i]}' for i in range(n_cls)]
    prompt = f'''Classify HAR from 561-dim sensor features (12 classes including transitions).
Classes: {', '.join(descs)}
Key features (first 3 channels): 
- acc_means: {[f"{x:.3f}" for x in means[:3]]}
- acc_stds: {[f"{x:.3f}" for x in stds[:3]]}
- energy: {[f"{x:.3f}" for x in energy[:3]]}
Static: WALKING, WALKING_UP, WALKING_DOWN, SITTING, STANDING, LAYING
Transitions: STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, LIE_TO_STAND
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
    print("  生成UCI-HAR-New软标签")
    print("=" * 50)
    
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_uci_new()
    print(f"  Train: {len(X_tr)} samples")
    
    out_file = "/home/fandy/workplace/thesis/results/soft_labels/uci_har_new_soft.npy"
    y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        print(f"  Class {c}({cn[c]}): {n} samples", end='', flush=True)
        
        for i, idx in enumerate(sampled):
            y_soft[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn)
            time.sleep(13)
            if (i + 1) % 50 == 0: print(f" {i+1}", end='', flush=True)
        print()
        np.save(out_file, y_soft)
    
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0
    
    np.save(out_file, y_soft)
    print(f"\n✅ UCI-HAR-New软标签已保存: {out_file}, Shape: {y_soft.shape}")