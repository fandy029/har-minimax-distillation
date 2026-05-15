#!/usr/bin/env python3
"""
合并18个类的软标签 → 输出A/B/C三版
用法: python merge_kuhar_soft.py [--quick]
  --quick: 只合并 --quick 生成的部分数据
"""
import os, sys, json, re, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR   = THESIS_DIR
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand',
    'Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle',
    'Run','Stair-up','Stair-down','Table-tennis']
N_CLS = len(CLASS_NAMES)

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
QUICK_MODE = args.quick

# ===== 路径 =====
OUT_BASE = os.path.join(SCRIPT_DIR, 'output')
PER_CLASS_DIR = os.path.join(OUT_BASE, 'per_class')
SOFT_DIR  = os.path.join(OUT_BASE, 'soft_labels')
LOG_DIR   = os.path.join(OUT_BASE, 'logs')
os.makedirs(SOFT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

SOFT_ALL      = os.path.join(SOFT_DIR, 'kuhar_soft_all.npy')
SOFT_FILTERED = os.path.join(SOFT_DIR, 'kuhar_soft_filtered.npy')
SOFT_CORRECT  = os.path.join(SOFT_DIR, 'kuhar_soft_correct_only.npy')
LOG_ALL_M     = os.path.join(LOG_DIR, 'all.log')
LOG_FILTERED_M = os.path.join(LOG_DIR, 'filtered.log')
LOG_CORRECT_M  = os.path.join(LOG_DIR, 'correct.log')

FILTER_ENT = 1.5; FILTER_GAP = 0.05; FILTER_CONF = 0.5

print("="*60)
print(" 合并 KuHar 按类软标签 → A/B/C 三版")
print("="*60)

# 加载全部样本索引
def load_kuhar_ids():
    base=os.path.join(BASE_DIR,'datasets','KuHar','1.Raw_time_domian_data')
    d,l = [],[]
    for folder in sorted(glob(os.path.join(base,'*'))):
        if not os.path.isdir(folder): continue
        parts=os.path.basename(folder).split('.'); cid=int(parts[0])
        if cid<0 or cid>=N_CLS: continue
        for f in sorted(glob(os.path.join(folder,'*.csv'))):
            try:
                df=pd.read_csv(f,header=None); data=df.values.astype(np.float32)
                for s in range(0,len(data)-127,64):
                    w=data[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d.append(0); l.append(cid)
            except: continue
    X=np.zeros((len(d),1),dtype=np.float32); y=np.array(l,dtype=np.int64)
    X,X_te,y,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X,X_vl,y,y_vl=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    return len(X), y

total_samples, labels = load_kuhar_ids()
print(f"全量训练样本: {total_samples}")

# 逐类合并
soft_all_records = np.zeros((total_samples, N_CLS), dtype=np.float32)
merged = 0

for c in range(N_CLS):
    class_dir = os.path.join(PER_CLASS_DIR, f'class_{c}')
    soft_file = os.path.join(class_dir, 'soft_all.npy')
    ckpt_file = os.path.join(OUT_BASE, 'checkpoints', f'ckpt_class_{c}.json')
    
    if not os.path.exists(soft_file):
        print(f"  class {c} ({CLASS_NAMES[c]}): ⚠️ 无软标签文件, 跳过")
        continue
    
    data = np.load(soft_file)
    if QUICK_MODE:
        done_indices = set()
        if os.path.exists(ckpt_file):
            with open(ckpt_file) as f:
                done_indices = set(json.load(f).get('done', []))
        for idx in done_indices:
            if data[idx].sum() > 0:
                soft_all_records[idx] = data[idx]
                merged += 1
    else:
        cidx = np.where(labels == c)[0]
        for idx in cidx:
            if data[idx].sum() > 0:
                soft_all_records[idx] = data[idx]
                merged += 1
    
    print(f"  class {c} ({CLASS_NAMES[c]:<15s}): found {int((data.sum(1)>0).sum())} labels")

print(f"\n合并完成: {merged}/{total_samples} 样本有软标签")

# ==== A版: 全部 ====
np.save(SOFT_ALL, soft_all_records)
print(f"A版: {SOFT_ALL}")

# ==== B版: 筛选 ====
# 对于每个样本, 检查是否通过质量筛选
soft_filtered = soft_all_records.copy()
filtered_count = 0

# 读取日志来判断筛选
for c in range(N_CLS):
    log_filtered_file = os.path.join(PER_CLASS_DIR, f'class_{c}', 'log_filtered.txt')
    if not os.path.exists(log_filtered_file): continue
    
    # 从 all.log 找非筛选的条目
    log_all_file = os.path.join(PER_CLASS_DIR, f'class_{c}', 'log_all.txt')
    if not os.path.exists(log_all_file): continue
    
    # 解析所有样本日志, 找 FAIL 或 不通过筛选的
    failed_idx = set()
    with open(log_all_file) as f:
        for l in f:
            if 'FAIL idx=' in l:
                m = re.search(r'idx=(\d+)', l)
                if m: failed_idx.add(int(m.group(1)))
    
    # 读筛选日志 → 通过的样本
    passed_idx = set()
    with open(log_filtered_file) as f:
        for l in f:
            m = re.search(r'true=.*?pred=.*?ent=([\d.]+).*?conf=([\d.]+).*?gap=([\d.]+)', l)
            if m:
                ent, conf, gap = float(m.group(1)), float(m.group(2)), float(m.group(3))
                if ent < FILTER_ENT and gap > FILTER_GAP and conf > FILTER_CONF:
                    # 查找 idx
                    m2 = re.search(r'#(\d+)/', l)
                    # 这里需要从all.log反查idx, 简化: 直接基于特征筛选
                    pass
    
    # 简化版: 直接从软标签本身筛
    cidx = np.where(labels == c)[0]
    for idx in cidx:
        probs = soft_filtered[idx]
        if probs.sum() == 0: continue
        ent = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1)))
        srt = sorted(enumerate(probs), key=lambda x: -x[1])
        gap = srt[0][1] - srt[1][1] if len(srt) >= 2 else 0
        max_prob = probs.max()
        if not (ent < FILTER_ENT and gap > FILTER_GAP and max_prob > FILTER_CONF):
            # 不通过筛选 → one-hot
            soft_filtered[idx] = 0
            soft_filtered[idx, labels[idx]] = 1.0
            filtered_count += 1

np.save(SOFT_FILTERED, soft_filtered)
filt_pct = (total_samples - filtered_count) / max(total_samples, 1) * 100
print(f"B版: {SOFT_FILTERED} (保留 {total_samples - filtered_count}/{total_samples} = {filt_pct:.1f}%)")

# ==== C版: correct-only ====
soft_correct = soft_all_records.copy()
wrong_count = 0
for c in range(N_CLS):
    cidx = np.where(labels == c)[0]
    for idx in cidx:
        probs = soft_correct[idx]
        if probs.sum() == 0: continue
        pred = int(np.argmax(probs))
        if pred != c:
            soft_correct[idx] = 0
            soft_correct[idx, c] = 1.0
            wrong_count += 1

np.save(SOFT_CORRECT, soft_correct)
corr_pct = (total_samples - wrong_count) / max(total_samples, 1) * 100
print(f"C版: {SOFT_CORRECT} (保留 {total_samples - wrong_count}/{total_samples} = {corr_pct:.1f}%)")

# ==== 合并日志 ====
print("\n合并日志...")
for log_type, src_pattern, dst_file in [
    ('all', 'log_all.txt', LOG_ALL_M),
    ('filtered', 'log_filtered.txt', LOG_FILTERED_M),
    ('correct', 'log_correct.txt', LOG_CORRECT_M),
]:
    with open(dst_file, 'w') as out:
        out.write(f"=== KuHar 合并日志 ({log_type}) ===\n\n")
        for c in range(N_CLS):
            src = os.path.join(PER_CLASS_DIR, f'class_{c}', src_pattern)
            if os.path.exists(src):
                out.write(f"\n--- Class {c} ({CLASS_NAMES[c]}) ---\n")
                with open(src) as f_in:
                    out.write(f_in.read())
    print(f"  {log_type}.log → {dst_file}")

print(f"\n{'='*60}")
print(f" 全部完成!")
print(f"  A: {SOFT_ALL}")
print(f"  B: {SOFT_FILTERED}")
print(f"  C: {SOFT_CORRECT}")
print(f"  日志: {LOG_DIR}/")
print(f"{'='*60}")
