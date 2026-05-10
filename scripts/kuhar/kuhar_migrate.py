#!/usr/bin/env python3
"""KuHar 软标签迁移: 旧软标签(60/20/20) → 新位置(70/15/15)"""
import os, numpy as np, hashlib, json, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(SCRIPT_DIR, 'output')
OLD_DIR = os.path.join(OUT_BASE, 'pre_migration_70_15_15')
N_CLS = 18

def window_hash(w):
    return hashlib.md5(w.tobytes()).hexdigest()

# === 1. Load old windows + soft labels
print("Loading old data...")
old_windows = {}       # hash → (soft_label, class)
old_soft = {}          # old_global_idx → soft_label (references)
old_labels = np.load(os.path.join(OLD_DIR, 'train_labels.npy'))

for c in range(N_CLS):
    wf = os.path.join(OLD_DIR, f'windows_class_{c}.npy')
    sf = os.path.join(OLD_DIR, f'soft_class_{c}.npy')
    if not os.path.exists(wf) or not os.path.exists(sf): 
        continue
    windows = np.load(wf)
    soft = np.load(sf)
    cidx = np.where(old_labels == c)[0]
    n = 0
    for i, gi in enumerate(cidx):
        if gi < len(soft) and soft[gi].sum() > 0:
            wh = window_hash(windows[i])
            old_windows[wh] = soft[gi]
            n += 1
    print(f"  class {c}: {n} valid soft labels from {len(cidx)} windows")

print(f"  Total old soft labels: {len(old_windows)}")

# === 2. Load new data
print("\nMapping to new positions...")
new_labels = np.load(os.path.join(OUT_BASE, 'train_labels.npy'))
new_soft = np.zeros((len(new_labels), N_CLS), dtype=np.float32)
migrated = 0
new_samples = 0

for c in range(N_CLS):
    wp = os.path.join(OUT_BASE, 'per_class', f'class_{c}', 'windows.npy')
    cidx = np.where(new_labels == c)[0]
    if not os.path.exists(wp): continue
    windows = np.load(wp)
    for i, gi in enumerate(cidx):
        wh = window_hash(windows[i])
        if wh in old_windows:
            new_soft[gi] = old_windows[wh]
            migrated += 1
        else:
            new_samples += 1

# === 3. Save migrated soft labels
for c in range(N_CLS):
    dd = os.path.join(OUT_BASE, 'per_class', f'class_{c}')
    sf = os.path.join(dd, 'soft_all.npy')
    # Create fresh array with new size → new_soft already has correct dims
    out = np.zeros((len(new_labels), N_CLS), dtype=np.float32)
    cidx = np.where(new_labels == c)[0]
    for gi in cidx:
        out[gi] = new_soft[gi]
    np.save(sf, out)

# Update checkpoints
for c in range(N_CLS):
    cidx = np.where(new_labels == c)[0]
    done_indices = [int(gi) for gi in cidx if new_soft[gi].sum() > 0]
    correct_indices = [int(gi) for gi in cidx if new_soft[gi].sum() > 0 and int(np.argmax(new_soft[gi])) == c]
    ckpt_dir = os.path.join(OUT_BASE, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, f'ckpt_class_{c}.json'), 'w') as f:
        json.dump({'done': done_indices, 'correct': correct_indices}, f)
    print(f"  class {c}: {len(done_indices)} done, {len(correct_indices)} correct")

print(f"\n✅ Migrated: {migrated} soft labels")
print(f"🆕 Need new API calls: {new_samples} samples")
print(f"💰 Saved API calls: {migrated}")
