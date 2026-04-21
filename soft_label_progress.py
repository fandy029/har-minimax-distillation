#!/usr/bin/env python3
"""
详细进度汇报脚本
每10分钟运行一次，向微信推送完整状态
"""
import os, time, subprocess, sys
import numpy as np

SOFT_DIR = "/home/fandy/workplace/thesis/results/soft_labels"
LOG_DIR = "/home/fandy/workplace/thesis/results/logs"
CHECKPOINT_DIR = "/home/fandy/workplace/thesis/results/checkpoints"
HISTORY_DIR = "/home/fandy/workplace/thesis/results/history"

DATASETS = ['pamap2', 'kuhar', 'uci_har', 'harth', 'uci_har_new', 'motionsense', 'gait', 'wisdm', 'motionsense_dm']
CFG = {
    'pamap2':         (5,  30),
    'kuhar':          (18, 200),
    'uci_har':        (6,  200),
    'harth':          (6,  200),
    'uci_har_new':    (12, 129),
    'motionsense':    (6,  200),
    'gait':           (4,  29),
    'wisdm':          (6,  200),
    'motionsense_dm': (6,  200),
}

# ============ 动态 target 计算（与 gen_soft_labels_unified.py 一致）============
def compute_dynamic_target(ds):
    """动态计算每类样本上限200、25%采样率的真实目标数"""
    if ds == 'pamap2':
        # PAMAP2: 5 classes × 30 = 150 (correct target, verified)
        return 150
    elif ds == 'kuhar':
        # 直接复用 gen_soft_labels_unified.py 的 loader，确保目标数完全一致
        try:
            import sys
            sys.path.insert(0, '/home/fandy/workplace/thesis')
            # 动态 import 避免循环引用
            loader_mod_name = 'gen_soft_labels_unified'
            import importlib
            loader_mod = importlib.import_module(loader_mod_name)
            load_fn = getattr(loader_mod, 'load_kuhar')
            X_tr, y_tr, _, _, _, _ = load_fn()
            from collections import Counter
            cnt = Counter(y_tr.tolist())
            return sum(min(200, max(1, int(n * 0.25))) for n in cnt.values())
        except Exception as e:
            print(f"[WARN] kuhar target fallback: {e}")
            return CFG[ds][0] * CFG[ds][1]
    elif ds == 'uci_har':
        try:
            import pandas as pd
            X = np.load('/home/fandy/workplace/thesis/datasets/UCI-HAR/train/X_train.npy')
            y = np.load('/home/fandy/workplace/thesis/datasets/UCI-HAR/train/y_train.npy')
            from collections import Counter
            cnt = Counter(y.flatten().tolist())
            return sum(min(200, max(1, int(n * 0.25))) for n in cnt.values())
        except: return CFG[ds][0] * CFG[ds][1]
    elif ds == 'harth':
        try:
            import pandas as pd
            labels = pd.read_csv('/home/fandy/workplace/thesis/datasets/HARTH/HARTH.csv', usecols=['label'])['label'].value_counts().to_dict()
            return sum(min(200, max(1, int(n * 0.25))) for n in labels.values())
        except: return CFG[ds][0] * CFG[ds][1]
    elif ds == 'uci_har_new':
        try:
            import pandas as pd
            base = '/home/fandy/workplace/thesis/datasets/UCI-HAR+'
            dfs = []
            for split in ['train', 'test']:
                for act in os.listdir(os.path.join(base, split)):
                    ad = os.path.join(base, split, act)
                    if os.path.isdir(ad):
                        for f in os.listdir(ad):
                            if f.endswith('.txt'):
                                dfs.append(pd.read_csv(os.path.join(ad, f), header=None, usecols=[0], nrows=100))
            if dfs:
                all_labels = pd.concat(dfs, ignore_index=True).iloc[:,0].value_counts().to_dict()
                return sum(min(200, max(1, int(n * 0.25))) for n in all_labels.values())
        except: return CFG[ds][0] * CFG[ds][1]
    elif ds == 'motionsense':
        try:
            import pandas as pd
            base = '/home/fandy/workplace/thesis/datasets/MotionSense'
            rows = []
            for person in sorted(os.listdir(os.path.join(base, 'data'))):
                for sess in sorted(os.listdir(os.path.join(base, 'data', person))):
                    for f in os.listdir(os.path.join(base, 'data', person, sess)):
                        if f.endswith('.csv'):
                            df = pd.read_csv(os.path.join(base, 'data', person, sess, f), nrows=100)
                            if 'activity_id' in df.columns: rows.extend(df['activity_id'].tolist())
            from collections import Counter
            if rows:
                cnt = Counter(rows)
                return sum(min(200, max(1, int(n * 0.25))) for n in cnt.values())
        except: return CFG[ds][0] * CFG[ds][1]
    elif ds == 'gait':
        try:
            base = '/home/fandy/workplace/thesis/datasets/Gait'
            rows = []
            for label in os.listdir(base):
                ldir = os.path.join(base, label)
                if not os.path.isdir(ldir): continue
                for f in os.listdir(ldir):
                    if f.endswith('.csv'):
                        import pandas as pd
                        try:
                            df = pd.read_csv(os.path.join(ldir, f), header=None, nrows=100)
                            rows.extend([label] * len(df))
                        except: pass
            from collections import Counter
            if rows:
                cnt = Counter(rows)
                return sum(min(200, max(1, int(n * 0.25))) for n in cnt.values())
        except: return CFG[ds][0] * CFG[ds][1]
    elif ds == 'wisdm':
        try:
            raw_path = '/home/fandy/workplace/thesis/datasets/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
            d, l = [], []
            label_map = {'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':3,'Sitting':4,'Standing':5}
            for line in open(raw_path):
                line = line.strip().rstrip(';')
                parts = line.split(',')
                if len(parts) != 6: continue
                try:
                    act = parts[1].strip()
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    if act in label_map: d.append([x,y,z]); l.append(label_map[act])
                except: continue
            from sklearn.model_selection import train_test_split
            window_size, step = 128, 64
            X, y_w = [], []
            for start in range(0, len(d)-window_size+1, step):
                X.append(d[start:start+window_size])
                y_w.append(l[start+window_size//2])
            X = np.array(X, dtype=np.float32); y_w = np.array(y_w, dtype=np.int64)
            X_tr_val, X_te, y_tr_val, y_te = train_test_split(X, y_w, test_size=0.2, random_state=42, stratify=y_w)
            X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr_val, y_tr_val, test_size=0.2, random_state=42, stratify=y_tr_val)
            from collections import Counter
            cnt = Counter(y_tr.tolist())
            return sum(min(200, max(1, int(n * 0.25))) for n in cnt.values())
        except: return CFG[ds][0] * CFG[ds][1]
    elif ds == 'motionsense_dm':
        try:
            base = '/home/fandy/workplace/thesis/datasets/MotionSense-DM'
            rows = []
            for person in sorted(os.listdir(os.path.join(base, 'data'))):
                for sess in sorted(os.listdir(os.path.join(base, 'data', person))):
                    for f in os.listdir(os.path.join(base, 'data', person, sess)):
                        if f.endswith('.csv'):
                            import pandas as pd
                            df = pd.read_csv(os.path.join(base, 'data', person, sess, f), nrows=100)
                            if 'activity_id' in df.columns: rows.extend(df['activity_id'].tolist())
            from collections import Counter
            if rows:
                cnt = Counter(rows)
                return sum(min(200, max(1, int(n * 0.25))) for n in cnt.values())
        except: return CFG[ds][0] * CFG[ds][1]
    return CFG[ds][0] * CFG[ds][1]

def get_running():
    try:
        r = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = [l for l in r.stdout.split('\n') if 'gen_soft_labels_unified' in l and 'grep' not in l]
        return lines
    except:
        return []

def get_training_running():
    try:
        r = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = [l for l in r.stdout.split('\n') if 'run_distill' in l and 'grep' not in l]
        return lines
    except:
        return []

def check_soft(ds):
    n_cls, spc = CFG[ds]
    # 使用动态计算的目标（与 gen_soft_labels_unified.py 一致）
    target = compute_dynamic_target(ds)
    f = f"{SOFT_DIR}/{ds}_soft.npy"
    log = f"{LOG_DIR}/gen_{ds}.log"
    
    result = {'name': ds, 'target': target, 'real': 0, 'onehot': 0, 'zero': 0, 'log': '', 'done': False}
    
    if ds == 'kuhar':
        # kuhar 使用并行生成，输出为 kuhar_soft_proc{N}.npy
        files = [f for f in os.listdir(SOFT_DIR) if f.startswith('kuhar_soft_proc') and f.endswith('.npy')]
        if not files:
            f_main = f"{SOFT_DIR}/kuhar_soft.npy"
            if os.path.exists(f_main):
                arr = np.load(f_main)
            else:
                arr = None
        else:
            arrays = [np.load(f"{SOFT_DIR}/{fn}") for fn in files]
            arr = np.sum(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    else:
        f_main = f"{SOFT_DIR}/{ds}_soft.npy"
        arr = np.load(f_main) if os.path.exists(f_main) else None
    
    if arr is not None:
        is_onehot = (arr > 0.99).sum(axis=1) == 1
        result['real'] = int(np.sum((arr.sum(axis=1) > 0) & ~is_onehot))
        result['onehot'] = int(np.sum(is_onehot))
        result['zero'] = int(np.sum(arr.sum(axis=1) == 0))
        result['done'] = result['real'] >= target
    
    # 最新日志（最后3行）
        # 最新日志（最后3行）- 用subprocess.run避免读取大文件
    try:
        r = subprocess.run(['"tail"', '-3', log], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            result['log'] = r.stdout.strip().replace('\n', ' | ')
    except:
        pass
        pass
    
    return result

def check_training(ds, ver):
    ckpt = f"{CHECKPOINT_DIR}/{ds}_{ver}_best.pt"
    hist = f"{HISTORY_DIR}/{ds}_{ver}_history.json"
    
    done = os.path.exists(ckpt)
    best_acc = None
    last_line = ''
    
    log_file = f"{LOG_DIR}/train_{ds}_{ver}.log"
    try:
        if os.path.exists(log_file):
            with open(log_file) as fh:
                lines = fh.readlines()
                for l in lines:
                    if 'val_acc' in l or 'best_val' in l:
                        last_line = l.strip()
    except:
        pass
    
    if os.path.exists(hist):
        try:
            import json
            with open(hist) as fh:
                h = json.load(fh)
                best_acc = h.get('best_val_acc', None)
        except:
            pass
    
    return {'done': done, 'best_acc': best_acc, 'last_line': last_line}

def main():
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    
    soft_procs = get_running()
    train_procs = get_training_running()
    
    running_soft = []
    for p in soft_procs:
        for ds in DATASETS:
            if ds in p:
                running_soft.append(ds); break
    
    running_train = []
    for p in train_procs:
        parts = p.split()
        for ds in DATASETS:
            if ds in p:
                running_train.append(ds); break
    
    # 检查各数据集软标签
    results = [check_soft(ds) for ds in DATASETS]
    total_target = sum(r['target'] for r in results)
    total_real = sum(r['real'] for r in results)
    
    # 检查训练
    train_status = {}
    for ds in DATASETS:
        train_status[ds] = {}
        for ver in ['pure_cnn', 'v1', 'v2', 'v3']:
            train_status[ds][ver] = check_training(ds, ver)
    
    # ===== 输出 =====
    print("=" * 60)
    print(f"  📊 详细进度汇报  {now}")
    print("=" * 60)
    
    # 软标签进度
    print(f"\n🗂️  软标签进度 (总: {total_real}/{total_target})")
    print(f"{'数据集':<16} {'真软标签':>8} {'one-hot':>8} {'未填充':>8} {'进度':>8} {'状态'}")
    print("-" * 65)
    for r in results:
        pct = r['real'] / r['target'] * 100
        bar = '█' * int(pct/5) + '░' * (20 - int(pct/5))
        status = '✅完成' if r['done'] else ('🔄' if r['name'] in running_soft else '⏸️')
        log_short = (r['log'][-30:] if r['log'] else '')[:30]
        print(f"{r['name']:<16} {r['real']:>6}/{r['target']:<6} {r['onehot']:>8} {r['zero']:>8} {bar} {status}")
        if r['log']:
            print(f"  ↳ {r['log'][:60]}")
    
    # 训练进度
    print(f"\n🏋️  训练进度")
    train_done = 0
    train_total = len(DATASETS) * 4
    for ds in DATASETS:
        statuses = []
        for ver in ['pure_cnn', 'v1', 'v2', 'v3']:
            t = train_status[ds][ver]
            if t['done']:
                acc_str = f"{t['best_acc']*100:.1f}%" if t['best_acc'] else ""
                statuses.append(f"{ver}✅{acc_str}")
                train_done += 1
            else:
                statuses.append(f"{ver}❌")
        print(f"  {ds:<16}: {' | '.join(statuses)}")
    
    print(f"\n📈 总训练: {train_done}/{train_total} 完成")
    print(f"🔧 软标签进程: {len(running_soft)} 运行中 ({', '.join(running_soft) if running_soft else '无'})")
    print(f"🔧 训练进程: {len(running_train)} 运行中 ({', '.join(running_train) if running_train else '无'})")
    
    if total_real >= total_target and train_done >= train_total:
        print(f"\n🎉 全部完成！")
    
    return f"汇报完毕"

if __name__ == "__main__":
    main()
