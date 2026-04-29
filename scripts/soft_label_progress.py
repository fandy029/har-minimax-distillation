#!/usr/bin/env python3
"""
详细进度汇报脚本
每10分钟运行一次，向微信推送完整状态
"""
import os, sys, time
import numpy as np

# 动态路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # thesis/ root
THESIS_DIR = os.path.dirname(SCRIPT_DIR)  # parent of scripts/ = thesis/
sys.path.insert(0, SCRIPT_DIR)  # scripts/ for importing gen_*

# 从 gen_soft_labels_unified.py 导入数据加载器（避免硬编码）
from gen_soft_labels_unified import LOADERS, DATASET_CONFIG
from gen_kuhar_parallel import load_or_create_data as load_kuhar_data

KUHAR_CN = ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay',
             'Lay-stand','Pick','Jump','Push-up','Sit-up','Walk',
             'Walk-backwards','Walk-circle','Run','Stair-up','Stair-down','Table-tennis']

SOFT_DIR = THESIS_DIR + "/results/soft_labels"
LOG_DIR = THESIS_DIR + "/results/logs"
CHECKPOINT_DIR = THESIS_DIR + "/results/checkpoints"
HISTORY_DIR = THESIS_DIR + "/results/history"


DATASETS = ['pamap2', 'kuhar', 'uci_har', 'harth', 'uci_har_new', 'motionsense', 'gait', 'wisdm', 'motionsense_dm']
# CFG 已移除，使用 DATASET_CONFIG（从 gen_soft_labels_unified 动态导入）

# ============ 动态 target 计算（实际加载数据，与 gen_soft_labels_unified.py 完全一致）============
DEFAULT_RATIO = 0.40
DEFAULT_LIMIT = 400

def compute_dynamic_target(ds):
    """加载实际数据，精确计算每类目标数（与 gen_soft_labels_unified.py 一致）"""
    try:
        if ds == 'kuhar':
            _, y_tr = load_kuhar_data()
            n_cls = len(KUHAR_CN)
        else:
            loader = LOADERS[ds]
            X_tr, y_tr, _, _, _, _ = loader()
            n_cls = len(DATASET_CONFIG[ds]["cn"])
        total = 0
        for c in range(n_cls):
            cnt = int(np.sum(y_tr == c))
            total += min(DEFAULT_LIMIT, max(1, int(cnt * DEFAULT_RATIO)))
        return total
    except Exception as e:
        print(f"    [WARN] compute_dynamic_target({ds}) failed: {e}")
        return None

def check_soft(ds):
    cn = KUHAR_CN if ds == 'kuhar' else DATASET_CONFIG[ds]['cn']
    n_cls = len(cn)
    target = compute_dynamic_target(ds)
    f = f"{SOFT_DIR}/{ds}_soft.npy"
    log = f"{LOG_DIR}/gen_{ds}.log"
    
    result = {'name': ds, 'target': target, 'real': 0, 'onehot': 0, 'zero': 0, 'log': '', 'done': False}
    
    if ds == 'kuhar':
        # kuhar 使用并行生成，输出为 kuhar_soft_proc{N}.npy
        files = [f for f in os.listdir(SOFT_DIR) if f.startswith('kuhar_soft_proc') and f.endswith('.npy')]
        if not files:
            f_main = f"{SOFT_DIR}/kuhar_soft.npy"
            arr = np.load(f_main) if os.path.exists(f_main) else None
        else:
            good = []
            for fn in files:
                try:
                    a = np.load(f"{SOFT_DIR}/{fn}")
                    good.append(a)
                except Exception:
                    pass  # 跳过损坏的 proc 文件
            arr = np.sum(good, axis=0) if len(good) > 1 else (good[0] if good else None)
    else:
        f_main = f"{SOFT_DIR}/{ds}_soft.npy"
        arr = np.load(f_main) if os.path.exists(f_main) else None
    
    if arr is not None:
        # one-hot: max probability > 0.95 → not valid (matches generation filtering)
        max_probs = np.max(arr, axis=1)
        is_valid = (arr.sum(axis=1) > 0) & (max_probs <= 0.95)
        is_onehot = max_probs > 0.95
        result['real'] = int(np.sum(is_valid))
        result['onehot'] = int(np.sum(is_onehot))
        result['zero'] = int(np.sum(arr.sum(axis=1) == 0))
        result['done'] = target is not None and result['real'] >= target
    
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

def get_running():
    """获取正在运行的软标签生成进程"""
    try:
        r = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=3)
        lines = r.stdout.strip().split('\n')
        procs = []
        for line in lines:
            if 'gen_soft_labels' in line or 'gen_kuhar_parallel' in line:
                procs.append(line)
        return procs
    except:
        return []

def get_training_running():
    """获取正在运行的训练进程"""
    try:
        r = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=3)
        lines = r.stdout.strip().split('\n')
        procs = []
        for line in lines:
            if 'run_distill.py' in line or 'run_train' in line:
                procs.append(line)
        return procs
    except:
        return []

def main():
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    
    soft_procs = get_running()
    train_procs = get_training_running()
    
    running_soft = []
    for p in soft_procs:
        for ds in DATASETS:
            # 精确匹配：dataset name 后跟空格或结束，或跟在 gen_xxx.py 后面
            import re
            if re.search(rf'(?:gen_\w+\.py\s+{re.escape(ds)}(?:\s|$))', p):
                running_soft.append(ds); break
    
    running_train = []
    for p in train_procs:
        parts = p.split()
        for ds in DATASETS:
            if ds in p:
                running_train.append(ds); break
    
    # 检查各数据集软标签
    results = [check_soft(ds) for ds in DATASETS]
    total_target = sum(r['target'] or 0 for r in results)
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
        pct = r['real'] / r['target'] * 100 if r['target'] else 0
        bar = '█' * int(pct/5) + '░' * (20 - int(pct/5))
        status = '✅完成' if r['done'] else ('🔄' if r['name'] in running_soft else '⏸️')
        log_short = (r['log'][-30:] if r['log'] else '')[:30]
        tgt_str = str(r['target'] or 'N/A')
        print(f"{r['name']:<16} {r['real']:>6}/{tgt_str:<6} {r['onehot']:>8} {r['zero']:>8} {bar} {status}")
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
