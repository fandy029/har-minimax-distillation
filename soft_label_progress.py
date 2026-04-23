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
    'pamap2':         (5,  35),
    'kuhar':          (18, 400),
    'uci_har':        (6,  400),
    'harth':          (6,  400),
    'uci_har_new':    (12, 35),
    'motionsense':    (6,  400),
    'gait':           (4,  35),
    'wisdm':          (6,  400),
    'motionsense_dm': (6,  400),
}

# ============ 动态 target 计算（与 gen_soft_labels_unified.py 一致）============
def compute_dynamic_target(ds):
    """用已知数据集信息计算目标（35%/400），每类单独计算"""
    # 各类样本数（用于精确计算每类目标）
    # HARTH 6类不均匀
    harth_per_class = {0:10290,1:2629,2:2320,3:629,4:578,5:6298}
    # Gait 4类
    gait_per_class = {0:220,1:153,2:159,3:141}
    # WISDM 6类不均匀
    wisdm_per_class = {0:4183,1:3363,2:1234,3:1003,4:599,5:482}
    # KuHar 18类（部分不均匀）
    kuhar_per_class = {0:5537,1:5496,2:5302,3:5499,4:6273,5:5395,6:5127,
                       7:3909,8:2030,9:1482,10:3060,11:2483,12:902,
                       13:748,14:1752,15:2475,16:2410,17:1330}
    # 通用：总数/n_cls 估算（用于均匀分布的数据集）
    counts = {
        'pamap2': 2826, 'kuhar': 61210, 'uci_har': 5881, 'harth': 22744,
        'uci_har_new': 6213, 'motionsense': 13993, 'motionsense_dm': 13784,
        'gait': 673, 'wisdm': 10864,
    }
    n_cls = {'pamap2':5,'kuhar':18,'uci_har':6,'harth':6,
             'uci_har_new':12,'motionsense':6,'motionsense_dm':6,'gait':4,'wisdm':6}
    try:
        if ds == 'harth':
            return sum(min(400, max(1, int(harth_per_class[c] * 0.35))) for c in range(6))
        elif ds == 'gait':
            return sum(min(400, max(1, int(gait_per_class[c] * 0.35))) for c in range(4))
        elif ds == 'wisdm':
            return sum(min(400, max(1, int(wisdm_per_class[c] * 0.35))) for c in range(6))
        elif ds == 'kuhar':
            return sum(min(400, max(1, int(kuhar_per_class[c] * 0.35))) for c in range(18))
        else:
            total = counts[ds]
            nc = n_cls[ds]
            avg = total / nc
            return min(400, max(1, int(avg * 0.35))) * nc
    except:
        return None

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
            if 'run_distill.py' in line or 'train_' in line:
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
