#!/usr/bin/env python3
"""
KuHar 按类并行生成 — 每个进程只生成指定类
用法: python kuhar_kuhar_gen.py --class N [--quick] [--force]
  --class N:   只生成第N类的软标签 (0-17)
  --quick:     测试模式, 每类只生成50个样本
  --force:     清除该类已有断点, 从头开始
"""
import os, sys, json, time, re, argparse
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import platform
try:
    import fcntl; HAS_FCNTL = True
except ImportError: HAS_FCNTL = False
from openai import OpenAI

_HERE = os.path.dirname(__file__)
GAIT_DIR = os.path.normpath(os.path.join(_HERE, '..'))
SCRIPTS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))
THESIS_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
# fixed
BASE_DIR   = THESIS_DIR
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))  # api_config
import api_config as _cfg

API_KEY          = _cfg.API_KEY
API_URL          = _cfg.API_URL
MODEL            = _cfg.MODEL
MAX_TOKENS       = _cfg.MAX_TOKENS
SLEEP_SEC        = _cfg.SLEEP_SEC
TIMEOUT          = _cfg.TIMEOUT
DISABLE_THINKING = _cfg.DISABLE_THINKING
TEMPERATURE      = _cfg.TEMPERATURE

CLASS_NAMES = ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand',
    'Pick','Jump','Push-up','Sit-up','Walk','Walk-backwards','Walk-circle',
    'Run','Stair-up','Stair-down','Table-tennis']
N_CLS = len(CLASS_NAMES)

# ===== 命令行参数 =====
ap = argparse.ArgumentParser()
ap.add_argument('--class', type=int, required=True, dest='target_class')
ap.add_argument('--force', action='store_true')
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
TARGET_CLS = args.target_class
FORCE_RESTART = args.force
QUICK_MODE = args.quick
QUICK_LIMIT = 50

assert 0 <= TARGET_CLS < N_CLS, f"class must be 0-17, got {TARGET_CLS}"

# ===== 输出路径 (在 kuhar/output/per_class/class_N/) =====
OUT_BASE = os.path.join(SCRIPT_DIR, 'output')
CLASS_DIR = os.path.join(OUT_BASE, 'per_class', f'class_{TARGET_CLS}')
LOG_DIR   = os.path.join(OUT_BASE, 'logs')
CKPT_DIR  = os.path.join(OUT_BASE, 'checkpoints')

for d in [CLASS_DIR, LOG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

SOFT_FILE     = os.path.join(CLASS_DIR, 'soft_all.npy')
LOG_ALL       = os.path.join(CLASS_DIR, 'log_all.txt')
LOG_FILTERED  = os.path.join(CLASS_DIR, 'log_filtered.txt')
LOG_CORRECT   = os.path.join(CLASS_DIR, 'log_correct.txt')
CKPT_FILE     = os.path.join(CKPT_DIR, f'ckpt_class_{TARGET_CLS}.json')
LOCK_FILE     = os.path.join(CLASS_DIR, '.lock')

# 质量筛选阈值
FILTER_ENT  = 1.5
FILTER_GAP  = 0.05
FILTER_CONF = 0.5

# ===== 工具函数 (复用 v2.1) =====
def entropy(probs):
    p = np.clip(probs, 1e-8, 1); return float(-(p*np.log(p)).sum())

def extract_probs(text):
    if not text: return None
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()
    for m in re.finditer(r'\{[^}]+\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                vals = [float(str(d[str(k)]).replace(',','.')) for k in range(N_CLS)]
                arr = np.clip(np.array(vals), 0, 1)
                if arr.sum()>0: return (arr/arr.sum()).tolist()
        except: pass
    return None

def call_api(prompt):
    from openai import RateLimitError
    for attempt in range(5):
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=TIMEOUT)
            r = client.chat.completions.create(model=MODEL, messages=[{'role':'user','content':prompt}],
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS, extra_body=DISABLE_THINKING)
            probs = extract_probs(r.choices[0].message.content.strip())
            if probs: return probs, None
            time.sleep(2)
        except RateLimitError: time.sleep(15*(2 if attempt>0 else 1))
        except Exception as e: time.sleep(5)
    return None, 'API failed'

def log_all(msg):
    ts=time.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_ALL,'a') as f: f.write(f"[{ts}] {msg}\n")

def log_filtered(msg):
    ts=time.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILTERED,'a') as f: f.write(f"[{ts}] {msg}\n")

def log_correct(msg):
    ts=time.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_CORRECT,'a') as f: f.write(f"[{ts}] {msg}\n")

def load_kuhar_data():
    """从预存文件加载 (先运行 prepare_per_class_data.py)"""
    # 加载本类的窗口数据和全局索引
    win_path = os.path.join(CLASS_DIR, 'windows.npy')
    idx_path = os.path.join(CLASS_DIR, 'indices.npy')
    label_path = os.path.join(OUT_BASE, 'train_labels.npy')
    
    if not os.path.exists(win_path):
        raise RuntimeError(f"请先运行 prepare_per_class_data.py: {win_path} 不存在")
    
    X_all = np.load(win_path)  # 仅本类的窗口
    indices = np.load(idx_path)  # 在全局train数组中的索引
    y_all = np.load(label_path)  # 全局标签 (57k)
    
    return X_all, y_all, indices

def compute_features(window):
    acc=window[:,1:4]; gyro=window[:,5:8]
    am=np.sqrt((acc**2).sum(1))
    am_m,am_s=am.mean(),am.std()
    acc_m=acc.mean(0); acc_s=acc.std(0); gyro_s=gyro.std(0)
    npa=int(np.sum((am[1:-1]>am[:-2])&(am[1:-1]>am[2:])))
    try: fft=np.abs(np.fft.rfft(am)); df=float(np.fft.rfftfreq(128,d=0.01)[np.argmax(fft[1:])+1])
    except: df=0.0
    def moms(s):
        m=s-s.mean(); m2=(m**2).mean(); m3=(m**3).mean(); m4=(m**4).mean()
        std=np.sqrt(m2)+1e-10; return float(m3/(std**3)),float(m4/(std**4)-3)
    _,kurt_az=moms(acc[:,2])
    return {'energy_acc':float((am**2).mean()),'impulsiveness':float(am.max()/(np.sqrt(am.mean()**2)+1e-10)),
        'z_grav':float(abs(acc_m[2])/(np.linalg.norm(acc_m)+1e-8)),
        'jerk':float(np.sqrt(np.mean((np.diff(acc,axis=0))**2))),
        'n_peaks_acc':npa,'dom_freq':df,'acc_mag_max':float(am.max()),
        'acc_mag_std':float(am_s),'acc_y_std':float(acc_s[1]),
        'gyro_x_std':float(gyro_s[0]),'gyro_y_std':float(gyro_s[1]),'kurt_az':kurt_az}

def build_prompt(window, hint=""):
    f = compute_features(window)
    e = f['energy_acc']
    return f"""You are a HAR expert. Classify this 1.28s waist-sensor window. CRITICAL: FIRST check energy to determine which TIER (1-4) the activity belongs to. Then ONLY compare with classes in THAT tier. Ignore classes from other tiers.

=== DATA-DRIVEN REFERENCE (p50 from actual KuHar) ===
 0=Stand      energy=0   z_grav=0.84 acc_y=0.02 impuls=2.30 gyro_x=0.01 gyro_y=0.01 jerk=0.02 n_peaks=33
 1=Sit        energy=0   z_grav=0.78 acc_y=0.02 impuls=2.39 gyro_x=0.00 gyro_y=0.01 jerk=0.02 n_peaks=36
 2=Talk-sit   energy=0   z_grav=0.72 acc_y=0.08 impuls=2.58 gyro_x=0.02 gyro_y=0.03 jerk=0.04 n_peaks=24
 3=Talk-stand energy=2   z_grav=0.61 acc_y=0.74 impuls=2.64 gyro_x=0.38 gyro_y=0.17 jerk=0.23 n_peaks=15
 4=Stand-sit  energy=2   z_grav=0.71 acc_y=0.34 impuls=3.04 gyro_x=0.11 gyro_y=0.77 jerk=0.21 n_peaks=18
 5=Lay        energy=0   z_grav=0.40 acc_y=0.02 impuls=2.67 gyro_x=0.00 gyro_y=0.01 jerk=0.02 n_peaks=32
 6=Lay-stand  energy=2   z_grav=0.54 acc_y=0.66 impuls=2.77 gyro_x=0.34 gyro_y=0.40 jerk=0.28 n_peaks=20
 7=Pick       energy=4   z_grav=0.75 acc_y=0.88 impuls=2.86 gyro_x=0.32 gyro_y=0.57 jerk=0.40 n_peaks=18
 8=Jump       energy=170 z_grav=0.71 acc_y=1.90 impuls=4.00 gyro_x=0.45 gyro_y=2.10 jerk=2.94 n_peaks=14
 9=Push-up    energy=9   z_grav=0.69 acc_y=0.74 impuls=2.08 gyro_x=0.24 gyro_y=0.59 jerk=0.35 n_peaks=20
10=Sit-up     energy=3   z_grav=0.64 acc_y=0.53 impuls=2.61 gyro_x=0.17 gyro_y=0.85 jerk=0.35 n_peaks=23
11=Walk       energy=21  z_grav=0.57 acc_y=1.89 impuls=2.43 gyro_x=0.49 gyro_y=0.42 jerk=0.97 n_peaks=16
12=Walk-back  energy=15  z_grav=0.67 acc_y=1.58 impuls=2.69 gyro_x=0.40 gyro_y=0.42 jerk=0.70 n_peaks=14
13=Walk-circle energy=16 z_grav=0.60 acc_y=1.82 impuls=2.49 gyro_x=0.53 gyro_y=0.38 jerk=0.79 n_peaks=16
14=Run        energy=300 z_grav=0.63 acc_y=5.96 impuls=3.11 gyro_x=1.47 gyro_y=2.36 jerk=3.95 n_peaks=17
15=Stair-up   energy=12  z_grav=0.49 acc_y=1.39 impuls=2.48 gyro_x=0.49 gyro_y=0.45 jerk=0.52 n_peaks=14
16=Stair-down energy=35  z_grav=0.61 acc_y=2.52 impuls=3.31 gyro_x=0.59 gyro_y=0.63 jerk=1.05 n_peaks=14
17=Ping-pong  energy=33  z_grav=0.46 acc_y=3.00 impuls=2.95 gyro_x=0.88 gyro_y=0.48 jerk=1.17 n_peaks=16

=== CURRENT WINDOW FEATURES ===
  energy={e:.1f}  z_grav={f['z_grav']:.3f}  acc_y_std={f['acc_y_std']:.3f}
  impulsiveness={f['impulsiveness']:.2f}  gyro_x_std={f['gyro_x_std']:.3f}  gyro_y_std={f['gyro_y_std']:.3f}
  jerk={f['jerk']:.3f}  n_peaks={f['n_peaks_acc']}  dom_freq={f['dom_freq']:.2f}

=== MATCHING RULES ===
FIRST DETERMINE TIER by energy:
  energy<0.1  → TIER 1 (static): compare with classes 0,1,2,5 only
  energy 0.1-5 → TIER 2 (transitional): compare with 3,4,6,7,10
  energy 5-50  → TIER 3 (dynamic): compare with 9,11,12,13,15,16,17
  energy>100  → TIER 4 (intense): compare with 8,14 ONLY

WITHIN EACH TIER, match by closest features:
TIER 1: z_grav<0.50→5=Lay. z_grav 0.70-0.78+energy>0.005+n_peaks<28→2=Talk-sit. z_grav>0.77+energy<0.005→1=Sit. else→0=Stand.
TIER 2: gyro_y>0.70+acc_y<0.45→4=Stand-sit. gyro_y>0.70+acc_y>0.45→10=Sit-up. energy>3+energy<6→7=Pick. z_grav<0.55→6=Lay-stand. else→3=Talk-stand.
TIER 3: impuls<2.25→9=Push-up. energy>25+gyro_x>0.75→17=Ping-pong. energy>25+gyro_x<0.75→16=Stairs-down. energy10-18+gyro_x<0.70→15=Stairs-up. energy18-25+acc_y>1.7→11=Walk. energy14-18→12=Walk-back or 13=Walk-circle.
TIER 4: impuls>3.5→8=Jump. else→14=Run.

{hint}

OUTPUT: explain which tier matches, then JSON: {{"0":p0,..."17":p17}} sum=1.0. Give high confidence only for definitive rules.
"""

def build_hint(true_label, pred_label, probs, f):
    hints = {0:"Stand: z_grav~0.84, energy<0.01",1:"Sit: z_grav~0.78",
        2:"Talk-sit: n_peaks<28,z_grav 0.70-0.78",3:"Talk-stand: energy~2,z_grav~0.61",
        4:"Stand-sit: gyro_y>0.70,acc_y<0.45",5:"Lay: z_grav<0.50 definitive",
        6:"Lay-stand: z_grav<0.55,energy~2",7:"Pick: energy>3,energy<6",
        8:"Jump: impuls>3.5,energy>100",9:"Push-up: impuls<2.25 UNIQUE",
        10:"Sit-up: gyro_y>0.70,acc_y>0.45",11:"Walk: energy 18-25,acc_y>1.7",
        12:"Walk-back: energy~15",13:"Walk-circle: energy~16",14:"Run: energy>100,impuls<3.5",
        15:"Stairs-up: energy 10-18",16:"Stairs-down: energy>25,gyro_x<0.75",
        17:"Ping-pong: energy>25,gyro_x>0.75"}
    return (f"REMINDER: True={CLASS_NAMES[true_label]}. "
            f"Features: energy={f['energy_acc']:.1f}, z_grav={f['z_grav']:.3f}. "
            f"Hints: {hints.get(true_label,'')}")

def main():
    lock_fd=open(LOCK_FILE,'w')
    if HAS_FCNTL:
        try: fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: print(f"class {TARGET_CLS} 已有进程在跑"); sys.exit(1)

    cname = CLASS_NAMES[TARGET_CLS]
    
    if FORCE_RESTART:
        for ff in [LOG_ALL, LOG_FILTERED, LOG_CORRECT, SOFT_FILE, CKPT_FILE]:
            if os.path.exists(ff): open(ff,'w').close()

    log_all(f"Class {TARGET_CLS} ({cname}) 软标签生成开始")
    log_all(f"  QUICK={QUICK_MODE} LIMIT={QUICK_LIMIT}")
    log_all(f"  温度={TEMPERATURE} (from api_config)")

    # 从预存文件加载 (仅本类数据, ~13MB)
    X_cls, y_all, global_indices = load_kuhar_data()
    log_all(f"  本类: {len(X_cls)}窗口 (仅加载本类, 内存友好)")

    # 本类在全局数组中的索引
    np.random.seed(42 + TARGET_CLS)
    take = min(QUICK_LIMIT, len(X_cls)) if QUICK_MODE else len(X_cls)
    chosen = np.random.choice(len(X_cls), size=take, replace=False)
    sample_local_indices = np.random.permutation(chosen)  # 本地索引
    sample_global_indices = global_indices[sample_local_indices]  # 全局索引
    total = len(sample_local_indices)
    log_all(f"  实际生成: {total} 窗口")

    # 断点续传
    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f: ckpt = json.load(f)
            done_set = set(ckpt.get('done', []))
            log_all(f"  续传: {len(done_set)}/{total}")
        except: pass

    # 计算全量数组大小
    global_N = len(y_all)
    soft_all = np.zeros((global_N, N_CLS), dtype=np.float32)
    done_count = 0
    true_correct = 0
    filtered_count = 0
    correct_indices = []

    # 恢复已有
    if done_set and os.path.exists(SOFT_FILE) and not QUICK_MODE:
        saved = np.load(SOFT_FILE)
        for gidx in done_set:
            if gidx < global_N and saved[gidx].sum() > 0:
                soft_all[gidx] = saved[gidx]
                done_count += 1
                if int(np.argmax(saved[gidx])) == TARGET_CLS:
                    true_correct += 1
                    correct_indices.append(gidx)

    for i, (local_idx, orig_idx) in enumerate(zip(sample_local_indices, sample_global_indices)):
        if QUICK_MODE and done_count >= QUICK_LIMIT: break
        if orig_idx in done_set: continue

        true_label = TARGET_CLS  # 本类
        window = X_cls[local_idx]  # 从预存的类数据读取
        prompt = build_prompt(window)

        probs, err = call_api(prompt)
        retry = 0
        while probs is None and retry < 3:
            time.sleep(5)
            probs, err = call_api(prompt)
            retry += 1

        if probs is None:
            log_all(f"FAIL idx={orig_idx}")
            soft_all[orig_idx, true_label] = 1.0
            done_set.add(orig_idx); done_count += 1
            continue

        ent = entropy(probs); max_prob = max(probs)
        gap = sorted(enumerate(probs), key=lambda x:-x[1])[0][1] - sorted(enumerate(probs), key=lambda x:-x[1])[1][1]
        pred_label = int(np.argmax(probs))
        ok = (pred_label == true_label)

        # Self-correction (1 round)
        if not ok:
            hint = build_hint(true_label, pred_label, probs, compute_features(window))
            probs2, _ = call_api(build_prompt(window, hint=hint))
            if probs2 is not None:
                ent2 = entropy(probs2); max_prob2 = max(probs2)
                gap2 = sorted(enumerate(probs2), key=lambda x:-x[1])[0][1] - sorted(enumerate(probs2), key=lambda x:-x[1])[1][1]
                pred2 = int(np.argmax(probs2))
                if pred2 == true_label and max_prob2 > 0.6:
                    probs, ent, max_prob, gap = probs2, ent2, max_prob2, gap2
                    pred_label = pred2; ok = True
            time.sleep(SLEEP_SEC)

        if not QUICK_MODE:
            soft_all[orig_idx] = probs
        done_set.add(orig_idx); done_count += 1

        # 三版日志
        status = "✓" if ok else "✗"
        base_line = (f"#{done_count:04d}/{total:05d} | true={cname}({TARGET_CLS}) | "
                     f"pred={CLASS_NAMES[pred_label]}({pred_label}) | {status} | "
                     f"ent={ent:.3f} | conf={max_prob:.3f} | gap={gap:.3f}")

        # A版: 全部
        log_all(base_line)

        # B版: 通过筛选的
        if ent < FILTER_ENT and gap > FILTER_GAP and max_prob > FILTER_CONF:
            log_filtered(base_line)
            filtered_count += 1

        # C版: 预测正确的
        if ok:
            log_correct(base_line)
            true_correct += 1
            correct_indices.append(orig_idx)

        if done_count % 20 == 0:
            acc = true_correct / max(done_count, 1) * 100
            log_all(f"  [{done_count}/{total}] acc={true_correct}/{done_count}={acc:.1f}% filtered={filtered_count}/{done_count}")

        if not QUICK_MODE:  # 每次迭代都保存checkpoint, 防止崩溃后日志编号重复
            np.save(SOFT_FILE, soft_all)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': [int(x) for x in done_set], 'correct': [int(x) for x in correct_indices]}, f)

        time.sleep(SLEEP_SEC)

    if not QUICK_MODE:
        np.save(SOFT_FILE, soft_all)
        with open(CKPT_FILE, 'w') as f:
            json.dump({'done': [int(x) for x in done_set], 'correct': [int(x) for x in correct_indices]}, f)

    acc = true_correct / max(done_count, 1) * 100
    log_all(f"Done: {true_correct}/{done_count} ({acc:.1f}%) filtered={filtered_count}/{done_count}")
    log_correct(f"Done: {true_correct}/{done_count} ({acc:.1f}%)")
    log_filtered(f"Done: {filtered_count}/{done_count} kept")

if __name__ == '__main__':
    main()
