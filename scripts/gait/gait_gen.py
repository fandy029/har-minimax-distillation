#!/usr/bin/env python3
"""
Gait 软标签生成脚本
数据集: Gait_Classification (S1_Dataset + S2_Dataset)
类别: 0=sit_on_bed, 1=sit_on_chair, 2=lying, 3=ambulating

策略:
  - 每类 ≤3000 样本：全部生成
  - 每类 >3000 样本：采样 3000 个
  - 支持 --limit 和 --force 参数
"""
import os, sys, json, time, re
import numpy as np
import pandas as pd
import glob
import fcntl
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d
from openai import OpenAI

# ============ 项目路径 (相对) ============
_SCRIPT = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT)))
SCRIPT_DIR = os.path.dirname(_SCRIPT)

# ============ API 配置 (Mimo) ============
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
import api_config as _api_cfg
API_KEY     = _api_cfg.API_KEY
API_URL     = _api_cfg.API_URL
MODEL       = _api_cfg.MODEL
TEMPERATURE = _api_cfg.TEMPERATURE
MAX_TOKENS  = _api_cfg.MAX_TOKENS
SLEEP_SEC   = _api_cfg.SLEEP_SEC
TIMEOUT     = _api_cfg.TIMEOUT
DISABLE_THINKING = _api_cfg.DISABLE_THINKING

# ============ 类别配置 ============
CLASS_NAMES = {0: 'sit_on_bed', 1: 'sit_on_chair', 2: 'lying', 3: 'ambulating'}
N_CLS = 4
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
MAX_PER_CLASS = 3000   # 每类上限

# ============ 输出路径 ============
OUT_DIR = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR = os.path.join(BASE_DIR, 'results', 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SOFT_FILE    = os.path.join(OUT_DIR, 'gait_soft.npy')
CORRECT_FILE = os.path.join(OUT_DIR, 'gait_soft_correct_only.npy')
LOG_FILE     = os.path.join(LOG_DIR, 'gen_gait.log')
ERR_FILE     = os.path.join(LOG_DIR, 'gen_gait_errors.log')
CORR_LOG     = os.path.join(LOG_DIR, 'gen_gait_correct.log')
CKPT_FILE    = os.path.join(LOG_DIR, 'gen_gait_checkpoint.json')
LOCK_FILE    = os.path.join(OUT_DIR, '.gen_gait.lock')

# ============ 参数 ============
FORCE_RESTART = '--force' in sys.argv
LIMIT_OVERRIDE = None
for i, arg in enumerate(sys.argv):
    if arg == '--limit' and i + 1 < len(sys.argv):
        LIMIT_OVERRIDE = int(sys.argv[i + 1])
LIMIT = LIMIT_OVERRIDE if LIMIT_OVERRIDE is not None else MAX_PER_CLASS

# ============ 软标签有效性判断 ============
def is_valid(probs):
    if probs is None:
        return False
    row = np.array(probs)
    if not np.isclose(row.sum(), 1.0, atol=0.01):
        return False
    if row.max() >= 0.95:
        return False
    return True

# ============ JSON 解析 ============
def extract_probs(text):
    if not text:
        return None
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<RESULT>.*?</RESULT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()
    for m in re.finditer(r'\{[^}]+\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                vals = [float(d[str(k)]) for k in range(N_CLS)]
                s = np.clip(np.array(vals), 0, 1)
                if s.sum() > 0:
                    return (s / s.sum()).tolist()
        except:
            pass
    return None

# ============ 日志函数 ============
def log(msg, to_stdout=True):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    if to_stdout:
        print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def log_correct(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(CORR_LOG, 'a') as f:
        f.write(f"[{ts}] {msg}\n")

def log_err(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(ERR_FILE, 'a') as f:
        f.write(line + '\n')

# ============ 数据加载 ============
def load_gait_data():
    base = os.path.join(BASE_DIR, 'datasets', 'Gait_Classification')
    d, l = [], []
    for folder in ['S1_Dataset', 'S2_Dataset']:
        for f in glob.glob(os.path.join(base, folder, '*')):
            if f.endswith('.txt') or 'README' in f:
                continue
            try:
                df = pd.read_csv(f, header=None)
                acc = df.iloc[:, 1:4].values.astype(np.float32)
                labels = df.iloc[:, 8].astype(int)
                for i in range(0, len(df) - 127, 64):
                    w = acc[i:i + 128]
                    label = int(labels.iloc[i]) if hasattr(labels, 'iloc') else int(labels[i])
                    if w.shape[0] == 128 and not np.any(np.isnan(w)) and label in LABEL_MAP:
                        d.append(w)
                        l.append(LABEL_MAP[label])
            except:
                continue
    X = np.array(d, dtype=np.float32)
    y = np.array(l, dtype=np.int64)
    if len(X) == 0:
        raise RuntimeError('No gait data loaded!')
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

# ============ 特征计算 ============
def compute_features(window):
    gfr = float(uniform_filter1d(window[:, 0], 10, axis=0).mean())
    gve = float(uniform_filter1d(window[:, 1], 10, axis=0).mean())
    gla = float(uniform_filter1d(window[:, 2], 10, axis=0).mean())
    free = window - np.array([gfr, gve, gla])
    free_mag = np.sqrt((free ** 2).sum(axis=1))
    acc_mag = np.sqrt((window ** 2).sum(axis=1))
    peaks = int(np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])))
    return {
        'gfr': gfr, 'gve': gve, 'gla': gla,
        'free_mag_mean': float(free_mag.mean()),
        'free_mag_std': float(free_mag.std()),
        'acc_mag_mean': float(acc_mag.mean()),
        'acc_mag_std': float(acc_mag.std()),
        'peaks': peaks,
    }

# ============ Prompt ============
def build_prompt(window):
    f = compute_features(window)
    descs = [f'{i}={CLASS_NAMES[i]}' for i in range(N_CLS)]
    return f'''You are classifying human body posture/activity from 3-axis accelerometer data.
(128 steps @ 40Hz, 3 channels: frontal=forward-back, vertical=up-down, lateral=left-right)

The {N_CLS} activity classes are:
  {", ".join(descs)}

=== PER-WINDOW FEATURES (measured) ===
  gfr (gravity frontal)      = {f['gfr']:+.4f}  G  (+ = forward tilt)
  gve (gravity vertical)     = {f['gve']:+.4f}  G  (+ = upright, − = inverted)
  gla (gravity lateral)      = {f['gla']:+.4f}  G  (+ = left tilt)
  free_mag_mean             = {f['free_mag_mean']:.4f}  G  (motion intensity, gravity-removed)
  free_mag_std              = {f['free_mag_std']:.4f}   (variability of motion)
  acc_mag_std               = {f['acc_mag_std']:.4f}   (total signal variability)
  peaks                     = {f['peaks']}       (gait cycle peaks in window)

=== PER-CLASS FEATURE SIGNATURES ===
Compare this window against each class:

  class 2 (lying):        gfr ~+0.82, gve ~+0.14, gla ~-0.36 — body horizontal, gve near 0 (no gravity along vertical), lateral tilt
  class 0 (sit_on_bed):   gve ~+0.77, gfr ~+0.47, gla ~0.00 — seated upright, slight forward lean, minimal lateral
  class 1 (sit_on_chair): gve ~+0.69, gfr ~+0.61, gla ~+0.04 — seated upright, more forward lean than bed, slight right lateral
  class 3 (ambulating):   gve ~+0.61, gfr ~+0.56, free_mag_std ~0.076 — walking/moving, most dynamic

=== OVERLAPPING PAIRS — how to distinguish ===
  • sit_on_bed vs sit_on_chair: both upright seated, gve similar (~0.7-0.8), but sit_on_bed has lower gfr (~0.47 vs ~0.61). Distinguish by gfr.
  • sitting vs lying: sitting has high gve (~0.7), lying has low gve (~0.14). Primary distinguish by gve.
  • ambulating vs sitting: ambulating has higher free_mag_std (~0.076), sitting is nearly still (free_mag_std ~0.04-0.07, but ambulating also has more peaks).

=== OUTPUT ===
Output ONLY valid JSON with {N_CLS} probabilities that sum to 1:
{{"0":p0,"1":p1,"2":p2,"3":p3}}
Do NOT output one-hot. Keep probabilities in the 0.05–0.75 range.
Give higher probability to classes whose feature signature best matches this window.
'''

# ============ API 调用 ============
def call_api(prompt):
    from openai import RateLimitError
    last_err = None
    for attempt in range(5):
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=TIMEOUT)
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body=DISABLE_THINKING,
            )
            content = r.choices[0].message.content or ''
            result = extract_probs(content)
            if result is not None:
                return result, None
            last_err = f'JSON解析失败 len={len(content)}'
            time.sleep(2)
        except RateLimitError:
            last_err = f'429限流 (attempt {attempt+1})'
            time.sleep(15 * (2 if attempt > 0 else 1))
        except Exception as e:
            last_err = f'API错误: {e}'
            time.sleep(5)
    return None, last_err

# ============ 主生成逻辑 ============
def main():
    # 单例锁
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: 已有实例在运行，请先停止后再启动")
        sys.exit(1)

    # --force: 清空日志和断点
    if FORCE_RESTART:
        for f in [LOG_FILE, ERR_FILE, CORR_LOG, SOFT_FILE, CORRECT_FILE, CKPT_FILE]:
            if os.path.exists(f):
                open(f, 'w').close()
        log(f"--force: 清除旧日志和断点，从头开始", to_stdout=False)

    log(f"Gait 软标签生成开始 (scripts/gait/gait_gen.py)")
    log(f"Mimo API: {API_URL}")
    log(f"Model: {MODEL}")
    log(f"Temperature: {TEMPERATURE}")

    # 加载数据
    log("加载 Gait 数据...")
    X, y, _, _, _, _ = load_gait_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")

    # 每类统计
    per_cls = {}
    for c in range(N_CLS):
        cnt = int(np.sum(y == c))
        per_cls[c] = cnt
        log(f"  class {c} ({CLASS_NAMES[c]}): {cnt} 样本")

    # 采样：每类上限 LIMIT 个
    np.random.seed(42)
    sample_indices = []
    for c in range(N_CLS):
        cidx = np.where(y == c)[0]
        take = min(LIMIT, len(cidx))
        chosen = np.random.choice(cidx, size=take, replace=False)
        sample_indices.extend(chosen.tolist())
    sample_indices = np.array(sample_indices)
    np.random.shuffle(sample_indices)
    total = len(sample_indices)
    log(f"总计采样: {total} 个窗口（每类 ~{LIMIT}）")

    # 加载断点
    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f:
                ckpt = json.load(f)
            done_set = set(ckpt.get('done', []))
            log(f"  断点续传: {len(done_set)}/{total} 已处理")
        except (json.JSONDecodeError, IOError):
            log(f"  断点文件损坏，从头开始")
    else:
        log(f"  断点续传: 无，从头开始")

    # 初始化
    soft_all = np.zeros((len(X), N_CLS), dtype=np.float32)

    # 续跑恢复统计
    done_count = 0
    true_correct = 0
    correct_indices = []
    class_gen = [0] * N_CLS
    class_corr = [0] * N_CLS
    if done_set and os.path.exists(SOFT_FILE):
        saved = np.load(SOFT_FILE)
        for idx in done_set:
            if saved[idx].sum() > 0:
                soft_all[idx] = saved[idx]
                tl = int(y[idx])
                pl = int(np.argmax(saved[idx]))
                class_gen[tl] += 1
                done_count += 1
                if pl == tl:
                    true_correct += 1
                    class_corr[tl] += 1
                    correct_indices.append(idx)
        log(f"  续跑恢复: 完成={done_count}, 正确={true_correct}, 正确样本数={len(correct_indices)}")

    for pos, orig_idx in enumerate(sample_indices):
        if orig_idx in done_set:
            continue

        true_label = int(y[orig_idx])
        prompt = build_prompt(X[orig_idx])

        # API 调用
        raw_result, err = call_api(prompt)
        retry_count = 0
        while raw_result is None and retry_count < 3:
            time.sleep(5)
            raw_result, err = call_api(prompt)
            retry_count += 1

        if raw_result is None:
            log_err(f"API_FAILED idx={orig_idx} true={true_label} err={err} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            done_set.add(orig_idx)
            continue

        probs = raw_result if isinstance(raw_result, list) else extract_probs(raw_result)
        if not is_valid(np.array(probs)):
            for _ in range(5):
                time.sleep(2)
                rr2, _ = call_api(prompt)
                if rr2:
                    p2 = rr2 if isinstance(rr2, list) else extract_probs(rr2)
                    if p2 and is_valid(np.array(p2)):
                        probs = p2
                        raw_result = rr2
                        break

        if not is_valid(np.array(probs)):
            log_err(f"ONEHOT_REJECT idx={orig_idx} true={true_label} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            class_gen[true_label] += 1
        else:
            soft_all[orig_idx] = probs
            pred_label = int(np.argmax(probs))
            ok = "✓" if pred_label == true_label else "✗"
            ent = float(-(np.array(probs) * np.log(np.clip(probs, 1e-8, 1))).sum())
            top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
            line = (f"  [{len(done_set)+1}/{total}] idx={orig_idx} "
                    f"true={true_label}({CLASS_NAMES[true_label]}) "
                    f"pred={pred_label}({CLASS_NAMES[pred_label]})[{ok}] "
                    f"ent={ent:.3f} "
                    f"top=[{top2[0][0]}:{top2[0][1]:.3f},{top2[1][0]}:{top2[1][1]:.3f}]")
            log(line)
            class_gen[true_label] += 1
            if ok == "✓":
                true_correct += 1
                class_corr[true_label] += 1
                correct_indices.append(orig_idx)
                log_correct(line)
            else:
                pass

        done_set.add(orig_idx)
        done_count += 1

        # 每100个打印一次每类统计
        if done_count % 100 == 0:
            stats = []
            for c in range(N_CLS):
                g = class_gen[c]
                cc = class_corr[c]
                pct = cc / g * 100 if g > 0 else 0
                stats.append(f"{CLASS_NAMES[c]}={cc}/{g}({pct:.0f}%)")
            acc_pct = true_correct / done_count * 100
            log(f"  === 100轮统计: 整体={true_correct}/{done_count}({acc_pct:.1f}%) " + " ".join(stats))

        # 每5个保存一次
        if done_count % 5 == 0:
            np.save(SOFT_FILE, soft_all)
            soft_correct = soft_all[correct_indices]
            np.save(CORRECT_FILE, soft_correct)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': [int(x) for x in done_set], 'class_names': CLASS_NAMES}, f)
            valid_n = int((soft_all.sum(axis=1) > 0).sum())
            acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
            log(f"  进度: {len(done_set)}/{total}, 准确率={acc_pct:.1f}%, 正确样本={len(correct_indices)}, 已保存")

        time.sleep(SLEEP_SEC)

if __name__ == '__main__':
    main()
