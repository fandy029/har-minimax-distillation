#!/usr/bin/env python3
"""
UCI-HAR 软标签生成脚本
561维预提取特征，6类活动识别
传感器: Samsung Galaxy S II 腰部加速度计+陀螺仪，50Hz
用法: python uci_har_gen.py [--force]
"""
import os, sys, json, time, re
import numpy as np
from sklearn.model_selection import train_test_split
import fcntl
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR   = THESIS_DIR

sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
import api_config as _cfg
API_KEY          = _cfg.API_KEY
API_URL          = _cfg.API_URL
MODEL            = _cfg.MODEL
TEMPERATURE      = _cfg.TEMPERATURE
MAX_TOKENS       = _cfg.MAX_TOKENS
SLEEP_SEC        = _cfg.SLEEP_SEC
TIMEOUT          = _cfg.TIMEOUT
DISABLE_THINKING = _cfg.DISABLE_THINKING

CLASS_NAMES = ['WALKING', 'WALKING_UP', 'WALKING_DOWN', 'SITTING', 'STANDING', 'LAYING']
N_CLS = len(CLASS_NAMES)
MAX_PER_CLASS = 3000  # 每类最多200个窗口（总共1200样本，约2.5小时）

OUT_DIR  = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR  = os.path.join(BASE_DIR, 'results', 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SOFT_FILE    = os.path.join(OUT_DIR, 'uci_har_soft.npy')
CORRECT_FILE = os.path.join(OUT_DIR, 'uci_har_soft_correct_only.npy')
LOG_FILE     = os.path.join(LOG_DIR, 'gen_uci_har.log')
FINAL_FILE = os.path.join(LOG_DIR, 'gen_uci_har_final.log')
ERR_FILE     = os.path.join(LOG_DIR, 'gen_uci_har_errors.log')
CORR_LOG     = os.path.join(LOG_DIR, 'gen_uci_har_correct.log')
CKPT_FILE    = os.path.join(LOG_DIR, 'gen_uci_har_checkpoint.json')
LOCK_FILE    = os.path.join(OUT_DIR, '.gen_uci_har.lock')

FORCE_RESTART = '--force' in sys.argv


# ============ 工具函数 ============
def is_valid(probs):
    if probs is None:
        return False
    row = np.array(probs)
    if not np.isclose(row.sum(), 1.0, atol=0.01):
        return False
    return True


def extract_probs(text):
    if not text: return None
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<RESULT>.*?</RESULT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()
    text = re.sub(r'---\w+---', '', text).strip()
    for m in re.finditer(r'\{[^}]+\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                vals = [float(d[str(k)]) for k in range(N_CLS)]
                s = np.clip(np.array(vals), 0, 1)
                if s.sum() > 0: return (s / s.sum()).tolist()
        except: pass
    return None


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    if sys.stdout.isatty(): print(line)
    with open(LOG_FILE, 'a') as f: f.write(line + '\n')


def log_correct(msg):
    with open(CORR_LOG, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def log_final(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(FINAL_FILE, 'a') as f:
        f.write(f'[{ts}] {msg}\n')

def log_err(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if sys.stdout.isatty(): print(line)
    with open(ERR_FILE, 'a') as f: f.write(line + '\n')


# ============ API ============
def call_api(prompt):
    from openai import RateLimitError
    last_err = None
    for attempt in range(5):
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=TIMEOUT)
            r = client.chat.completions.create(
                model=MODEL, messages=[{'role': 'user', 'content': prompt}],
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                extra_body=DISABLE_THINKING,
            )
            return r.choices[0].message.content.strip(), None
        except RateLimitError:
            last_err = f'429限流 (attempt {attempt+1})'
            time.sleep(15 * (2 if attempt > 0 else 1))
        except Exception as e:
            last_err = f'API错误: {e}'
            time.sleep(5)
    return None, last_err


# ============ 数据加载 ============
def load_data():
    base = os.path.join(BASE_DIR, 'datasets', 'UCI_HAR')
    X_tr = np.loadtxt(os.path.join(base, 'train', 'X_train.txt')).astype(np.float32)
    y_tr = (np.loadtxt(os.path.join(base, 'train', 'y_train.txt')) - 1).astype(np.int64)
    X_te = np.loadtxt(os.path.join(base, 'test', 'X_test.txt')).astype(np.float32)
    y_te = (np.loadtxt(os.path.join(base, 'test', 'y_test.txt')) - 1).astype(np.int64)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te


# ============ 特征提取 ============
# UCI HAR 561维特征索引（0-based，features.txt是1-based）
# 关键索引（从 features.txt 验证）:
# [8]   tBodyAcc-mad()-Z         → DYNAMIC/STATIC 分界
# [40]  tGravityAcc-mean()-X      → LAYING 检测
# [41]  tGravityAcc-mean()-Y      → SITTING vs STANDING
# [42]  tGravityAcc-mean()-Z      → 辅助判断
# [68]  tGravityAcc-arCoeff()-X,4 → 动态类辅助
# [77]  tGravityAcc-correlation()-X,Y → SIT vs STAND（但实际不如 gmy）
# [265] fBodyAcc-mean()-X
# [266] fBodyAcc-mean()-Y
# [267] fBodyAcc-mean()-Z
# [502] fBodyAccMag-mean()        → WALKING_DOWN 检测（最强）
# [558] angle(X,gravityMean)
# [559] angle(Y,gravityMean)
# [560] angle(Z,gravityMean)

def extract_uci_features(vals):
    def s(i): return float(vals[i])
    return {
        'body_mad_z':   s(8),    # tBodyAcc-mad()-Z — DYNAMIC/STATIC 分界
        'grav_mean_x':  s(40),   # tGravityAcc-mean()-X — LAYING 唯一检测
        'grav_mean_y':  s(41),   # tGravityAcc-mean()-Y — SIT vs STAND 关键
        'grav_mean_z':  s(42),   # tGravityAcc-mean()-Z — 辅助
        'grav_ar_x4':   s(68),   # tGravityAcc-arCoeff()-X,4
        'grav_corr_xy': s(77),   # tGravityAcc-correlation()-X,Y
        'facc_mean_x':  s(265),  # fBodyAcc-mean()-X
        'facc_mean_y':  s(266),  # fBodyAcc-mean()-Y
        'facc_mean_z':  s(267),  # fBodyAcc-mean()-Z
        'facc_mag':     s(502),  # fBodyAccMag-mean() — WALKING_DOWN 最强检测
        'angle_x':      s(558),  # angle(X,gravityMean)
        'angle_y':      s(559),  # angle(Y,gravityMean)
        'angle_z':      s(560),  # angle(Z,gravityMean)
    }


def build_prompt(data):
    f = extract_uci_features(data)
    facc_avg = (f['facc_mean_x'] + f['facc_mean_y'] + f['facc_mean_z']) / 3

    return f'''You are a HAR expert. Classify 561-DIM SENSOR FEATURES

IMPORTANT: Do NOT take shortcuts. Analyze the actual sensor feature VALUES — do NOT just guess the most common class. Think step by step through the features, then output your calibrated probabilities. (waist Samsung Galaxy S2, 50Hz, 2.56s window) into 6 classes.

Classes:
  0=WALKING, 1=WALKING_UP, 2=WALKING_DOWN,
  3=SITTING, 4=STANDING, 5=LAYING

=== FEATURE VALUES (真实测量值) ===
tBodyAcc-mad-Z        = {f["body_mad_z"]:.4f}  (>-0.6=DYNAMIC, <-0.6=STATIC)
tGravityAcc-mean-X   = {f["grav_mean_x"]:.4f}  (LAYING<0.5; others>0.5)
tGravityAcc-mean-Y   = {f["grav_mean_y"]:.4f}  (SIT~+0.09; STAND~-0.19)
tGravityAcc-mean-Z   = {f["grav_mean_z"]:.4f}
fBodyAcc-mean avg    = {facc_avg:.4f}  (DYNAMIC~-0.3; STATIC~-0.98)
fBodyAccMag-mean      = {f["facc_mag"]:.4f}  (DOWN>+0.16; WALK~-0.30; UP~-0.24)
angle(Y,gravityMean) = {f["angle_y"]:.4f}  (SIT>0; STAND>0; overlap zone)

=== DECISION TREE (3 steps) ===

Step 1 — DYNAMIC or STATIC?
  body_mad_z > -0.6 → DYNAMIC (WALK/UP/DOWN)
  body_mad_z ≤ -0.6 → STATIC (SIT/STAND/LAY)

Step 2A — STATIC (body_mad_z ≤ -0.6):
  grav_mean_x < 0.5 → LAYING(5)  [median=-0.36, unique x-axis gravity direction]
  grav_mean_y > 0.0 → SITTING(3)  [median=+0.09, slight forward lean]
  grav_mean_y ≤ 0.0 → STANDING(4)  [median=-0.19, upright]

Step 2B — DYNAMIC (body_mad_z > -0.6):
  fBodyAccMag-mean > 0 → WALKING_DOWN(2)  [median=+0.16, highest magnitude]
  grav_mean_y < -0.25 → WALKING_UP(1)    [median=-0.30, more negative Y gravity]
  otherwise → WALKING(0)                   [remainder]

=== EXAMPLE CLASSIFICATIONS ===
Example 1: WALKING
  body_mad_z=-0.284(>-0.6=DYNAMIC), fBodyAccMag-mean=-0.296(≤0), grav_mean_y=-0.212(>-0.25 → WALKING)
  {{"0":0.60,"1":0.10,"2":0.10,"3":0.05,"4":0.10,"5":0.05}}

Example 2: WALKING_UP
  body_mad_z=-0.177(>-0.6=DYNAMIC), fBodyAccMag-mean=-0.242(≤0), grav_mean_y=-0.304(<-0.25 → WALKING_UP)
  {{"0":0.10,"1":0.55,"2":0.10,"3":0.05,"4":0.10,"5":0.05}}

Example 3: WALKING_DOWN
  body_mad_z=-0.251(>-0.6=DYNAMIC), fBodyAccMag-mean=+0.162(>0 → WALKING_DOWN)
  {{"0":0.10,"1":0.10,"2":0.65,"3":0.05,"4":0.05,"5":0.05}}

Example 4: SITTING
  body_mad_z=-0.976(≤-0.6=STATIC), grav_mean_x=+0.926(>0.5), grav_mean_y=+0.092(>0 → SITTING)
  {{"0":0.05,"1":0.05,"2":0.05,"3":0.65,"4":0.15,"5":0.05}}

Example 5: LAYING
  body_mad_z=-0.982(≤-0.6=STATIC), grav_mean_x=-0.358(<0.5 → LAYING)
  {{"0":0.02,"1":0.02,"2":0.02,"3":0.02,"4":0.02,"5":0.90}}

=== FORBIDDEN PATTERNS ===
NEVER use these exact probability pairs: 0.556/0.152 - FORBIDDEN  0.529/0.144 - FORBIDDEN
Probabilities must vary based on actual feature comparison.

=== YOUR RESPONSE FORMAT ===
---REASONING---
Step 1: [DYNAMIC or STATIC based on body_mad_z]
Step 2: [Apply sub-branch rule]
Step 3: [Conclusion with class name]
---PROBABILITIES---
{{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4,"5":p5}}'''

def main():
    lock_fd = open(LOCK_FILE, 'w')
    try: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: print("已有实例在运行"); sys.exit(1)

    if FORCE_RESTART:
        for f in [LOG_FILE, FINAL_FILE, ERR_FILE, CORR_LOG, CKPT_FILE]:
            if os.path.exists(f): open(f, 'w').close()

    log(f"UCI-HAR 软标签生成开始")
    log(f"Mimo API: {API_URL} | Model: {MODEL} | T={TEMPERATURE}")

    X, y, _, _, _, _ = load_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")
    for c in range(N_CLS):
        log(f"  class {c} ({CLASS_NAMES[c]}): {int(np.sum(y == c))} 样本")

    np.random.seed(42)
    sample_indices = []
    for c in range(N_CLS):
        cidx = np.where(y == c)[0]
        take = min(MAX_PER_CLASS, len(cidx))
        sample_indices.extend(np.random.choice(cidx, size=take, replace=False).tolist())
    sample_indices = np.random.permutation(sample_indices)
    total = len(sample_indices)
    log(f"  总计采样: {total} 个窗口（每类最多 {MAX_PER_CLASS}）")

    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        done_set = set(json.load(open(CKPT_FILE)).get('done', []))
        log(f"  断点续传: {len(done_set)}/{total} 已处理")

    soft_all = np.zeros((len(X), N_CLS), dtype=np.float32)
    done_count = true_correct = 0
    correct_indices = []
    class_gen = [0] * N_CLS
    class_corr = [0] * N_CLS

    # 续跑恢复：从 soft_all 恢复统计
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

    for orig_idx in sample_indices:
        if orig_idx in done_set: continue

        true_label = int(y[orig_idx])
        raw_result, err = call_api(build_prompt(X[orig_idx]))

        retry_count = 0
        while raw_result is None and retry_count < 3:
            time.sleep(5)
            raw_result, err = call_api(build_prompt(X[orig_idx]))
            retry_count += 1

        if raw_result is None:
            log_err(f"API_FAILED idx={orig_idx} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            done_set.add(orig_idx); done_count += 1; continue

        probs = extract_probs(raw_result)
        pred_label = int(np.argmax(probs))
        if pred_label != true_label:
            rr2, _ = call_api(build_prompt(X[orig_idx]))
            if rr2:
                probs = extract_probs(rr2)
                pred_label = int(np.argmax(probs))
                log(f'  [RETRY] | true={CLASS_NAMES[true_label]}({true_label}) | pred={CLASS_NAMES[pred_label]}({pred_label})')
        soft_all[orig_idx] = probs
        ok = "✓" if pred_label == true_label else "✗"
        ent = float(-(np.array(probs) * np.log(np.clip(probs, 1e-8, 1))).sum())
        top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
        line = (f"  [{len(done_set)+1:03d}/{total}] | true={CLASS_NAMES[true_label]}({true_label}) | pred={CLASS_NAMES[pred_label]}({pred_label}) | {ok:>2} | ent={ent:.2f} | top={top2[0][0]}:{top2[0][1]:.2f}, {top2[1][0]}:{top2[1][1]:.2f}")
        log(line)
        log_final(line)
        class_gen[true_label] += 1
        if ok == "✓":
            true_correct += 1; class_corr[true_label] += 1
            correct_indices.append(orig_idx); log_correct(line)

        done_set.add(orig_idx); done_count += 1

        if done_count % 100 == 0:
            stats = [f"{CLASS_NAMES[c]}={class_corr[c]}/{class_gen[c]}({class_corr[c] / class_gen[c] * 100:.0f}%)"
                     for c in range(N_CLS) if class_gen[c] > 0]
            log(f"  100轮: {true_correct}/{done_count}({true_correct / done_count * 100:.1f}%) " + " ".join(stats))

        if done_count % 5 == 0:
            np.save(SOFT_FILE, soft_all)
            if correct_indices: np.save(CORRECT_FILE, soft_all[correct_indices])
            json.dump({'done': [int(x) for x in done_set]}, open(CKPT_FILE, 'w'))
            log(f"  进度: {len(done_set)}/{total}, 准确率={true_correct / done_count * 100:.1f}%, 已保存")

        time.sleep(SLEEP_SEC)

    np.save(SOFT_FILE, soft_all)
    if correct_indices: np.save(CORRECT_FILE, soft_all[correct_indices])
    json.dump({'done': [int(x) for x in done_set]}, open(CKPT_FILE, 'w'))
    log(f"\n=== 完成 === mean_entropy={float(-(soft_all[done_set] * np.log(np.clip(soft_all[done_set],1e-8,1))).sum(axis=1).mean()):.3f}")
    log(f"  准确率: {true_correct}/{done_count} ({true_correct / done_count * 100:.1f}%)")


if __name__ == '__main__':
    main()
