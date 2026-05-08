#!/usr/bin/env python3
"""
PAMAP2 软标签生成脚本
IMU 传感器: 加速度计 + 陀螺仪 (6通道), 100Hz, 128步窗口
5活动: lying, sitting, standing, walking, jogging

用法: python pamap2_gen.py [--force]
"""
import os, sys, json, time, re
import numpy as np
import pandas as pd
from glob import glob
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

CLASS_NAMES = ['lying', 'sitting', 'standing', 'walking', 'jogging']
N_CLS = len(CLASS_NAMES)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
MAX_PER_CLASS = 3000

OUT_DIR  = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR  = os.path.join(BASE_DIR, 'results', 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SOFT_FILE    = os.path.join(OUT_DIR, 'pamap2_soft.npy')
CORRECT_FILE = os.path.join(OUT_DIR, 'pamap2_soft_correct_only.npy')
LOG_FILE     = os.path.join(LOG_DIR, 'gen_pamap2.log')
FINAL_FILE = os.path.join(LOG_DIR, 'gen_pamap2_final.log')
ERR_FILE     = os.path.join(LOG_DIR, 'gen_pamap2_errors.log')
CORR_LOG     = os.path.join(LOG_DIR, 'gen_pamap2_correct.log')
CKPT_FILE    = os.path.join(LOG_DIR, 'gen_pamap2_checkpoint.json')
LOCK_FILE    = os.path.join(OUT_DIR, '.gen_pamap2.lock')

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
    if not text:
        return None
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
                if s.sum() > 0:
                    return (s / s.sum()).tolist()
        except:
            pass
    return None


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    if sys.stdout.isatty():
        print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def log_correct(msg):
    with open(CORR_LOG, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def log_final(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(FINAL_FILE, 'a') as f:
        f.write(f'[{ts}] {msg}\n')

def log_err(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if sys.stdout.isatty():
        print(line)
    with open(ERR_FILE, 'a') as f:
        f.write(line + '\n')


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
    """
    PAMAP2 数据加载
    原始数据: column 1=activity, columns 9-14=IMU (acc_x/y/z + gyro_x/y/z, 6通道)
    IMU: 心率带上的加速度计 + 陀螺仪, 100Hz
    活动: 1=lying, 2=sitting, 3=standing, 4=walking, 5=jogging
    """
    base = os.path.join(BASE_DIR, 'datasets', 'PAMAP2', 'PAMAP2_Dataset')
    d, l = [], []

    for folder in ['Protocol', 'Optional']:
        folder_path = os.path.join(base, folder)
        if not os.path.exists(folder_path):
            continue
        files = sorted(glob(os.path.join(folder_path, '*.dat')))
        for f in files:
            try:
                df = pd.read_csv(f, sep=' ', header=None)
                acts = df.iloc[:, 1].values
                imu = df.iloc[:, 9:15].values.astype(np.float32)
                for aid, cid in LABEL_MAP.items():
                    mask = (acts == aid)
                    idx = np.where(mask)[0]
                    for s in range(0, len(idx) - 127, 64):
                        w = imu[idx[s:s + 128]]
                        if w.shape[0] == 128 and not np.any(np.isnan(w)):
                            d.append(w)
                            l.append(cid)
            except Exception:
                pass

    X = np.array(d, dtype=np.float32)
    y = np.array(l, dtype=np.int64)

    if len(X) == 0:
        raise RuntimeError("PAMAP2 数据加载失败")

    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te


# ============ 特征计算 ============
def compute_features(window):
    acc = window[:, :3]
    gyro = window[:, 3:6]
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))

    energy = float((acc_mag ** 2).mean())
    jerk = float(np.sqrt(np.mean(np.diff(acc, axis=0) ** 2)))
    n_peaks = int(np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])))
    gyro_mag_mean = float(gyro_mag.mean())
    gyro_x_mean = float(gyro[:, 0].mean())

    fft_v = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag) // 2])
    dom_freq = float(np.fft.fftfreq(len(acc_mag), 1 / 100)[np.argmax(fft_v) + 1]) if len(fft_v) > 0 and fft_v.max() > 0 else 0.0

    impulsiveness = float(acc_mag.max() / (acc_mag.std() + 1e-8))

    acc_std = float(acc.std())

    return {
        'energy': energy,
        'jerk': jerk,
        'n_peaks': n_peaks,
        'gyro_mag': gyro_mag_mean,
        'gyro_x': gyro_x_mean,
        'gyro_std': float(gyro.std()),
        'gyro_y': float(gyro[:, 1].mean()),
        'gyro_z': float(gyro[:, 2].mean()),
        'corr_gyro_yz': float(np.corrcoef(gyro[:,1], gyro[:,2])[0,1]),
        'corr_accz_gyroy': float(np.corrcoef(acc[:,2], gyro[:,1])[0,1]),
        'acc_std': acc_std,
        'acc_range': float(acc.max() - acc.min()),
        'dom_freq': dom_freq,
        'impulsiveness': impulsiveness,
        'acc_mean_x': float(acc[:, 0].mean()),
        'acc_mean_y': float(acc[:, 1].mean()),
        'acc_mean_z': float(acc[:, 2].mean()),
    }


# ============ Prompt 构建 ============
def build_prompt(window):
    f = compute_features(window)
    descs = ', '.join([f'{i}:{CLASS_NAMES[i]}' for i in range(N_CLS)])

    # 判断能量等级
    en = f['energy']
    if en < 10:
        band = 'A'
    elif en < 18:
        band = 'B'
    else:
        band = 'C'

    return f'''Classify hand-IMU data

IMPORTANT: Do NOT take shortcuts. Analyze the actual sensor feature VALUES — do NOT just guess the most common class. Think step by step through the features, then output your calibrated probabilities. (128@100Hz, 3acc+3gyro).

Classes: {descs}

=== MEASURED ===
  jerk={f['jerk']:.4f}  | dom_freq={f['dom_freq']:.2f}Hz | gyro_std={f['gyro_std']:.1f}
  acc_mean_x={f['acc_mean_x']:.2f} | gyro_y={f['gyro_y']:.1f} | acc_range={f['acc_range']:.3f}

=== PRIORITY DECISION TREE ===
FIRST: jerk > 0.10 → DYNAMIC (walk/jog). Otherwise STATIC (lying/sit/stand).

IF STATIC (jerk <= 0.10):
  MOST IMPORTANT: acc_mean_x. Median: standing=2.4, lying=7.3, sitting=6.9.
  • acc_mean_x < 4.0 → STRONG standing signal (hand hangs near vertical)
  • acc_mean_x > 5.0 → lying or sitting (hand more horizontal)
  • gyro_std < 16 AND acc_mean_x < 4 → standing
  • gyro_std > 16 AND acc_mean_x > 5 → lying/sitting (give ~0.5/0.5)

IF DYNAMIC (jerk > 0.10):
  • dom_freq <= 1.5Hz → walking (arm swing ~0.8Hz)
  • dom_freq > 1.5Hz → jogging (arm swing ~2-3Hz)
  • If dom_freq ambiguous, use jerk: jerk>0.25→jogging, jerk<0.25→walking

No probability > 0.90. If features match multiple classes, distribute probability.

---REASONING---
1: [Static or Dynamic?]
2: [Decision based on tree above]
---PROBABILITIES---
{{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4}}'''


# ============ 主循环 ============
def main():
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("已有实例在运行")
        sys.exit(1)

    if FORCE_RESTART:
        for f in [LOG_FILE, FINAL_FILE, ERR_FILE, CORR_LOG, SOFT_FILE, CORRECT_FILE, CKPT_FILE, LOCK_FILE]:
            if os.path.exists(f):
                open(f, 'w').close()

    log(f"PAMAP2 软标签生成开始")
    log(f"Mimo API: {API_URL} | Model: {MODEL} | T={TEMPERATURE}")

    X, y, _, _, _, _ = load_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")

    for c in range(N_CLS):
        cnt = int(np.sum(y == c))
        log(f"  class {c} ({CLASS_NAMES[c]}): {cnt} 样本")

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
    else:
        log(f"  断点续传: 无，从头开始")

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
        if orig_idx in done_set:
            continue

        true_label = int(y[orig_idx])
        raw_result, err = call_api(build_prompt(X[orig_idx]))

        retry_count = 0
        while raw_result is None and retry_count < 2:
            time.sleep(5)
            raw_result, err = call_api(build_prompt(X[orig_idx]))
            retry_count += 1

        if raw_result is None:
            log_err(f"API_FAILED idx={orig_idx} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            done_set.add(orig_idx)
            done_count += 1
            continue

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
        line = (f"  [{done_count+1:03d}/{total}] | true={CLASS_NAMES[true_label]}({true_label}) | pred={CLASS_NAMES[pred_label]}({pred_label}) | {ok:>2} | ent={ent:.2f} | top={top2[0][0]}:{top2[0][1]:.2f}, {top2[1][0]}:{top2[1][1]:.2f}")
        log(line)
        log_final(line)
        class_gen[true_label] += 1
        if ok == "✓":
            true_correct += 1
            class_corr[true_label] += 1
            correct_indices.append(orig_idx)
            log_correct(line)

        done_set.add(orig_idx)
        done_count += 1

        if done_count % 100 == 0:
            stats = [f"{CLASS_NAMES[c]}={class_corr[c]}/{class_gen[c]}({class_corr[c] / class_gen[c] * 100:.0f}%)"
                     for c in range(N_CLS) if class_gen[c] > 0]
            log(f"  100轮: {true_correct}/{done_count}({true_correct / done_count * 100:.1f}%) " + " ".join(stats))

        if done_count % 5 == 0:
            np.save(SOFT_FILE, soft_all)
            if correct_indices:
                np.save(CORRECT_FILE, soft_all[correct_indices])
            json.dump({'done': [int(x) for x in done_set]}, open(CKPT_FILE, 'w'))
            log(f"  进度: {done_count}/{total}, 准确率={true_correct / done_count * 100:.1f}%, 正确样本数={len(correct_indices)}, 已保存")

        time.sleep(SLEEP_SEC)

    # 最终保存
    np.save(SOFT_FILE, soft_all)
    if correct_indices:
        np.save(CORRECT_FILE, soft_all[correct_indices])
    json.dump({'done': [int(x) for x in done_set]}, open(CKPT_FILE, 'w'))

    log(f"\n=== 生成完成 ===")
    log(f"  mean_entropy={float(-(soft_all[done_set] * np.log(np.clip(soft_all[done_set], 1e-8, 1))).sum(axis=1).mean()):.3f}")
    log(f"  mean_max_prob={float(soft_all[done_set].max(axis=1).mean()):.3f}")
    log(f"  整体准确率: {true_correct}/{done_count} ({true_correct / done_count * 100:.1f}%)")
    log(f"  输出: {SOFT_FILE}")


if __name__ == '__main__':
    main()
