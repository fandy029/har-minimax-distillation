#!/usr/bin/env python3
"""
MotionSense 软标签生成脚本
数据集: /datasets/MotionSense/
类别: downstairs, jogging, sitting, standing, upstairs, walking (6类)
通道: x/y/z userAcceleration, 128步窗口, 20Hz

用法: python gen.py [--force]
"""
import os, sys, json, time, re
import numpy as np, pandas as pd
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

CLASS_NAMES = ['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']
N_CLS       = len(CLASS_NAMES)
LABEL_MAP   = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}
MAX_PER_CLASS = 3000

OUT_DIR  = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR  = os.path.join(BASE_DIR, 'results', 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SOFT_FILE    = os.path.join(OUT_DIR, 'motionsense_soft.npy')
CORRECT_FILE = os.path.join(OUT_DIR, 'motionsense_soft_correct_only.npy')
LOG_FILE     = os.path.join(LOG_DIR, 'gen_motionsense.log')
ERR_FILE     = os.path.join(LOG_DIR, 'gen_motionsense_errors.log')
CORR_LOG     = os.path.join(LOG_DIR, 'gen_motionsense_correct.log')
CKPT_FILE    = os.path.join(LOG_DIR, 'gen_motionsense_checkpoint.json')
LOCK_FILE    = os.path.join(OUT_DIR, '.gen_motionsense.lock')

FORCE_RESTART = '--force' in sys.argv


# ============ 工具函数 ============
def is_valid(probs):
    if probs is None:
        return False
    row = np.array(probs)
    if not np.isclose(row.sum(), 1.0, atol=0.01):
        return False
    if row.max() >= 0.95:
        return False
    return True

def extract_probs(text):
    """从 API 响应中提取概率向量"""
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

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    # nohup 时 stdout 重定向到日志文件，避免重复写入
    if sys.stdout.isatty():
        print(line)
    with open(LOG_FILE, 'a') as f: f.write(line + '\n')

def log_correct(msg):
    with open(CORR_LOG, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def log_err(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if sys.stdout.isatty():
        print(line)
    with open(ERR_FILE, 'a') as f: f.write(line + '\n')


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
    MotionSense 数据加载
    数据来源: /datasets/MotionSense/
    结构: 每个活动一个文件夹 (dws_1, jog_1, sit_1, std_1, ups_1, wlk_1)
    每个文件夹内: sub_1.csv ~ sub_24.csv
    CSV 列: index, x, y, z (userAcceleration)
    采样率: 20Hz，窗口: 128步 (6.4秒)
    """
    base = os.path.join(BASE_DIR, 'datasets', 'MotionSense')
    d, l = [], []
    label_map = LABEL_MAP

    folders = sorted(glob(os.path.join(base, '*/')))
    for folder in folders:
        folder_name = os.path.basename(os.path.dirname(folder))
        label_prefix = folder_name.split('_')[0]
        if label_prefix not in label_map:
            continue
        label = label_map[label_prefix]

        csv_files = sorted(glob(os.path.join(folder, 'sub_*.csv')))
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                data = df.iloc[:, 1:4].values.astype(np.float32)  # (N, 3)
                for start in range(0, len(data) - 127, 64):
                    window = data[start:start + 128]
                    if window.shape[0] == 128 and not np.any(np.isnan(window)):
                        d.append(window)
                        l.append(label)
            except Exception:
                continue

    X = np.array(d, dtype=np.float32)
    y = np.array(l, dtype=np.int64)

    if len(X) == 0:
        raise RuntimeError("MotionSense 数据加载失败")

    # 80/20 分割后，再从训练集分 75/25 给 train/val（最终 60/20/20）
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te


# ============ Prompt 构建 ============
def build_prompt(data):
    """
    data: (128, 3) 窗口，x/y/z 加速度 (userAcceleration from iPhone waist, 20Hz)
    """
    acc = data  # (128, 3)
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))

    acc_mean = acc.mean(axis=0)
    acc_std = acc.std(axis=0)
    y_mean = acc[:, 1].mean()
    z_mean = acc[:, 2].mean()
    # xz_norm: 重力在x/z轴的合成幅度，standing时接近0(重力在y轴)，sitting时接近1(重力分散)
    xz_norm = float(np.sqrt(acc_mean[0]**2 + acc_mean[2]**2))

    n_peaks = int(np.sum(
        (acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])
    ))

    fft_v = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag) // 2])
    if len(fft_v) > 0 and fft_v.max() > 0:
        dom_freq_idx = np.argmax(fft_v) + 1
        dom_freq = float(np.fft.fftfreq(len(acc_mag), 1.0 / 20.0)[dom_freq_idx])
    else:
        dom_freq = 0.0

    # jerk: 加速度变化率的均值（平滑度指标，上楼最低，下楼中等，走路最高）
    jerk = np.diff(acc_mag)
    jerk_mean = float(np.abs(jerk).mean())

    class_list = ', '.join([f'{i}:{CLASS_NAMES[i]}' for i in range(N_CLS)])

    return f"""You are classifying human activity from 128-step (6.4 second) 20Hz accelerometer window (x, y, z axes, iPhone waist position, userAcceleration).
The {N_CLS} classes are: {class_list}

=== PER-WINDOW FEATURES (measured) ===
  magnitude_mean = {acc_mag.mean():.3f} G
  mean_x = {acc_mean[0]:.3f}  mean_y = {y_mean:.3f}  mean_z = {z_mean:.3f}  G
  std_x = {acc_std[0]:.3f}  std_y = {acc_std[1]:.3f}  std_z = {acc_std[2]:.3f}  G
  xz_norm = {xz_norm:.3f} G  (sqrt(x_mean^2+z_mean^2), standing~0, sitting~1)
  peak_count = {n_peaks}
  dom_freq = {dom_freq:.2f} Hz
  jerk_mean = {jerk_mean:.3f} G/s (motion smoothness: <0.18=smooth, >0.25=impact-heavy)

=== DECISION RULES (apply in order) ===
  1. std_x/y/z ALL < 0.05 -> sitting or standing (only if truly static)
       - y_mean > 0.7 AND xz_norm < 0.25 -> standing
       - y_mean < 0.5 AND xz_norm > 0.20 -> sitting
       - If ambiguous: prefer sitting if y_mean < 0.5, else prefer standing
  2. std_x > 0.7 AND std_y > 1.0 AND magnitude > 1.3 -> jogging
  3. jerk_mean < 0.18 AND std_y < 0.45 -> upstairs (smooth ascending)
  4. jerk_mean > 0.26 AND std_y > 0.55 -> walking (high variability)
  5. jerk_mean 0.18-0.25 AND std_y 0.30-0.55 AND dom_freq 0.7-1.5Hz -> downstairs
  6. When uncertain upstairs vs downstairs: upstairs has jerk<0.18, downstairs has jerk>0.20
  2. std_x > 0.7 AND std_y > 1.0 AND magnitude > 1.3 -> jogging
  3. jerk_mean < 0.18 AND std_y < 0.45 -> upstairs (controlled, smooth ascending motion)
  4. jerk_mean > 0.26 AND std_y > 0.55 -> walking (high variability, more impact)
  5. jerk_mean 0.18-0.25 AND std_y 0.30-0.55 AND dom_freq 0.7-1.5Hz -> downstairs
       (controlled descent: smoother than walking but more impact than upstairs)
  6. When uncertain between upstairs/downstairs: upstairs is SMOOTHER (jerk<0.18), downstairs has MORE impact (jerk>0.20)

=== OUTPUT ===
Output ONLY JSON with {N_CLS} probabilities summing to 1:
{{"0":p0,"1":p1,...}}
Keep probabilities in the 0.05-0.75 range. Do NOT output one-hot."""


# ============ 主循环 ============
def main():
    lock_fd = open(LOCK_FILE, 'w')
    try: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: print("已有实例在运行"); sys.exit(1)

    if FORCE_RESTART:
        for f in [LOG_FILE, ERR_FILE, CORR_LOG, CKPT_FILE]:
            if os.path.exists(f): open(f, 'w').close()

    log(f"MotionSense 软标签生成开始")
    log(f"Mimo API: {API_URL} | Model: {MODEL} | T={TEMPERATURE}")

    X, y, _, _, _, _ = load_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")

    for c in range(N_CLS):
        cnt = int(np.sum(y == c))
        log(f"  class {c} ({CLASS_NAMES[c]}): {cnt} 样本")

    # 每类采样
    np.random.seed(42)
    sample_indices = []
    for c in range(N_CLS):
        cidx = np.where(y == c)[0]
        take = min(MAX_PER_CLASS, len(cidx))
        sample_indices.extend(np.random.choice(cidx, size=take, replace=False).tolist())
    sample_indices = np.random.permutation(sample_indices)
    total = len(sample_indices)
    log(f"  总计采样: {total} 个窗口（每类最多 {MAX_PER_CLASS}）")

    # 断点续传
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
        if not is_valid(np.array(probs)):
            for _ in range(5):
                time.sleep(2)
                rr2, _ = call_api(build_prompt(X[orig_idx]))
                if rr2:
                    p2 = extract_probs(rr2)
                    if p2 and is_valid(np.array(p2)):
                        probs = p2; raw_result = rr2; break

        if not is_valid(np.array(probs)):
            log_err(f"ONEHOT_REJECT idx={orig_idx} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
        else:
            soft_all[orig_idx] = probs
            pred = int(np.argmax(probs))
            ok = "✓" if pred == true_label else "✗"
            ent = float(-(np.array(probs) * np.log(np.clip(probs, 1e-8, 1))).sum())
            top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
            line = (f"  [{done_count+1}/{total}] idx={orig_idx} "
                    f"true={true_label}({CLASS_NAMES[true_label]}) "
                    f"pred={pred}({CLASS_NAMES[pred]})[{ok}] "
                    f"ent={ent:.3f} "
                    f"top=[{top2[0][0]}:{top2[0][1]:.3f},{top2[1][0]}:{top2[1][1]:.3f}]")
            log(line)
            class_gen[true_label] += 1
            if ok == "✓":
                true_correct += 1; class_corr[true_label] += 1
                correct_indices.append(orig_idx); log_correct(line)

        done_set.add(orig_idx); done_count += 1

        if done_count % 100 == 0:
            stats = [f"{CLASS_NAMES[c]}={class_corr[c]}/{class_gen[c]}({class_corr[c]/class_gen[c]*100:.0f}%)"
                     for c in range(N_CLS) if class_gen[c] > 0]
            log(f"  100轮: {true_correct}/{done_count}({true_correct/done_count*100:.1f}%) " + " ".join(stats))

        if done_count % 5 == 0:
            np.save(SOFT_FILE, soft_all)
            np.save(CORRECT_FILE, soft_all[correct_indices])
            json.dump({'done': [int(x) for x in done_set]}, open(CKPT_FILE, 'w'))
            log(f"  进度: {done_count}/{total}, 准确率={true_correct/done_count*100:.1f}%, 已保存")

        time.sleep(SLEEP_SEC)

    # 最终保存
    np.save(SOFT_FILE, soft_all)
    np.save(CORRECT_FILE, soft_all[correct_indices])
    json.dump({'done': [int(x) for x in done_set]}, open(CKPT_FILE, 'w'))

    log(f"\n=== 生成完成 ===")
    log(f"  有效软标签: {len(correct_indices)} 个正确预测")
    log(f"  整体准确率: {true_correct}/{done_count} ({true_correct/done_count*100:.1f}%)")
    ent_mean = float(-(soft_all[done_set] * np.log(np.clip(soft_all[done_set], 1e-8, 1))).sum(axis=1).mean())
    maxp_mean = float(soft_all[done_set].max(axis=1).mean())
    log(f"  mean_entropy={ent_mean:.3f}, mean_max_prob={maxp_mean:.3f}")
    log(f"  输出: {SOFT_FILE}")
    log(f"  仅正确: {CORRECT_FILE}")


if __name__ == '__main__':
    main()
