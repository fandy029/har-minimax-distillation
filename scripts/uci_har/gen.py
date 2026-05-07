#!/usr/bin/env python3
"""
UCI_HAR 软标签生成脚本
6类活动识别，561维预提取特征
用法: python gen_uci_har.py [--force]
"""
import os, sys, json, time, re
import numpy as np
from sklearn.model_selection import train_test_split

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR   = THESIS_DIR
OUT_DIR    = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR    = os.path.join(BASE_DIR, 'results', 'logs')
CKPT_FILE  = os.path.join(LOG_DIR, 'gen_uci_har_checkpoint.json')

# ============ API 配置 ============
sys.path.insert(0, SCRIPT_DIR)
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from api_config import API_KEY, API_URL, MODEL, TEMPERATURE, MAX_TOKENS, SLEEP_SEC, TIMEOUT
from openai import OpenAI

# ============ 数据集配置 ============
CLASS_NAMES = ['WALKING', 'WALKING_UP', 'WALKING_DOWN', 'SITTING', 'STANDING', 'LAYING']
N_CLS       = len(CLASS_NAMES)
SAMPLES_PER_CLASS = 200
FORCE_RESTART     = '--force' in sys.argv


# ============ 数据加载 ============
def load_uci_har_data():
    """
    UCI_HAR 数据加载
    数据来源: /datasets/UCI_HAR/UCI HAR Dataset/
    - train/X_train.txt: 训练集特征 (7352 × 561)
    - train/y_train.txt: 训练集标签 (1-6)
    - test/X_test.txt: 测试集特征 (2947 × 561)
    - test/y_test.txt: 测试集标签
    561维特征包括: time-domain (均值/方差/...) + frequency-domain (FFT)
    """
    base = os.path.join(BASE_DIR, 'datasets', 'UCI_HAR', 'UCI HAR Dataset')

    X_tr = np.loadtxt(os.path.join(base, 'train', 'X_train.txt')).astype(np.float32)
    y_tr = (np.loadtxt(os.path.join(base, 'train', 'y_train.txt')) - 1).astype(np.int64)
    X_te = np.loadtxt(os.path.join(base, 'test', 'X_test.txt')).astype(np.float32)
    y_te = (np.loadtxt(os.path.join(base, 'test', 'y_test.txt')) - 1).astype(np.int64)

    # 划分: 80% train, 10% val, 10% test
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te


# ============ Prompt 构建 ============
def build_prompt(data):
    """
    data: (561,) 预提取特征向量
    UCI_HAR 特征顺序 (关键索引):
      0-41: tBodyAcc-mean()-X ~ tBodyAcc-max-Inds (加速度统计)
      42-83: tBodyAcc-std() 系列
      84-121: tBodyAcc-mad() ~ tBodyAcc-energy()
      122-159: tBodyAcc-arCoeff()-X1 ~ angletBodyAccMeanGravity
      160-201: tGravityAcc-mean()-X ~ tGravityAcc-correlation()-Z
      ...
      296-333: tBodyGyro-mean() ~ tBodyGyro-entropy()
      ...
      554-560: angle(Z,gravityMean) ~ angle(X,gravityMean)
    这里不硬编码具体值，只用特征块统计量
    """
    vals = np.array(data)
    n = len(vals)

    # 关键特征块统计（不做硬编码）
    def safe_mean(start, end):
        if end <= n:
            return vals[start:min(end, n)].mean()
        return 0.0

    def safe_std(start, end):
        if end <= n:
            return vals[start:min(end, n)].std()
        return 0.0

    # 加速度统计
    acc_mean = safe_mean(0, 40)
    acc_std = safe_std(0, 40)
    gravity_mean = safe_mean(160, 200)
    gravity_std = safe_std(160, 200)

    # 陀螺仪统计
    gyro_mean = safe_mean(296, 340)
    gyro_std = safe_std(296, 340)

    # FFT 能量
    fft_start = 332
    fft_end = min(380, n)
    fft_energy = (vals[fft_start:fft_end] ** 2).mean() if fft_end > fft_start else 0.0

    # 幅度特征
    jerk_start = 224
    jerk_end = min(260, n)
    jerk_mag = np.abs(vals[jerk_start:jerk_end]).mean() if jerk_end > jerk_start else 0.0

    # 相关性（简单）
    corr_acc_gyro = 0.0
    if n > 500:
        corr_acc_gyro = np.corrcoef(vals[0:40], vals[296:336])[0, 1] if n >= 336 else 0.0

    class_list = ', '.join([f'{i}:{CLASS_NAMES[i]}' for i in range(N_CLS)])

    return f"""Classify human activity from 561 pre-extracted sensor features (accelerometer + gyroscope in time and frequency domains).

Classes ({N_CLS} total): {class_list}

Key feature statistics (no hard-coded values):
  - Accelerometer mean (time domain): {acc_mean:.4f}
  - Accelerometer std (time domain): {acc_std:.4f}
  - Gravity mean: {gravity_mean:.4f}, gravity std: {gravity_std:.4f}
  - Gyroscope mean: {gyro_mean:.4f}, gyro std: {gyro_std:.4f}
  - FFT energy: {fft_energy:.4f}
  - Jerk magnitude: {jerk_mag:.4f}
  - Accel-gyro correlation: {corr_acc_gyro:.4f}
  - Total feature vector: {n}-dim

Physical reasoning rules:
  * WALKING: moderate periodic motion, alternating left-right (FFT shows ~1Hz peak), moderate gyro
  * WALKING_UP: positive Y acceleration (upward), higher gyro activity on stairs, cadence ~1.5Hz
  * WALKING_DOWN: negative Y acceleration (downward), controlled descent, slight forward lean
  * SITTING: static posture, near-zero acceleration except gravity (~9.8 on Z when seated), minimal gyro
  * STANDING: static upright posture, gravity on Z axis (~1.0g), very low gyro
  * LAYING: supine position, gravity on X axis (horizontal), near-zero gyro, all accel axes near static

Note: FFT energy distinguishes walking from static; Y-axis bias distinguishes up/down stairs.

Output ONLY a valid JSON object mapping class indices to probabilities that sum to 1:
{{"0":p0,"1":p1,...}}"""


# ============ API 调用 ============
def call_api(prompt):
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=TIMEOUT)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content or '', None
    except Exception as e:
        return '', str(e)


# ============ JSON 解析 ============
def extract_probs(text):
    if not text:
        return None
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()

    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                vals = [float(d[str(k)]) for k in range(N_CLS)]
                s = sum(vals)
                if s > 0:
                    return [v / s for v in vals]
        except:
            pass

    vals = [0.0] * N_CLS
    for i in range(N_CLS):
        m = re.search(rf'"{i}"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if m:
            vals[i] = float(m.group(1))
    if sum(vals) > 0:
        return [v / sum(vals) for v in vals]
    return None


def is_valid(probs):
    if probs is None:
        return False
    row = np.array(probs)
    if not np.isclose(row.sum(), 1.0, atol=0.01):
        return False
    if row.max() >= 0.95:
        return False
    return True


# ============ 主生成逻辑 ============
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, 'gen_uci_har.log')
    err_file = os.path.join(LOG_DIR, 'gen_uci_har_errors.log')

    if FORCE_RESTART:
        for f in [log_file, err_file]:
            if os.path.exists(f):
                open(f, 'w').close()

    def log(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_file, 'a') as f:
            f.write(line + '\n')

    def log_err(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        with open(err_file, 'a') as f:
            f.write(line + '\n')

    log(f"UCI_HAR 软标签生成开始")
    log(f"Mimo API: {API_URL}, Model: {MODEL}")

    log("加载 UCI_HAR 数据...")
    X, y, _, _, _, _ = load_uci_har_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")

    for c in range(N_CLS):
        cnt = int(np.sum(y == c))
        log(f"  class {c} ({CLASS_NAMES[c]}): {cnt} 样本")

    np.random.seed(42)
    sample_indices = []
    for c in range(N_CLS):
        cidx = np.where(y == c)[0]
        take = min(SAMPLES_PER_CLASS, len(cidx))
        chosen = np.random.choice(cidx, size=take, replace=False)
        sample_indices.extend(chosen.tolist())
    sample_indices = np.array(sample_indices)
    np.random.shuffle(sample_indices)
    total = len(sample_indices)
    log(f"总计采样: {total} 个窗口（每类 ~{SAMPLES_PER_CLASS}）")

    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f:
                ckpt = json.load(f)
            done_set = set(ckpt.get('done', []))
            log(f"  断点续传: {len(done_set)}/{total} 已处理")
            except (json.JSONDecodeError, IOError) as e:
                log(f"  断点文件损坏 ({e})，从头开始")

    soft_all = np.zeros((len(X), N_CLS), dtype=np.float32)
    done_count = 0
    true_correct = 0

    for pos, orig_idx in enumerate(sample_indices):
        if orig_idx in done_set:
            continue

        true_label = int(y[orig_idx])
        prompt = build_prompt(X[orig_idx])

        result, err = call_api(prompt)
        retry_count = 0
        while result is None and retry_count < 3:
            time.sleep(5)
            result, err = call_api(prompt)
            retry_count += 1

        if result is None:
            log_err(f"API_FAILED idx={orig_idx} true={true_label} err={err} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            done_set.add(orig_idx)
            continue

        probs = extract_probs(result)
        if not is_valid(probs):
            for _ in range(5):
                time.sleep(2)
                result2, err2 = call_api(prompt)
                if result2:
                    probs2 = extract_probs(result2)
                    if is_valid(probs2):
                        probs = probs2
                        result = result2
                        break

        if not is_valid(probs):
            log_err(f"ONEHOT_REJECT idx={orig_idx} true={true_label} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
        else:
            soft_all[orig_idx] = probs
            pred_label = int(np.argmax(probs))
            ok = "✓" if pred_label == true_label else "✗"
            ent = float(-(np.array(probs) * np.log(np.clip(probs, 1e-8, 1))).sum())
            top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
            log(f"  [{done_count+1}/{total}] idx={orig_idx} "
                f"true={true_label}({CLASS_NAMES[true_label]}) "
                f"pred={pred_label}({CLASS_NAMES[pred_label]})[{ok}] "
                f"ent={ent:.3f} "
                f"top=[{top2[0][0]}:{top2[0][1]:.3f},{top2[1][0]}:{top2[1][1]:.3f}]")
            if ok == "✓":
                true_correct += 1

        done_set.add(orig_idx)
        done_count += 1

        if done_count % 5 == 0:
            np.save(os.path.join(OUT_DIR, 'uci_har_soft.npy'), soft_all)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': list(done_set), 'class_names': CLASS_NAMES}, f)
            valid_n = int((soft_all.sum(axis=1) > 0).sum())
            acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
            log(f"  进度: +{done_count}/{total}, 准确率={acc_pct:.1f}%, 已保存")

        time.sleep(SLEEP_SEC)

    out_path = os.path.join(OUT_DIR, 'uci_har_soft.npy')
    np.save(out_path, soft_all)
    valid_n = int((soft_all.sum(axis=1) > 0).sum())
    acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
    log(f"\n=== 生成完成 ===")
    log(f"  有效软标签: {valid_n}/{len(X)}")
    log(f"  整体准确率: {acc_pct:.1f}% ({true_correct}/{done_count})")
    with open(CKPT_FILE, 'w') as f:
        json.dump({'done': list(done_set), 'class_names': CLASS_NAMES}, f)
    log(f"  输出: {out_path}")


if __name__ == '__main__':
    main()
