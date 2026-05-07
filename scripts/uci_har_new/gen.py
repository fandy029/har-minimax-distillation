#!/usr/bin/env python3
"""
UCI_HAR_New 软标签生成脚本
12类（6个基础类 + 6个过渡类），561维预提取特征
用法: python gen_uci_har_new.py [--force]
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
CKPT_FILE  = os.path.join(LOG_DIR, 'gen_uci_har_new_checkpoint.json')

# ============ API 配置 ============
sys.path.insert(0, SCRIPT_DIR)
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from api_config import API_KEY, API_URL, MODEL, TEMPERATURE, MAX_TOKENS, SLEEP_SEC, TIMEOUT
from openai import OpenAI

# ============ 数据集配置 ============
CLASS_NAMES = [
    'WALKING', 'WALKING_UP', 'WALKING_DOWN', 'SITTING', 'STANDING', 'LAYING',
    'SIT_TO_STAND', 'STAND_TO_SIT', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND'
]
N_CLS       = len(CLASS_NAMES)
SAMPLES_PER_CLASS = 129
FORCE_RESTART     = '--force' in sys.argv


# ============ 数据加载 ============
def load_uci_har_new_data():
    """
    UCI_HAR_New 数据加载
    数据来源: /datasets/UCI_HAR_New/
    - Train/X_train.txt: 训练集特征
    - Train/y_train.txt: 训练集标签 (1-12)
    - Test/X_test.txt: 测试集特征
    - Test/y_test.txt: 测试集标签
    12类: 6 基础类 + 6 过渡类
    """
    base = os.path.join(BASE_DIR, 'datasets', 'UCI_HAR_New')

    X_tr = np.loadtxt(os.path.join(base, 'Train', 'X_train.txt')).astype(np.float32)
    y_tr = (np.loadtxt(os.path.join(base, 'Train', 'y_train.txt')) - 1).astype(np.int64)
    X_te = np.loadtxt(os.path.join(base, 'Test', 'X_test.txt')).astype(np.float32)
    y_te = (np.loadtxt(os.path.join(base, 'Test', 'y_test.txt')) - 1).astype(np.int64)

    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te


# ============ Prompt 构建 ============
def build_prompt(data):
    """
    data: (561,) 预提取特征向量
    12类: 6 基础静态/动态 + 6 过渡状态
    过渡类特征：短时间窗口内的姿势变化，加速度/陀螺仪非平稳
    """
    vals = np.array(data)
    n = len(vals)

    def safe_mean(start, end):
        end = min(end, n)
        if end <= start:
            return 0.0
        return float(vals[start:end].mean())

    def safe_std(start, end):
        end = min(end, n)
        if end <= start:
            return 0.0
        return float(vals[start:end].std())

    # 关键块统计
    acc_mean = safe_mean(0, 40)
    acc_std = safe_std(0, 40)
    gravity_mean = safe_mean(160, 200)
    gyro_mean = safe_mean(296, 340)
    gyro_std = safe_std(296, 340)

    fft_start = 332
    fft_end = min(380, n)
    fft_energy = float((vals[fft_start:fft_end] ** 2).mean()) if fft_end > fft_start else 0.0

    jerk_start = 224
    jerk_end = min(260, n)
    jerk_mag = float(np.abs(vals[jerk_start:jerk_end]).mean()) if jerk_end > jerk_start else 0.0

    # 过渡类检测用: 变化率（非平稳性）
    # 用前半段和后半段的均值差异作为非平稳性代理
    mid = n // 2
    first_half_var = float(vals[:mid].std())
    second_half_var = float(vals[mid:].std())
    nonstationarity = abs(first_half_var - second_half_var)

    class_list = ', '.join([f'{i}:{CLASS_NAMES[i]}' for i in range(N_CLS)])

    return f"""Classify human activity from 561 pre-extracted sensor features (12 classes: 6 static/dynamic + 6 transition states).

Classes ({N_CLS} total): {class_list}

Key feature statistics:
  - Accelerometer mean: {acc_mean:.4f}, std: {acc_std:.4f}
  - Gravity mean: {gravity_mean:.4f}
  - Gyroscope mean: {gyro_mean:.4f}, std: {gyro_std:.4f}
  - FFT energy: {fft_energy:.4f}
  - Jerk magnitude: {jerk_mag:.4f}
  - Non-stationarity (var diff between halves): {nonstationarity:.4f}

Physical reasoning rules:
  * Static activities (low motion): SITTING, STANDING, LAYING — minimal acceleration variation, near-zero gyro
  * Dynamic activities (periodic motion): WALKING, WALKING_UP, WALKING_DOWN — rhythmic FFT patterns, moderate gyro
    - WALKING_UP: positive Y acceleration bias (ascending)
    - WALKING_DOWN: negative Y acceleration bias (descending)
  * Transition activities (posture change, short duration):
    - SIT_TO_STAND: rising motion, Y acceleration goes from low to high, brief transition
    - STAND_TO_SIT: sitting down, Y acceleration goes from high to low
    - SIT_TO_LIE: large angle change, Z/X axis shift (gravity reorientation), prolonged transition
    - LIE_TO_SIT: opposite of SIT_TO_LIE
    - STAND_TO_LIE: large angular change, prolonged transition
    - LIE_TO_STAND: rising from supine, highest Y acceleration spike

Transitions are characterized by: high non-stationarity, brief duration of the motion phase, gravity axis reorientation.

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

    log_file = os.path.join(LOG_DIR, 'gen_uci_har_new.log')
    err_file = os.path.join(LOG_DIR, 'gen_uci_har_new_errors.log')

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

    log(f"UCI_HAR_New 软标签生成开始")
    log(f"Mimo API: {API_URL}, Model: {MODEL}")

    log("加载 UCI_HAR_New 数据...")
    X, y, _, _, _, _ = load_uci_har_new_data()
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
            np.save(os.path.join(OUT_DIR, 'uci_har_new_soft.npy'), soft_all)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': list(done_set), 'class_names': CLASS_NAMES}, f)
            valid_n = int((soft_all.sum(axis=1) > 0).sum())
            acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
            log(f"  进度: +{done_count}/{total}, 准确率={acc_pct:.1f}%, 已保存")

        time.sleep(SLEEP_SEC)

    out_path = os.path.join(OUT_DIR, 'uci_har_new_soft.npy')
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
