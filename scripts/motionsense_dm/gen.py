#!/usr/bin/env python3
"""
MotionSense_DeviceMotion 软标签生成脚本
6类活动识别，从 DeviceMotion 数据加载，128步×3通道（userAcceleration）
用法: python gen_motionsense_dm.py [--force]
"""
import os, sys, json, time, re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR   = THESIS_DIR
OUT_DIR    = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR    = os.path.join(BASE_DIR, 'results', 'logs')
CKPT_FILE  = os.path.join(LOG_DIR, 'gen_motionsense_dm_checkpoint.json')

# ============ API 配置 ============
sys.path.insert(0, SCRIPT_DIR)
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from api_config import API_KEY, API_URL, MODEL, TEMPERATURE, MAX_TOKENS, SLEEP_SEC, TIMEOUT
from openai import OpenAI

# ============ 数据集配置 ============
CLASS_NAMES = ['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']
N_CLS       = len(CLASS_NAMES)
SAMPLES_PER_CLASS = 200
FORCE_RESTART     = '--force' in sys.argv


# ============ 数据加载 ============
def load_motionsense_dm_data():
    """
    MotionSense DeviceMotion 数据加载
    数据来源: /datasets/MotionSense_DeviceMotion/A_DeviceMotion_data/
    结构: 每个活动一个文件夹 (dws_1, jog_1, sit_1, std_1, ups_1, wlk_1)
    每个文件夹内: sub_1.csv ~ sub_24.csv
    CSV 列 (13列): attitude, gravity, rotationRate, userAcceleration(x,y,z)
    取 userAcceleration.x, userAcceleration.y, userAcceleration.z
    """
    base = os.path.join(BASE_DIR, 'datasets', 'MotionSense_DeviceMotion', 'A_DeviceMotion_data')
    d, l = [], []
    label_map = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}

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
                # 找 userAcceleration 列
                cols = [c for c in df.columns if 'userAcceleration' in c]
                if len(cols) >= 3:
                    data = df[cols].values.astype(np.float32)  # (N, 3)
                    for start in range(0, len(data) - 127, 64):
                        window = data[start:start + 128]
                        if window.shape[0] == 128 and not np.any(np.isnan(window)):
                            d.append(window)
                            l.append(label)
            except Exception:
                continue

    X = np.array(d, dtype=np.float32)  # (N, 128, 3)
    y = np.array(l, dtype=np.int64)

    if len(X) == 0:
        raise RuntimeError("MotionSense_DeviceMotion 数据加载失败")

    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te


# ============ Prompt 构建 ============
def build_prompt(data):
    """
    data: (128, 3) 窗口，userAcceleration x/y/z from DeviceMotion
    与 MotionSense 相同类别，物理特征相同
    """
    acc = data
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    acc_mean = acc.mean(axis=0)
    acc_std = acc.std(axis=0)
    y_mean = acc[:, 1].mean()
    n_peaks = int(np.sum(
        (acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])
    ))
    fft_v = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag) // 2])
    if len(fft_v) > 0 and fft_v.max() > 0:
        dom_freq_idx = np.argmax(fft_v) + 1
        dom_freq = float(np.fft.fftfreq(len(acc_mag), 1.0 / 20.0)[dom_freq_idx])
    else:
        dom_freq = 0.0

    class_list = ', '.join([f'{i}:{CLASS_NAMES[i]}' for i in range(N_CLS)])

    return f"""Classify physical activity from 128-step DeviceMotion accelerometer window (userAcceleration x, y, z axes).

Classes ({N_CLS} total): {class_list}

Physics-based features:
  - accel magnitude mean: {acc_mag.mean():.3f}
  - accel mean [x, y, z]: [{acc_mean[0]:.3f}, {acc_mean[1]:.3f}, {acc_mean[2]:.3f}]
  - accel std [x, y, z]: [{acc_std[0]:.3f}, {acc_std[1]:.3f}, {acc_std[2]:.3f}]
  - Y-axis mean (vertical): {y_mean:.3f}
  - peak count: {n_peaks}
  - dominant frequency: {dom_freq:.2f} Hz

Physical reasoning rules:
  * downstairs: negative Y bias (descending), moderate periodic, ~1-1.5Hz
  * upstairs: positive Y bias (ascending), ~1.5Hz
  * jogging: high magnitude (~1.5-2g), strong periodic, ~2-3Hz
  * walking: moderate magnitude, regular periodic, ~1-1.5Hz
  * sitting: near-zero net acceleration, minimal motion
  * standing: near-zero net acceleration, minimal motion, gravity on Z

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

    log_file = os.path.join(LOG_DIR, 'gen_motionsense_dm.log')
    err_file = os.path.join(LOG_DIR, 'gen_motionsense_dm_errors.log')

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

    log(f"MotionSense_DeviceMotion 软标签生成开始")
    log(f"Mimo API: {API_URL}, Model: {MODEL}")

    log("加载 MotionSense_DeviceMotion 数据...")
    X, y, _, _, _, _ = load_motionsense_dm_data()
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
            np.save(os.path.join(OUT_DIR, 'motionsense_dm_soft.npy'), soft_all)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': list(done_set), 'class_names': CLASS_NAMES}, f)
            valid_n = int((soft_all.sum(axis=1) > 0).sum())
            acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
            log(f"  进度: +{done_count}/{total}, 准确率={acc_pct:.1f}%, 已保存")

        time.sleep(SLEEP_SEC)

    out_path = os.path.join(OUT_DIR, 'motionsense_dm_soft.npy')
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
