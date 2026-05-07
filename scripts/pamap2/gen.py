#!/usr/bin/env python3
"""
PAMAP2 软标签生成脚本
用法: python gen_pamap2.py [--force]
"""
import os, sys, json, time, re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR   = THESIS_DIR
OUT_DIR    = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR    = os.path.join(BASE_DIR, 'results', 'logs')
CKPT_FILE  = os.path.join(LOG_DIR, 'gen_pamap2_checkpoint.json')

sys.path.insert(0, SCRIPT_DIR)
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from api_config import API_KEY, API_URL, MODEL, TEMPERATURE, MAX_TOKENS, SLEEP_SEC, TIMEOUT
from openai import OpenAI

# ============ 数据集配置 ============
# PAMAP2 原始 labels: 1=lying, 2=sitting, 3=standing, 4=walking, 5=jogging (下楼梯另作9但这里用1-5)
CLASS_NAMES = ['lying', 'sitting', 'standing', 'walking', 'jogging']
N_CLS        = len(CLASS_NAMES)
SAMPLES_PER_CLASS = 300
FORCE_RESTART     = '--force' in sys.argv

# ============ 数据加载 ============
def load_pamap2_data():
    """
    PAMAP2 数据加载
    原始数据格式: column 1=activity, columns 9-14=IMU (acc+gyro for 2 sensors)
    实际使用: acc+gyro 共 6 通道
    原始 activity labels: 1=lying, 2=sitting, 3=standing, 4=walking, 5=jogging
    """
    base = os.path.join(BASE_DIR, 'datasets', 'PAMAP2', 'PAMAP2_Dataset')
    d, l = [], []
    # raw label -> class_id
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4}

    for folder in ['Protocol', 'Optional']:
        folder_path = os.path.join(base, folder)
        if not os.path.exists(folder_path):
            continue
        files = sorted(glob(os.path.join(folder_path, '*.dat')))
        for f in files:
            try:
                df = pd.read_csv(f, sep=' ', header=None)
                # Column 1 = activity label
                acts = df.iloc[:, 1].values
                # Columns 9-14 = IMU (accel + gyro, 6 values)
                imu = df.iloc[:, 9:15].values.astype(np.float32)
                for aid, unlabel in label_map.items():
                    mask = (acts == aid)
                    idx = np.where(mask)[0]
                    for s in range(0, len(idx)-127, 64):
                        w = imu[idx[s:s+128]]
                        if w.shape[0]==128 and not np.any(np.isnan(w)):
                            d.append(w); l.append(unlabel)
            except Exception as e:
                pass

    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

# ============ Prompt ============
def build_prompt(window):
    """
    PAMAP2 专用 prompt
    window: (128, 6) = accelerometer(3) + gyroscope(3) @ 100Hz
    """
    acc  = window[:, :3]
    gyro = window[:, 3:6]

    acc_mag  = np.sqrt((acc**2).sum(axis=1))
    gyro_mag = np.sqrt((gyro**2).sum(axis=1))

    # 统计特征
    acc_mean  = acc.mean(axis=0)
    gyro_mean = gyro.mean(axis=0)
    acc_std   = acc.std(axis=0)
    gyro_std  = gyro.std(axis=0)

    # FFT 分析步态周期
    fft_v = np.abs(np.fft.fft(acc_mag)[1:len(acc_mag)//2])
    dom_f = np.fft.fftfreq(len(acc_mag), 1/100)[np.argmax(fft_v)+1] if len(fft_v) > 0 else 0

    # 峰值计数（步态周期）
    peaks = int(np.sum((acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])))

    descs = [f'{i}={CLASS_NAMES[i]}' for i in range(N_CLS)]

    return f"""Classify physical activity from IMU sensor data (accelerometer + gyroscope, 128 steps @ 100Hz, 6 channels).

Classes:
  {", ".join(descs)}

=== SENSOR FEATURES (this window) ===
  acc_mag_mean  = {acc_mag.mean():.4f}  (movement intensity)
  gyro_mag_mean = {gyro_mag.mean():.4f}  (rotation intensity)
  acc_mean      = {[f'{v:.3f}' for v in acc_mean]}
  gyro_mean     = {[f'{v:.3f}' for v in gyro_mean]}
  acc_std       = {[f'{v:.4f}' for v in acc_std]}
  gyro_std      = {[f'{v:.4f}' for v in gyro_std]}
  dominant_freq = {dom_f:.1f}Hz
  step_peaks   = {peaks}

=== PHYSICS-BASED RULES ===
  - lying:    acc_mag ≈ 1.0g (gravity only), gyro_mag ≈ 0, person is supine
  - sitting:  acc_mag ≈ 1.0g, acc tilted, minimal movement, gyro ≈ 0
  - standing: acc_mag ≈ 1.0g, upright orientation, minimal movement
  - walking:  acc_mag 1.0-2.0g with periodic peaks, dom_freq 1-2Hz, peaks 5-15
  - jogging:  acc_mag 2.0-4.0g with high peaks, dom_freq 2-4Hz, peaks > 15

=== OUTPUT ===
Estimate confusion probabilities for this window.
Output ONLY valid JSON with {N_CLS} probabilities that sum to 1:
{{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4}}
Do NOT output one-hot. Probabilities should be moderate (0.05-0.60)."""

# ============ API ============
def call_api(prompt):
    client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=TIMEOUT)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)

def extract_probs(text):
    if not text:
        return None
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<[^>]+>', '', text).strip()
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                vals = [float(d[str(k)]) for k in range(N_CLS)]
                s = sum(vals)
                if s > 0:
                    return [v/s for v in vals]
        except:
            pass
    vals = [0.0] * N_CLS
    for i in range(N_CLS):
        m = re.search(rf'"{i}"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if m:
            vals[i] = float(m.group(1))
    if sum(vals) > 0:
        return [v/sum(vals) for v in vals]
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

# ============ 主函数 ============
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, 'gen_pamap2.log')
    err_file = os.path.join(LOG_DIR, 'gen_pamap2_errors.log')

    if FORCE_RESTART:
        for f in [log_file, err_file]:
            if os.path.exists(f):
                open(f, 'w').close()
        if os.path.exists(CKPT_FILE):
            os.remove(CKPT_FILE)

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

    log(f"PAMAP2 软标签生成开始")
    log(f"API: {API_URL} | Model: {MODEL} | T={TEMPERATURE}")

    log("加载 PAMAP2 数据...")
    X, y, _, _, _, _ = load_pamap2_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")
    for c in range(N_CLS):
        log(f"  class {c} ({CLASS_NAMES[c]}): {int(np.sum(y==c))} 样本")

    np.random.seed(42)
    sample_indices = []
    for c in range(N_CLS):
        cidx = np.where(y == c)[0]
        take = min(SAMPLES_PER_CLASS, len(cidx))
        sample_indices.extend(np.random.choice(cidx, size=take, replace=False).tolist())
    sample_indices = np.array(sample_indices)
    np.random.shuffle(sample_indices)
    total = len(sample_indices)
    log(f"总计采样: {total} 个窗口")

    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f:
                ckpt = json.load(f)
            done_set = set(ckpt.get('done', []))
            log(f"断点续传: {len(done_set)}/{total} 已处理")
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
        for retry in range(3):
            if result:
                break
            time.sleep(5)
            result, err = call_api(prompt)

        if result is None:
            log_err(f"API_FAILED idx={orig_idx} true={true_label} err={err} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            done_set.add(orig_idx)
            done_count += 1
            continue

        probs = extract_probs(result)
        for _ in range(5):
            if is_valid(probs):
                break
            time.sleep(2)
            result2, _ = call_api(prompt)
            if result2:
                probs2 = extract_probs(result2)
                if is_valid(probs2):
                    probs = probs2

        if not is_valid(probs):
            log_err(f"ONEHOT_REJECT idx={orig_idx} true={true_label} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
        else:
            soft_all[orig_idx] = probs
            pred = int(np.argmax(probs))
            ok = "✓" if pred == true_label else "✗"
            ent = float(-(np.array(probs) * np.log(np.clip(probs, 1e-8, 1))).sum())
            top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
            log(f"  [{done_count+1}/{total}] idx={orig_idx} "
                f"true={true_label}({CLASS_NAMES[true_label]}) "
                f"pred={pred}({CLASS_NAMES[pred]})[{ok}] "
                f"ent={ent:.3f} top=[{top2[0][0]}:{top2[0][1]:.3f},{top2[1][0]}:{top2[1][1]:.3f}]")
            if ok == "✓":
                true_correct += 1

        done_set.add(orig_idx)
        done_count += 1

        if done_count % 5 == 0:
            np.save(os.path.join(OUT_DIR, 'pamap2_soft.npy'), soft_all)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': list(done_set)}, f)
            acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
            log(f"  进度: +{done_count}/{total}, 准确率={acc_pct:.1f}%, 已保存")

        time.sleep(SLEEP_SEC)

    out_path = os.path.join(OUT_DIR, 'pamap2_soft.npy')
    np.save(out_path, soft_all)
    valid_n = int((soft_all.sum(axis=1) > 0).sum())
    acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
    log(f"\n=== 完成 ===")
    log(f"  有效软标签: {valid_n}/{len(X)}, 准确率: {acc_pct:.1f}%")
    log(f"  输出: {out_path}")
    with open(CKPT_FILE, 'w') as f:
        json.dump({'done': list(done_set)}, f)

if __name__ == '__main__':
    main()
