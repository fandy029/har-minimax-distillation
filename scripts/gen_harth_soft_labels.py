#!/usr/bin/env python3
"""
HARTH 软标签生成脚本（独立版）
从 .env 读取 Mimo API 配置

用法:
  python scripts/gen_harth_soft_labels.py           # 继续（断点续传）
  python scripts/gen_harth_soft_labels.py --force  # 强制从头开始
"""

import os, sys, time, json, argparse, ssl, http.client, re
import numpy as np
from glob import glob

# ============ 加载 .env ============
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k] = v

API_KEY    = os.environ.get('MIMO_API_KEY', '')
API_URL    = os.environ.get('MIMO_API_URL', 'https://token-plan-cn.xiaomimimo.com/v1')
MODEL      = os.environ.get('MIMO_MODEL', 'mimo-v2.5-pro')
TEMPERATURE = float(os.environ.get('MIMO_TEMPERATURE', '0.25'))
MAX_TOKENS  = int(os.environ.get('MIMO_MAX_TOKENS', '2000'))
SLEEP_SEC   = float(os.environ.get('MIMO_SLEEP_SEC', '0.3'))
N_ENSEMBLE  = 3  # 每个样本采样次数

# ============ 路径 ============
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE   = f"{BASE_DIR}/results/soft_labels/harth_soft.npy"
LOG_DIR    = f"{BASE_DIR}/results/logs"
OUT_LOG    = f"{LOG_DIR}/gen_harth.log"
ERR_LOG    = f"{LOG_DIR}/gen_harth_errors.log"
CKPT_FILE  = f"{LOG_DIR}/gen_harth_checkpoint.json"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ============ 数据加载 ============
def load_harth():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    base = f"{BASE_DIR}/datasets/HARTH/harth"
    files = sorted(glob(f"{base}/*.csv"))
    d, l = [], []
    # HARTH 原始 code → 类索引（7类）
    # 1=stand, 2=stairs_up, 3=sit, 4=lie, 5=walk, 6=stand_still, 7=stairs_down
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
    for f in files:
        try:
            df = pd.read_csv(f)
            back = df.iloc[:, 1:4].values.astype(np.float32)
            thigh = df.iloc[:, 4:7].values.astype(np.float32)
            x = np.concatenate([back, thigh], axis=1)
            y_ = df.iloc[:, 7].values.astype(int)
            for i in range(0, len(x)-127, 64):
                w = x[i:i+128]
                label = y_[i]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                    d.append(w); l.append(label_map[label])
        except: pass
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    return X, y

# ============ 特征计算 ============
def compute_features(sample):
    """计算用于分类的特征"""
    sample = np.clip(sample, -5.0, 5.0)
    back  = sample[:, :3]
    thigh = sample[:, 3:6]

    b_std  = back.std(axis=0)
    t_std  = thigh.std(axis=0)
    b_mag  = np.sqrt((back**2).sum(axis=1))
    t_mag  = np.sqrt((thigh**2).sum(axis=1))
    b_dyn  = b_mag.std() / (b_mag.mean() + 1e-6)
    t_dyn  = t_mag.std() / (t_mag.mean() + 1e-6)
    b_mean = back.mean(axis=0)
    t_mean = thigh.mean(axis=0)
    b_gn   = b_mean / (np.linalg.norm(b_mean) + 1e-10)
    t_gn   = t_mean / (np.linalg.norm(t_mean) + 1e-10)

    # Jerk per axis
    bj = np.abs(np.diff(back, axis=0)).mean(axis=0)
    tj = np.abs(np.diff(thigh, axis=0)).mean(axis=0)

    # P75 per axis
    b_p75 = np.percentile(back, 75, axis=0) - np.percentile(back, 25, axis=0)
    t_p75 = np.percentile(thigh, 75, axis=0) - np.percentile(thigh, 25, axis=0)

    # FFT
    fft_t = np.abs(np.fft.rfft(t_mag - t_mag.mean()))
    freqs = np.fft.rfftfreq(len(t_mag), d=1.0/50)
    dom_f = freqs[np.argmax(fft_t[1:])+1] if len(fft_t) > 1 else 0

    return {
        'b_std_x': b_std[0], 'b_std_y': b_std[1], 'b_std_z': b_std[2],
        't_std_x': t_std[0], 't_std_y': t_std[1], 't_std_z': t_std[2],
        'b_dyn': b_dyn, 't_dyn': t_dyn,
        'b_gn_x': b_gn[0], 'b_gn_y': b_gn[1], 'b_gn_z': b_gn[2],
        't_gn_x': t_gn[0], 't_gn_y': t_gn[1], 't_gn_z': t_gn[2],
        'bj_x': bj[0], 'bj_y': bj[1], 'bj_z': bj[2],
        'tj_x': tj[0], 'tj_y': tj[1], 'tj_z': tj[2],
        'b_p75_x': b_p75[0], 'b_p75_y': b_p75[1], 'b_p75_z': b_p75[2],
        't_p75_x': t_p75[0], 't_p75_y': t_p75[1], 't_p75_z': t_p75[2],
        'dom_freq': dom_f,
    }

# ============ Prompt 构建 ============
CN = ['stand', 'stairs_up', 'sit', 'lie', 'walk', 'stand_still', 'stairs_down']

def build_prompt(sample):
    f = compute_features(sample)
    descs = [f"{i}={CN[i]}" for i in range(7)]
    return f"""Classify dual-IMU human activities (back IMU + thigh IMU, 128 steps @ 50Hz).
Classes: {', '.join(descs)}

=== CLASS PATTERNS (from HARTH data analysis, 7 classes) ===
- class 0 stand: grav≈1.0, b_dyn 0.06-0.25, b_std moderate — upright, slight sway
- class 1 stairs_up: grav≈1.0, b_dyn 0.12-0.30, b_std moderate — upward motion, opposing gravity
- class 2 sit: grav≈1.0, b_dyn 0.01-0.26, b_std very low — seated, minimal torso movement
- class 3 lie: grav≈1.0, b_dyn 0.21-0.30, b_std moderate — horizontal posture
- class 4 walk: grav≈1.0, b_dyn 0.38-0.50, t_std high — rhythmic leg swings, dom_freq 2-4Hz
- class 5 stand_still: grav≈1.0, b_dyn < 0.20, b_std low — motionless standing
- class 6 stairs_down: grav≈1.0, b_dyn 0.07-0.17, t_dyn moderate — controlled descent

=== KEY DISCRIMINATORS ===
1. sit vs stand_still: sit has lower b_dyn (0.01-0.10) vs stand_still (0.15-0.20)
2. stand_still vs stairs_down: stand_still has lower b_dyn and t_dyn than stairs_down
3. walk vs stairs_down/up: walk has significantly higher t_dyn (0.35+) and dom_freq (2-4Hz)
4. stairs_up vs stairs_down: stairs_up has higher b_dyn (0.12-0.30) vs stairs_down (0.07-0.17)
5. stand vs lie: similar grav but lie has different b_gn_z orientation

Give probabilistic predictions: if borderline, distribute probability across similar classes.

=== THIS SAMPLE ===
b_std_x={f['b_std_x']:.4f}, b_std_y={f['b_std_y']:.4f}, b_std_z={f['b_std_z']:.4f}
t_std_x={f['t_std_x']:.4f}, t_std_y={f['t_std_y']:.4f}, t_std_z={f['t_std_z']:.4f}
b_dyn={f['b_dyn']:.4f}, t_dyn={f['t_dyn']:.4f}
b_gn=[{f['b_gn_x']:+.3f}, {f['b_gn_y']:+.3f}, {f['b_gn_z']:+.3f}]
t_gn=[{f['t_gn_x']:+.3f}, {f['t_gn_y']:+.3f}, {f['t_gn_z']:+.3f}]
bj_x={f['bj_x']:.4f}, bj_y={f['bj_y']:.4f}, bj_z={f['bj_z']:.4f}
t_p75_x={f['t_p75_x']:.4f}, dom_freq={f['dom_freq']:.1f}Hz

Output ONLY valid JSON: {{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4,"5":p5,"6":p6}}. Make predictions probabilistic — avoid near-one-hot distributions unless the evidence is overwhelming."""

# ============ API 调用 ============
def call_api(prompt, max_retries=5):
    for k in list(os.environ.keys()):
        if 'proxy' in k.lower(): os.environ.pop(k, None)
    ctx = ssl.create_default_context()
    hostname = API_URL.replace('https://', '').split('/')[0]
    path = '/v1/chat/completions'
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "thinking": {"type": "disabled"}
    }).encode()

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            conn = http.client.HTTPSConnection(hostname, timeout=30, context=ctx)
            conn.request('POST', path, body=payload, headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
                'Content-Length': str(len(payload))
            })
            resp = conn.getresponse()
            data = json.loads(resp.read().decode('utf-8', errors='replace'))
            conn.close()
            if 'choices' not in data:
                raise ValueError(f"No choices: {data}")
            content = data['choices'][0]['message']['content']
            # 提取 JSON
            m = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if not m:
                m2 = re.search(r'\{.*\}', content, re.DOTALL)
                if m2: m = m2
            if m:
                obj = json.loads(m.group())
                vals = [float(obj.get(str(i), 0)) for i in range(7)]
                s = sum(vals)
                if s > 0: return [v/s for v in vals], None
            return None, 'no valid json found'
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(10)
    return None, last_err

def call_api_multi(prompt, n=3):
    results, errors = [], []
    for _ in range(n):
        r, e = call_api(prompt)
        if r is not None:
            results.append(np.array(r))
        else:
            errors.append(e)
        time.sleep(SLEEP_SEC)
    if not results:
        return None, errors[0] if errors else 'all failed'
    return np.mean(results, axis=0).tolist(), None

# ============ 检查点 ============
def load_checkpoint():
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f:
            return json.load(f)
    return {'done': [], 'soft': {}}

def save_checkpoint(idx, soft_row):
    ckpt = load_checkpoint()
    ckpt['done'].append(idx)
    ckpt['soft'][str(idx)] = soft_row
    with open(CKPT_FILE, 'w') as f:
        json.dump(ckpt, f)

def is_valid(v):
    try:
        return (isinstance(v, (list, np.ndarray, tuple))
                and len(v) == 7
                and all(isinstance(x, (int, float)) for x in v)
                and sum(v) > 0.01)
    except:
        return False

# ============ 主循环 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='强制从头开始（同时清空日志）')
    args = parser.parse_args()

    if args.force:
        # 清空日志和检查点
        open(OUT_LOG, 'w').close()
        open(ERR_LOG, 'w').close()
        if os.path.exists(CKPT_FILE):
            os.remove(CKPT_FILE)

    print(f"Mimo API: {API_URL}")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Ensemble: {N_ENSEMBLE}")

    X, y = load_harth()
    n = len(X)
    print(f"Loaded HARTH: {n} samples")

    # 初始化或加载软标签
    if os.path.exists(OUT_FILE) and not args.force:
        y_soft = np.load(OUT_FILE)
        print(f"Loaded existing soft labels: {y_soft.shape}")
    else:
        y_soft = np.zeros((n, 7), dtype=np.float32)

    # 断点续传
    ckpt = load_checkpoint()
    done_set = set(ckpt['done'])
    for idx_str, row in ckpt['soft'].items():
        idx = int(idx_str)
        if is_valid(row):
            y_soft[idx] = row
            done_set.add(idx)

    start_time = time.time()
    done = 0
    total_done = len([i for i in range(n) if is_valid(y_soft[i])])

    for i in range(n):
        if not args.force and i in done_set:
            continue

        prompt = build_prompt(X[i])
        result, err = call_api_multi(prompt, n=N_ENSEMBLE)

        if result is None or not is_valid(result):
            # retry
            result2, err2 = call_api_multi(prompt, n=N_ENSEMBLE)
            if result2 is not None and is_valid(result2):
                result = result2; err = None
            else:
                with open(ERR_LOG, 'a') as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} idx={i} true={y[i]} err={err}\n")
                result = None

        if result is not None:
            y_soft[i] = result
            done += 1
            total_done += 1
            save_checkpoint(i, result)

            pred = float(np.argmax(result))
            ok = "✓" if int(pred) == int(y[i]) else "✗"
            print(f"[{i:4d}/{n}] true={int(y[i])} pred={int(pred)}[{ok}] "
                  f"soft=[{','.join(f'{v:.2f}' for v in result)}]")

        if (done > 0 and done % 20 == 0) or done == 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remain = (n - total_done) / rate if rate > 0 else 0
            print(f"  Progress: {total_done}/{n} | {rate:.1f}/s | ETA: {remain/60:.0f}min")

        sys.stdout.flush()

    np.save(OUT_FILE, y_soft.astype(np.float32))
    print(f"\nDone. Saved: {OUT_FILE}")
    print(f"Shape: {y_soft.shape}")

    # 质量报告
    preds = y_soft.argmax(axis=1)
    acc = (preds == y).mean()
    print(f"\n=== Quality Report ===")
    print(f"Overall accuracy: {acc:.4f}")
    for c, name in enumerate(CN):
        mask = y == c
        if mask.sum() > 0:
            c_acc = (preds[mask] == y[mask]).mean()
            print(f"  {c} ({name:12s}): n={mask.sum()}, acc={c_acc:.4f}")

if __name__ == '__main__':
    main()
