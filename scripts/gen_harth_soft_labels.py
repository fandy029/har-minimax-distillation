#!/usr/bin/env python3
"""
HARTH 软标签生成脚本（方案A）
只对能可靠区分的3类生成软标签，其他类用one-hot fallback

能区分的3类:
  - class 3 (lie):     back_gn_z > +0.60
  - class 4 (walk):    thigh_std > 0.25
  - class 6 (stairs_down): thigh_gn_z > +0.60

其他4类 (0=stand, 1=stairs_up, 2=sit, 5=stand_still):
  → 软标签用 one-hot fallback（CNN 自己能学会这些简单的类）

采样策略：
  - class 3, 4, 6: 每类 1000 个（可靠软标签）
  - class 0, 1, 2, 5: 每类 200 个（one-hot fallback）
  - 总计: 3600 个样本，软标签只覆盖 3000 个

输出: 7维 soft label（维度不变，保持与训练脚本兼容）
"""

import os, sys, time, json, argparse, ssl, http.client, re
import logging
import numpy as np
from glob import glob

# ============ 日志配置 ============
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR  = os.path.join(_BASE_DIR, 'results', 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)

for fn in ['gen_harth.log', 'gen_harth_errors.log']:
    fp = os.path.join(_LOG_DIR, fn)
    if os.path.exists(fp): open(fp, 'w').close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, 'gen_harth.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ============ 加载 .env ============
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k] = v

API_KEY     = os.environ.get('MIMO_API_KEY', '')
API_URL     = os.environ.get('MIMO_API_URL', 'https://token-plan-cn.xiaomimimo.com/v1')
MODEL       = os.environ.get('MIMO_MODEL', 'mimo-v2.5-pro')
TEMPERATURE = float(os.environ.get('MIMO_TEMPERATURE', '0.7'))
MAX_TOKENS  = int(os.environ.get('MIMO_MAX_TOKENS', '2000'))
SLEEP_SEC   = float(os.environ.get('MIMO_SLEEP_SEC', '0.3'))
N_ENSEMBLE  = 3

# 采样配置
SAMPLES_RELIABLE = 1000  # class 3,4,6 每类采样数（可靠软标签）
SAMPLES_FALLBACK = 200   # class 0,1,2,5 每类采样数（one-hot fallback）
N_CLS = 7

# ============ 路径 ============
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE  = f"{BASE_DIR}/results/soft_labels/harth_soft.npy"
ERR_LOG   = f"{_LOG_DIR}/gen_harth_errors.log"
CKPT_FILE = f"{_LOG_DIR}/gen_harth_checkpoint.json"

# ============ 数据加载（6通道） ============
def load_harth():
    import pandas as pd
    base = f"{BASE_DIR}/datasets/HARTH/harth"
    files = sorted(glob(f'{base}/*.csv'))
    d, l, s = [], [], []
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
    for f in files:
        subject = f.split('/')[-1].replace('.csv', '')
        try:
            df = pd.read_csv(f)
            back  = df.iloc[:, 1:4].values.astype(np.float32)
            thigh = df.iloc[:, 4:7].values.astype(np.float32)
            x = np.concatenate([back, thigh], axis=1)
            y_ = df.iloc[:, 7].values.astype(int)
            for i in range(0, len(x)-127, 64):
                w = x[i:i+128]
                label = y_[i]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                    d.append(w); l.append(label_map[label]); s.append(subject)
        except:
            pass
    X = np.array(d, dtype=np.float32)
    y = np.array(l, dtype=np.int64)
    subjects = np.array(s)
    return X, y, subjects

# ============ 特征计算 ============
def compute_features(sample):
    sample = np.clip(sample, -5.0, 5.0)
    back  = sample[:, :3]
    thigh = sample[:, 3:6]
    FS = 100.0

    b_mean = back.mean(axis=0)
    t_mean = thigh.mean(axis=0)
    b_gn = b_mean / (np.linalg.norm(b_mean) + 1e-10)
    t_gn = t_mean / (np.linalg.norm(t_mean) + 1e-10)

    b_std_total = float(np.mean([back[:,0].std(), back[:,1].std(), back[:,2].std()]))
    t_std_total = float(np.mean([thigh[:,0].std(), thigh[:,1].std(), thigh[:,2].std()]))

    t_detrend = np.sqrt((thigh**2).sum(axis=1)) - np.sqrt((thigh**2).sum(axis=1)).mean()
    fft_t = np.abs(np.fft.rfft(t_detrend))
    freqs = np.fft.rfftfreq(len(t_detrend), d=1.0/FS)
    fft_peak = float(freqs[np.argmax(fft_t[1:])+1]) if len(fft_t) > 1 else 0.0

    return {
        'b_gn_z': float(b_gn[2]), 't_gn_z': float(t_gn[2]),
        'b_gn_x': float(b_gn[0]),
        'b_std_total': b_std_total,
        't_std_total': t_std_total,
        'fft_peak': fft_peak,
    }

# ============ Prompt（只区分3类）============
CN = ['stand', 'stairs_up', 'sit', 'lie', 'walk', 'stand_still', 'stairs_down']

def build_prompt(sample, true_label):
    f = compute_features(sample)
    cls_name = ['stand', 'stairs_up', 'sit', 'lie', 'walk', 'stand_still', 'stairs_down'][true_label]
    return f"""You are classifying human activity from back IMU + thigh IMU (128 steps @ 100Hz).

The 7 activity classes are:
  0=stand, 1=stairs_up, 2=sit, 3=lie, 4=walk, 5=stand_still, 6=stairs_down

=== GROUND TRUTH (for this sample) ===
This window's true label is class {true_label} ({cls_name}).
Your task: estimate class confusion — what other classes could this sensor pattern be mistaken for?

=== SENSOR FEATURES (this window) ===
  thigh_gn_z = {f['t_gn_z']:+.3f}   (+ = leg up/back, - = leg down)
  back_gn_z  = {f['b_gn_z']:+.3f}   (+ = lying face-up, - = upright)
  thigh_std   = {f['t_std_total']:.4f}  (leg motion magnitude)
  back_std    = {f['b_std_total']:.4f}   (body motion magnitude)

=== CONFUSION ESTIMATION ===
Based on the features above and the fact this is truly class {true_label} ({cls_name}):
- Which classes have similar sensor signatures to class {true_label}?
- Assign HIGHER probability to those confusable classes (0.15-0.35)
- Assign LOWER probability to classes with different signatures (0.02-0.15)
- Keep the true class {true_label} as highest but NOT one-hot (0.35-0.55)

Key patterns:
  - class 6 (stairs_down): thigh_gn_z > +0.60
  - class 3 (lie): back_gn_z > +0.60
  - class 4 (walk): thigh_std > 0.25
  - classes 0,1,2,5: overlap heavily — NOT separable

=== OUTPUT FORMAT ===
Return ONLY valid JSON (no explanation):
{{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4,"5":p5,"6":p6}}
All probabilities must sum to 1. Do NOT output one-hot."""

# ============ JSON 解析 ============
def extract_json_probs(text, n_cls):
    if not text: return None
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()
    candidates = []
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(n_cls)):
                vals = [float(d[str(k)]) for k in range(n_cls)]
                s = sum(vals)
                if s > 0: candidates.append(vals)
        except: pass
    if candidates:
        vals = candidates[-1]; s = sum(vals)
        return [v/s for v in vals]
    vals = [0.0]*n_cls
    for i in range(n_cls):
        m = re.search(rf'"{i}"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if m: vals[i] = float(m.group(1))
    if sum(vals) > 0:
        s = sum(vals); return [v/s for v in vals]
    return None

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
            if 'choices' not in data: raise ValueError(f"No choices: {data}")
            content = data['choices'][0]['message']['content']
            result = extract_json_probs(content, N_CLS)
            if result is not None: return result, None
            return None, 'no valid json found'
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries: time.sleep(10)
    return None, last_err

def call_api_multi(prompt, n=3):
    results, errors = [], []
    for _ in range(n):
        r, e = call_api(prompt)
        if r is not None: results.append(np.array(r))
        else: errors.append(e)
        time.sleep(SLEEP_SEC)
    if not results: return None, errors[0] if errors else 'all failed'
    return np.mean(results, axis=0).tolist(), None

# ============ 检查点 ============
def load_checkpoint():
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f: return json.load(f)
    return {'done': [], 'soft': {}}

def save_checkpoint(done_list, soft_dict):
    with open(CKPT_FILE, 'w') as f:
        json.dump({'done': done_list, 'soft': soft_dict}, f)

def is_valid(v):
    try:
        v = list(v)
        if len(v) != N_CLS or not all(isinstance(x,(int,float)) for x in v): return False
        if any(x < 0 for x in v): return False
        s = sum(v)
        if s < 0.01: return False
        v_norm = [x/s for x in v]
        if max(v_norm) > 0.90: return False  # 拒绝过度的 one-hot
        return True
    except: return False

# ============ 主循环 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='强制从头开始')
    args = parser.parse_args()

    if args.force:
        if os.path.exists(CKPT_FILE): os.remove(CKPT_FILE)

    logger.info("=" * 60)
    logger.info("HARTH 软标签生成 (方案A: 只区分3类)")
    logger.info(f"Mimo API: {API_URL}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Ensemble: {N_ENSEMBLE}")
    logger.info(f"Reliable classes (3,4,6): {SAMPLES_RELIABLE}/class")
    logger.info(f"Fallback classes (0,1,2,5): {SAMPLES_FALLBACK}/class")
    logger.info("=" * 60)

    X, y, subjects = load_harth()
    n_total = len(X)
    n_cls = N_CLS
    logger.info(f"原始 HARTH: {n_total} 窗口, {len(set(subjects))} 受试者")

    logger.info("\n原始类别分布:")
    for c in range(n_cls):
        cnt = (y == c).sum()
        logger.info(f"  {c} ({CN[c]:12s}): {cnt:>6d} ({100*cnt/n_total:5.1f}%)")

    # ============ 分层采样 ============
    np.random.seed(42)
    per_class_samples = {}
    sample_indices = []

    # 可靠的3类: class 3, 4, 6 → 软标签
    for c in [3, 4, 6]:
        cls_idx = np.where(y == c)[0]
        n_take = min(SAMPLES_RELIABLE, len(cls_idx))
        chosen = cls_idx[np.random.permutation(len(cls_idx))[:n_take]]
        per_class_samples[c] = chosen.tolist()
        sample_indices.extend(chosen.tolist())
        logger.info(f"  软标签 {c} ({CN[c]:12s}): {n_take}/{len(cls_idx)}")

    # fallback的4类: class 0, 1, 2, 5 → one-hot
    for c in [0, 1, 2, 5]:
        cls_idx = np.where(y == c)[0]
        n_take = min(SAMPLES_FALLBACK, len(cls_idx))
        chosen = cls_idx[np.random.permutation(len(cls_idx))[:n_take]]
        per_class_samples[c] = chosen.tolist()
        sample_indices.extend(chosen.tolist())
        logger.info(f"  One-hot  {c} ({CN[c]:12s}): {n_take}/{len(cls_idx)}")

    sample_indices = np.array(sample_indices)
    n_samples = len(sample_indices)
    shuffle_idx = np.random.permutation(n_samples)
    sample_indices = sample_indices[shuffle_idx]

    logger.info(f"\n总计采样: {n_samples} 个 (可靠软标签: {3*SAMPLES_RELIABLE}, one-hot: {4*SAMPLES_FALLBACK})")

    y_soft = np.zeros((n_samples, n_cls), dtype=np.float32)

    # 断点续传
    ckpt = load_checkpoint()
    done_global = set(ckpt['done'])
    idx_map = {int(oi): ni for ni, oi in enumerate(sample_indices)}

    soft_restore = {}
    for oi_str, row in ckpt['soft'].items():
        oi = int(oi_str)
        if oi in idx_map and is_valid(row):
            ni = idx_map[oi]
            arr = np.array(row, dtype=np.float32)
            y_soft[ni] = arr
            soft_restore[ni] = arr

    done_count = len(soft_restore)
    logger.info(f"断点续传: 已处理 {done_count}/{n_samples}")

    def get_cls_progress():
        per_cls = {}
        for c in range(n_cls):
            done_c = sum(1 for oi in per_class_samples[c] if oi in done_global)
            per_cls[c] = (done_c, len(per_class_samples[c]))
        return per_cls

    # 标记哪些类需要 API 调用（3,4,6 需要，0,1,2,5 直接 one-hot）
    NEED_API = {0, 1, 2, 3, 4, 5, 6}  # 所有类都走 API（条件软标签）
    start_time = time.time()

    for pos in range(n_samples):
        orig_idx = int(sample_indices[pos])
        if pos in soft_restore:
            continue

        true_label = int(y[orig_idx])

        # fallback 类直接用 one-hot
        if true_label not in NEED_API:
            y_soft[pos] = np.zeros(n_cls, dtype=np.float32)
            y_soft[pos, true_label] = 1.0
            soft_restore[pos] = y_soft[pos].copy()
            done_global.add(orig_idx)
            # 定期保存检查点
            if (pos + 1) % 100 == 0:
                save_checkpoint(list(done_global), {str(int(k)): v.tolist() for k, v in soft_restore.items()})
            continue

        # 可靠的3类: 调用 API
        prompt = build_prompt(X[orig_idx], true_label)
        result, err = call_api_multi(prompt, n=N_ENSEMBLE)

        if result is None or not is_valid(result):
            result2, err2 = call_api_multi(prompt, n=N_ENSEMBLE)
            if result2 is not None and is_valid(result2):
                result = result2; err = None
            else:
                with open(ERR_LOG, 'a') as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} orig_idx={orig_idx} "
                             f"pos={pos} true={true_label} err={err}\n")
                # API 失败: 用 one-hot fallback
                result = np.zeros(n_cls, dtype=np.float32)
                result[true_label] = 1.0
                logger.warning(f"[%d/%d] orig_idx=%d true=%d API_FAILED → one-hot", pos+1, n_samples, orig_idx, true_label)

        y_soft[pos] = result
        soft_restore[pos] = np.array(result, dtype=np.float32)
        done_global.add(orig_idx)

        save_checkpoint(list(done_global), {str(int(k)): v.tolist() for k, v in soft_restore.items()})

        pred = int(np.argmax(result))
        ok = "✓" if pred == true_label else "✗"
        entropy = -sum(p * np.log(p + 1e-10) for p in result if p > 0)
        logger.info(f"[%d/%d] cls=%d->%d[%s] E=%.3f [%s]",
                    pos+1, n_samples, true_label, pred, ok, entropy,
                    ','.join(f'{v:.2f}' for v in result))

        if (pos + 1) % 20 == 0 or (pos + 1) == n_samples:
            elapsed = time.time() - start_time
            done_now = len(soft_restore)
            rate = done_now / elapsed if elapsed > 0 else 0
            remain = (n_samples - done_now) / rate if rate > 0 else 0
            per_cls = get_cls_progress()
            cls_str = " | ".join(f"c{c}={d}/{t}" for c, (d, t) in sorted(per_cls.items()))
            logger.info(f"  → %d/%d (%d%%) | %.1f/s | ETA %.0fmin | %s",
                        done_now, n_samples, int(100*done_now/n_samples),
                        rate, remain/60, cls_str)

    # ============ 保存 ============
    meta = {
        'sample_indices': sample_indices.tolist(),
        'subjects': subjects[sample_indices].tolist(),
        'true_labels': y[sample_indices].tolist(),
        'n_cls': n_cls,
        'class_names': CN,
        'sampling': {
            'reliable_classes': list(NEED_API),
            'samples_per_reliable_class': SAMPLES_RELIABLE,
            'samples_per_fallback_class': SAMPLES_FALLBACK,
        },
    }
    meta_file = OUT_FILE.replace('.npy', '_meta.json')
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    np.save(OUT_FILE, y_soft.astype(np.float32))
    logger.info(f"\n✅ 完成！软标签: {OUT_FILE} shape={y_soft.shape}")
    logger.info(f"   元数据: {meta_file}")

    # ============ 质量报告 ============
    preds = y_soft.argmax(axis=1)
    true_labels = y[sample_indices]
    overall_acc = (preds == true_labels).mean()
    logger.info(f"\n=== 质量报告 ===")
    logger.info(f"Overall accuracy: {overall_acc:.4f}")

    # 区分可靠类的准确率
    for c in range(n_cls):
        mask = true_labels == c
        if mask.sum() > 0:
            c_acc = (preds[mask] == true_labels[mask]).mean()
            soft_rows = y_soft[mask]
            mean_max = np.mean(soft_rows.max(axis=1))
            is_onehot = np.allclose(soft_rows.max(axis=1), 1.0, atol=0.01) if c not in NEED_API else False
            tag = "(one-hot)" if is_onehot else ""
            logger.info(f"  %d (%12s): n=%3d acc=%.3f mean_max=%.3f %s",
                         c, CN[c], mask.sum(), c_acc, mean_max, tag)

if __name__ == '__main__':
    main()
