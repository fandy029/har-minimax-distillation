#!/usr/bin/env python3
"""
HARTH 软标签生成脚本
用法: python gen.py [--force]
"""
import os, sys, json, time, re, math
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d
import fcntl

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # /.../thesis/
BASE_DIR   = THESIS_DIR
OUT_DIR    = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR    = os.path.join(BASE_DIR, 'results', 'logs')
CKPT_FILE  = os.path.join(LOG_DIR, 'gen_harth_checkpoint.json')

# ============ API 配置（使用本地 api_config.py）============
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from api_config import API_KEY, API_URL, MODEL, TEMPERATURE, MAX_TOKENS, SLEEP_SEC, TIMEOUT  # 降低随机性，提高准确率
from openai import OpenAI

# ============ 数据集配置 ============
# label_map={1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7} 与训练一致
# UCI 官方: 1=walk, 2=run, 3=shuffle, 4=stairs_up, 5=stairs_down, 6=stand, 7=sit, 8=lying
CLASS_NAMES = ['walk', 'run', 'shuffle', 'stairs_up', 'stairs_down', 'stand', 'sit', 'lying']
N_CLS       = len(CLASS_NAMES)
MAX_PER_CLASS = 3000  # 每类上限
FORCE_RESTART     = '--force' in sys.argv

# ============ 数据加载 ============
def load_harth_data():
    """
    HARTH 数据加载
    原始数据: back IMU (3) + thigh IMU (3) = 6通道
    label_map 与训练一致: {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7}
    8类: walk(0)/run(1)/shuffle(2)/stairs_up(3)/stairs_down(4)/stand(5)/sit(6)/lying(7)
    被过滤: 13-140(Cycling variants)
    """
    base = os.path.join(BASE_DIR, 'datasets', 'HARTH', 'harth')
    files = sorted(glob(os.path.join(base, '*.csv')))
    d, l = [], []
    # label_map: raw_label -> class_id (与训练一致)
    label_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}

    for f in files:
        try:
            df = pd.read_csv(f)
            # Some files have 8 cols (timestamp,back,thigh,label), some have 9 (+index column)
            n_cols = len(df.columns)
            if n_cols == 9:
                back  = df.iloc[:, 2:5].values.astype(np.float32)   # skip index+timestamp
                thigh = df.iloc[:, 5:8].values.astype(np.float32)
                y_    = df.iloc[:, 8].values.astype(int)
            else:  # 8 columns
                back  = df.iloc[:, 1:4].values.astype(np.float32)
                thigh = df.iloc[:, 4:7].values.astype(np.float32)
                y_    = df.iloc[:, 7].values.astype(int)
            for i in range(0, len(y_)-127, 64):
                w = np.concatenate([back[i:i+128], thigh[i:i+128]], axis=1)  # (128, 6)
                label = y_[i]
                if w.shape[0]==128 and not np.any(np.isnan(w)) and label in label_map:
                    d.append(w); l.append(label_map[label])
        except Exception as e:
            print(f"  加载失败 {f}: {e}")

    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

# ============ Prompt 构建 ============
def build_prompt(window):
    """
    HARTH 专用 prompt
    window: (128, 6) = back(3) + thigh(3)
    """
    back  = window[:, :3]
    thigh = window[:, 3:6]

    # 低通滤波提取重力方向（每个轴）
    back_gf  = uniform_filter1d(back,  10, axis=0)   # (128, 3)
    thigh_gf = uniform_filter1d(thigh, 10, axis=0)   # (128, 3)

    # 每个轴的均值
    back_mean  = back.mean(axis=0)
    thigh_mean = thigh.mean(axis=0)
    # 重力方向（低通后均值）
    back_gz  = back_gf[:, 2].mean()
    thigh_gz = thigh_gf[:, 2].mean()

    # 各轴 std
    back_std  = back.std(axis=0)
    thigh_std = thigh.std(axis=0)

    # 幅值均值（区分 stand/walk/lie 的关键特征）
    back_mag  = np.sqrt((back**2).sum(axis=1)).mean()
    thigh_mag = np.sqrt((thigh**2).sum(axis=1)).mean()

    # 幅值 std（关键特征：区分活动强度）
    thigh_m = np.sqrt((thigh**2).sum(axis=1))
    back_m  = np.sqrt((back**2).sum(axis=1))
    thigh_std_mag = thigh_m.std()
    back_std_mag  = back_m.std()

    # 步态峰值
    peaks = int(np.sum((thigh_m[1:-1] > thigh_m[:-2]) & (thigh_m[1:-1] > thigh_m[2:])))

    # 计算到每个类签名的欧氏距离（标准化）
    # 签名: [thigh_std, thigh_mag, thigh_z, back_std, back_mag]
    sigs = np.array([
        [0.630, 1.325, -0.072, 0.295, 1.054],  # 0 walk
        [1.128, 2.201, -0.140, 0.881, 1.338],  # 1 run
        [0.142, 1.049, -0.065, 0.067, 1.020],  # 2 shuffle
        [0.424, 1.153, -0.011, 0.223, 1.036],  # 3 stairs_up
        [0.530, 1.277, -0.034, 0.345, 1.052],  # 4 stairs_down
        [0.042, 1.013, -0.117, 0.024, 1.012],  # 5 stand
        [0.011, 1.004, +0.898, 0.007, 1.012],  # 6 sit
        [0.012, 1.017, +0.127, 0.007, 0.969],  # 7 lying
    ])
    feat = np.array([thigh_std_mag, thigh_mag, thigh_gz, back_std_mag, back_mag])
    # 标准化距离（各维度除以全局 std）
    global_std = np.array([0.35, 0.45, 0.35, 0.22, 0.10])  # 近似值
    dists = np.sqrt(((sigs - feat) / global_std) ** 2).sum(axis=1)
    ranked = np.argsort(dists)  # 从近到远

    descs = [f'{i}={CLASS_NAMES[i]}' for i in range(N_CLS)]
    rank_lines = []
    for rank, ci in enumerate(ranked):
        tag = ' <-- BEST MATCH' if rank == 0 else ''
        rank_lines.append(f'  #{rank+1} class {ci} ({CLASS_NAMES[ci]:12s}) dist={dists[ci]:.2f}{tag}')

    return f"""Classify this 2.56s window of back+thigh IMU sensor data.

Classes: {", ".join(descs)}

=== THIS WINDOW ===
  thigh_std = {thigh_std_mag:.3f}  (leg motion intensity -- most important)
  thigh_mag = {thigh_mag:.3f}  (leg acceleration magnitude)
  thigh_z   = {thigh_gz:+.3f}  (thigh gravity: +~sitting, ~0~standing)
  back_std  = {back_std_mag:.3f}  (torso motion intensity)
  back_mag  = {back_mag:.3f}  (torso acceleration magnitude)
  back_z    = {back_gz:+.3f}  (torso z-axis gravity)

=== DISTANCE RANKING (precomputed Euclidean distance to each class) ===
{chr(10).join(rank_lines)}

=== EXAMPLE REASONING (for Walk) ===
Your reasoning (2-3 sentences, plain text, in English):
Based on the features: thigh_std=0.630, thigh_mag=1.325, thigh_z=-0.072, back_std=0.295
Comparing to class signatures: Walk(0) has thigh_std~0.630, thigh_mag~1.325. Stairs-down(4) has thigh_std~0.530, thigh_mag~1.277. Shuffle(2) has thigh_std~0.142.
My conclusion: thigh_std=0.630 matches Walk(0). Stairs-down has slightly lower values, Shuffle has much lower intensity.
{{"0":0.55,"1":0.02,"2":0.02,"3":0.05,"4":0.15,"5":0.02,"6":0.02,"7":0.02}}

=== YOUR TASK ===
First, write 2-3 sentences of REASONING in plain text explaining which class this window most likely belongs to and why. Then output the probability distribution as JSON.

=== REASONING FORMAT ===
Your reasoning (2-3 sentences, plain text, in English):
Based on the features: [mention thigh_std, thigh_mag, thigh_z first]
Comparing to class signatures: [compare to 2-3 most similar classes from the ranking]
My conclusion: [state the most likely class]

=== JSON OUTPUT ===
After your reasoning, output the probabilities as valid JSON:
{{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4,"5":p5,"6":p6,"7":p7}}
IMPORTANT: Probabilities must sum to 1. Primary=0.45-0.65, close neighbors=0.05-0.20, others=0.02-0.05. Never output one-hot. Never use 0.833 or 0.056 patterns."""

# ============ API 调用 ============
def call_api(prompt):
    client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=TIMEOUT)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content.strip()
        return content, None
    except Exception as e:
        return None, str(e)

def extract_probs(text):
    """从 API 响应中提取概率向量"""
    if not text:
        return None
    # 去掉 thought 标签和 markdown
    # 去掉各种 thinking/result 标签及其内容
    text = re.sub(r'<THOUGHT>[\s\S]*?</THOUGHT>', '', text)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<RESULT>[\s\S]*?</RESULT>', '', text)
    text = re.sub(r'<[^>]+>', '', text).strip()

    # 找 JSON 对象
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

    # fallback: 正则提取
    vals = [0.0] * N_CLS
    for i in range(N_CLS):
        m = re.search(rf'"{i}"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if m:
            vals[i] = float(m.group(1))
    if sum(vals) > 0:
        return [v/sum(vals) for v in vals]
    return None

def is_valid(probs):
    """判断软标签是否有效（非 one-hot，概率和=1）"""
    if probs is None:
        return False
    row = np.array(probs)
    if not np.isclose(row.sum(), 1.0, atol=0.01):
        return False
    if row.max() >= 0.95:  # 接近 one-hot
        return False
    return True

# ============ 主生成逻辑 ============
def main():
    # 单例锁，防止多进程并发（先确保目录存在）
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    lock_path = os.path.join(OUT_DIR, '.gen_harth.lock')
    lock_fd = open(lock_path, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: 已有实例在运行，请先停止后再启动")
        sys.exit(1)

    # 日志设置
    log_file = os.path.join(LOG_DIR, 'gen_harth.log')
    err_file = os.path.join(LOG_DIR, 'gen_harth_errors.log')
    correct_file = os.path.join(LOG_DIR, 'gen_harth_correct.log')
    ts_fmt = time.strftime('%Y-%m-%d %H:%M:%S')

    # --force: 清空日志和断点（从头开始）
    if FORCE_RESTART:
        for f in [log_file, err_file, correct_file, CKPT_FILE]:
            if os.path.exists(f):
                open(f, 'w').close()
        print(f"[{ts_fmt}] --force: 清除旧日志和断点，从头开始")

    def log(msg, to_file=True, to_stdout=True):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        if to_stdout:
            print(line)
        if to_file:
            with open(log_file, 'a') as f:
                f.write(line + '\n')

    def log_correct(msg):
        """仅写入正确预测日志（安静，不打印）"""
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        with open(correct_file, 'a') as f:
            f.write(line + '\n')

    def log_err(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        with open(err_file, 'a') as f:
            f.write(line + '\n')

    log(f"HARTH 软标签生成开始 (scripts/harth/gen.py)")
    log(f"Mimo API: {API_URL}")
    log(f"Model: {MODEL}")
    log(f"Temperature: {TEMPERATURE}")

    # 加载数据
    log("加载 HARTH 数据...")
    X, y, _, _, _, _ = load_harth_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")

    # 每类统计
    per_cls = {}
    for c in range(N_CLS):
        cnt = int(np.sum(y == c))
        per_cls[c] = cnt
        log(f"  class {c} ({CLASS_NAMES[c]}): {cnt} 样本")

    # 采样：每类上限 MAX_PER_CLASS 个
    np.random.seed(42)
    sample_indices = []
    for c in range(N_CLS):
        cidx = np.where(y == c)[0]
        take = min(MAX_PER_CLASS, len(cidx))
        chosen = np.random.choice(cidx, size=take, replace=False)
        sample_indices.extend(chosen.tolist())
    sample_indices = np.array(sample_indices)
    np.random.shuffle(sample_indices)
    total = len(sample_indices)
    log(f"总计采样: {total} 个窗口（每类 ~{MAX_PER_CLASS}）")

    # 加载断点
    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f:
                ckpt = json.load(f)
            done_set = set(ckpt.get('done', []))
            log(f"  断点续传: {len(done_set)}/{total} 已处理")
        except (json.JSONDecodeError, IOError) as e:
            log(f"  断点文件损坏 ({e})，从头开始")
    else:
        log(f"  断点续传: 无，从头开始")

    # 初始化输出数组（shape: len(X) × N_CLS）
    soft_all = np.zeros((len(X), N_CLS), dtype=np.float32)

    # 续跑时从已保存的soft labels恢复 correct_indices 和统计
    done_count = 0
    true_correct = 0
    correct_indices = []
    class_gen = [0] * N_CLS
    class_corr = [0] * N_CLS
    saved_soft_path = os.path.join(OUT_DIR, 'harth_soft.npy')
    if done_set and os.path.exists(saved_soft_path):
        saved_soft = np.load(saved_soft_path)
        for idx in done_set:
            if saved_soft[idx].sum() > 0:
                soft_all[idx] = saved_soft[idx]
                true_label = int(y[idx])
                pred_label = int(np.argmax(saved_soft[idx]))
                class_gen[true_label] += 1
                done_count += 1
                if pred_label == true_label:
                    true_correct += 1
                    class_corr[true_label] += 1
                    correct_indices.append(idx)
        log(f"  续跑恢复: 完成={done_count}, 正确={true_correct}, "
            f"正确样本数={len(correct_indices)}")

    for pos, orig_idx in enumerate(sample_indices):
        if orig_idx in done_set:
            continue

        true_label = int(y[orig_idx])
        prompt = build_prompt(X[orig_idx])

        # API 调用
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
            # 重新生成
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
                log_correct(line)  # 仅正确样本写入

        done_set.add(orig_idx)
        done_count += 1

        # 每100个打印一次每类正确率统计
        if done_count % 100 == 0:
            stats_lines = []
            for c in range(N_CLS):
                g = class_gen[c]
                cc = class_corr[c]
                pct = cc / g * 100 if g > 0 else 0
                stats_lines.append(f"{CLASS_NAMES[c]}={cc}/{g}({pct:.0f}%)")
            acc_pct = true_correct / done_count * 100
            log(f"  === 100轮统计: 整体={true_correct}/{done_count}({acc_pct:.1f}%) "
                + " ".join(stats_lines) + " ===")

        # 每5个保存一次（两份：全量 + 仅正确预测）
        if done_count % 5 == 0:
            np.save(os.path.join(OUT_DIR, 'harth_soft.npy'), soft_all)
            soft_correct = soft_all[correct_indices]
            np.save(os.path.join(OUT_DIR, 'harth_soft_correct_only.npy'), soft_correct)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': [int(x) for x in done_set], 'class_names': CLASS_NAMES}, f)
            valid_n = int((soft_all.sum(axis=1) > 0).sum())
            acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
            log(f"  进度: {len(done_set)}/{total}, 准确率={acc_pct:.1f}%, "
                f"正确样本={len(correct_indices)}, 已保存")

        time.sleep(SLEEP_SEC)

    # 最终保存（两份：全量 + 仅正确预测）
    out_path = os.path.join(OUT_DIR, 'harth_soft.npy')
    out_correct = os.path.join(OUT_DIR, 'harth_soft_correct_only.npy')
    np.save(out_path, soft_all)
    soft_correct = soft_all[correct_indices]
    np.save(out_correct, soft_correct)
    valid_n = int((soft_all.sum(axis=1) > 0).sum())
    acc_pct = true_correct / done_count * 100 if done_count > 0 else 0

    # 每类统计
    log(f"\n=== 生成完成 ===")
    log(f"  有效软标签: {valid_n}/{len(X)}")
    log(f"  整体准确率: {acc_pct:.1f}% ({true_correct}/{done_count})")
    log(f"  仅正确版本: {out_correct} ({len(correct_indices)} 样本)")
    for c in range(N_CLS):
        g = class_gen[c]
        cc = class_corr[c]
        pct = cc / g * 100 if g > 0 else 0
        log(f"  class {c} ({CLASS_NAMES[c]}): 生成={g}, 正确={cc}, 准确率={pct:.1f}%")

    with open(CKPT_FILE, 'w') as f:
        json.dump({'done': [int(x) for x in done_set], 'class_names': CLASS_NAMES}, f)
    log(f"  输出: {out_path}")

if __name__ == '__main__':
    main()
