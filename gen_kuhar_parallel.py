#!/usr/bin/env python3
"""
kuhar 软标签并行生成脚本（完全独立，不依赖 gen_soft_labels_unified.py）

数据只加载一次保存为 mmap，并行进程只读自己需要的类索引，不重复加载全量数据。
  Proc 0: 类 0,1,2
  Proc 1: 类 3,4,5
  Proc 2: 类 6,7,8
  Proc 3: 类 9,10,11
  Proc 4: 类 12,13,14
  Proc 5: 类 15,16,17

用法：
  python3 gen_kuhar_parallel.py start      # 启动全部6个进程
  python3 gen_kuhar_parallel.py start 0   # 只启动进程0
  python3 gen_kuhar_parallel.py merge     # 合并结果
  python3 gen_kuhar_parallel.py status     # 查看进度
"""

import os, sys, time, json, subprocess, glob, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============ 配置 ============
BASE_DIR  = "/home/fandy/workplace/thesis"
DATA_DIR  = f"{BASE_DIR}/datasets/KuHar"
OUT_DIR  = f"{BASE_DIR}/results/soft_labels"
CKPT_DIR = f"{OUT_DIR}/kuhar_par_checkpoints"
LOG_DIR  = f"{BASE_DIR}/results/logs"
X_CACHE = f"{OUT_DIR}/kuhar_X_tr.npy"
y_CACHE = f"{OUT_DIR}/kuhar_y_tr.npy"
CPP     = 3

# API 配置（与原脚本一致）
API_URL    = "https://api.minimaxi.com/v1"
MODEL      = "MiniMax-M2.7-highspeed"
API_KEY    = "sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc"
TEMP       = 0.7
MAX_TOKENS = 50000
SLEEP_SEC  = 1.0

CLASS_NAMES = [
    "Stand","Sit","Talk-sit","Talk-stand","Stand-sit","Lay",
    "Lay-stand","Pick","Jump","Push-up","Sit-up","Walk",
    "Walk-backwards","Walk-circle","Run","Stair-up","Stair-down","Table-tennis"
]
N_CLS = 18

# ============ 数据加载（一次性缓存到 npz）============

def load_or_create_data():
    """加载数据并缓存到 npz，以后只从 npz 读取"""
    if os.path.exists(X_CACHE) and os.path.exists(y_CACHE):
        print(f"[数据] 从缓存加载: X_tr={X_CACHE}")
        X_tr = np.load(X_CACHE, mmap_mode='r')
        y_tr = np.load(y_CACHE, mmap_mode='r')
        return X_tr, y_tr
    print("[数据] 第一次运行，正在加载全量数据（需约10秒）...")
    d_list, l_list = [], []
    for folder in sorted(glob.glob(f"{DATA_DIR}/1.Raw_time_domian_data/*/")):
        label = int(os.path.basename(folder.rstrip("/")).split(".")[0])
        for f in sorted(glob.glob(f"{folder}/*.csv")):
            try:
                df = pd.read_csv(f, header=None)
                data = df.values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        d_list.append(w); l_list.append(label)
            except: pass
    X = np.array(d_list, dtype=np.float32)
    y = np.array(l_list, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    np.save(X_CACHE, X)
    np.save(y_CACHE, y)
    print(f"[数据] 已缓存: X_tr={X.shape}, y_tr={y.shape}")
    return X, y

# ============ 软标签判断（与原脚本一致）============

def is_valid_soft_label(row):
    s = row.sum()
    if s < 0.99: return False
    second = np.sort(row)[-2]
    if row.max() > 0.90 and second < 0.20: return False
    return True

# ============ API 调用（与原脚本一致）============

def call_api(prompt, n_cls=N_CLS):
    import re, json as json_mod
    from openai import OpenAI
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=120.0)
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMP,
            extra_body={"reasoning_split": True}
        )
        msg = r.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        # 优先从 content 提取 JSON，fallback 到 reasoning
        text = content.strip() if content.strip() else reasoning.strip()
        if not text:
            return None, "empty content"
    except Exception as e:
        return None, str(e)[:100]
    
    try:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        obj = json_mod.loads(text)
        if isinstance(obj, dict) and "probabilities" in obj:
            obj = obj["probabilities"]
        probs = [float(obj.get(str(i), 0.0)) for i in range(n_cls)]
        s = sum(probs)
        if s > 0: probs = [p/s for p in probs]
        return np.array(probs, dtype=np.float32), None
    except Exception as e:
        return None, f"JSON解析失败: {str(e)[:50]}"

# ============ Prompt（kuhar 专用版，数据: 128×8）============

def build_prompt(data, cn):
    acc = data[:, :3]
    gyro = data[:, 3:6] if data.shape[1] >= 6 else np.zeros_like(acc)
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    descs = [f"{i}={c}" for i, c in enumerate(cn)]
    return f"""Classify physical activity from IMU sensor data (accelerometer + gyroscope).
Classes: {', '.join(descs)}
Features: acc_mag={acc_mag.mean():.3f}±{acc_mag.std():.3f}, acc_mean={[f"{v:.3f}" for v in acc.mean(axis=0)[:3]]}, gyro_mean={[f"{v:.3f}" for v in gyro.mean(axis=0)[:3]]}
Physics: Stand/Sit/Lay=stationary, Walk/Run/Jump=periodic motion, Stair-up/down=vertical pattern
Output JSON with probability distribution: {{"0":0.8,"1":0.1,"2":0.05,...}}}}"""

# ============ 单进程生成 =====================

def run_process(proc_id, verbose=True):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    my_cls = list(range(proc_id * CPP, (proc_id + 1) * CPP))
    ckpt_file = f"{CKPT_DIR}/proc{proc_id}_state.json"
    out_file   = f"{OUT_DIR}/kuhar_soft_proc{proc_id}.npy"
    err_file   = f"{LOG_DIR}/gen_kuhar_par{proc_id}_err.log"

    # 错开启动时间，避免 6 路同时打 API 被限流
    # proc_id=0 立即开始，proc_id=1 等 30s，proc_id=2 等 60s...
    # 这样每批 6 个请求错开 30s，API 能承受
    stagger = proc_id * 30
    if verbose:
        print(f"[Proc {proc_id}] 启动，{stagger}秒后开始（避免并发冲突）")
    time.sleep(stagger)

    # 只加载需要的类索引，不加载全量数据
    X_tr, y_tr = load_or_create_data()
    n_samples = len(y_tr)
    n_cls = N_CLS

    # 只取本进程需要的类索引
    my_cidx = {}
    for c in my_cls:
        my_cidx[c] = np.where(y_tr == c)[0]

    # 初始化或恢复
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f: ckpt = json.load(f)
        y_soft = np.load(out_file) if os.path.exists(out_file) else np.zeros((n_samples, n_cls), dtype=np.float32)
    else:
        ckpt = {"done_classes": [], "current_class": my_cls[0], "current_idx": 0}
        y_soft = np.zeros((n_samples, n_cls), dtype=np.float32)

    with open(err_file, "a") as ef:
        ef.write(f"\n=== Proc {proc_id} 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"); ef.flush()

        for c in my_cls:
            if c in ckpt["done_classes"]:
                if verbose: print(f"[Proc {proc_id}] 类 {c} 已完成，跳过")
                continue
            cidx = my_cidx[c]
            tgt = min(200, max(1, int(len(cidx) * 0.25)))
            already = sum(1 for i in cidx if is_valid_soft_label(y_soft[i]))
            need = max(0, tgt - already)
            print(f"[Proc {proc_id}] 类 {c} ({CLASS_NAMES[c]}): {len(cidx)}样本, 目标{tgt}, 已有{already}, 需生成{need}")
            if need == 0:
                ckpt["done_classes"].append(c)
                continue
            # 重启后如果 current_class 指向已完成的类，直接跳到下一个
            if ckpt["current_class"] == c and c in ckpt["done_classes"]:
                print(f"[Proc {proc_id}] 类 {c} 已完成但 checkpoint 未更新，跳过")
                continue
            done = 0
            start_i = ckpt["current_idx"] if ckpt["current_class"] == c else 0
            t_class_start = time.time()
            for i, global_idx in enumerate(cidx):
                if i < start_i: continue
                if done >= need: break
                if is_valid_soft_label(y_soft[global_idx]): continue
                t_req = time.time()
                prompt = build_prompt(X_tr[global_idx], CLASS_NAMES)
                res, err = call_api(prompt, n_cls)
                latency_ms = (time.time() - t_req) * 1000
                if res is None:
                    wait = 30  # API 失败后等 30 秒再重试，避免被连续拦截
                    print(f"  [{done+1}/{need}] ⚠️ API失败 ({err[:40]}), 等{wait}s后重试...")
                    time.sleep(wait)
                    for retry in range(2):
                        res, err = call_api(prompt, n_cls)
                        if res is not None: break
                        print(f"  [{done+1}/{need}] ⚠️ 重试失败 ({err[:40]}), 等{wait}s...")
                        time.sleep(wait)
                    if res is None:
                        y_soft[global_idx, y_tr[global_idx]] = 1.0
                        ef.write(f"FALLBACK idx={global_idx} class={y_tr[global_idx]} err={err}\n"); ef.flush()
                        done += 1
                        print(f"  [{done}/{need}] ❌FALLBACK idx={global_idx}")
                        continue
                y_soft[global_idx] = res
                is_oh = (res > 0.99).sum() == 1 and res.sum() > 0.99
                if is_oh:
                    ef.write(f"ONEHOT  idx={global_idx} class={y_tr[global_idx]}\n"); ef.flush()
                else:
                    done += 1
                pct = done / need * 100
                elapsed_class = time.time() - t_class_start
                eta = (elapsed_class / done * (need - done)) if done > 0 else 0
                print(f"  [{done}/{need} · {pct:.0f}%] idx={global_idx}: {'🔄ONEHOT' if is_oh else '✅REAL'} max={res.max():.3f} ({latency_ms:.0f}ms) ETA={eta:.0f}s")
                # 每生成1个有效软标签后立即保存checkpoint，避免重启丢失进度
                ckpt["current_class"] = c; ckpt["current_idx"] = i + 1
                np.save(out_file, y_soft)
                with open(ckpt_file, "w") as f: json.dump(ckpt, f)
                if done % 20 == 0:
                    print(f"  💾 [{done}/{need}] 进度已保存")
                time.sleep(SLEEP_SEC)
            ckpt["done_classes"].append(c)
            ckpt["current_class"] = c; ckpt["current_idx"] = 0
            np.save(out_file, y_soft)
            with open(ckpt_file, "w") as f: json.dump(ckpt, f)
            real_c = sum(1 for j in cidx if is_valid_soft_label(y_soft[j]))
            elapsed = time.time() - t_class_start
            print(f"[Proc {proc_id}] ✅ 类 {c} ({CLASS_NAMES[c]}) 完成: {real_c}/{tgt} 耗时{elapsed:.0f}s")

        ef.write(f"=== Proc {proc_id} 完成 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"); ef.flush()

    np.save(out_file, y_soft)
    if verbose: print(f"[Proc {proc_id}] 全部完成 → {out_file}")
    return True

# ============ 合并 =====================

def merge_results():
    _, y_tr = load_or_create_data()
    n = len(y_tr)
    merged = np.zeros((n, N_CLS), dtype=np.float32)
    for pid in range(6):
        part = f"{OUT_DIR}/kuhar_soft_proc{pid}.npy"
        if not os.path.exists(part):
            print(f"[Merge] 警告: {part} 不存在"); continue
        y = np.load(part)
        merged += y
        cids = list(range(pid*CPP, (pid+1)*CPP))
        real = sum(1 for i in range(n) if is_valid_soft_label(merged[i]))
        print(f"[Merge] Proc {pid} (类{cids}): {real} 有效软标签")
    out = f"{OUT_DIR}/kuhar_soft.npy"
    np.save(out, merged)
    total = sum(1 for i in range(n) if is_valid_soft_label(merged[i]))
    print(f"\n[Merge] 完成: {out}")
    print(f"  Shape: {merged.shape}")
    print(f"  总有效软标签: {total}")
    for pid in range(6):
        for f in [f"{OUT_DIR}/kuhar_soft_proc{pid}.npy",
                  f"{CKPT_DIR}/proc{pid}_state.json"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"[Merge] 清理: {f}")

# ============ 状态 =====================

def show_status():
    _, y_tr = load_or_create_data()
    n = len(y_tr)
    total_tgt = sum(min(200, max(1, int(np.sum(y_tr==c)*0.25))) for c in range(N_CLS))
    print(f"\n{'='*60}")
    print(f" KuHar 并行生成状态 (目标: {total_tgt})")
    print(f"{'='*60}")
    done_total = 0
    all_done = True
    for pid in range(6):
        my_c = list(range(pid*CPP, (pid+1)*CPP))
        part  = f"{OUT_DIR}/kuhar_soft_proc{pid}.npy"
        ckpt  = f"{CKPT_DIR}/proc{pid}_state.json"
        if os.path.exists(part):
            y = np.load(part)
            rd = sum(1 for i in range(n) if is_valid_soft_label(y[i]))
            pid_tgt = sum(min(200, max(1, int(np.sum(y_tr==c)*0.25))) for c in my_c)
            pct = rd/pid_tgt*100 if pid_tgt > 0 else 0
            print(f"  Proc {pid} (类{my_c}): {rd}/{pid_tgt} {'✅' if rd>=pid_tgt else '🔄'} ({pct:.1f}%)")
            done_total += rd
        else:
            print(f"  Proc {pid} (类{my_c}): 未启动 ⏸️")
            all_done = False
    print(f"\n  总计: {done_total}/{total_tgt} ({(done_total/total_tgt*100) if total_tgt else 0:.1f}%)")
    r = subprocess.run(["ps","aux"], capture_output=True, text=True)
    running = [l for l in r.stdout.split('\n') if "gen_kuhar_parallel" in l and "python3" in l and "grep" not in l]
    if running:
        print(f"\n  运行中: {len(running)} 个进程")
        for l in running:
            print(f"    {l.strip()}")
    else:
        print(f"\n  无运行中进程")

    # 打印各进程最近日志
    for pid in range(6):
        log_file = f"{LOG_DIR}/gen_kuhar_par{pid}.log"
        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()
            last = lines[-20:] if len(lines) > 20 else lines
            if last:
                print(f"\n  --- Proc {pid} 最近日志 ({len(lines)}行) ---")
                for l in last:
                    print(f"    {l.rstrip()}")
    if all_done and os.path.exists(f"{OUT_DIR}/kuhar_soft_proc0.npy"):
        print(f"\n  ✅ 全部完成，可运行: python3 gen_kuhar_parallel.py merge")
    print(f"{'='*60}\n")

# ============ 启动 =====================

def start_workers(only=None):
    ids = [only] if only is not None else list(range(6))
    for pid in ids:
        log = f"{LOG_DIR}/gen_kuhar_par{pid}.log"
        cmd = [sys.executable, "-u", sys.argv[0], "worker", str(pid)]
        with open(log, "w", buffering=1) as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        time.sleep(2)  # 错开启动时刻，确保各进程真正错开
        print(f"[启动] Proc {pid} PID={p.pid} → {log}")
    print(f"\n已启动 {len(ids)} 个进程")

# ============ main =====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "start":
        only = int(sys.argv[2]) if len(sys.argv) > 2 else None
        start_workers(only)
    elif cmd == "worker":
        run_process(int(sys.argv[2]))
    elif cmd == "merge":
        merge_results()
    elif cmd == "status":
        show_status()
    else:
        print(f"未知命令: {cmd}"); print(__doc__); sys.exit(1)
