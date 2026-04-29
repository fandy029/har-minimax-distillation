#!/usr/bin/env python3
"""
kuhar 软标签并行生成脚本（完全独立，不依赖 gen_soft_labels_unified.py）

数据只加载一次保存为 mmap，并行进程只读自己需要的类索引，不重复加载全量数据。
  Proc 0:  类 0        Proc 9:  类 9
  Proc 1:  类 1        Proc 10: 类 10
  ...                     ...
  Proc 8:  类 8        Proc 17: 类 17
  共18个进程，每进程处理1个类。

用法：
  python3 gen_kuhar_parallel.py start      # 启动全部18个进程
  python3 gen_kuhar_parallel.py start 0   # 只启动进程0
  python3 gen_kuhar_parallel.py merge     # 合并结果
  python3 gen_kuhar_parallel.py status    # 查看进度
"""

import os, sys, time, json, subprocess, glob, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============ 配置 ============
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = f"{BASE_DIR}/datasets/KuHar"
OUT_DIR  = f"{BASE_DIR}/results/soft_labels"
CKPT_DIR = f"{OUT_DIR}/kuhar_par_checkpoints"
LOG_DIR  = f"{BASE_DIR}/results/logs"
X_CACHE = f"{OUT_DIR}/kuhar_X_tr.npy"
y_CACHE = f"{OUT_DIR}/kuhar_y_tr.npy"
CPP     = 1   # 每进程处理类数，18进程×1类=18类全覆盖
RATIO   = 0.40
LIMIT   = 400

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
    # one-hot: 最大值 > 0.97，且第二大值 < 0.20 → 舍弃，重新生成
    if row.max() > 0.97 and second < 0.20: return False
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

class TimestampedLogger:
    """同时写文件（带时间戳）和 stdout"""
    def __init__(self, path):
        self.file = open(path, 'a', buffering=1)  # 'a' 模式追加写入，保留旧日志
        self.buf = ''
    def write(self, msg):
        self.buf += msg
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            self.file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
            self.file.flush()
            sys.__stdout__.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
            sys.__stdout__.flush()
    def flush(self):
        if self.buf:
            self.file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.buf}\n")
            self.file.flush()
            sys.__stdout__.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.buf}\n")
            sys.__stdout__.flush()
            self.buf = ''
        self.file.flush()
    def close(self):
        self.flush()
        self.file.close()

def _write_ckpt(ckpt_file, ckpt):
    """原子写入 checkpoint：先写临时文件再 rename，避免损坏"""
    tmp = ckpt_file + ".tmp"
    with open(tmp, "w") as f:
        # numpy int64 → Python int，避免 JSON 序列化失败
        raw = {
            "done_classes": [int(x) for x in ckpt["done_classes"]],
            "current_class": int(ckpt["current_class"]),
            "current_idx": int(ckpt["current_idx"]),
            "exhausted_idxs": [int(x) for x in ckpt.get("exhausted_idxs", [])],
        }
        json.dump(raw, f)
    os.replace(tmp, ckpt_file)  # 原子替换，旧文件自动消失

def _load_ckpt(ckpt_file):
    """加载 checkpoint，文件损坏时返回 None"""
    if not os.path.exists(ckpt_file):
        return None
    try:
        with open(ckpt_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def run_process(proc_id, log_path=None, verbose=True):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    my_cls = list(range(proc_id * CPP, (proc_id + 1) * CPP))
    ckpt_file = f"{CKPT_DIR}/proc{proc_id}_state.json"
    out_file   = f"{OUT_DIR}/kuhar_soft_proc{proc_id}.npy"
    err_file   = f"{LOG_DIR}/gen_kuhar_{proc_id}_err.log"
    
    # 重定向 stdout 到带时间戳的日志文件
    if log_path is not None:
        logger = TimestampedLogger(log_path)
        sys.stdout = logger
        sys.stderr = logger

    # 错开启动时间，避免 18 路同时打 API 被限流
    # proc_id=0 立即开始，proc_id=1 等 5s，proc_id=2 等 10s...
    # 这样 18 个请求每5秒错开一批，API 能承受
    stagger = proc_id * 5
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

    # 初始化或恢复（损坏的 checkpoint 自动跳过，从头开始）
    ckpt = _load_ckpt(ckpt_file)
    if ckpt is not None and os.path.exists(out_file):
        try:
            y_soft = np.load(out_file)
        except Exception as e:
            print(f'[Proc {proc_id}] 警告: 软标签文件损坏，从头开始 ({e})')
            y_soft = np.zeros((n_samples, n_cls), dtype=np.float32)
    else:
        ckpt = {"done_classes": [], "current_class": my_cls[0], "current_idx": 0, "exhausted_idxs": []}
        y_soft = np.zeros((n_samples, n_cls), dtype=np.float32)

    with open(err_file, "a") as ef:
        ef.write(f"\n=== Proc {proc_id} 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"); ef.flush()

        for c in my_cls:
            if c in ckpt["done_classes"]:
                if verbose: print(f"[Proc {proc_id}] 类 {c} 已完成，跳过")
                continue
            cidx = my_cidx[c]
            tgt = min(LIMIT, max(1, int(len(cidx) * RATIO)))
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
            # 追踪已耗尽的样本（API彻底失败后设置one-hot，跳过不再重试）
            exhausted = set(ckpt.get("exhausted_idxs", []))
            done = 0
            start_i = ckpt["current_idx"] if ckpt["current_class"] == c else 0
            t_class_start = time.time()
            for i, global_idx in enumerate(cidx):
                if i < start_i: continue
                if done >= need: break
                if global_idx in exhausted: continue  # 已耗尽，跳过
                if is_valid_soft_label(y_soft[global_idx]): continue
                
                t_req = time.time()
                prompt = build_prompt(X_tr[global_idx], CLASS_NAMES)
                res, err = call_api(prompt, n_cls)
                latency_ms = (time.time() - t_req) * 1000
                
                if res is None:
                    # API 调用失败，最多重试 3 次，每次等 30 秒
                    for retry in range(2):
                        time.sleep(30)
                        res, err = call_api(prompt, n_cls)
                        if res is not None: break
                    if res is None:
                        # 3次全失败 → 标记为耗尽，保存one-hot，跳过不再重试
                        exhausted.add(global_idx)
                        y_soft[global_idx, y_tr[global_idx]] = 1.0
                        ckpt["exhausted_idxs"] = list(exhausted)
                        ckpt["current_class"] = c; ckpt["current_idx"] = i + 1
                        np.save(out_file, y_soft)
                        _write_ckpt(ckpt_file, ckpt)
                        ef.write(f"FALLBACK idx={global_idx} class={y_tr[global_idx]} err={err}\n"); ef.flush()
                        print(f"  ⚠️ idx={global_idx} API彻底失败，标记为耗尽，设置one-hot")
                        continue  # 不增加 done，该样本已耗尽，跳过
                
                # one-hot → 舍弃，重新生成（最多重试 MAX_ONEHOT_RETRY 次）
                MAX_ONEHOT_RETRY = 10
                reject_count = 0
                while not is_valid_soft_label(res):
                    reject_count += 1
                    if reject_count >= MAX_ONEHOT_RETRY:
                        # 重试次数用尽 → 强制接受该 one-hot，不再重试
                        y_soft[global_idx, y_tr[global_idx]] = 1.0
                        ef.write(f"ONEHOT_FORCE idx={global_idx} class={y_tr[global_idx]} reject={reject_count}\n"); ef.flush()
                        print(f"  ⚠️ idx={global_idx} 重试{MAX_ONEHOT_RETRY}次仍为one-hot，强制接受")
                        break
                    if reject_count > 1:
                        print(f"  ⚠️ idx={global_idx} one-hot被舍弃，重新生成 (第{reject_count}次)")
                        ef.write(f"ONEHOT_REJECT idx={global_idx} class={y_tr[global_idx]} reject={reject_count}\n"); ef.flush()
                    t_req = time.time()
                    res, err = call_api(prompt, n_cls)
                    latency_ms = (time.time() - t_req) * 1000
                    if res is None:
                        for retry in range(2):
                            time.sleep(30)
                            res, err = call_api(prompt, n_cls)
                            if res is not None: break
                        if res is None:
                            # API 彻底失败 → 标记为耗尽，设置 one-hot，跳过不再重试
                            exhausted.add(global_idx)
                            y_soft[global_idx, y_tr[global_idx]] = 1.0
                            ckpt["exhausted_idxs"] = list(exhausted)
                            ckpt["current_class"] = c; ckpt["current_idx"] = i + 1
                            np.save(out_file, y_soft)
                            _write_ckpt(ckpt_file, ckpt)
                            ef.write(f"FALLBACK idx={global_idx} class={y_tr[global_idx]} err={err}\n"); ef.flush()
                            print(f"  ⚠️ idx={global_idx} 重试耗尽，标记为耗尽，设置one-hot")
                            break
                    # API 成功但仍是 one-hot → 继续重试
                
                if is_valid_soft_label(res):
                    y_soft[global_idx] = res
                    done += 1
                    pct = done / need * 100
                    elapsed_class = time.time() - t_class_start
                    eta = (elapsed_class / done * (need - done)) if done > 0 else 0
                    extra = f" ({reject_count}次重试后)" if reject_count > 0 else ""
                    print(f"  [{done}/{need} · {pct:.0f}%] idx={global_idx}: ✅REAL max={res.max():.3f} ({latency_ms:.0f}ms){extra} ETA={eta:.0f}s")
                    
                    ckpt["current_class"] = c; ckpt["current_idx"] = i + 1
                    np.save(out_file, y_soft)
                    _write_ckpt(ckpt_file, ckpt)
                    if done % 20 == 0:
                        print(f"  💾 [{done}/{need}] 进度已保存")
                    time.sleep(SLEEP_SEC)
            ckpt["done_classes"].append(c)
            ckpt["current_class"] = c; ckpt["current_idx"] = 0
            np.save(out_file, y_soft)
            _write_ckpt(ckpt_file, ckpt)
            real_c = sum(1 for j in cidx if is_valid_soft_label(y_soft[j]))
            elapsed = time.time() - t_class_start
            print(f"[Proc {proc_id}] ✅ 类 {c} ({CLASS_NAMES[c]}) 完成: {real_c}/{tgt} 耗时{elapsed:.0f}s")

        ef.write(f"=== Proc {proc_id} 完成 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"); ef.flush()

    np.save(out_file, y_soft)
    if verbose: print(f"[Proc {proc_id}] 全部完成 → {out_file}")
    if log_path is not None:
        logger.close()
    return True

# ============ 合并 =====================

def merge_results():
    _, y_tr = load_or_create_data()
    n = len(y_tr)
    merged = np.zeros((n, N_CLS), dtype=np.float32)
    for pid in range(18):
        part = f"{OUT_DIR}/kuhar_soft_proc{pid}.npy"
        if not os.path.exists(part):
            print(f"[Merge] 警告: {part} 不存在"); continue
        y = np.load(part)
        merged += y
        cids = list(range(pid*CPP, (pid+1)*CPP))
        real = sum(1 for i in range(n) if is_valid_soft_label(merged[i]))
        cls_name = CLASS_NAMES[cids[0]] if cids else '?'
        print(f"[Merge] Proc {pid} (类{cids[0] if cids else '?'}={cls_name}): {real} 有效软标签")
    out = f"{OUT_DIR}/kuhar_soft.npy"
    np.save(out, merged)
    total = sum(1 for i in range(n) if is_valid_soft_label(merged[i]))
    print(f"\n[Merge] 完成: {out}")
    print(f"  Shape: {merged.shape}")
    print(f"  总有效软标签: {total}")
    # 合并后保留 proc 文件不断点记录，供后续续传使用
    # （如需清理请手动删除 kuhar_soft_proc*.npy 和 kuhar_par_checkpoints/）

# ============ 状态 =====================

def show_status():
    _, y_tr = load_or_create_data()
    n = len(y_tr)
    total_tgt = sum(min(LIMIT, max(1, int(np.sum(y_tr==c) * RATIO))) for c in range(N_CLS))
    print(f"\n{'='*60}")
    print(f" KuHar 并行生成状态 (目标: {total_tgt})")
    print(f"{'='*60}")
    done_total = 0
    all_done = True
    for pid in range(18):
        my_c = list(range(pid*CPP, (pid+1)*CPP))
        part  = f"{OUT_DIR}/kuhar_soft_proc{pid}.npy"
        ckpt  = f"{CKPT_DIR}/proc{pid}_state.json"
        if os.path.exists(part):
            y = np.load(part)
            rd = sum(1 for i in range(n) if is_valid_soft_label(y[i]))
            pid_tgt = sum(min(LIMIT, max(1, int(np.sum(y_tr==c) * RATIO))) for c in my_c)
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
    for pid in range(18):
        log_file = f"{LOG_DIR}/gen_kuhar_{pid}.log"
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
def start_workers(only=None, wait=True):
    ids = [only] if only is not None else list(range(18))
    workers = []
    for pid in ids:
        log = f"{LOG_DIR}/gen_kuhar_{pid}.log"
        cmd = [sys.executable, sys.argv[0], "worker", str(pid), "--ratio", str(RATIO), "--limit", str(LIMIT)]
        # stdout 定向到 /dev/null（子进程自己写日志文件）
        with open(os.devnull, 'w') as devnull:
            p = subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
        workers.append((pid, p))
        time.sleep(2)  # 错开启动时刻
        print(f"[启动] Proc {pid} PID={p.pid} → {log}")
    print(f"\n已启动 {len(ids)} 个进程，正在等待完成...")
    if wait:
        for pid, p in workers:
            ret = p.wait()
            print(f"[Proc {pid}] 进程结束，exit={ret}")
        print("\n所有进程已完成，开始合并...\n")
        merge_results()

# ============ main =====================

if __name__ == "__main__":
    # Parse --ratio and --limit before cmd processing
    new_argv = [sys.argv[0]]  # keep script name
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == '--ratio' and i+1 < len(sys.argv):
            RATIO = float(sys.argv[i+1]); i += 2; continue
        elif a == '--limit' and i+1 < len(sys.argv):
            LIMIT = int(sys.argv[i+1]); i += 2; continue
        elif a.startswith('--'):
            i += 1; continue  # skip unknown flags
        new_argv.append(a); i += 1
    sys.argv = new_argv

    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "start":
        only = int(sys.argv[2]) if len(sys.argv) > 2 else None
        start_workers(only)
    elif cmd == "worker":
        # Handle --ratio and --limit that come after pid
        argv2 = sys.argv[2]
        if argv2.startswith('--'):
            # No pid given, argv2 is actually --ratio
            RATIO = float(sys.argv[3]) if len(sys.argv) > 3 else RATIO
            LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else LIMIT
            run_process(None, log_path=None)  # 无 pid 则不写日志文件
        else:
            pid = int(argv2)
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == '--ratio' and i+1 < len(sys.argv):
                    RATIO = float(sys.argv[i+1]); i += 2
                elif sys.argv[i] == '--limit' and i+1 < len(sys.argv):
                    LIMIT = int(sys.argv[i+1]); i += 2
                else:
                    i += 1
            log_path = f"{LOG_DIR}/gen_kuhar_{pid}.log"
            run_process(pid, log_path=log_path)
    elif cmd == "merge":
        merge_results()
    elif cmd == "status":
        show_status()
    else:
        print(f"未知命令: {cmd}"); print(__doc__); sys.exit(1)
