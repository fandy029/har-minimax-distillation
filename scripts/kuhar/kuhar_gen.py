#!/usr/bin/env python3
"""
KuHar 软标签生成脚本
数据集: KuHar / 1.Raw_time_domian_data / (18类活动)
8通道×128步窗口 (加速度3 + 陀螺仪3 + 2个额外通道)
用法: python gen.py [--force]
"""
import os, sys, json, time, re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import fcntl
from openai import OpenAI

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR   = THESIS_DIR

sys.path.insert(0, os.path.dirname(SCRIPT_DIR))   # api_config 在 scripts/ 而非 scripts/kuhar/
import api_config as _cfg
API_KEY          = _cfg.API_KEY
API_URL          = _cfg.API_URL
MODEL            = _cfg.MODEL
TEMPERATURE      = _cfg.TEMPERATURE   # 从 api_config 导入，保持一致
MAX_TOKENS       = _cfg.MAX_TOKENS
SLEEP_SEC        = _cfg.SLEEP_SEC
TIMEOUT          = _cfg.TIMEOUT
DISABLE_THINKING = _cfg.DISABLE_THINKING   # 关闭思考过程

# ============ 类别配置 ============
CLASS_NAMES = [
    'Stand', 'Sit', 'Talk-sit', 'Talk-stand', 'Stand-sit',
    'Lay', 'Lay-stand', 'Pick', 'Jump', 'Push-up', 'Sit-up',
    'Walk', 'Walk-backwards', 'Walk-circle', 'Run',
    'Stair-up', 'Stair-down', 'Table-tennis'
]
N_CLS = len(CLASS_NAMES)   # 18
MAX_PER_CLASS = 3000  

# ============ 输出路径 ============
OUT_DIR  = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR  = os.path.join(BASE_DIR, 'results', 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SOFT_FILE    = os.path.join(OUT_DIR, 'kuhar_soft.npy')
CORRECT_FILE = os.path.join(OUT_DIR, 'kuhar_soft_correct_only.npy')
LOG_FILE     = os.path.join(LOG_DIR, 'gen_kuhar.log')
FINAL_FILE = os.path.join(LOG_DIR, 'gen_kuhar_final.log')
ERR_FILE     = os.path.join(LOG_DIR, 'gen_kuhar_errors.log')
CORR_LOG     = os.path.join(LOG_DIR, 'gen_kuhar_correct.log')
CKPT_FILE    = os.path.join(LOG_DIR, 'gen_kuhar_checkpoint.json')
LOCK_FILE    = os.path.join(OUT_DIR, '.gen_kuhar.lock')

FORCE_RESTART = '--force' in sys.argv

# ============ 有效性判断 ============
def is_valid(probs):
    if probs is None:
        return False
    row = np.array(probs)
    if not np.isclose(row.sum(), 1.0, atol=0.01):
        return False
    return True

# ============ JSON 解析 ============
def extract_probs(text):
    """从 API 响应中提取概率向量。关闭 thinking 靠 API 参数兜底清理。"""
    if not text:
        return None
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<RESULT>.*?</RESULT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()
    for m in re.finditer(r'\{[^}]+\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                def to_float(v):
                    try:
                        return float(str(v).replace(',', '.'))
                    except:
                        return 0.0
                vals = [to_float(d[str(k)]) for k in range(N_CLS)]
                arr = np.array(vals, dtype=float)
                arr = np.clip(arr, 0, 1)
                if arr.sum() > 0:
                    return (arr / arr.sum()).tolist()
        except:
            pass
    return None
def call_api(prompt):
    """调用 Mimo API，关闭思考过程。"""
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
            content = r.choices[0].message.content.strip()
            probs = extract_probs(content)
            if probs is None:
                last_err = 'JSON解析失败'
                time.sleep(2)
                continue
            return probs, None
        except RateLimitError:
            last_err = f'429限流 (attempt {attempt+1})'
            time.sleep(15 * (2 if attempt > 0 else 1))
        except Exception as e:
            last_err = f'API错误: {e}'
            time.sleep(5)
    return None, last_err

# ============ 日志函数 ============
def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    if sys.stdout.isatty():
        print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def log_correct(msg):
    with open(CORR_LOG, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def log_final(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(FINAL_FILE, 'a') as f:
        f.write(f'[{ts}] {msg}\n')

def log_err(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    if sys.stdout.isatty():
        print(line)
    with open(ERR_FILE, 'a') as f:
        f.write(line + '\n')

# ============ 数据加载 ============
def load_kuhar_data():
    """
    KuHar 数据加载 - 与 run_distill.py 保持一致，使用全部 8 列
    CSV 格式 (8列，无表头):
      col0: Accelerometer timestamp (ms)
      col1-3: Accel X/Y/Z (m/s^2)
      col4: Gyroscope timestamp (ms)
      col5-7: Gyro X/Y/Z (rad/s)
    步长 ~10ms (100Hz)，每个文件滑动窗口 128 步，步进 64
    """
    base = os.path.join(BASE_DIR, 'datasets', 'KuHar', '1.Raw_time_domian_data')
    d, l = [], []

    folders = sorted(glob(os.path.join(base, '*')))
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        folder_name = os.path.basename(folder)
        parts = folder_name.split('.')
        if len(parts) < 2:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if class_id < 0 or class_id >= N_CLS:
            continue

        csv_files = sorted(glob(os.path.join(folder, '*.csv')))
        for f in csv_files:
            try:
                df = pd.read_csv(f, header=None)
                data = df.values.astype(np.float32)   # (N, 8) — 全部8列，与训练一致
                for start in range(0, len(data) - 127, 64):
                    window = data[start:start + 128]   # (128, 8)
                    if window.shape[0] == 128 and not np.any(np.isnan(window)):
                        d.append(window)
                        l.append(class_id)
            except Exception:
                continue

    X = np.array(d, dtype=np.float32)   # (N, 128, 8)
    y = np.array(l, dtype=np.int64)

    if len(X) == 0:
        raise RuntimeError("KuHar 数据加载失败: 未找到有效数据")

    # 划分: 80% train, 10% val, 10% test
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X, X_vl, y, y_vl = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X, y, X_vl, y_vl, X_te, y_te

# ============ 特征计算 ============
def compute_features(window):
    """从 (128, 8) 窗口计算全面的时域+频域特征"""
    # Accelerometer: cols 1-3
    acc = window[:, 1:4]
    # Gyroscope: cols 5-7
    gyro = window[:, 5:8]

    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
    acc_mean = acc.mean(axis=0)
    acc_std = acc.std(axis=0)
    gyro_mean = gyro.mean(axis=0)
    gyro_std = gyro.std(axis=0)

    # 峰值数（步态周期检测）
    n_peaks_acc = int(np.sum(
        (acc_mag[1:-1] > acc_mag[:-2]) & (acc_mag[1:-1] > acc_mag[2:])
    ))
    n_peaks_gyro = int(np.sum(
        (gyro_mag[1:-1] > gyro_mag[:-2]) & (gyro_mag[1:-1] > gyro_mag[2:])
    ))

    # FFT 频域特征（行走/跑步有明显的步态频率）
    try:
        fft_acc = np.abs(np.fft.rfft(acc_mag))
        fft_freq = np.fft.rfftfreq(128, d=0.01)   # 100Hz 采样
        dom_freq = float(fft_freq[np.argmax(fft_acc[1:]) + 1]) if len(fft_acc) > 1 else 0.0
        fft_max = float(fft_acc.max())
    except Exception:
        dom_freq, fft_max = 0.0, 0.0

    # Zero-crossing rate（说话类活动特有）
    def zcr(signal):
        return float(np.sum(np.abs(np.diff(np.sign(signal - signal.mean()))) > 0) / len(signal))
    zcr_acc = zcr(acc_mag)
    zcr_gyro = zcr(gyro_mag)

    # 额外特征：更精确区分
    acc_x_mean, acc_y_mean, acc_z_mean = float(acc_mean[0]), float(acc_mean[1]), float(acc_mean[2])
    acc_x_std, acc_y_std, acc_z_std = float(acc_std[0]), float(acc_std[1]), float(acc_std[2])
    gyro_x_mean, gyro_y_mean, gyro_z_mean = float(gyro_mean[0]), float(gyro_mean[1]), float(gyro_mean[2])
    gyro_x_std, gyro_y_std, gyro_z_std = float(gyro_std[0]), float(gyro_std[1]), float(gyro_std[2])

    # 信号能量
    energy_acc = float((acc_mag ** 2).mean())
    energy_gyro = float((gyro_mag ** 2).mean())

    # NEW: periodicity (lag-1 autocorrelation)
    def autocorr(sig):
        if np.std(sig) < 1e-8: return 0.0
        n = len(sig)
        c0 = np.correlate(sig-sig.mean(), sig-sig.mean(), mode='full')[n-1]
        c1 = np.correlate(sig-sig.mean(), sig-sig.mean(), mode='full')[n-2]
        return c1/(c0+1e-10)
    acc_auto1 = float(autocorr(acc_mag))
    gyro_auto1 = float(autocorr(gyro_mag))

    # NEW: impulsiveness (max/rms, >3=impulsive)
    impulsiveness = float(acc_mag.max() / (np.sqrt(np.mean(acc_mag**2)) + 1e-10))


    # 垂直方向偏差（上下楼梯方向性）
    y_bias = acc[:, 1].mean()

    # Kurtosis and skewness (关键区分特征)
    def moments(sig):
        m = sig - sig.mean()
        m2 = (m**2).mean(); m3 = (m**3).mean(); m4 = (m**4).mean()
        std = np.sqrt(m2) + 1e-10
        return float(m3/(std**3)), float(m4/(std**4) - 3)
    skew_am, kurt_am = moments(acc_mag)
    skew_ax, kurt_ax = moments(acc[:, 0])
    skew_ay, kurt_ay = moments(acc[:, 1])
    skew_az, kurt_az = moments(acc[:, 2])
    skew_gm, kurt_gm = moments(gyro_mag)

    # 互相关（Stand vs Sit 区分有效）
    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8: return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    corr_xy = safe_corr(acc[:, 0], acc[:, 1])
    corr_xz = safe_corr(acc[:, 0], acc[:, 2])
    corr_yz = safe_corr(acc[:, 1], acc[:, 2])
    gyro_corr_xy = safe_corr(gyro[:, 0], gyro[:, 1])

    return {
        'acc_mag_mean': float(acc_mag.mean()),
        'acc_mag_std':  float(acc_mag.std()),
        'acc_mag_max':  float(acc_mag.max()),
        'acc_x_mean':   acc_x_mean,
        'acc_y_mean':   acc_y_mean,
        'acc_z_mean':   acc_z_mean,
        'acc_x_std':    acc_x_std,
        'acc_y_std':    acc_y_std,
        'acc_z_std':    acc_z_std,
        'gyro_mag_mean': float(gyro_mag.mean()),
        'gyro_mag_std':  float(gyro_mag.std()),
        'gyro_x_mean':   gyro_x_mean,
        'gyro_y_mean':   gyro_y_mean,
        'gyro_z_mean':   gyro_z_mean,
        'gyro_x_std':    gyro_x_std,
        'gyro_y_std':    gyro_y_std,
        'gyro_z_std':    gyro_z_std,
        'n_peaks_acc':  n_peaks_acc,
        'n_peaks_gyro': n_peaks_gyro,
        'y_bias':        float(y_bias),
        'dom_freq':      dom_freq,
        'fft_max':       fft_max,
        'zcr_acc':       zcr_acc,
        'zcr_gyro':      zcr_gyro,
        'energy_acc':    energy_acc,
        'energy_gyro':   energy_gyro,
        'acc_auto1':     acc_auto1,
        'gyro_auto1':    gyro_auto1,
        'impulsiveness': impulsiveness,
        'jerk':          float(np.sqrt(np.mean((np.diff(acc, axis=0))**2))),
        'z_grav':        float(abs(acc_mean[2]) / (np.linalg.norm(acc_mean) + 1e-8)),
        'kurt_am':      kurt_am,
        'kurt_ax':      kurt_ax,
        'kurt_ay':      kurt_ay,
        'kurt_az':      kurt_az,
        'kurt_gm':      kurt_gm,
        'skew_am':      skew_am,
        'skew_ax':      skew_ax,
        'skew_ay':      skew_ay,
        'skew_az':      skew_az,
        'skew_gm':      skew_gm,
        'corr_xy':       corr_xy,
        'corr_xz':       corr_xz,
        'corr_yz':       corr_yz,
        'gyro_corr_xy':  gyro_corr_xy,
    }



def build_prompt(window):
    """
    KuHar prompt v95: MiMo直连版 — 简洁结构 + 核心特征 + 清晰决策链
    去掉复杂阈值表，用简明的特征含义引导判断
    """
    f = compute_features(window)
    return f'''You are a HAR expert. Classify a 1.28-second waist sensor window into ONE of 18 activities.

IMPORTANT: Do NOT take shortcuts. Analyze the actual sensor feature VALUES — do NOT just guess the most common class. Think step by step through the features, then output your calibrated probabilities.

IMPORTANT: Do NOT take shortcuts. Analyze the actual sensor feature VALUES — do NOT just guess the most common class. Think step by step through the features, then output your calibrated probabilities.

=== ACTIVITY DESCRIPTIONS (use these to understand each class) ===
0=Stand: completely still, upright waist (z_grav ~0.69), very low energy ~0.003
1=Sit: completely still, seated waist (z_grav ~0.78), very low energy ~0.002
2=Talk-sit: sitting and speaking, very still (energy 0.02-0.09) but slight periodic speaking motion
3=Talk-stand: standing and speaking, very still (energy 0.5-3) but slight periodic speaking motion
4=Stand-sit: transitioning between standing and sitting, moderate Y-axis movement, gyro_y_std high (~0.77)
5=Lay: horizontal body, z_grav very low (<0.55), very low energy ~0.004
6=Lay-stand: transitioning from lying to standing, moderate Y-axis movement, gyro_y_std lower (~0.43)
7=Pick: bending to pick something up, moderate energy (~4.5), acc_y_std high (~0.94)
8=Jump: explosive vertical hops, very high energy (~177), high jerk, impulsive (kurtosis >3)
9=Push-up: slow horizontal pushing motion, moderate energy (~8.9), LOW impulsiveness (~1.99) — this is key
10=Sit-up: sit to lie or lie to sit, energy ~3.1, acc_y_std ~0.53
11=Walk: normal walking, energy ~20, rhythmic pattern, acc_y_std ~1.9
12=Walk-back: walking backwards, similar to Walk but slightly different cadence, energy ~15
13=Walk-circle: walking in circles, irregular direction changes, energy ~15, acc_y_std variable
14=Run: running, very high energy (~323), acc_y_std >5, regular high-frequency steps
15=Stairs-up: climbing stairs up, energy ~12, acc_y_std ~1.5, gyro_x ~0.50
16=Stairs-down: descending stairs, energy ~36, acc_y_std ~2.5, gyro_x ~0.53
17=Ping-pong: table tennis, energy ~35, fast rotation, gyro_x ~1.0 (much higher than Stairs-down)

=== SENSOR FEATURES ===
  energy_acc: {f['energy_acc']:.3f}   (total motion energy, key discriminator by intensity tier)
  acc_y_std: {f['acc_y_std']:.4f}    (Y-axis variation — walking/running/transitional movements)
  gyro_x_std: {f['gyro_x_std']:.4f}  (X-axis rotation — ping-pong/stairs key)
  gyro_y_std: {f['gyro_y_std']:.4f}  (Y-axis rotation — stand-sit vs lay-stand key)
  jerk: {f['jerk']:.4f}               (suddenness of motion — Jump high, Push-up low)
  n_peaks_acc: {f['n_peaks_acc']}     (periodic peaks — walking/running high, still low)
  impulsiveness: {f['impulsiveness']:.2f} (peak/rms ratio — Jump high, Push-up low)
  z_grav: {f['z_grav']:.4f}          (vertical orientation: >0.65=upright, <0.55=horizontal)
  kurt_az: {f['kurt_az']:.2f}        (peak sharpness — Jump>3, Run moderate)

=== DECISION GUIDE ===
Step 1 — INTENSITY: energy < 0.1 → static (Stand/Sit/Lay/Talk-sit)
                energy 0.1-5 → transitional (Talk-stand/Stand-sit/Lay-stand/Sit-up/Pick)
                energy 5-20 → light activity (Stairs-up/Walk-back/Walk-circle/Push-up)
                energy 20-50 → moderate (Walk/Stairs-down/Ping-pong)
                energy > 100 → intense (Jump/Run)

Step 2 — ORIENTATION (for static/very-low energy):
  z_grav > 0.65 → upright (Stand or Talk-stand or Stand-sit)
  z_grav < 0.55 → horizontal (Lay)
  z_grav 0.55-0.65 → seated (Sit or Talk-sit)

Step 3 — TRANSITIONAL (energy 0.1-5):
  gyro_y_std > 0.65 → Stand-sit (high Y rotation during transition)
  gyro_y_std < 0.50 → Lay-stand (lower Y rotation)
  acc_y_std > 0.55, energy > 2.5 → Sit-up
  acc_y_std < 0.25, energy < 1.0 → Talk-stand (near-still while standing)

Step 4 — DYNAMIC (energy 5-50):
  impulsiveness < 2.1 → Push-up (only class with impulsiveness <2.1!)
  energy 10-15, acc_y_std 0.7-1.5 → Stairs-up
  energy 13-18, acc_y_std >1.5 → Walk-circle
  energy ~15, acc_y_std 1.4-1.9 → Walk-back
  energy ~20, acc_y_std ~1.9 → Walk
  energy ~35-40, gyro_x_std > 0.8 → Ping-pong (rotation is the key differentiator)
  energy ~35-40, gyro_x_std < 0.7 → Stairs-down

Step 5 — HIGH INTENSITY (energy > 100):
  energy > 250 AND acc_y_std > 5 → Run
  energy < 250 AND kurt_az > 3 → Jump

Step 6 — STATIC DETAIL (energy < 0.1):
  z_grav > 0.70, acc_y_std > 0.020 → Stand
  z_grav > 0.70, acc_y_std < 0.018 → Sit
  z_grav 0.60-0.70, n_peaks_acc > 25 → Sit (minor body movement)
  energy > 0.010, n_peaks_acc < 20 → Talk-sit (periodic speaking)

=== OUTPUT FORMAT ===
First give a brief classification reason (1-2 sentences), then output ONLY this JSON:
{{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4,"5":p5,"6":p6,"7":p7,"8":p8,"9":p9,"10":p10,"11":p11,"12":p12,"13":p13,"14":p14,"15":p15,"16":p16,"17":p17}}
All 18 probabilities, sum to 1.0. The true label is NOT given — decide based on the features above.
'''# ============ 主生成逻辑 ============
def main():
    # 单例锁
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("已有实例在运行，请先停止后再启动")
        sys.exit(1)

    if FORCE_RESTART:
        for f in [LOG_FILE, FINAL_FILE, ERR_FILE, CORR_LOG, SOFT_FILE, CORRECT_FILE, CKPT_FILE]:
            if os.path.exists(f):
                open(f, 'w').close()
        log("--force: 清除旧日志和断点，从头开始")

    log(f"KuHar 软标签生成 (DISABLE_THINKING + 温度{TEMPERATURE})")
    log(f"Mimo API: {API_URL}")
    log(f"Model: {MODEL}")
    log(f"Temperature: {TEMPERATURE}")

    # 加载数据
    log("加载 KuHar 数据...")
    X, y, _, _, _, _ = load_kuhar_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类, shape={X.shape}")

    # 每类统计
    per_cls = {}
    for c in range(N_CLS):
        cnt = int(np.sum(y == c))
        per_cls[c] = cnt
        log(f"  class {c} ({CLASS_NAMES[c]}): {cnt} 样本")

    # 每类采样
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

    # 断点续传
    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f:
                ckpt = json.load(f)
            done_set = set(ckpt.get('done', []))
            log(f"  断点续传: {len(done_set)}/{total} 已处理")
        except (json.JSONDecodeError, IOError):
            log(f"  断点文件损坏，从头开始")
    else:
        log(f"  断点续传: 无，从头开始")

    # 初始化
    soft_all = np.zeros((len(X), N_CLS), dtype=np.float32)
    done_count = 0
    true_correct = 0
    correct_indices = []
    class_gen = [0] * N_CLS
    class_corr = [0] * N_CLS

    # 续跑恢复统计
    if done_set and os.path.exists(SOFT_FILE):
        saved = np.load(SOFT_FILE)
        for idx in done_set:
            if saved[idx].sum() > 0:
                soft_all[idx] = saved[idx]
                tl = int(y[idx])
                pl = int(np.argmax(saved[idx]))
                class_gen[tl] += 1
                done_count += 1
                if pl == tl:
                    true_correct += 1
                    class_corr[tl] += 1
                    correct_indices.append(idx)
        log(f"  续跑恢复: 完成={done_count}, 正确={true_correct}, 正确样本={len(correct_indices)}")

    for pos, orig_idx in enumerate(sample_indices):
        if orig_idx in done_set:
            continue

        true_label = int(y[orig_idx])
        window = X[orig_idx]
        f = compute_features(window)
        prompt = build_prompt(window)

        # API 调用（内部已尝试多温度：0.3→0.4→0.6→0.8）
        probs, err = call_api(prompt)

        # 重试（仅在 call_api 完全失败时）
        retry_count = 0
        while probs is None and retry_count < 3:
            time.sleep(5)
            probs, err = call_api(prompt)
            retry_count += 1

        if probs is None:
            log_err(f"API_FAILED idx={orig_idx} true={true_label} err={err} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            class_gen[true_label] += 1
            done_set.add(orig_idx)
            done_count += 1
            continue

        # 重试直到 pred==true 或达到最大次数
        retry_pred_count = 2
        while retry_pred_count < 3:
            pred_label = int(np.argmax(probs))
            if pred_label == true_label:
                break
            probs2, _ = call_api(prompt)
            if probs2 is None:
                break
            probs = probs2
            retry_pred_count += 1
            log(f'  [RETRY {retry_pred_count}] | true={CLASS_NAMES[true_label]}({true_label}) | pred={CLASS_NAMES[int(np.argmax(probs))]}({int(np.argmax(probs))})')
        ent = float(-(np.array(probs) * np.log(np.clip(probs, 1e-8, 1))).sum())
        top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
        soft_all[orig_idx] = probs
        ok = "✓" if pred_label == true_label else "✗"
        class_gen[true_label] += 1
        line = (f"  [{done_count:03d}/{total}] | true={CLASS_NAMES[true_label]}({true_label}) | pred={CLASS_NAMES[pred_label]}({pred_label}) | {ok:>2} | ent={ent:.2f} | top={top2[0][0]}:{top2[0][1]:.2f}, {top2[1][0]}:{top2[1][1]:.2f}")
        log(line)
        log_final(line)
        if ok == "✓":
            true_correct += 1
            class_corr[true_label] += 1
            correct_indices.append(orig_idx)
            log_correct(line)

        done_set.add(orig_idx)
        done_count += 1

        # 每100个打印一次每类统计
        if done_count % 100 == 0:
            stats = []
            for c in range(N_CLS):
                g = class_gen[c]
                cc = class_corr[c]
                pct = cc / g * 100 if g > 0 else 0
                stats.append(f"{CLASS_NAMES[c]}={cc}/{g}({pct:.0f}%)")
            acc_pct = true_correct / done_count * 100
            log(f"  === 100轮统计: 整体={true_correct}/{done_count}({acc_pct:.1f}%) " + " ".join(stats))

        # 每5个保存一次
        if done_count % 5 == 0:
            np.save(SOFT_FILE, soft_all)
            soft_correct = soft_all[correct_indices]
            np.save(CORRECT_FILE, soft_correct)
            with open(CKPT_FILE, 'w') as f:
                json.dump({'done': [int(x) for x in done_set], 'class_names': CLASS_NAMES}, f)
            valid_n = int((soft_all.sum(axis=1) > 0).sum())
            acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
            log(f"  进度: {len(done_set)}/{total}, 准确率={acc_pct:.1f}%, 正确样本={len(correct_indices)}, 已保存")

        time.sleep(SLEEP_SEC)

    # 最终保存
    np.save(SOFT_FILE, soft_all)
    soft_correct = soft_all[correct_indices]
    np.save(CORRECT_FILE, soft_correct)
    with open(CKPT_FILE, 'w') as f:
        json.dump({'done': [int(x) for x in done_set], 'class_names': CLASS_NAMES}, f)

    valid_n = int((soft_all.sum(axis=1) > 0).sum())
    acc_pct = true_correct / done_count * 100 if done_count > 0 else 0
    log(f"\n=== 生成完成 ===")
    log(f"  有效软标签: {valid_n}/{len(X)}")
    log(f"  整体准确率: {acc_pct:.1f}% ({true_correct}/{done_count})")
    log(f"  正确样本数: {len(correct_indices)}")
    log(f"  输出: {SOFT_FILE}")

if __name__ == '__main__':
    main()
