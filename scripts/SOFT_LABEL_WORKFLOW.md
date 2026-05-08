# 软标签生成标准流程

参考实现：`scripts/kuhar/kuhar_gen.py`（最新，经 KuHar 18 类实战验证）

---

## 核心原则

1. **软标签反映真实不确定性，不追求猜对**
2. **熵是核心质量指标，不是准确率**
3. **特征重叠的类对无法通过 prompt 解决**
4. **Prompt 必须根据具体数据集单独定制，绝不复用其他数据集的 prompt** — 每个数据集的传感器类型、安装位置、采样率、用户群体不同，特征的物理含义和区分度也不同，照搬会导致严重错误
5. **温度保持低位（0.3）** — 细粒度分类需要确定性，不要为提高多样性而提高温度
6. **MAX_TOKENS 必须充足（≥ 10000）** — 强制推理 + 完整 18 类 JSON 需要大量 token

---

## 目录结构

```
scripts/
  api_config.py                           # 统一配置
  <dataset_name>/
   <dataset>_gen.py                              # 生成脚本

results/
  soft_labels/                              # OUT_DIR
<dataset>_soft.npy                  # 全量软标签 (N, N_CLS)
    <dataset>_soft_correct_only.npy     # 仅正确预测的软标签
    .gen_<dataset>.lock                  # 单例锁
  logs/                                     # LOG_DIR
    gen_<dataset>.log                   # 主日志
    gen_<dataset>_errors.log            # 错误/失败日志
    gen_<dataset>_correct.log           # 正确样本日志
    gen_<dataset>_checkpoint.json       # 断点续传
```

---

## 代码模板

### 1. 固定配置区

**创建设置脚本时，在同级目录下同时创建 README.md 模板**（见文档章节「新数据集 Checklist → 文档生成」），待代码调试完成后用实际数据填充。

```python
#!/usr/bin/env python3
"""<Dataset> 软标签生成脚本"""
import os, sys, json, time, re
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import fcntl
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR   = THESIS_DIR

sys.path.insert(0, os.path.dirname(SCRIPT_DIR))   # api_config 在 scripts/ 而非 scripts/<dataset>/
import api_config as _cfg
API_KEY          = _cfg.API_KEY
API_URL          = _cfg.API_URL
MODEL            = _cfg.MODEL
TEMPERATURE      = _cfg.TEMPERATURE   # 保持 0.3，不要为提高多样性调高
MAX_TOKENS       = _cfg.MAX_TOKENS    # ≥ 10000
SLEEP_SEC        = _cfg.SLEEP_SEC
TIMEOUT          = _cfg.TIMEOUT
DISABLE_THINKING = _cfg.DISABLE_THINKING   # 关闭 API 端思考，靠 prompt 内推理

CLASS_NAMES = ['class0', 'class1', ...]
N_CLS       = len(CLASS_NAMES)
LABEL_MAP   = {raw_label: class_id, ...}
MAX_PER_CLASS = 3000

OUT_DIR  = os.path.join(BASE_DIR, 'results', 'soft_labels')
LOG_DIR  = os.path.join(BASE_DIR, 'results', 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SOFT_FILE    = os.path.join(OUT_DIR, '<dataset>_soft.npy')
CORRECT_FILE = os.path.join(OUT_DIR, '<dataset>_soft_correct_only.npy')
LOG_FILE     = os.path.join(LOG_DIR, 'gen_<dataset>.log')
ERR_FILE     = os.path.join(LOG_DIR, 'gen_<dataset>_errors.log')
CORR_LOG     = os.path.join(LOG_DIR, 'gen_<dataset>_correct.log')
CKPT_FILE    = os.path.join(LOG_DIR, 'gen_<dataset>_checkpoint.json')
LOCK_FILE    = os.path.join(OUT_DIR, '.gen_<dataset>.lock')

FORCE_RESTART = '--force' in sys.argv
```

### 2. 工具函数

```python
def is_valid(probs):
    """概率向量合法性检查"""
    if probs is None:
        return False
    row = np.array(probs)
    if not np.isclose(row.sum(), 1.0, atol=0.01):
        return False
    if row.max() >= 0.95:        # 拒绝真正的 one-hot
        return False
    return True

def extract_probs(text):
    """
    从 API 响应中提取概率向量。
    支持：带标签的思考文本、纯推理文本、严格格式标记。
    """
    if not text:
        return None
    # 清理 API thinking 标签
    text = re.sub(r'<THOUGHT>.*?</THOUGHT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<RESULT>.*?</RESULT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()

    # 清理严格格式标记（如 ---REASONING--- / ---PROBABILITIES---）
    text = re.sub(r'---\w+---', '', text).strip()

    for m in re.finditer(r'\{[^}]+\}', text):
        try:
            d = json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                vals = [float(d[str(k)]) for k in range(N_CLS)]
                s = np.clip(np.array(vals), 0, 1)
                if s.sum() > 0:
                    return (s / s.sum()).tolist()
        except:
            pass
    return None

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    if sys.stdout.isatty():
        print(line)
    with open(LOG_FILE, 'a') as f: f.write(line + '\n')

def log_correct(msg):
    with open(CORR_LOG, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def log_err(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if sys.stdout.isatty():
        print(line)
    with open(ERR_FILE, 'a') as f: f.write(line + '\n')
```

### 3. API 调用

```python
def call_api(prompt):
    """调用 API，关闭思考过程减少 token 开销。返回原始响应文本。"""
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
                extra_body=DISABLE_THINKING,  # 关闭 API 思考
            )
            return r.choices[0].message.content.strip(), None
        except RateLimitError:
            last_err = f'429限流 (attempt {attempt+1})'
            time.sleep(15 * (2 if attempt > 0 else 1))
        except Exception as e:
            last_err = f'API错误: {e}'
            time.sleep(5)
    return None, last_err
```

### 4. Prompt 结构（经 KuHar 实战验证）

```
=== 分层 + 强制推理 + 禁止模板值 ===

You classify human activity from {传感器描述}.
IMPORTANT: {数据集特有说明（如重力已去除）}

CLASSES (grouped by energy level):
  GROUP A — STATIC (energy<{阈值1}):    class_id=class_name, ...
  GROUP B — LOW MODERATE (energy {阈值范围}): ...
  GROUP C — MODERATE (energy {阈值范围}): ...
  GROUP D — HIGH (energy>{阈值2}): ...

=== THIS SAMPLE ===
  energy={value}   jerk={value}   n_peaks={value}
  gyro_mag={value}   gyro_x={value}
  impulsiveness={value}   dom_freq={value}Hz

=== YOUR REASONING (3 STEPS) ===
Step 1 — Energy: energy={value} -> GROUP {A/B/C/D}.
Step 2 — Within group, compare jerk, n_peaks, gyro_mag. Which 2-3 classes are closest?
Step 3 — Conclude with the strongest match.

=== GROUP SIGNATURES ===
GROUP A:
  Class(N):  e={val}, jerk={val}, n_peaks={val}(EXTREME标记), gyro_mag={val}
...

=== EXAMPLE 1: Static -> {正确类} ===
Sample: energy={val}, n_peaks={val}, ...
Step 1: e={val} < {thresh} -> GROUP A
Step 2: n_peaks={val} matches {class}({ref}) vs {other}({ref}). ...
Step 3: {class}. {other} has different {feat}.
输出: {{"0":0.20,"1":0.55,"其他":0.01}}  ← 所有18个key都写全

=== EXAMPLE 2: Moderate -> {正确类} ===
...

=== FORBIDDEN TEMPLATE PATTERNS ===
NEVER use these exact probability pairs (robotic defaults):
  0.556/0.152 - FORBIDDEN   0.529/0.144 - FORBIDDEN
  0.514/0.280 - FORBIDDEN   0.833/0.056 - FORBIDDEN
Probabilities must vary based on actual feature comparison.

=== YOUR RESPONSE FORMAT ===
YOUR ENTIRE RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

---REASONING---
Step 1: [which energy group and why]
Step 2: [compare with 2-3 closest classes in that group]
Step 3: [conclusion: most likely class]

---PROBABILITIES---
{{"0":p0,"1":p1,...,"N":pN}}
```

### 5. 主循环

```python
def main():
    lock_fd = open(LOCK_FILE, 'w')
    try: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: print("已有实例在运行"); sys.exit(1)

    if FORCE_RESTART:
        for f in [LOG_FILE, ERR_FILE, CORR_LOG, CKPT_FILE]:
            if os.path.exists(f): open(f, 'w').close()

    log(f"<Dataset> 软标签生成开始")
    log(f"Mimo API: {API_URL} | Model: {MODEL} | T={TEMPERATURE}")

    X, y, _, _, _, _ = load_data()
    log(f"  训练数据: {len(X)} 样本, {N_CLS} 类")

    np.random.seed(42)
    sample_indices = []
    for c in range(N_CLS):
        cidx = np.where(y == c)[0]
        take = min(MAX_PER_CLASS, len(cidx))
        sample_indices.extend(np.random.choice(cidx, size=take, replace=False).tolist())
    sample_indices = np.random.permutation(sample_indices)
    total = len(sample_indices)

    done_set = set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        done_set = set(json.load(open(CKPT_FILE)).get('done', []))
        log(f"  断点续传: {len(done_set)}/{total} 已处理")

    soft_all = np.zeros((len(X), N_CLS), dtype=np.float32)
    done_count = true_correct = 0
    correct_indices = []
    class_gen = [0] * N_CLS
    class_corr = [0] * N_CLS

    for orig_idx in sample_indices:
        if orig_idx in done_set: continue

        true_label = int(y[orig_idx])
        raw_result, err = call_api(build_prompt(X[orig_idx]))

        retry_count = 0
        while raw_result is None and retry_count < 3:
            time.sleep(5)
            raw_result, err = call_api(build_prompt(X[orig_idx]))
            retry_count += 1

        if raw_result is None:
            log_err(f"API_FAILED idx={orig_idx} → one-hot")
            soft_all[orig_idx, true_label] = 1.0
            done_set.add(orig_idx); done_count += 1; continue

        probs = extract_probs(raw_result)
        if not is_valid(np.array(probs)):
            for _ in range(5):
                time.sleep(2)
                rr2, _ = call_api(build_prompt(X[orig_idx]))
                if rr2:
                    p2 = extract_probs(rr2)
                    if p2 and is_valid(np.array(p2)):
                        probs = p2; raw_result = rr2; break

        ent = float(-(np.array(probs) * np.log(np.clip(probs, 1e-8, 1))).sum())
        pred_label = int(np.argmax(probs))
        ok = "✓" if pred_label == true_label else "✗"

        if not is_valid(np.array(probs)):
            # 模型过于确信 (max>=0.95)：保留原始分布，不替换为 one-hot
            log_err(f"HIGH_CONFIDENCE idx={orig_idx} true={true_label} ent={ent:.3f}")
            soft_all[orig_idx] = probs      # ← 保留模型原始判断
            class_gen[true_label] += 1
            if ok == "✓":
                true_correct += 1
                class_corr[true_label] += 1
                correct_indices.append(orig_idx)
                log_correct(line)
        else:
            soft_all[orig_idx] = probs
            class_gen[true_label] += 1
            top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
            line = (f"  [{done_count+1}/{total}] idx={orig_idx} "
                    f"true={true_label}({CLASS_NAMES[true_label]}) "
                    f"pred={pred_label}({CLASS_NAMES[pred_label]})[{ok}] "
                    f"ent={ent:.3f} "
                    f"top=[{top2[0][0]}:{top2[0][1]:.3f},{top2[1][0]}:{top2[1][1]:.3f}]")
            log(line)
            if ok == "✓":
                true_correct += 1; class_corr[true_label] += 1
                correct_indices.append(orig_idx); log_correct(line)

        done_set.add(orig_idx); done_count += 1

        if done_count % 100 == 0:
            stats = [f"{CLASS_NAMES[c]}={class_corr[c]}/{class_gen[c]}({class_corr[c]/class_gen[c]*100:.0f}%)"
                     for c in range(N_CLS) if class_gen[c] > 0]
            log(f"  100轮: {true_correct}/{done_count}({true_correct/done_count*100:.1f}%) " + " ".join(stats))

        if done_count % 5 == 0:
            np.save(SOFT_FILE, soft_all)
            np.save(CORRECT_FILE, soft_all[correct_indices])
            json.dump({'done': [int(x) for x in done_set]}, open(CKPT_FILE, 'w'))
            log(f"  进度: {done_count}/{total}, 准确率={true_correct/done_count*100:.1f}%, 已保存")

        time.sleep(SLEEP_SEC)
```

---

## 已知 Bug & 避坑

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| API 响应含 `<THOUGHT>` 标签 | 模型返回思考过程 | `extra_body={"thinking":{"type":"disabled"}}` |
| 响应含 `<RESULT>` 标签 | 同上 | API 层解决 + `extract_probs` 兜底清理 |
| `is_valid(None)` 抛异常 | 未检查 None | `is_valid` 第一行必须 `if probs is None: return False` |
| one-hot 输出 | 模型过于确信 | `is_valid` 拒绝 `max >= 0.95` |
| 重试逻辑运算符优先级 | `and` > `or` | 用 `while result is None and retry < N:` |
| 日志每行重复两遍 | nohup print + 文件写入 | 用 `if sys.stdout.isatty(): print()` |
| `ModuleNotFoundError: api_config` | sys.path 指向错误目录 | 插入 `os.path.dirname(SCRIPT_DIR)` |
| extract_probs 提取到错误 JSON | 响应含标记性标签 | 必须清理 `<RESULT>` + `<...>` |
| Prompt 签名值与实际不匹配 | 估算值而非真实统计 | 用实际数据统计分析 |
| **模型输出固定模板（如0.556/0.152）** | 模型放弃推理，套用安全模板 | 强制推理 + 明确禁止模板值（见下方章节） |
| **is_valid 不通过时替换为 one-hot** | 错误地覆盖模型判断 | 保留原始分布，只修正统计计数 |

---

## LLM 分类特殊问题

### 常见固定模板输出模式

LLM 在放弃推理时会输出固定数值模式。不同类数对应不同模板值：

| 类数 | 模板模式（top2 概率） | 产生方式 |
|------|---------------------|----------|
| 6 类 | [X:0.833, Y:0.056] | ≈5/6 + 1/18 |
| 8 类 | [X:0.550, Y:0.150] | ≈联立方程 |
| 18 类 | [X:0.556, Y:0.152] | ≈10/18 + 随机分配 |
| 18 类 | [X:0.529, Y:0.144] | 变形 |
| 18 类 | [X:0.514, Y:0.280] | 变形 |

### 打破模板输出的策略（按重要性排序）

**策略1：能量分层分类（最有效）**

将类按 energy 值分组（A=静态, B=低-中, C=中-高, D=极高），让模型先判断能量组再精确分类。这大幅减少了每个层级的分类选择数（从 N 类降到 3-6 类）。

```python
# 自动判断并在 prompt 中提示
if e < 0.1:    band = 'A'
elif e < 10:   band = 'B'
elif e < 80:   band = 'C'
else:          band = 'D'
```

**策略2：强制文本推理 + 具体示例**

在 prompt 中要求模型先写推理再输出 JSON，并且提供至少 2 个不同能量层的具体推理示例：
- 1 个静态活动示例（展示 n_peaks 比较）
- 1 个中等活动示例（展示 gyro_x 等独特特征）

```
=== EXAMPLE 1: Static -> {class} ===
Sample: energy=0.002, n_peaks=36, jerk=0.016
Step 1: e=0.002 < 0.1 -> GROUP A
Step 2: n_peaks=36 matches Sit(36) vs Stand(33) vs Lay(32)
Step 3: Sit(1). Lay has lower n_peaks, Stand has middle n_peaks.
{{"0":0.20,"1":0.55,"2":0.05,...}}
```

**策略3：明确禁止模板数值（必须做）**

```python
=== FORBIDDEN TEMPLATE PATTERNS ===
NEVER use these exact probability pairs:
  0.556/0.152 - FORBIDDEN    0.529/0.144 - FORBIDDEN
  0.514/0.280 - FORBIDDEN    0.833/0.056 - FORBIDDEN
```

**策略4：严格响应格式**

强制模型使用标记结构，便于解析和防止偏离：

```
---REASONING---
Step 1: ...
Step 2: ...
Step 3: ...
---PROBABILITIES---
{"0":..., "1":..., ...}
```

### extract_probs 兼容性验证

```python
# 验证 extract_probs 能从"推理 + 严格格式"中提取
test = '''---REASONING---
Step 1: energy=15.1 -> GROUP C
Step 2: gyro_x=+0.66 is strongly positive
Step 3: Walk-circle(13)
---PROBABILITIES---
{"0":0.01,...,"13":0.60,...}'''
result = extract_probs(test)
assert result is not None
assert np.argmax(result) == 13

# 验证禁止模板值后概率是否自然变化
test2 = '{"0":0.556,"1":0.152,"2":0.056,"3":0.056,"4":0.056,"5":0.056,...}'
result2 = extract_probs(test2)
assert result2 is not None
assert max(result2) >= 0.95  # 模版值被 is_valid 拒绝
```

### LLM HAR 分类的本质局限

| 难分类对 | 原因 | 可改进空间 |
|---------|------|-----------|
| Stand vs Sit vs Lay | 静态姿势，重力去除后传感器差异极小 | 通过 n_peaks + gyro_mag + auto1 组合可部分区分 |
| Walk-backwards vs Walk | 步态模式几乎相同 | energy 差 ~1.9x，有区分力 |
| Jump vs Run (部分) | 爆发性能量范围重叠 | impulsiveness > 3.0 vs < 3.0 有区分力 |
| Transitions（Stand-sit/Sit-up/Lay-stand）| 过渡动作特征重叠 | 有限区分力 |

**18 类 HAR 上 40-55% 准确率是 LLM 真实上限**。不要期望更高。软标签的价值在于"反映不确定性"而非"猜对"。如果需要更高准确率：
1. 减少类数量（合并高度重叠的类）
2. 使用 CNN 等端到端方法
3. 软标签只用于可区分的类对

---

## 质量判断

| 指标 | 理想值 | 含义 |
|------|--------|------|
| mean_entropy | 1.0–1.5 | 概率有区分度 |
| mean_max_prob | 0.45–0.70 | 有确信度但不过度 |
| mean_entropy > 1.8 | — | prompt 太模糊 |
| mean_max_prob > 0.85 | — | 模型过度确信 |
| mean_entropy < 0.8 | — | 接近 one-hot |
| 模板值出现频率 | < 5% | 模板值（如0.556/0.152）越少越好 |

---

## 新数据集 Prompt 制作流程

### 第一阶段：数据理解（必须做）

- [ ] **读取数据集描述文档**：特别注意传感器类型、采样率、安装位置。每一类至少看20个数据从而总结规律。
- [ ] **确认重力状态**：
  ```python
  # 检查静态类的 acc_mean
  # 如果 acc_mean 在 0.001~0.05 范围 → 重力已去除
  # 如果 acc_mean 的 z 轴在 ±9.8 附近 → 重力保留
  # 重力已去除时，z_grav 特征无效！
  ```
- [ ] **检查特征方向一致性**：例如 Walk-circle 的 gyro_x 是否全部同向，如果不是，prompt 应写 `|gyro_x| > threshold` 而非 `gyro_x > +threshold`

### 第二阶段：特征分析

- [ ] 每类 50+ 文件，提取窗口（步长与训练一致），计算 `compute_features`
- [ ] 统计每类各特征的中位数、10% 分位数、90% 分位数
- [ ] 找出**类间差异最大的特征**（用于 prompt 主体）和**重叠最多的特征**（用于提示"此特征对这个类对无效"）
- [ ] 对比 prompt 签名值是否与 `compute_features` 实际输出一致

### 第三阶段：Prompt 构建

- [ ] 按能量层级分组（A/B/C/D）
- [ ] 每个分组的签名只展示 3-5 个核心特征
- [ ] 在每类签名旁标注极端值（如 `(HIGHEST)` / `(UNIQUE)` / `(LOWEST)`)
- [ ] 写 2 个不同能量层的推理示例
- [ ] 加入禁止模板值段落
- [ ] 使用严格响应格式

### 第四阶段：验证

- [ ] 先跑 20-50 个样本
- [ ] 检查 top-2 概率是否多样（而不是全变成 0.556/0.152）
- [ ] 检查 per-class 准确率：确认哪些类特别差
- [ ] 如果模板值仍出现 → 加强禁止表述 + 增加对应的示例
- [ ] 如果某些类准确率极低 → 检查特征签名值是否与实际匹配

### 第五阶段：全量运行

- [ ] `--force` 从头开始
- [ ] 定期检查日志，确保没有大规模模板输出

---

## HAR 有用的通用特征

在 `compute_features` 中实现：

| 特征 | 计算方式 | 用途 |
|------|---------|------|
| `energy` | `(acc_mag^2).mean()` | 主强度区分器，跨量级（0.002 ~ 258） |
| `jerk` | `sqrt(mean(diff(acc)^2))` | 运动平滑度，静态<0.03、walk~0.96、jump~3.0 |
| `n_peaks` | 加速度幅值峰值计数 | 静态>30, 动态<20 |
| `gyro_mag` | 陀螺仪幅值均值 | 旋转强度，Push-up 最低(0.51) |
| `gyro_x` | X 轴陀螺仪均值 | **Walk-circle 核心区分特征**（±0.6 方向偏差）|
| `impulsiveness` | `max(acc_mag)/rms(acc_mag)` | 区分 Jump(>3.0) vs Run(<3.0) |
| `dom_freq` | FFT 主频 | 步频分析：walk~3.9Hz, stairs~2.3-3.1Hz |
| `zcr_acc` | zero-crossing rate | 静态>0.22, 动态<0.17 |
| `acc_auto1` | lag-1 自相关 | 周期性，Talk-sit 最高(0.83) |
| `z_grav` | `\|acc_mean_z\| / \|acc_mean\|` | **仅在重力保留时有效！** 重力去除时无意义 |

---

## 新数据集 Checklist

### 准备阶段
- [ ] 从 `api_config.py` 导入所有配置（含 `DISABLE_THINKING`）
- [ ] 建立 `CLASS_NAMES`、`LABEL_MAP`、数据加载函数
- [ ] `call_api` 传入 `extra_body=DISABLE_THINKING`
- [ ] `extract_probs` 必须清理 `<THOUGHT>` + `<RESULT>` + `<...>` 三种标签 + `---标记---`
- [ ] `is_valid` 必须处理 None、sum≈1.0、max<0.95
- [ ] 重试循环用 `while result is None and retry < N:` 形式
- [ ] **MAX_TOKENS ≥ 10000**（推理文字 + 完整 JSON 需要足够空间）

### 数据验证
- [ ] 读取数据集描述，确认**重力是否已去除**
- [ ] 每类 50+ 文件运行 `compute_features`，统计中位数[10%,90%]
- [ ] 检查奇异值：Walk-circle gyro_x 是否全同向？特征方向是否有反例？
- [ ] 确认 prompt 签名值与 `compute_features` 实际输出一致（偏差 < 30%）

### Prompt 构建
- [ ] **按能量分层分组**（A/B/C/D），不要平铺所有类
- [ ] 每类签名标注极端值（如 `(HIGHEST)`、`(UNIQUE)`）
- [ ] 至少 2 个推理示例（不同能量层）
- [ ] 明确禁止模板数值
- [ ] 严格响应格式（`---REASONING---` / `---PROBABILITIES---`）

### 文档生成
- [ ] 在同级目录下创建 `README.md`，包含以下待填充模板：
  ```markdown
  # {数据集名} 软标签生成

  ## 数据集概览
  - 来源：{数据集名称、来源论文}
  - 传感器：{传感器类型、采样率}
  - 参与者：{人数、性别分布}
  - 重力状态：{已去除/保留/分离到独立通道}

  ## 数据统计
  - 类列表：{类ID → 类名 映射表}
  - 每类文件数：{
    类名: X 文件
  }
  - 每类窗口数：{
    类名: X 窗口
  }
  - 输入形状：{形状说明}
  - 划分方式：{训练/验证/测试比例}

  ## 软标签生成方法
  - 脚本：{gen.py 路径}
  - API 配置：{模型名、温度、max_tokens}
  - 每类采样上限：{MAX_PER_CLASS}
  - Prompt 策略：{简述 prompt 设计核心策略}
  - 使用的特征：{列出使用的特征及其作用}
  - 未使用的特征及原因：{}

  ## 软标签训练参考
  - 输出文件路径：{soft.npy, soft_correct_only.npy 等}
  - 软标签格式：{形状、范围、特殊说明}
  - 训练注意事项：{
    - 蒸馏损失推荐
    - 采样平衡建议
    - CNN 架构参考参数
  }
  ```
- [ ] 代码调试完成后，用实际数据（文件数、窗口数、准确率等）填充 README

### 运行验证
- [ ] 先跑 10-20 个样本验证解析是否正常
- [ ] 检查 top-2 概率分布是否多样（无模板模式）
- [ ] 检查 per-class 准确率，确认可接受的类对
- [ ] 全量运行时用 `--force` 从头开始
