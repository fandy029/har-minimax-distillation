"""
生成WISDM MiniMax软标签 - 修复版
使用ARFF中的43个统计特征，每类200样本，6类共1200次API调用
"""
import os, sys, json, time, re
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
SAMPLES_PER_CLASS = 200
cn = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
n_cls = 6

def load_wisdm():
    """加载WISDM ARFF格式数据（43个统计特征）"""
    arff_path = '/home/fandy/workplace/thesis/datasets/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff'
    d, l = [], []
    # 标签映射：ARFF中的class值 -> 索引
    label_map = {
        'Walking': 0, 'Jogging': 1,
        'Upstairs': 2, 'Downstairs': 3,
        'Sitting': 4, 'Standing': 5
    }

    with open(arff_path, 'r') as f:
        content = f.read()

    in_data = False
    for line in content.split('\n'):
        line = line.strip()
        if line == '@data':
            in_data = True
            continue
        if not in_data or not line:
            continue
        parts = line.split(',')
        if len(parts) < 46:
            continue
        try:
            # features = columns 2-44 (43 features: X0-X9, Y0-Y9, Z0-Z9, XAVG, YAVG, ZAVG, XPEAK, YPEAK, ZPEAK, XABSOLDEV, YABSOLDEV, ZABSOLDEV, XSTANDDEV, YSTANDDEV, ZSTANDDEV, RESULTANT)
            features = [float(parts[i]) for i in range(2, 45)]
            label_str = parts[45].strip().strip('"')
            if label_str in label_map:
                d.append(features)
                l.append(label_map[label_str])
        except (ValueError, IndexError):
            continue

    X = np.array(d, dtype=np.float32)
    y = np.array(l, dtype=np.int64)
    print(f"  Loaded {len(X)} samples, {X.shape[1]} features, classes: {n_cls}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 80/10/10 split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te


def get_soft_label(features, true_label, n_cls, cn):
    """调用MiniMax API获取单样本软标签"""
    # 从43个特征中提取关键统计量供API判断
    x_deciles = features[0:10]
    y_deciles = features[10:20]
    z_deciles = features[20:30]
    xavg, yavg, zavg = features[30], features[31], features[32]
    xpeak, ypeak, zpeak = features[33], features[34], features[35]
    xabs, yabs, zabs = features[36], features[37], features[38]
    xstd, ystd, zstd = features[39], features[40], features[41]
    resultant = features[42]

    acc_mag = np.sqrt(xavg**2 + yavg**2 + zavg**2)

    prompt = f"""Classify physical activity from smartphone accelerometer statistics.
Classes: 0=Walking, 1=Jogging, 2=Upstairs, 3=Downstairs, 4=Sitting, 5=Standing

Key features:
- X-axis deciles: {[round(v,3) for v in x_deciles]}
- Y-axis deciles: {[round(v,3) for v in y_deciles]}
- Z-axis deciles: {[round(v,3) for v in z_deciles]}
- Averages: X={xavg:.3f}, Y={yavg:.3f}, Z={zavg:.3f}, resultant={resultant:.3f}
- Peaks: X={xpeak:.1f}, Y={ypeak:.1f}, Z={zpeak:.1f}
- StdDev: X={xstd:.3f}, Y={ystd:.3f}, Z={zstd:.3f}

Physics: Walking/Jogging show periodic motion patterns and higher stddev;
Upstairs/Downstairs show vertical bias; Sitting/Standing are static postures.
Output JSON with probabilities for all 6 classes: {{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4,"5":p5}}"""

    try:
        from openai import OpenAI
        c = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        r = c.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=150,
            extra_body={'reasoning_split': True}
        )
        msg = r.choices[0].message
        reasoning = msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content = msg.content

        # 尝试从content中提取JSON
        for m in re.findall(r'\{[^{}]*\}', content, re.DOTALL):
            try:
                d = json.loads(m)
                if all(str(k) in d for k in range(n_cls)):
                    s = np.clip(np.array([float(d[str(k)]) for k in range(n_cls)]), 0, 1)
                    if s.sum() > 0:
                        return s / s.sum()
            except:
                pass

        # 备用：从reasoning中提取数值
        nums = re.findall(r'(?:p|prob)?\s*[0-9]\s*[:＝]\s*([0-9.]+)', reasoning, re.IGNORECASE)
        if len(nums) >= n_cls:
            s = np.clip(np.array([float(n) for n in nums[:n_cls]]), 0, 1)
            if s.sum() > 0:
                return s / s.sum()
    except Exception as e:
        print(f" [ERR {e}]", end='', flush=True)

    # Fallback: one-hot
    s = np.zeros(n_cls)
    s[true_label] = 1.0
    return s


if __name__ == "__main__":
    print("=" * 60)
    print("  WISDM 软标签生成 (修复版 - MLP特征)")
    print("=" * 60)
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_wisdm()
    print(f"  Train: {len(X_tr)}, Val: {len(X_vl)}, Test: {len(X_te)}")

    out_file = "/home/fandy/workplace/thesis/results/soft_labels/wisdm_soft.npy"
    y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)

    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]
        n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        print(f"  Class {c}({cn[c]}): {n} samples", end='', flush=True)
        for i, idx in enumerate(sampled):
            y_soft[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn)
            time.sleep(0.12)
            if (i + 1) % 50 == 0:
                print(f" {i+1}", end='', flush=True)
        print()
        np.save(out_file, y_soft)

    # 填充未生成的条目
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3:
            y_soft[i, y_tr[i]] = 1.0

    np.save(out_file, y_soft)
    print(f"\n✅ WISDM软标签已保存: {out_file}, Shape: {y_soft.shape}")
    print(f"   Soft label stats: min={y_soft.min():.3f}, max={y_soft.max():.3f}, mean={y_soft.mean():.3f}")
