"""
处理并训练3个新数据集: KuHar, UCI_HAR_New, MotionSense_DeviceMotion
"""
import os, json, time, re, argparse
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

API_KEY  = "sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc"
API_URL = "https://api.minimaxi.com/v1"
MODEL   = "MiniMax-M2.7-highspeed"
DEVICE  = torch.device("cpu")
EPOCHS  = 80
BATCH   = 64
SAMPLES_PER_CLASS = 50
MAX_TRAIN = 20000

DS_DIR = "/home/fandy/workplace/thesis/datasets"
OUT_DIR = "/home/fandy/workplace/thesis/new_results"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "kuhar": {
        "name": "KuHar", "path": f"{DS_DIR}/KuHar/1.Raw_time_domian_data",
        "num_classes": 18, "channels": 8, "cn": [
            "Stand","Sit","Talk-sit","Talk-stand","Stand-sit","Lay","Lay-stand",
            "Pick","Jump","Push-up","Sit-up","Walk","Walk-backwards","Walk-circle","Run","Stair-up","Stair-down","Table-tennis"
        ],
    },
    "uci_har_new": {
        "name": "UCI-HAR-New", "path": f"{DS_DIR}/UCI_HAR_New",
        "num_classes": 12, "channels": 561, "features_dim": 561, "cn": [
            "WALKING","WALKING_UP","WALKING_DOWN","SITTING","STANDING","LAYING",
            "STAND_TO_SIT","SIT_TO_STAND","SIT_TO_LIE","LIE_TO_SIT","STAND_TO_LIE","LIE_TO_STAND"
        ],
    },
    "motion_sense_dm": {
        "name": "MotionSense-DM", "path": f"{DS_DIR}/MotionSense_DeviceMotion/A_DeviceMotion_data",
        "num_classes": 6, "channels": 12, "cn": ["downstairs","jogging","sitting","standing","upstairs","walking"],
        "folders": {"dws_1":0,"dws_11":0,"dws_2":0,"jog_16":1,"jog_9":1,"sit_13":2,"sit_5":2,"std_14":3,"std_6":3,"ups_12":4,"ups_3":4,"ups_4":4,"wlk_15":5,"wlk_7":5,"wlk_8":5},
    },
}

def load_kuhar():
    base = DATASETS["kuhar"]["path"]
    d, l = [], []
    folders = sorted(glob(f"{base}/*/"))
    for folder in folders:
        label_name = os.path.basename(folder.rstrip("/"))
        label = int(label_name.split(".")[0])  # 0.Stand -> 0, 1.Sit -> 1
        for f in glob(f"{folder}/*.csv"):
            try:
                df = pd.read_csv(f, header=None)
                data = df.values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0] == 128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label)
            except: pass
    if len(d) < 100: return None
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    print(f"  KuHar: {len(X)} samples, dist={np.bincount(y, minlength=18).tolist()}")
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if len(X_tr) > MAX_TRAIN:
        idx = np.random.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_uci_har_new():
    base = DATASETS["uci_har_new"]["path"]
    X_tr = np.loadtxt(f"{base}/Train/X_train.txt").astype(np.float32)
    y_tr = np.loadtxt(f"{base}/Train/y_train.txt").astype(np.int64) - 1
    X_te = np.loadtxt(f"{base}/Test/X_test.txt").astype(np.float32)
    y_te = np.loadtxt(f"{base}/Test/y_test.txt").astype(np.int64) - 1
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    print(f"  UCI-HAR-New: Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}")
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

def load_motion_sense_dm():
    base = DATASETS["motion_sense_dm"]["path"]
    folders = DATASETS["motion_sense_dm"]["folders"]
    d, l = [], []
    for folder, label in folders.items():
        fp = f"{base}/{folder}"
        if not os.path.exists(fp): continue
        for f in glob(f"{fp}/*.csv"):
            try:
                df = pd.read_csv(f)
                cols = [c for c in df.columns if c != ""]
                data = df[cols].values[1:].astype(np.float32)  # skip header row
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0] == 128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label)
            except: pass
    if len(d) < 100: return None
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
    print(f"  MotionSense-DM: {len(X)} samples, dist={np.bincount(y, minlength=6).tolist()}")
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

LOADERS = {"kuhar": load_kuhar, "uci_har_new": load_uci_har_new, "motion_sense_dm": load_motion_sense_dm}

class DeepCNN(nn.Module):
    def __init__(self, in_ch=6, n_cls=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 7, 2, 3); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 5, 2, 2); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, 2, 1); self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, 3, 1, 1); self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 128); self.fc2 = nn.Linear(128, 64); self.fc3 = nn.Linear(64, n_cls)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none"); pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

class CombinedLoss(nn.Module):
    def __init__(self, T=3.0, alpha=0.6):
        super().__init__(); self.T = T; self.alpha = alpha; self.focal = FocalLoss()
    def forward(self, logits, hard, soft):
        hl = self.focal(logits, hard)
        sl = F.kl_div(F.log_softmax(logits/self.T, dim=1), F.softmax(soft/self.T, dim=1), reduction="batchmean") * (self.T**2)
        return self.alpha * hl + (1-self.alpha) * sl

def get_soft_label(data, true_label, n_cls, cn, sr=50):
    acc = data[:,:3]; acc_m = np.sqrt((acc**2).sum(axis=1)); y_a = acc[:,1]
    fft_v = np.abs(np.fft.fft(acc_m)[1:len(acc_m)//2])
    dom_f = np.fft.fftfreq(len(acc_m), 1/sr)[np.argmax(fft_v)+1]
    descs = [f"{i}={cn[i]}" for i in range(n_cls)]
    prompt = f"""Classify IMU window. Classes: {', '.join(descs)}
Features: acc_mag={acc_m.mean():.2f}±{acc_m.std():.2f}, y_mean={y_a.mean():.4f}, peaks={np.sum((acc_m[1:-1]>acc_m[:-2])&(acc_m[1:-1]>acc_m[2:]))}, freq={dom_f:.1f}Hz
Physics: upstairs=posY~1Hz, downstairs=negY~1Hz, walk=posY~1-2Hz, jog=high~2-4Hz, sit/stand=low~0Hz
Output JSON: {{"0":p0,"1":p1,...}}"""
    try:
        from openai import OpenAI
        c = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        r = c.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], max_tokens=120, extra_body={"reasoning_split":True})
        msg = r.choices[0].message
        reasoning = msg.reasoning_details[0]["text"] if msg.reasoning_details else ""
        content = msg.content
        for m in re.findall(r'\{[^{}]*\}', content, re.DOTALL):
            try:
                d = json.loads(m)
                if all(str(k) in d for k in range(n_cls)):
                    s = np.clip(np.array([float(d[str(k)]) for k in range(n_cls)]), 0, 1)
                    if s.sum() > 0: return s / s.sum()
            except: pass
        nums = re.findall(r'(?:p|prob)?\s*[0-9]\s*[:＝]\s*([0-9.]+)', reasoning, re.IGNORECASE)
        if len(nums) >= n_cls:
            s = np.clip(np.array([float(n) for n in nums[:n_cls]]), 0, 1)
            if s.sum() > 0: return s / s.sum()
    except: pass
    s = np.zeros(n_cls); s[true_label] = 1.0; return s

def train(model, X_tr, y_tr, y_soft, X_vl, y_vl, n_cls, epochs, lr, batch, distill):
    device = DEVICE
    Xt = torch.FloatTensor(X_tr); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl); yv = torch.LongTensor(y_vl)
    ys = torch.FloatTensor(y_soft) if y_soft is not None else None
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = CombinedLoss() if distill else FocalLoss()
    best_acc, best_state = 0, None; t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i+batch]; bx = Xt[idx].to(device); bh = yt[idx].to(device)
            if distill and ys is not None:
                bs = ys[idx].to(device); bx_ = bx + torch.randn_like(bx)*0.02 if ep >= 60 else bx
                out = model(bx_); loss = crit(out, bh, bs)
            else:
                bx = bx + torch.randn_like(bx)*0.02; out = model(bx); loss = crit(out, bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv.to(device)).argmax(1).cpu().numpy() == yv.numpy()).mean()
        if acc > best_acc:
            best_acc = acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep{ep:>3}: {acc*100:.1f}% best={best_acc*100:.1f}% ({time.time()-t0:.0f}s)")
    model.load_state_dict(best_state); return model, best_acc

def evaluate(model, X_te, y_te, n_cls, cn):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te).to(DEVICE)).argmax(1).cpu().numpy()
    acc = (preds == y_te).mean()
    ca = {}
    for c in range(n_cls):
        mask = y_te == c
        if mask.sum() > 0: ca[cn[c]] = (preds[mask] == y_te[mask]).mean()
    return float(acc), ca

def run_dataset(ds_key):
    cfg = DATASETS[ds_key]
    print(f"\n{'='*55}\n  [{ds_key}] {cfg['name']} ({cfg['channels']}ch,{cfg['num_classes']}类)\n{'='*55}")
    t0 = time.time()
    result = LOADERS[ds_key]()
    if result is None: print("  加载失败"); return None
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = result
    n_cls = cfg["num_classes"]; cn = cfg["cn"]; in_ch = cfg["channels"]
    print(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)} | {time.time()-t0:.0f}s")
    mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std

    print(f"\n  [A] Pure CNN ({EPOCHS} epochs)...")
    tp = time.time()
    mp = DeepCNN(in_ch, n_cls); mp, _ = train(mp, X_tr_n, y_tr, None, X_vl_n, y_vl, n_cls, EPOCHS, 5e-4, 64, False)
    ap, cap = evaluate(mp, X_te_n, y_te, n_cls, cn)
    print(f"  Pure CNN: {ap*100:.2f}% ({time.time()-tp:.0f}s)")

    print(f"\n  [B] CNN + MiniMax蒸馏 ({EPOCHS} epochs)...")
    y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
    for c in range(n_cls):
        cidx = np.where(y_tr == c)[0]; n = min(SAMPLES_PER_CLASS, len(cidx))
        sampled = np.random.choice(cidx, n, replace=False)
        print(f"  {c}({cn[c]}): {n}", end="", flush=True)
        for i, idx in enumerate(sampled):
            y_soft[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn, 50); time.sleep(0.12)
            if (i+1) % 25 == 0: print(f" {i+1}", end="", flush=True)
        print()
    for i in range(len(X_tr)):
        if y_soft[i].sum() < 1e-3: y_soft[i, y_tr[i]] = 1.0
    tp = time.time()
    mk = DeepCNN(in_ch, n_cls); mk, _ = train(mk, X_tr_n, y_tr, y_soft, X_vl_n, y_vl, n_cls, EPOCHS, 5e-4, 64, True)
    ak, cak = evaluate(mk, X_te_n, y_te, n_cls, cn)
    print(f"  CNN+MiniMax: {ak*100:.2f}% ({time.time()-tp:.0f}s)")

    result_data = {
        "dataset": cfg["name"], "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": round(ap*100, 2), "cnn_minimax": round(ak*100, 2),
        "improvement": round((ak-ap)*100, 2),
        "pure_class_acc": cap, "kd_class_acc": cak,
    }
    out_path = f"{OUT_DIR}/{ds_key}_result.json"
    with open(out_path, "w") as f: json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ 结果已保存: {out_path}")
    return result_data

if __name__ == "__main__":
    results = []
    for ds in ["kuhar", "uci_har_new", "motion_sense_dm"]:
        r = run_dataset(ds)
        if r: results.append(r)
    if results:
        print(f"\n\n{'='*65}\n  📊 新数据集结果汇总\n{'='*65}")
        for r in results:
            imp = r.get("improvement", 0)
            print(f"  {r['dataset']:<20} {r['num_classes']:>3} {r['train']:>6} {r['pure_cnn']:>7.1f}% {r['cnn_minimax']:>7.1f}% {'+' if imp > 0 else ''}{imp:.2f}%")
