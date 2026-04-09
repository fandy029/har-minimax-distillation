"""
第2步：MiniMax蒸馏 for all 3 new datasets
每类150个软标签
"""
import os, json, time, re
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
EPOCHS  = 60
BATCH   = 64
SAMPLES_PER_CLASS = 50
MAX_TRAIN = 20000
DS_DIR = "/home/fandy/workplace/thesis/datasets"
OUT_DIR = "/home/fandy/workplace/thesis/new_results"

DATASETS = {
    "kuhar": {
        "name": "KuHar", "path": f"{DS_DIR}/KuHar/1.Raw_time_domian_data",
        "num_classes": 18, "channels": 8, "model": "cnn",
        "cn": ["Stand","Sit","Talk-sit","Talk-stand","Stand-sit","Lay","Lay-stand","Pick","Jump","Push-up","Sit-up","Walk","Walk-backwards","Walk-circle","Run","Stair-up","Stair-down","Table-tennis"],
    },
    "uci_har_new": {
        "name": "UCI-HAR-New", "path": f"{DS_DIR}/UCI_HAR_New",
        "num_classes": 12, "channels": 561, "model": "mlp",
        "cn": ["WALKING","WALKING_UP","WALKING_DOWN","SITTING","STANDING","LAYING","STAND_TO_SIT","SIT_TO_STAND","SIT_TO_LIE","LIE_TO_SIT","STAND_TO_LIE","LIE_TO_STAND"],
    },
    "motion_sense_dm": {
        "name": "MotionSense-DM", "path": f"{DS_DIR}/MotionSense_DeviceMotion/A_DeviceMotion_data",
        "num_classes": 6, "channels": 12, "model": "cnn",
        "cn": ["downstairs","jogging","sitting","standing","upstairs","walking"],
        "folders": {"dws_1":0,"dws_11":0,"dws_2":0,"jog_16":1,"jog_9":1,"sit_13":2,"sit_5":2,"std_14":3,"std_6":3,"ups_12":4,"ups_3":4,"ups_4":4,"wlk_15":5,"wlk_7":5,"wlk_8":5},
    },
}

def load_kuhar():
    base = DATASETS["kuhar"]["path"]
    d, l = [], []
    folders = sorted(glob(f"{base}/*/"))
    for folder in folders:
        label = int(os.path.basename(folder.rstrip("/")).split(".")[0])
        for f in glob(f"{folder}/*.csv"):
            try:
                df = pd.read_csv(f, header=None)
                data = df.values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0] == 128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label)
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
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
                data = df.iloc[1:, 1:].values.astype(np.float32)
                for s in range(0, len(data)-127, 64):
                    w = data[s:s+128]
                    if w.shape[0] == 128 and not np.any(np.isnan(w)):
                        d.append(w); l.append(label)
            except: pass
    X, y = np.array(d, dtype=np.float32), np.array(l, dtype=np.int64)
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

class MLP(nn.Module):
    def __init__(self, in_dim=561, n_cls=12):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128); self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64); self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, n_cls)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

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

def train_distill(model, X_tr, y_tr, y_soft, X_vl, y_vl, n_cls, epochs, lr, batch):
    device = DEVICE
    Xt = torch.FloatTensor(X_tr); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl); yv = torch.LongTensor(y_vl)
    ys = torch.FloatTensor(y_soft)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = CombinedLoss()
    best_acc, best_state = 0, None; t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i+batch]
            bx = Xt[idx].to(device); bh = yt[idx].to(device); bs = ys[idx].to(device)
            bx_ = bx + torch.randn_like(bx)*0.02 if ep >= 40 else bx
            out = model(bx_); loss = crit(out, bh, bs)
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
        if mask.sum() > 0: ca[cn[c]] = float((preds[mask] == y_te[mask]).mean())
    return float(acc), ca

for ds_key in ["kuhar", "uci_har_new", "motion_sense_dm"]:
    result_file = f"{OUT_DIR}/{ds_key}_kd.json"
    if os.path.exists(result_file):
        print(f"  {ds_key} already done, skipping")
        continue
    cfg = DATASETS[ds_key]
    print(f"\n{'='*55}\n  [{ds_key}] {cfg['name']} ({cfg['channels']}ch,{cfg['num_classes']}类)\n{'='*55}")
    t0 = time.time()
    result = LOADERS[ds_key]()
    if result is None: continue
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = result
    n_cls = cfg["num_classes"]; cn = cfg["cn"]; in_ch = cfg["channels"]
    print(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}")
    
    if cfg["model"] == "mlp":
        mean = X_tr.mean(axis=0, keepdims=True); std = X_tr.std(axis=0, keepdims=True) + 1e-8
    else:
        mean = X_tr.mean(axis=(0,1), keepdims=True); std = X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    print(f"\n  Generating MiniMax soft labels ({SAMPLES_PER_CLASS}/class)...")
    soft_file = f"{OUT_DIR}/{ds_key}_soft.npy"
    if os.path.exists(soft_file):
        print(f"  Soft labels already exist, loading...")
        y_soft = np.load(soft_file)
    else:
        y_soft = np.zeros((len(X_tr), n_cls), dtype=np.float32)
        for c in range(n_cls):
            cidx = np.where(y_tr == c)[0]; n = min(SAMPLES_PER_CLASS, len(cidx))
            sampled = np.random.choice(cidx, n, replace=False)
            print(f"  Class {c}({cn[c]}): {n}", end="", flush=True)
            for i, idx in enumerate(sampled):
                try:
                    y_soft[idx] = get_soft_label(X_tr[idx], y_tr[idx], n_cls, cn, 50)
                except Exception as e:
                    print(f" Error: {e}")
                    y_soft[idx, y_tr[idx]] = 1.0
                time.sleep(0.12)
                if (i+1) % 50 == 0: print(f" {i+1}", end="", flush=True)
            print()
        for i in range(len(X_tr)):
            if y_soft[i].sum() < 1e-3: y_soft[i, y_tr[i]] = 1.0
        np.save(soft_file, y_soft)
        print(f"  Soft labels saved: {soft_file}")
    
    print(f"\n  CNN/MLP + MiniMax distillation ({EPOCHS} epochs)...")
    tp = time.time()
    if cfg["model"] == "mlp":
        model = MLP(in_ch, n_cls)
    else:
        model = DeepCNN(in_ch, n_cls)
    model, _ = train_distill(model, X_tr_n, y_tr, y_soft, X_vl_n, y_vl, n_cls, EPOCHS, 5e-4, BATCH)
    ak, cak = evaluate(model, X_te_n, y_te, n_cls, cn)
    print(f"  CNN+MiniMax Test: {ak*100:.2f}% ({time.time()-tp:.0f}s)")
    
    # Load pure result
    pure_file = f"{OUT_DIR}/{ds_key}_pure.json"
    pure_data = {}
    if os.path.exists(pure_file):
        with open(pure_file) as f: pure_data = json.load(f)
    ap = pure_data.get("pure_cnn", 0)
    
    result_data = {
        "dataset": cfg["name"], "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": ap, "cnn_minimax": round(ak*100, 2),
        "improvement": round(ak*100 - ap, 2) if ap else 0,
        "pure_class_acc": pure_data.get("pure_class_acc", {}),
        "kd_class_acc": cak,
    }
    with open(result_file, "w") as f: json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ {ds_key} DONE! Pure CNN: {ap:.2f}% → CNN+MiniMax: {ak*100:.2f}% ({(ak*100-ap):+.2f}%)")
