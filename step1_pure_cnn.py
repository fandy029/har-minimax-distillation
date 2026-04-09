"""
第1步：Pure CNN for 2 remaining new datasets
UCI-HAR-New 用MLP (561维特征)，KuHar和MotionSense-DM用CNN
"""
import os, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
EPOCHS = 60
BATCH = 64
MAX_TRAIN = 20000
DS_DIR = "/home/fandy/workplace/thesis/datasets"
OUT_DIR = "/home/fandy/workplace/thesis/new_results"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
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
                # 排除 Unnamed: 0 索引列，只保留传感器数据列
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

LOADERS = {"uci_har_new": load_uci_har_new, "motion_sense_dm": load_motion_sense_dm}

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

def train(model, X_tr, y_tr, X_vl, y_vl, n_cls, epochs, lr, batch):
    device = DEVICE
    Xt = torch.FloatTensor(X_tr); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl); yv = torch.LongTensor(y_vl)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = FocalLoss()
    best_acc, best_state = 0, None; t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i+batch]
            bx = Xt[idx].to(device) + torch.randn_like(Xt[idx]) * 0.02
            bh = yt[idx].to(device)
            out = model(bx); loss = crit(out, bh)
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
    model.load_state_dict(best_state)
    return model, best_acc

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

for ds_key in ["uci_har_new", "motion_sense_dm"]:
    result_file = f"{OUT_DIR}/{ds_key}_pure.json"
    if os.path.exists(result_file):
        print(f"  {ds_key} already done, skipping")
        continue
    cfg = DATASETS[ds_key]
    print(f"\n{'='*55}\n  [{ds_key}] {cfg['name']} ({cfg['channels']}ch,{cfg['num_classes']}类, {cfg['model']})\n{'='*55}")
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
    
    print(f"\n  Pure CNN/MLP ({EPOCHS} epochs)...")
    tp = time.time()
    if cfg["model"] == "mlp":
        model = MLP(in_ch, n_cls)
    else:
        model = DeepCNN(in_ch, n_cls)
    model, _ = train(model, X_tr_n, y_tr, X_vl_n, y_vl, n_cls, EPOCHS, 5e-4, BATCH)
    ap, cap = evaluate(model, X_te_n, y_te, n_cls, cn)
    print(f"  Test Accuracy: {ap*100:.2f}% ({time.time()-tp:.0f}s)")
    result_data = {"dataset": cfg["name"], "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
                  "pure_cnn": round(ap*100, 2), "pure_class_acc": cap}
    with open(result_file, "w") as f: json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved: {result_file}")
