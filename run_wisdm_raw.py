"""WISDM pure CNN from raw data - saturated dataset"""
import os, sys, json, time
import numpy as np
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cpu")
cn = ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
WINDOW = 128
STEP = 64

def load_wisdm_raw():
    base = '/home/fandy/workplace/thesis/datasets/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    d, l = [], []
    lbl_map = {'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':3,'Sitting':4,'Standing':5}
    
    print("  Loading raw WISDM data...")
    # Group by user+activity to create windows
    curr_x, curr_y, curr_z, curr_act, curr_user = [], [], [], None, None
    
    for line in open(base):
        line = line.strip()
        if not line or line.startswith('33,Jogging,49'): continue  # skip header-like
        try:
            parts = line.replace(';', '').split(',')
            if len(parts) < 6: continue
            user = int(parts[0])
            act = parts[1].strip()
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            
            if act not in lbl_map: continue
            
            if curr_act != act or curr_user != user:
                # New segment - save windows from previous
                if len(curr_x) >= WINDOW:
                    for s in range(0, len(curr_x)-WINDOW+1, STEP):
                        w_x = np.array(curr_x[s:s+WINDOW], dtype=np.float32)
                        w_y = np.array(curr_y[s:s+WINDOW], dtype=np.float32)
                        w_z = np.array(curr_z[s:s+WINDOW], dtype=np.float32)
                        window = np.stack([w_x, w_y, w_z], axis=0)  # (3, 128)
                        if not np.any(np.isnan(window)):
                            d.append(window); l.append(lbl_map[curr_act])
                curr_x, curr_y, curr_z = [], [], []
                curr_act = act; curr_user = user
            
            curr_x.append(x); curr_y.append(y); curr_z.append(z)
        except: pass
    
    print(f"  Total windows: {len(d)}")
    X = np.array(d, dtype=np.float32); y = np.array(l, dtype=np.int64)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te

class CNN(nn.Module):
    def __init__(self, in_ch=3, n_cls=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 7, 2, 3); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 5, 2, 2); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, 2, 1); self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, 3, 1, 1); self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 128); self.fc2 = nn.Linear(128, 64); self.fc3 = nn.Linear(64, n_cls)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(F.relu(self.fc1(x))); x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

if __name__ == "__main__":
    print("\n=== WISDM Raw Pure CNN ===")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_wisdm_raw()
    print(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}")
    
    mean = X_tr.mean(axis=(0,2), keepdims=True); std = X_tr.std(axis=(0,2), keepdims=True) + 1e-8
    X_tr_n = (X_tr-mean)/std; X_vl_n = (X_vl-mean)/std; X_te_n = (X_te-mean)/std
    
    X_tr_t = torch.FloatTensor(X_tr_n); y_tr_t = torch.LongTensor(y_tr)
    X_vl_t = torch.FloatTensor(X_vl_n); y_vl_t = torch.LongTensor(y_vl)
    X_te_t = torch.FloatTensor(X_te_n); y_te_t = torch.LongTensor(y_te)
    
    model = CNN(3, 6).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = nn.CrossEntropyLoss()
    
    EPOCHS = 15
    best_state = None; best_val_acc = 0
    
    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 64):
            idx = perm[i:i+64]
            bx = X_tr_t[idx].to(DEVICE); bh = y_tr_t[idx].to(DEVICE)
            out = model(bx); loss = crit(out, bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(X_vl_t.to(DEVICE)).argmax(1).cpu().numpy() == y_vl_t.numpy()).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep}: {val_acc*100:.1f}%")
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = (model(X_te_t.to(DEVICE)).argmax(1).cpu().numpy() == y_te_t.numpy()).mean()
    print(f"  Pure CNN test: {test_acc*100:.2f}%")
    
    result = {"dataset": "WISDM", "num_classes": 6, "train": len(X_tr), "test": len(X_te),
              "pure_cnn": round(test_acc*100, 2)}
    with open("/home/fandy/workplace/thesis/results/wisdm_pure_cnn.json", "w") as f:
        json.dump(result, f, indent=2)
    print("  ✅ DONE!")
