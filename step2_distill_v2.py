"""
step2_distill_v2.py - 优化版MiniMax蒸馏 v2
==========================================
核心问题诊断：MiniMax对所有样本都高置信度(>0.85)，但很多是"自信且错误"

改进点（v2核心）:
1. CNN-Minimax一致性过滤: 只蒸馏CNN和MiniMax意见一致的样本
2. 错误软标签惩罚: CNN强预测≠MiniMax预测时，大幅降低蒸馏权重
3. 自适应温度: CNN和MiniMax一致时T=1.5强化，不一致时T=4.0保护
4. 课程学习: 先训高一致性样本，逐步加入低一致性样本
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
EPOCHS  = 80
BATCH   = 64
SAMPLES_PER_CLASS = 50
MAX_TRAIN = 20000
DS_DIR = "/home/fandy/workplace/thesis/datasets"
OUT_DIR = "/home/fandy/workplace/thesis/new_results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# 核心优化参数
T_HIGH = 4.0    # CNN和MiniMax不一致时用高温度（软化）
T_LOW = 1.5     # CNN和MiniMax一致时用低温度（强化）
T_NORMAL = 3.0  # 默认温度
ALPHA = 0.75    # 提高Focal Loss权重（更重视硬标签）
WRONG_KD_WEIGHT = 0.15  # CNN和MiniMax不一致时，蒸馏权重降到15%

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

# ============================================================
# 核心优化: CNN-Minimax一致性蒸馏损失
# ============================================================
class CNNAwareDistillLoss(nn.Module):
    """
    CNN感知蒸馏损失 - 核心改进
    
    利用Pure CNN的预测来判断MiniMax软标签是否可靠:
    - CNN预测softmax概率高 + MiniMax预测一致 → T=1.5强化（双重确认）
    - CNN预测softmax概率高 + MiniMax预测不一致 → 只用Focal，大幅降低蒸馏
    - CNN预测softmax概率低 → T=4.0软化（CNN自己也不确定）
    """
    def __init__(self, T_high=4.0, T_low=1.5, alpha=0.75, wrong_kd_weight=0.15):
        super().__init__()
        self.T_high = T_high
        self.T_low = T_low
        self.alpha = alpha
        self.wrong_kd_weight = wrong_kd_weight
        self.focal = FocalLoss()
    
    def forward(self, logits, hard_labels, soft_labels, cnn_logits):
        """
        logits: 模型输出 (B, n_cls)
        hard_labels: 硬标签 (B,)
        soft_labels: MiniMax软标签 (B, n_cls)
        cnn_logits: Pure CNN logits (B, n_cls) - 用于判断一致性
        """
        # CNN预测概率
        cnn_probs = F.softmax(cnn_logits, dim=1)
        cnn_max_prob, cnn_pred = cnn_probs.max(dim=1)
        mm_pred = soft_labels.argmax(dim=1)
        
        # 判断CNN和MiniMax是否一致
        cnn_agree = (cnn_pred == mm_pred)
        
        # 逐样本计算损失
        total_loss = 0.0
        n = logits.size(0)
        
        for i in range(n):
            h = hard_labels[i]
            s = soft_labels[i]
            cm = cnn_max_prob[i].item()
            agree = cnn_agree[i].item()
            
            fl = self.focal(logits[i:i+1], h.unsqueeze(0))
            
            if agree and cm > 0.7:
                # 情况1: CNN和MiniMax一致，且CNN高置信 → T=1.5强化
                T = self.T_low
                w_kd = (1.0 - self.alpha) * 1.3  # 略提高蒸馏权重
            elif not agree and cm > 0.7:
                # 情况2: CNN高置信但和MiniMax不一致 → 大幅降低蒸馏
                T = self.T_high
                w_kd = (1.0 - self.alpha) * self.wrong_kd_weight
            elif cm < 0.4:
                # 情况3: CNN自己不确定 → T=4.0软化
                T = self.T_high
                w_kd = (1.0 - self.alpha) * 0.3
            else:
                # 情况4: 中等置信度
                T = self.T_normal if hasattr(self, 'T_normal') else 3.0
                w_kd = (1.0 - self.alpha) * 0.5
            
            if w_kd > 0:
                log_student = F.log_softmax(logits[i] / T, dim=0)
                soft_teacher = F.softmax(s / T, dim=0)
                kl = F.kl_div(log_student, soft_teacher, reduction="batchmean") * (T ** 2)
                loss = self.alpha * fl + w_kd * kl
            else:
                loss = fl
            
            total_loss += loss
        
        return total_loss / n

# ============================================================
# 课程学习: 一致性分数调度
# ============================================================
class ConsistencyCurriculum:
    """
    按CNN-Minimax一致性分数做课程学习
    - 早期: 只用高一致性样本（agreement=True, cnn_prob>0.7）
    - 中期: 加入中等一致性
    - 后期: 全部样本
    """
    def __init__(self, y_soft, cnn_logits, y_trues, thresholds=[0.7, 0.4], milestones=[20, 50]):
        self.y_soft = y_soft
        self.n_samples = len(y_soft)
        
        # 计算一致性分数
        cnn_probs = F.softmax(cnn_logits, dim=1)
        cnn_max_prob = cnn_probs.max(dim=1)[0]
        cnn_pred = cnn_probs.argmax(dim=1)
        mm_pred = torch.from_numpy(y_soft.argmax(axis=1))
        
        # 一致性: CNN高置信且预测相同
        self.agreement = ((cnn_pred == mm_pred) & (cnn_max_prob > 0.7)).float()
        self.cnn_conf = cnn_max_prob.numpy()
        self.thresholds = thresholds
        self.milestones = milestones
    
    def get_active_mask(self, epoch):
        if epoch < self.milestones[0]:
            # 只用高一致性
            return self.agreement >= 0.5
        elif epoch < self.milestones[1]:
            # 用CNN conf > 0.4
            return torch.from_numpy(self.cnn_conf >= self.thresholds[1])
        else:
            return torch.ones(self.n_samples, dtype=torch.bool)
    
    def get_stats(self, epoch):
        mask = self.get_active_mask(epoch)
        return mask.sum().item(), mask.float().mean().item()

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

def train_distill_v2(model, X_tr, y_tr, y_soft, X_vl, y_vl, n_cls, epochs, lr, batch):
    """v2蒸馏训练: 需要先训练Pure CNN获取CNN logits用于一致性判断"""
    device = DEVICE
    Xt = torch.FloatTensor(X_tr); yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl); yv = torch.LongTensor(y_vl)
    ys = torch.FloatTensor(y_soft)
    
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = FocalLoss()
    
    # 第一阶段: 先训练Pure CNN用于后续一致性判断
    print("    [Stage 1] Training Pure CNN for consistency filtering...")
    t0 = time.time()
    best_state_cnn = None; best_acc_cnn = 0
    
    for ep in range(1, 41):  # 40 epochs足够
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i+batch]
            bx = Xt[idx].to(device); bh = yt[idx].to(device)
            bx = bx + torch.randn_like(bx) * 0.02
            out = model(bx); loss = crit(out, bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xv.to(device)).argmax(1).cpu().numpy() == yv.numpy()).mean()
            if acc > best_acc_cnn:
                best_acc_cnn = acc; best_state_cnn = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            print(f"      Pure CNN ep{ep}: {acc*100:.1f}% ({time.time()-t0:.0f}s)")
    
    # 保存Pure CNN用于一致性判断
    model.load_state_dict(best_state_cnn)
    model.eval()
    with torch.no_grad():
        cnn_logits = model(Xt.to(device)).cpu()
    
    pure_acc = best_acc_cnn
    
    # 第二阶段: CNN感知蒸馏
    print(f"\n    [Stage 2] CNN-Aware Distillation ({epochs} epochs)...")
    print(f"    Pure CNN accuracy: {pure_acc*100:.2f}%")
    
    # 初始化课程学习和损失函数
    curriculum = ConsistencyCurriculum(ys.numpy(), cnn_logits, yt.numpy())
    distill_crit = CNNAwareDistillLoss(T_HIGH, T_LOW, ALPHA, WRONG_KD_WEIGHT)
    
    # 重新初始化模型
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None: nn.init.zeros_(layer.bias)
    model.train()
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    
    best_acc, best_state = 0, None
    t0 = time.time()
    
    for ep in range(1, epochs+1):
        active_mask = curriculum.get_active_mask(ep)
        active_idx = torch.where(active_mask)[0]
        if len(active_idx) == 0:
            active_idx = torch.arange(len(Xt))
        
        n_active, pct = curriculum.get_stats(ep)
        
        perm = torch.randperm(len(active_idx))
        total_loss = 0.0; n_batches = 0
        
        for i in range(0, len(active_idx), batch):
            idx = perm[i:i+batch]
            sample_idx = active_idx[idx]
            
            bx = Xt[sample_idx].to(device)
            bh = yt[sample_idx].to(device)
            bs = ys[sample_idx].to(device)
            bc = cnn_logits[sample_idx].to(device)
            
            bx_ = bx + torch.randn_like(bx) * 0.02 if ep >= 50 else bx
            
            out = model(bx_)
            loss = distill_crit(out, bh, bs, bc)
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); n_batches += 1
        
        sch.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv.to(device)).argmax(1).cpu().numpy() == yv.numpy()).mean()
        
        if acc > best_acc:
            best_acc = acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep{ep:>3}: acc={acc*100:.1f}% best={best_acc*100:.1f}% | active={n_active}({pct*100:.0f}%) | loss={total_loss/max(1,n_batches):.4f} ({time.time()-t0:.0f}s)")
    
    model.load_state_dict(best_state)
    return model, best_acc, pure_acc

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

# ============================================================
# 主流程
# ============================================================
for ds_key in ["kuhar", "uci_har_new", "motion_sense_dm"]:
    result_file = f"{OUT_DIR}/{ds_key}_kd_v2.json"
    if os.path.exists(result_file):
        print(f"  {ds_key} already done, skipping")
        continue
    
    cfg = DATASETS[ds_key]
    print(f"\n{'='*55}\n  [{ds_key}] {cfg['name']} v2蒸馏 (CNN一致性过滤)\n{'='*55}")
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
    
    # 软标签
    v1_soft = f"/home/fandy/workplace/thesis/new_results/{ds_key}_soft.npy"
    if os.path.exists(v1_soft):
        y_soft = np.load(v1_soft)[:MAX_TRAIN] if len(np.load(v1_soft)) > MAX_TRAIN else np.load(v1_soft)
    else:
        print(f"  ERROR: No soft label file found for {ds_key}")
        continue
    
    if len(y_soft) < MAX_TRAIN:
        padding = np.zeros((MAX_TRAIN - len(y_soft), n_cls), dtype=np.float32)
        y_soft = np.vstack([y_soft, padding])
    
    # 软标签分析
    max_probs = y_soft.max(axis=1)
    mm_pred = y_soft.argmax(axis=1)
    print(f"\n  Soft label quality:")
    print(f"    Max prob > 0.85: {(max_probs>0.85).sum()} ({(max_probs>0.85).mean()*100:.1f}%)")
    print(f"    Max prob < 0.4: {(max_probs<0.4).sum()} ({(max_probs<0.4).mean()*100:.1f}%)")
    print(f"    Teacher pred == true_label: {(mm_pred == y_tr[:len(y_soft)]).sum()} ({(mm_pred == y_tr[:len(y_soft)]).mean()*100:.1f}%)")
    
    # 蒸馏
    print(f"\n  CNN-Aware Distillation v2...")
    tp = time.time()
    if cfg["model"] == "mlp":
        model = MLP(in_ch, n_cls)
    else:
        model = DeepCNN(in_ch, n_cls)
    model, best_acc, pure_acc = train_distill_v2(model, X_tr_n, y_tr, y_soft, X_vl_n, y_vl, n_cls, EPOCHS, 5e-4, BATCH)
    ak, cak = evaluate(model, X_te_n, y_te, n_cls, cn)
    print(f"\n  v2 Test: {ak*100:.2f}% | Pure CNN was: {pure_acc*100:.2f}%")
    
    # 对比v1
    v1_file = f"/home/fandy/workplace/thesis/new_results/{ds_key}_kd.json"
    v1_acc = 0; v1_pure = 0
    if os.path.exists(v1_file):
        with open(v1_file) as f: v1_data = json.load(f)
        v1_acc = v1_data.get("cnn_minimax", 0)
        v1_pure = v1_data.get("pure_cnn", 0)
    
    result_data = {
        "dataset": cfg["name"], "num_classes": n_cls, "train": len(X_tr), "test": len(X_te),
        "pure_cnn": round(pure_acc*100, 2),
        "v1_kd": v1_acc,
        "v2_kd": round(ak*100, 2),
        "v1_vs_pure": round(v1_acc - v1_pure, 2) if v1_pure else 0,
        "v2_vs_pure": round(ak*100 - pure_acc*100, 2),
        "v2_vs_v1": round(ak*100 - v1_acc, 2) if v1_acc else 0,
        "kd_class_acc": cak,
    }
    with open(result_file, "w") as f: json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ {ds_key} DONE!")
    print(f"     Pure CNN: {pure_acc*100:.2f}%")
    print(f"     v1 KD:   {v1_acc:.2f}% (vs pure: {v1_acc-v1_pure:+.2f}%)")
    print(f"     v2 KD:   {ak*100:.2f}% (vs pure: {ak*100-pure_acc*100:+.2f}%)")
    print(f"     v2 vs v1: {ak*100-v1_acc:+.2f}%")
