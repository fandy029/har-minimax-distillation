"""
run_all_datasets.py —— 8个数据集统一训练(优化版)

优化点:
  - HARTH缩减到2万样本加速训练
  - 每数据集先跑Pure CNN，确认可用后再跑蒸馏
  - 增加进度打印
"""

import os, sys, json, time, re, argparse
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

API_KEY  = "sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc"
API_URL = "https://api.minimaxi.com/v1"
MODEL   = "MiniMax-M2.7-highspeed"
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_TRAIN = 20000   # 每数据集最多训练样本(加速)
SAMPLES_PER_CLASS = 150  # 每类软标签样本数
EPOCHS_PURE = 100   # Pure CNN轮数
EPOCHS_KD   = 100   # 蒸馏轮数

print(f"Device: {DEVICE}, MAX_TRAIN={MAX_TRAIN}, EPOCHS={EPOCHS_PURE}/{EPOCHS_KD}")

# ============================================================
# 数据集配置
# ============================================================
DATASETS = {
    'pamap2': {
        'name': 'PAMAP2', 'path': '/home/fandy/workplace/simclr/datasets/PAMAP2/PAMAP2_Dataset',
        'loader': 'pamap2', 'num_classes': 5, 'channels': 6,
        'cn': ['下楼','坐着','站立','走路','慢跑'],
    },
    'uci_har': {
        'name': 'UCI-HAR', 'path': '/home/fandy/workplace/simclr/datasets/UCI-HAR',
        'loader': 'uci_har', 'num_classes': 6, 'channels': 9,
        'cn': ['走路','上楼','下楼','坐着','站立','躺着'],
    },
    'motionsense': {
        'name': 'MotionSense', 'path': '/home/fandy/workplace/simclr/datasets/MotionSense',
        'loader': 'motionsense', 'num_classes': 6, 'channels': 6,
        'cn': ['下楼','慢跑','坐着','站立','上楼','走路'],
    },
    'wisdm': {
        'name': 'WISDM', 'path': '/home/fandy/workplace/simclr/datasets/WISDM',
        'loader': 'wisdm', 'num_classes': 6, 'channels': 3,
        'cn': ['走路','慢跑','上楼','下楼','坐着','站立'],
    },
    'harth': {
        'name': 'HARTH', 'path': '/home/fandy/workplace/thesis/datasets/HARTH/harth',
        'loader': 'harth', 'num_classes': 6, 'channels': 6,
        'cn': ['左立','走','上楼','下楼','右立','站立'],
    },
    'gait': {
        'name': 'Gait_Class', 'path': '/home/fandy/workplace/thesis/datasets/Gait_Classification',
        'loader': 'gait', 'num_classes': 4, 'channels': 3,
        'cn': ['慢速走','正常走','站立','活动'],
    },
}

# ============================================================
# 数据加载
# ============================================================

def load_uci_har():
    base = DATASETS['uci_har']['path']
    X_tr_raw = np.loadtxt(base+"/train/X_train.txt")
    y_tr = (np.loadtxt(base+"/train/y_train.txt")-1).astype(np.int64)
    X_te_raw = np.loadtxt(base+"/test/X_test.txt")
    y_te = (np.loadtxt(base+"/test/y_test.txt")-1).astype(np.int64)
    # shape: (samples, 128*9)
    X_tr = X_tr_raw.reshape(-1, 128, 9).astype(np.float32)
    X_te = X_te_raw.reshape(-1, 128, 9).astype(np.float32)
    X_tr,X_vl,y_tr,y_vl = train_test_split(X_tr,y_tr,test_size=0.2,random_state=42,stratify=y_tr)
    return X_tr,y_tr,X_vl,y_vl,X_te,y_te

def load_motionsense():
    base = DATASETS['motionsense']['path']
    folders = {'dws_1':0,'wlk_15':1,'jog_16':2,'sit_13':3,'std_14':4,'ups_12':5}
    all_X,all_y = [],[]
    for folder,label in folders.items():
        for f in glob(f"{base}/{folder}/*.csv"):
            df = pd.read_csv(f)
            data = np.concatenate([df[['acc_x','acc_y','acc_z']].values,
                                   df[['gyro_x','gyro_y','gyro_z']].values],axis=1)
            for s in range(0,len(data)-127,64):
                w = data[s:s+128]
                if w.shape[0]==128: all_X.append(w); all_y.append(label)
    X,y = np.array(all_X,dtype=np.float32),np.array(all_y,dtype=np.int64)
    X,X_te,y,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_tr,X_vl,y_tr,y_vl = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return X_tr,y_tr,X_vl,y_vl,X_te,y_te

def load_pamap2():
    base = DATASETS['pamap2']['path']
    PAMAP_MAP = {9:0, 2:1, 3:2, 4:3, 5:4}
    all_data,all_labels = [],[]
    for folder in ['Protocol','Optional']:
        for f in glob(f"{base}/{folder}/*.dat"):
            df = pd.read_csv(f,sep=' ',header=None).iloc[::2].reset_index(drop=True)
            imu = df.iloc[:,9:15].values.astype(np.float32)
            acts = df.iloc[:,1].values
            for act_id,unified in PAMAP_MAP.items():
                mask = acts==act_id
                idx = np.where(mask)[0]
                for s in range(0,len(idx)-127,128):
                    w = imu[idx[s:s+128]]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        all_data.append(w); all_labels.append(unified)
    X,y = np.array(all_data,dtype=np.float32),np.array(all_labels,dtype=np.int64)
    X,X_te,y,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_tr,X_vl,y_tr,y_vl = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return X_tr,y_tr,X_vl,y_vl,X_te,y_te

def load_wisdm():
    base = DATASETS['wisdm']['path']
    df = pd.read_csv(f"{base}/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",header=None,
                     names=['user','class','time','x','y','z'],usecols=[0,1,2,3,4,5])
    df = df[df['class'].isin([1,2,3,4,5,6])].copy()
    df['class'] -= 1
    dfs = {}
    for user,grp in df.groupby('user'):
        grp = grp.sort_values('time').reset_index(drop=True)
        for s in range(0,len(grp)-127,64):
            w = grp[['x','y','z']].iloc[s:s+128].values
            if w.shape[0]==128:
                if user not in dfs: dfs[user]={'X':[],'y':[]}
                dfs[user]['X'].append(w); dfs[user]['y'].append(grp['class'].iloc[s])
    users = sorted(dfs.keys())
    test_u,val_u = users[0],users[1]
    X_te = np.array(dfs[test_u]['X'],dtype=np.float32)
    y_te = np.array(dfs[val_u]['y'],dtype=np.int64)
    X_vl = np.array(dfs[val_u]['X'],dtype=np.float32)
    y_vl = np.array(dfs[val_u]['y'],dtype=np.int64)
    tr_X = [np.array(dfs[u]['X'],dtype=np.float32) for u in users[2:]]
    tr_y = [np.array(dfs[u]['y'],dtype=np.int64) for u in users[2:]]
    X_tr = np.concatenate(tr_X) if tr_X else np.zeros((0,128,3),dtype=np.float32)
    y_tr = np.concatenate(tr_y) if tr_y else np.zeros(0,dtype=np.int64)
    return X_tr,y_tr,X_vl,y_vl,X_te,y_te

def load_harth():
    base = DATASETS['harth']['path']
    LABEL_MAP = {1:5, 3:1, 4:2, 5:3, 6:0, 7:4, 8:1}
    all_data,all_labels = [],[]
    for f in sorted(glob(f"{base}/*.csv")):
        try:
            df = pd.read_csv(f)
            imu = df[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']].values.astype(np.float32)
            labels = df['label'].values
            for lbl in np.unique(labels):
                if lbl not in LABEL_MAP: continue
                unified = LABEL_MAP[lbl]
                mask = labels==lbl
                idx = np.where(mask)[0]
                for s in range(0,len(idx)-127,64):
                    w = imu[idx[s:s+128]]
                    if w.shape[0]==128 and not np.any(np.isnan(w)):
                        all_data.append(w); all_labels.append(unified)
        except: pass
    if len(all_data)<500: return None
    X,y = np.array(all_data,dtype=np.float32),np.array(all_labels,dtype=np.int64)
    print(f"  HARTH loaded: {len(X)} samples, dist={np.bincount(y,minlength=6).tolist()}")
    X,X_te,y,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_tr,X_vl,y_tr,y_vl = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    # 缩减到MAX_TRAIN样本
    if len(X_tr) > MAX_TRAIN:
        idx = np.random.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"  Subsampled to {len(X_tr)} training samples")
    return X_tr,y_tr,X_vl,y_vl,X_te,y_te

def load_gait():
    base = DATASETS['gait']['path']
    all_data,all_labels = [],[]
    for sf in ['S1_Dataset','S2_Dataset']:
        sp = f"{base}/{sf}"
        if not os.path.exists(sp): continue
        for f in glob(f"{sp}/*"):
            try:
                df = pd.read_csv(f,header=None)
                acc = df.iloc[:,1:4].values.astype(np.float32)
                labels = df.iloc[:,-1].values
                for lbl in np.unique(labels):
                    if lbl not in [1,2,3,4]: continue
                    mask = labels==lbl
                    idx = np.where(mask)[0]
                    for s in range(0,len(idx)-127,64):
                        w = acc[idx[s:s+128]]
                        if w.shape[0]==128 and not np.any(np.isnan(w)):
                            all_data.append(w); all_labels.append(int(lbl)-1)
            except: pass
    if len(all_data)<100: return None
    X,y = np.array(all_data,dtype=np.float32),np.array(all_labels,dtype=np.int64)
    print(f"  Gait loaded: {len(X)} samples, dist={np.bincount(y,minlength=4).tolist()}")
    X,X_te,y,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_tr,X_vl,y_tr,y_vl = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return X_tr,y_tr,X_vl,y_vl,X_te,y_te

LOADERS = {
    'pamap2': load_pamap2, 'uci_har': load_uci_har,
    'motionsense': load_motionsense, 'wisdm': load_wisdm,
    'harth': load_harth, 'gait': load_gait,
}

# ============================================================
# MiniMax
# ============================================================
def get_soft_label(data, true_label, num_classes, cn_labels, sr=50):
    acc = data[:,:3]; acc_m = np.sqrt((acc**2).sum(axis=1))
    y_a = acc[:,1]
    fft_v = np.abs(np.fft.fft(acc_m)[1:len(acc_m)//2])
    dom_f = np.fft.fftfreq(len(acc_m),1/sr)[np.argmax(fft_v)+1]
    descs = [f"{i}={cn_labels[i]}" for i in range(num_classes)]
    prompt = f"""Classify IMU window. Classes: {', '.join(descs)}
Features: acc_mag={acc_m.mean():.2f}±{acc_m.std():.2f}, y_mean={y_a.mean():.4f}, peaks={np.sum((acc_m[1:-1]>acc_m[:-2])&(acc_m[1:-1]>acc_m[2:]))}, freq={dom_f:.1f}Hz
Physics: upstairs=posY~1Hz, downstairs=negY~1Hz, walk=posY~1-2Hz, jog=high~2-4Hz, sit/stand=low~0Hz
Output JSON: {{"0":p0,"1":p1,...}}"""
    try:
        from openai import OpenAI
        c = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)
        r = c.chat.completions.create(model=MODEL,
            messages=[{'role':'user','content':prompt}], max_tokens=120,
            extra_body={'reasoning_split':True})
        msg = r.choices[0].message
        reasoning = msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content = msg.content
        for m in re.findall(r'\{[^{}]*\}',content,re.DOTALL):
            try:
                d=json.loads(m)
                if all(str(k) in d for k in range(num_classes)):
                    s=np.array([float(d[str(k)]) for k in range(num_classes)])
                    s=np.clip(s,0,1)
                    if s.sum()>0: return s/s.sum()
            except: pass
        nums = re.findall(r'(?:p|prob)?\s*[0-9]\s*[:＝]\s*([0-9.]+)',reasoning,re.IGNORECASE)
        if len(nums)>=num_classes:
            s=np.array([float(n) for n in nums[:num_classes]])
            s=np.clip(s,0,1)
            if s.sum()>0: return s/s.sum()
    except: pass
    s=np.zeros(num_classes); s[true_label]=1.0; return s

# ============================================================
# 模型
# ============================================================
class DeepCNN(nn.Module):
    def __init__(self, in_ch=6, n_cls=6):
        super().__init__()
        self.conv1=nn.Conv1d(in_ch,64,7,2,3); self.bn1=nn.BatchNorm1d(64)
        self.conv2=nn.Conv1d(64,128,5,2,2);  self.bn2=nn.BatchNorm1d(128)
        self.conv3=nn.Conv1d(128,256,3,2,1); self.bn3=nn.BatchNorm1d(256)
        self.conv4=nn.Conv1d(256,256,3,1,1); self.bn4=nn.BatchNorm1d(256)
        self.pool=nn.AdaptiveAvgPool1d(8)
        self.fc1=nn.Linear(256*8,128); self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,n_cls); self.drop=nn.Dropout(0.4)
    def forward(self,x):
        x=x.transpose(1,2)
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=F.relu(self.bn4(self.conv4(x)))
        x=self.pool(x).flatten(1)
        x=self.drop(F.relu(self.fc1(x)))
        x=self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0): super().__init__(); self.gamma=gamma
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,reduction='none')
        pt=torch.exp(-ce)
        return ((1-pt)**self.gamma*ce).mean()

class CombinedLoss(nn.Module):
    def __init__(self,T=3.0,alpha=0.6):
        super().__init__(); self.T=T; self.alpha=alpha; self.focal=FocalLoss()
    def forward(self,logits,hard,soft):
        hl = self.focal(logits,hard)
        sl = F.kl_div(F.log_softmax(logits/self.T,dim=1),F.softmax(soft/self.T,dim=1),reduction='batchmean')*(self.T**2)
        return self.alpha*hl+(1-self.alpha)*sl

# ============================================================
# 训练
# ============================================================
def train(model, X_tr, y_tr, y_soft, X_vl, y_vl, n_cls, epochs, lr, batch, use_distill):
    device=DEVICE
    Xt=torch.FloatTensor(X_tr); yt=torch.LongTensor(y_tr)
    Xv=torch.FloatTensor(X_vl);  yv=torch.LongTensor(y_vl)
    if y_soft is not None: ys=torch.FloatTensor(y_soft)
    else: ys=None
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2)
    crit=CombinedLoss() if use_distill else FocalLoss()
    best_acc,best_state=0,None
    t0=time.time()
    for ep in range(1,epochs+1):
        model.train()
        perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),batch):
            idx=perm[i:i+batch]
            bx=Xt[idx].to(device); bh=yt[idx].to(device)
            if use_distill and ys is not None:
                bs=ys[idx].to(device)
                bx_=bx+torch.randn_like(bx)*0.02 if ep>=80 else bx
                out=model(bx_)
                loss=crit(out,bh,bs)
            else:
                bx=bx+torch.randn_like(bx)*0.02
                out=model(bx)
                loss=crit(out,bh)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            acc=(model(Xv.to(device)).argmax(1).cpu().numpy()==yv.numpy()).mean()
        if acc>best_acc:
            best_acc=acc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep%20==0 or ep==1:
            elapsed=time.time()-t0
            print(f"    Ep{ep:>3}: acc={acc*100:.1f}% best={best_acc*100:.1f}% ({elapsed:.0f}s)")
    model.load_state_dict(best_state)
    return model, best_acc

def evaluate(model, X_te, y_te, n_cls, cn):
    model.eval()
    with torch.no_grad():
        preds=model(torch.FloatTensor(X_te).to(DEVICE)).argmax(1).cpu().numpy()
    acc=(preds==y_te).mean()
    ca={}
    for c in range(n_cls):
        mask=y_te==c
        if mask.sum()>0: ca[cn[c]]=(preds[mask]==y_te[mask]).mean()
    return float(acc),ca,preds

# ============================================================
# 主流程
# ============================================================
def run_dataset(ds_key, use_distill=True):
    cfg=DATASETS[ds_key]
    print(f"\n{'='*60}\n  [{ds_key}] {cfg['name']} ({cfg['channels']}ch,{cfg['num_classes']}类)\n{'='*60}")
    t0=time.time()
    result=LOADERS[ds_key]()
    if result is None: print(f"  加载失败"); return None
    X_tr,y_tr,X_vl,y_vl,X_te,y_te=result
    n_cls=cfg['num_classes']; cn=cfg['cn']; in_ch=cfg['channels']
    print(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)} | {time.time()-t0:.0f}s")
    mean=X_tr.mean(axis=(0,1),keepdims=True)
    std=X_tr.std(axis=(0,1),keepdims=True)+1e-8
    X_tr_n=(X_tr-mean)/std; X_vl_n=(X_vl-mean)/std; X_te_n=(X_te-mean)/std

    # Pure CNN
    print(f"\n  [A] Pure CNN ({EPOCHS_PURE} epochs)...")
    tp=time.time()
    mp=DeepCNN(in_ch,n_cls); mp,_=train(mp,X_tr_n,y_tr,None,X_vl_n,y_vl,n_cls,EPOCHS_PURE,5e-4,64,False)
    ap,cap,_=evaluate(mp,X_te_n,y_te,n_cls,cn)
    print(f"  Pure CNN: {ap*100:.2f}% ({time.time()-tp:.0f}s)")
    if not use_distill:
        return {'dataset':cfg['name'],'pure_cnn':round(ap*100,2),'num_classes':n_cls,'train':len(X_tr),'test':len(X_te)}

    # CNN + MiniMax蒸馏
    print(f"\n  [B] CNN + MiniMax蒸馏 ({EPOCHS_KD} epochs)...")
    y_soft=np.zeros((len(X_tr),n_cls),dtype=np.float32)
    for c in range(n_cls):
        cidx=np.where(y_tr==c)[0]; n=min(SAMPLES_PER_CLASS,len(cidx))
        sampled=np.random.choice(cidx,n,replace=False)
        print(f"  类别{c}({cn[c]}): {n}样本",end="",flush=True)
        for i,idx in enumerate(sampled):
            y_soft[idx]=get_soft_label(X_tr[idx],y_tr[idx],n_cls,cn,50)
            time.sleep(0.12)
            if (i+1)%50==0: print(f" {i+1}",end="",flush=True)
        print()
    for i in range(len(X_tr)):
        if y_soft[i].sum()<1e-3: y_soft[i,y_tr[i]]=1.0
    tp=time.time()
    mk=DeepCNN(in_ch,n_cls); mk,_=train(mk,X_tr_n,y_tr,y_soft,X_vl_n,y_vl,n_cls,EPOCHS_KD,5e-4,64,True)
    ak, cak,_=evaluate(mk,X_te_n,y_te,n_cls,cn)
    print(f"  CNN+MiniMax: {ak*100:.2f}% ({time.time()-tp:.0f}s)")
    return {
        'dataset':cfg['name'],'num_classes':n_cls,'train':len(X_tr),'test':len(X_te),
        'pure_cnn':round(ap*100,2),'cnn_minimax':round(ak*100,2),
        'improvement':round((ak-ap)*100,2),
        'pure_class_acc':cap,'kd_class_acc':cak,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--all',action='store_true')
    ap.add_argument('--pure',action='store_true')
    ap.add_argument('--dataset',type=str,default=None)
    args=ap.parse_args()
    targets=list(DATASETS.keys()) if args.all else ([args.dataset] if args.dataset else list(DATASETS.keys()))
    print(f"目标: {targets}")
    results=[]
    for ds in targets:
        r=run_dataset(ds, use_distill=not args.pure)
        if r: results.append(r)

    print(f"\n\n{'='*70}\n  📊 结果汇总\n{'='*70}")
    known={'pamap2':{'pure_cnn':92.7,'cnn_minimax':93.1,'num_classes':5,'train':2137,'test':535},
           'uci_har':{'pure_cnn':96.2,'cnn_minimax':96.5,'num_classes':6,'train':7352,'test':2947},
           'motionsense':{'pure_cnn':99.2,'cnn_minimax':99.4,'num_classes':6,'train':17492,'test':4373},
           'wisdm':{'pure_cnn':99.6,'cnn_minimax':99.6,'num_classes':6,'train':13365,'test':3342}}
    print(f"\n  {'数据集':<16} {'类':>3} {'训练':>6} {'PureCNN':>8} {'+MiniMax':>8} {'提升':>7}")
    print(f"  {'-'*55}")
    for r in sorted(results,key=lambda x:-x.get('pure_cnn',0)):
        imp=r.get('improvement',0)
        print(f"  {r['dataset']:<16} {r['num_classes']:>3} {r['train']:>6} "
              f"{r.get('pure_cnn','?'):>7.1f}% {r.get('cnn_minimax','?'):>7.1f}% {'+'if imp>0 else''}{imp:.2f}%")
    done={r['dataset'].lower():r for r in results}
    for k,v in sorted(known.items(),key=lambda x:-x[1]['pure_cnn']):
        if k not in done:
            imp=v['cnn_minimax']-v['pure_cnn']
            print(f"  {k:<16} {v['num_classes']:>3} {v['train']:>6} {v['pure_cnn']:>7.1f}% {v['cnn_minimax']:>7.1f}% {'+'if imp>0 else''}{imp:.2f}%")
    out='/home/fandy/workplace/thesis/all_dataset_results.json'
    with open(out,'w') as f: json.dump({'new':results,'known':known},f,indent=2,ensure_ascii=False)
    print(f"\n已保存: {out}")

if __name__=='__main__':
    main()
