"""
第1步：跑HARTH Pure CNN，保存模型
"""
import os, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cpu')
MAX_TRAIN = 20000
EPOCHS = 80
BATCH = 64

base='/home/fandy/workplace/thesis/datasets/HARTH/harth'
MAP={1:5,3:1,4:2,5:3,6:0,7:4,8:1}
d,l=[],[]
for f in sorted(glob(f'{base}/*.csv')):
    try:
        df=pd.read_csv(f)
        imu=df[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']].values.astype(np.float32)
        labels=df['label'].values
        for lb in np.unique(labels):
            if lb not in MAP: continue
            mask=labels==lb; idx=np.where(mask)[0]
            for s in range(0,len(idx)-127,64):
                w=imu[idx[s:s+128]]
                if w.shape[0]==128 and not np.any(np.isnan(w)): d.append(w); l.append(MAP[lb])
    except: pass
X,y=np.array(d,dtype=np.float32),np.array(l,dtype=np.int64)
print(f'HARTH: {len(X)} samples')
X,X_te,y,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_tr,X_vl,y_tr,y_vl=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
if len(X_tr)>MAX_TRAIN:
    idx=np.random.choice(len(X_tr),MAX_TRAIN,replace=False)
    X_tr,y_tr=X_tr[idx],y_tr[idx]
print(f'Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}')
mean=X_tr.mean(axis=(0,1),keepdims=True); std=X_tr.std(axis=(0,1),keepdims=True)+1e-8
X_tr_n=(X_tr-mean)/std; X_vl_n=(X_vl-mean)/std; X_te_n=(X_te-mean)/std
n_cls=6; in_ch=6

class DeepCNN(nn.Module):
    def __init__(self,in_ch=6,n_cls=6):
        super().__init__()
        self.conv1=nn.Conv1d(in_ch,64,7,2,3); self.bn1=nn.BatchNorm1d(64)
        self.conv2=nn.Conv1d(64,128,5,2,2); self.bn2=nn.BatchNorm1d(128)
        self.conv3=nn.Conv1d(128,256,3,2,1); self.bn3=nn.BatchNorm1d(256)
        self.conv4=nn.Conv1d(256,256,3,1,1); self.bn4=nn.BatchNorm1d(256)
        self.pool=nn.AdaptiveAvgPool1d(8)
        self.fc1=nn.Linear(256*8,128); self.fc2=nn.Linear(128,64); self.fc3=nn.Linear(64,n_cls)
        self.drop=nn.Dropout(0.4)
    def forward(self,x):
        x=x.transpose(1,2); x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x))); x=F.relu(self.bn3(self.conv3(x)))
        x=F.relu(self.bn4(self.conv4(x))); x=self.pool(x).flatten(1)
        x=self.drop(F.relu(self.fc1(x))); x=self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0): super().__init__(); self.gamma=gamma
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,reduction='none'); pt=torch.exp(-ce)
        return ((1-pt)**self.gamma*ce).mean()

Xt=torch.FloatTensor(X_tr_n); yt=torch.LongTensor(y_tr)
Xv=torch.FloatTensor(X_vl_n); yv=torch.LongTensor(y_vl)

print(f'Pure CNN {EPOCHS} epochs...')
t0=time.time()
model=DeepCNN(in_ch,n_cls).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2)
crit=FocalLoss()
best_acc,best_state=0,None
for ep in range(1,EPOCHS+1):
    model.train()
    perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),BATCH):
        idx=perm[i:i+BATCH]; bx=Xt[idx].to(DEVICE); bh=yt[idx].to(DEVICE)
        bx=bx+torch.randn_like(bx)*0.02
        out=model(bx); loss=crit(out,bh)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    sch.step()
    model.eval()
    with torch.no_grad(): acc=(model(Xv.to(DEVICE)).argmax(1).cpu().numpy()==yv.numpy()).mean()
    if acc>best_acc:
        best_acc=acc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if ep%20==0 or ep==1: print(f'  Ep{ep:>3}: {acc*100:.1f}% best={best_acc*100:.1f}% ({time.time()-t0:.0f}s)')
model.load_state_dict(best_state)
model.eval()
with torch.no_grad(): preds=model(torch.FloatTensor(X_te_n).to(DEVICE)).argmax(1).cpu().numpy()
ap=(preds==y_te).mean()
cn=['左立','走路','上楼','下楼','右立','站立']
cap={}
for c in range(n_cls):
    mask=y_te==c
    if mask.sum()>0: cap[cn[c]]=(preds[mask]==y_te[mask]).mean()
print(f'Pure CNN Test: {ap*100:.2f}%')

# 保存
os.makedirs('/home/fandy/workplace/thesis/checkpoints', exist_ok=True)
torch.save({'model':best_state,'mean':mean,'std':std,'n_cls':n_cls,'in_ch':in_ch}, '/home/fandy/workplace/thesis/checkpoints/harth_pure_cnn.pt')
result={'dataset':'HARTH','num_classes':6,'train':len(X_tr),'test':len(X_te),'pure_cnn':round(ap*100,2),'pure_class_acc':cap}
with open('/home/fandy/workplace/thesis/checkpoints/harth_pure_cnn_result.json','w') as f: json.dump(result,f,indent=2,ensure_ascii=False)
np.savez('/home/fandy/workplace/thesis/checkpoints/harth_data.npz', X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, X_vl=X_vl, y_vl=y_vl)
print(f'✅ Pure CNN DONE! Saved to checkpoints/')
print(f'Result: {ap*100:.2f}%')
