"""
第3步：HARTH CNN+MiniMax蒸馏，保存结果
"""
import os, json, time
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cpu')
EPOCHS = 80
BATCH = 64
cn=['左立','走路','上楼','下楼','右立','站立']
n_cls=6; in_ch=6

data=np.load('/home/fandy/workplace/thesis/checkpoints/harth_data.npz')
X_tr=data['X_tr']; y_tr=data['y_tr']
X_te=data['X_te']; y_te=data['y_te']
X_vl=data['X_vl']; y_vl=data['y_vl']
mean=X_tr.mean(axis=(0,1),keepdims=True); std=X_tr.std(axis=(0,1),keepdims=True)+1e-8
X_tr_n=(X_tr-mean)/std; X_vl_n=(X_vl-mean)/std; X_te_n=(X_te-mean)/std
y_soft=np.load('/home/fandy/workplace/thesis/checkpoints/harth_soft_labels.npy')
print(f'HARTH loaded: Train:{len(X_tr)} Test:{len(X_te)}, soft:{y_soft.shape}')

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

class CombinedLoss(nn.Module):
    def __init__(self,T=3.0,alpha=0.6):
        super().__init__(); self.T=T; self.alpha=alpha; self.focal=FocalLoss()
    def forward(self,logits,hard,soft):
        hl=self.focal(logits,hard)
        sl=F.kl_div(F.log_softmax(logits/self.T,dim=1),F.softmax(soft/self.T,dim=1),reduction='batchmean')*(self.T**2)
        return self.alpha*hl+(1-self.alpha)*sl

Xt=torch.FloatTensor(X_tr_n); yt=torch.LongTensor(y_tr)
Xv=torch.FloatTensor(X_vl_n); yv=torch.LongTensor(y_vl)
ys=torch.FloatTensor(y_soft)

print(f'Distillation {EPOCHS} epochs...')
t0=time.time()
model=DeepCNN(in_ch,n_cls).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2)
crit=CombinedLoss()
best_acc,best_state=0,None
for ep in range(1,EPOCHS+1):
    model.train()
    perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),BATCH):
        idx=perm[i:i+BATCH]; bx=Xt[idx].to(DEVICE); bh=yt[idx].to(DEVICE); bs=ys[idx].to(DEVICE)
        bx_=bx+torch.randn_like(bx)*0.02 if ep>=60 else bx
        out=model(bx_); loss=crit(out,bh,bs)
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
ak=(preds==y_te).mean()
cak={}
for c in range(n_cls):
    mask=y_te==c
    if mask.sum()>0: cak[cn[c]]=(preds[mask]==y_te[mask]).mean()
print(f'CNN+MiniMax Test: {ak*100:.2f}%')

result={'dataset':'HARTH','num_classes':6,'train':len(X_tr),'test':len(X_te),
        'cnn_minimax':round(ak*100,2),'kd_class_acc':cak}
with open('/home/fandy/workplace/thesis/checkpoints/harth_kd_result.json','w') as f: json.dump(result,f,indent=2,ensure_ascii=False)
print(f'✅ Distillation DONE! CNN+MiniMax: {ak*100:.2f}%')
