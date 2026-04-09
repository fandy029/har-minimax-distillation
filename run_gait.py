"""
Gait Pure CNN + 软标签 + 蒸馏
"""
import os, json, time, re
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
DEVICE = torch.device('cpu')
EPOCHS = 80
BATCH = 64
SAMPLES_PER_CLASS = 50

print('[1] Loading Gait...')
base='/home/fandy/workplace/thesis/datasets/Gait_Classification'
d,l=[],[]
for sf in ['S1_Dataset','S2_Dataset']:
    sp=f'{base}/{sf}'
    if not os.path.exists(sp): continue
    for f in glob(f'{sp}/*'):
        try:
            df=pd.read_csv(f,header=None)
            acc=df.iloc[:,1:4].values.astype(np.float32)
            labels=df.iloc[:,-1].values
            for lb in np.unique(labels):
                if lb not in [1,2,3,4]: continue
                mask=labels==lb; idx=np.where(mask)[0]
                for s in range(0,len(idx)-127,64):
                    w=acc[idx[s:s+128]]
                    if w.shape[0]==128 and not np.any(np.isnan(w)): d.append(w); l.append(int(lb)-1)
        except: pass
X,y=np.array(d,dtype=np.float32),np.array(l,dtype=np.int64)
print(f'Gait: {len(X)} samples, dist={np.bincount(y,minlength=4).tolist()}')
X,X_te,y,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_tr,X_vl,y_tr,y_vl=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f'Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}')
mean=X_tr.mean(axis=(0,1),keepdims=True); std=X_tr.std(axis=(0,1),keepdims=True)+1e-8
X_tr_n=(X_tr-mean)/std; X_vl_n=(X_vl-mean)/std; X_te_n=(X_te-mean)/std
n_cls=4; cn=['慢速走','正常走','站立','活动']; in_ch=3

class DeepCNN(nn.Module):
    def __init__(self,in_ch=3,n_cls=4):
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

# Pure CNN
print(f'\n[2] Pure CNN ({EPOCHS} epochs)...')
t0=time.time()
Xt=torch.FloatTensor(X_tr_n); yt=torch.LongTensor(y_tr)
Xv=torch.FloatTensor(X_vl_n); yv=torch.LongTensor(y_vl)
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
cap={}
for c in range(n_cls):
    mask=y_te==c
    if mask.sum()>0: cap[cn[c]]=(preds[mask]==y_te[mask]).mean()
print(f'  Pure CNN Test: {ap*100:.2f}%')

# Soft labels
print(f'\n[3] MiniMax soft labels...')
def get_soft_label(data,true_label,n_cls,cn,sr=50):
    acc=data[:,:3]; acc_m=np.sqrt((acc**2).sum(axis=1)); y_a=acc[:,1]
    fft_v=np.abs(np.fft.fft(acc_m)[1:len(acc_m)//2])
    dom_f=np.fft.fftfreq(len(acc_m),1/sr)[np.argmax(fft_v)+1]
    descs=[f'{i}={cn[i]}' for i in range(n_cls)]
    prompt=f'''Classify IMU window. Classes: {', '.join(descs)}
Features: acc_mag={acc_m.mean():.2f}±{acc_m.std():.2f}, y_mean={y_a.mean():.4f}, peaks={np.sum((acc_m[1:-1]>acc_m[:-2])&(acc_m[1:-1]>acc_m[2:]))}, freq={dom_f:.1f}Hz
Physics: slow_walk=low_freq, normal_walk=medium, stand=slow, active=mixed
Output JSON: {{"0":p0,"1":p1,...}}'''
    try:
        from openai import OpenAI
        c=OpenAI(api_key=API_KEY,base_url=API_URL,timeout=60.0)
        r=c.chat.completions.create(model=MODEL,messages=[{'role':'user','content':prompt}],max_tokens=120,extra_body={'reasoning_split':True})
        msg=r.choices[0].message
        reasoning=msg.reasoning_details[0]['text'] if msg.reasoning_details else ''
        content=msg.content
        for m in re.findall(r'\{[^{}]*\}',content,re.DOTALL):
            try:
                d=json.loads(m)
                if all(str(k) in d for k in range(n_cls)):
                    s=np.clip(np.array([float(d[str(k)]) for k in range(n_cls)]),0,1)
                    if s.sum()>0: return s/s.sum()
            except: pass
        nums=re.findall(r'(?:p|prob)?\s*[0-9]\s*[:＝]\s*([0-9.]+)',reasoning,re.IGNORECASE)
        if len(nums)>=n_cls:
            s=np.clip(np.array([float(n) for n in nums[:n_cls]]),0,1)
            if s.sum()>0: return s/s.sum()
    except: pass
    s=np.zeros(n_cls); s[true_label]=1.0; return s

y_soft=np.zeros((len(X_tr),n_cls),dtype=np.float32)
for c in range(n_cls):
    cidx=np.where(y_tr==c)[0]; n=min(SAMPLES_PER_CLASS,len(cidx))
    sampled=np.random.choice(cidx,n,replace=False)
    print(f'  Class {c}({cn[c]}): {n}',end='',flush=True)
    for i,idx in enumerate(sampled):
        y_soft[idx]=get_soft_label(X_tr[idx],y_tr[idx],n_cls,cn,50); time.sleep(0.12)
        if (i+1)%25==0: print(f' {i+1}',end='',flush=True)
    print()
for i in range(len(X_tr)):
    if y_soft[i].sum()<1e-3: y_soft[i,y_tr[i]]=1.0

# Distillation
print(f'\n[4] CNN + MiniMax distillation ({EPOCHS} epochs)...')
t0=time.time()
Xt2=torch.FloatTensor(X_tr_n); yt2=torch.LongTensor(y_tr)
Xv2=torch.FloatTensor(X_vl_n); yv2=torch.LongTensor(y_vl)
ys=torch.FloatTensor(y_soft)
model2=DeepCNN(in_ch,n_cls).to(DEVICE)
opt2=torch.optim.AdamW(model2.parameters(),lr=5e-4,weight_decay=1e-4)
sch2=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2,T_0=20,T_mult=2)
crit2=CombinedLoss()
best_acc2,best_state2=0,None
for ep in range(1,EPOCHS+1):
    model2.train()
    perm=torch.randperm(len(Xt2))
    for i in range(0,len(Xt2),BATCH):
        idx=perm[i:i+BATCH]; bx=Xt2[idx].to(DEVICE); bh=yt2[idx].to(DEVICE); bs=ys[idx].to(DEVICE)
        bx_=bx+torch.randn_like(bx)*0.02 if ep>=60 else bx
        out=model2(bx_); loss=crit2(out,bh,bs)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(),5.0); opt2.step()
    sch2.step()
    model2.eval()
    with torch.no_grad(): acc2=(model2(Xv2.to(DEVICE)).argmax(1).cpu().numpy()==yv2.numpy()).mean()
    if acc2>best_acc2:
        best_acc2=acc2; best_state2={k:v.cpu().clone() for k,v in model2.state_dict().items()}
    if ep%20==0 or ep==1: print(f'  Ep{ep:>3}: {acc2*100:.1f}% best={best_acc2*100:.1f}% ({time.time()-t0:.0f}s)')
model2.load_state_dict(best_state2)
model2.eval()
with torch.no_grad(): preds2=model2(torch.FloatTensor(X_te_n).to(DEVICE)).argmax(1).cpu().numpy()
ak=(preds2==y_te).mean()
cak={}
for c in range(n_cls):
    mask=y_te==c
    if mask.sum()>0: cak[cn[c]]=(preds2[mask]==y_te[mask]).mean()
print(f'  CNN+MiniMax Test: {ak*100:.2f}%')

# Save
result={'dataset':'Gait','num_classes':4,'train':len(X_tr),'test':len(X_te),
        'pure_cnn':round(ap*100,2),'cnn_minimax':round(ak*100,2),
        'improvement':round((ak-ap)*100,2),
        'pure_class_acc':cap,'kd_class_acc':cak}
with open('/home/fandy/workplace/thesis/checkpoints/gait_result.json','w') as f: json.dump(result,f,indent=2,ensure_ascii=False)
print(f'\n✅ GAIT DONE! Pure CNN: {ap*100:.2f}%, CNN+MiniMax: {ak*100:.2f}%')
