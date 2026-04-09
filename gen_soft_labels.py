"""
第2步：生成HARTH MiniMax软标签并保存
"""
import os, json, time, re
import numpy as np
import torch

API_KEY  = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
API_URL = 'https://api.minimaxi.com/v1'
MODEL   = 'MiniMax-M2.7-highspeed'
SAMPLES_PER_CLASS = 150
cn=['左立','走路','上楼','下楼','右立','站立']
n_cls=6

data=np.load('/home/fandy/workplace/thesis/checkpoints/harth_data.npz')
X_tr=data['X_tr']; y_tr=data['y_tr']
print(f'Loaded HARTH: {len(X_tr)} train samples')

def get_soft_label(data,true_label,n_cls,cn,sr=50):
    acc=data[:,:3]; acc_m=np.sqrt((acc**2).sum(axis=1)); y_a=acc[:,1]
    fft_v=np.abs(np.fft.fft(acc_m)[1:len(acc_m)//2])
    dom_f=np.fft.fftfreq(len(acc_m),1/sr)[np.argmax(fft_v)+1]
    descs=[f'{i}={cn[i]}' for i in range(n_cls)]
    prompt=f'''Classify IMU window. Classes: {', '.join(descs)}
Features: acc_mag={acc_m.mean():.2f}±{acc_m.std():.2f}, y_mean={y_a.mean():.4f}, peaks={np.sum((acc_m[1:-1]>acc_m[:-2])&(acc_m[1:-1]>acc_m[2:]))}, freq={dom_f:.1f}Hz
Physics: upstairs=posY~1Hz, downstairs=negY~1Hz, walk=posY~1-2Hz, jog=high~2-4Hz, sit/stand=low~0Hz
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
    print(f'  Class {c}({cn[c]}): {n} samples',end='',flush=True)
    for i,idx in enumerate(sampled):
        y_soft[idx]=get_soft_label(X_tr[idx],y_tr[idx],n_cls,cn,50); time.sleep(0.12)
        if (i+1)%50==0: print(f' {i+1}',end='',flush=True)
    print()
for i in range(len(X_tr)):
    if y_soft[i].sum()<1e-3: y_soft[i,y_tr[i]]=1.0

np.save('/home/fandy/workplace/thesis/checkpoints/harth_soft_labels.npy', y_soft)
print(f'✅ Soft labels saved! Shape: {y_soft.shape}')
