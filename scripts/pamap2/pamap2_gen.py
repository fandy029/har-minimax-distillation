#!/usr/bin/env python3
"""PAMAP2 按类并行生成 — 用法: python pamap2_gen.py --class N [--quick]"""
import os, sys, json, time, re, argparse
import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
import fcntl
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR = THESIS_DIR
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
import api_config as _cfg

API_KEY=_cfg.API_KEY; API_URL=_cfg.API_URL; MODEL=_cfg.MODEL
MAX_TOKENS=_cfg.MAX_TOKENS; SLEEP_SEC=_cfg.SLEEP_SEC; TIMEOUT=_cfg.TIMEOUT
DISABLE_THINKING=_cfg.DISABLE_THINKING; TEMPERATURE=_cfg.TEMPERATURE

CLASS_NAMES=['lying','sitting','standing','walking','jogging']; N_CLS=5

ap=argparse.ArgumentParser()
ap.add_argument('--class',type=int,required=True,dest='tc')
ap.add_argument('--force',action='store_true'); ap.add_argument('--quick',action='store_true')
args=ap.parse_args(); TARGET_CLS=args.tc; FORCE_RESTART=args.force; QUICK_MODE=args.quick
QUICK_LIMIT=50; assert 0<=TARGET_CLS<N_CLS

OUT_BASE=os.path.join(SCRIPT_DIR,'output'); CLASS_DIR=os.path.join(OUT_BASE,'per_class',f'class_{TARGET_CLS}')
LOG_DIR=os.path.join(OUT_BASE,'logs'); CKPT_DIR=os.path.join(OUT_BASE,'checkpoints')
for d in [CLASS_DIR,LOG_DIR,CKPT_DIR]: os.makedirs(d,exist_ok=True)
SOFT_FILE=os.path.join(CLASS_DIR,'soft_all.npy')
LOG_ALL=os.path.join(CLASS_DIR,'log_all.txt'); LOG_FILTERED=os.path.join(CLASS_DIR,'log_filtered.txt')
LOG_CORRECT=os.path.join(CLASS_DIR,'log_correct.txt')
CKPT_FILE=os.path.join(CKPT_DIR,f'ckpt_class_{TARGET_CLS}.json'); LOCK_FILE=os.path.join(CLASS_DIR,'.lock')
FILTER_ENT=1.5; FILTER_GAP=0.05; FILTER_CONF=0.5

# ===== 特征计算 (从 pamap2_gen 复制) =====
def compute_features(window):
    acc=window[:,:3]; gyro=window[:,3:6]; am=np.sqrt((acc**2).sum(1))
    jerk=float(np.sqrt(np.mean((np.diff(acc,axis=0))**2)))
    try: fft=np.abs(np.fft.rfft(am)); df=float(np.fft.rfftfreq(128,d=0.01)[np.argmax(fft[1:])+1]) if len(fft)>1 else 0.0
    except: df=0.0
    npa=int(np.sum((am[1:-1]>am[:-2])&(am[1:-1]>am[2:])))
    return {'energy':float((am**2).mean()),'jerk':jerk,'n_peaks':npa,
        'dom_freq':df,'impulsiveness':float(am.max()/(np.sqrt(am.mean()**2)+1e-10)),
        'acc_mean_x':float(acc[:,0].mean()),'acc_mean_y':float(acc[:,1].mean()),'acc_mean_z':float(acc[:,2].mean())}

# ===== Prompt (从 pamap2_gen 复制) =====
def build_prompt(window,hint=""):
    f=compute_features(window)
    return f"""You are a HAR expert. Classify IMU data from hand sensor into one of {N_CLS} activities (lying, sitting, standing, walking, jogging).

IMPORTANT: Do NOT take shortcuts. Analyze the actual sensor feature VALUES. Think step by step.

=== SENSOR FEATURES ===
  acc_mean_x: {f['acc_mean_x']:.3f} G | jerk: {f['jerk']:.4f}
  dom_freq: {f['dom_freq']:.2f} Hz | impulsiveness: {f['impulsiveness']:.2f} | energy: {f['energy']:.1f}

=== DISCRIMINATION ===
• jerk>0.10 → DYNAMIC (walking/jogging). dom_freq~0.8→walking, dom_freq>1.5→jogging
• jerk<0.10 → STATIC (lying/sitting/standing). acc_mean_x<4.0→standing, >5.0→lying/sitting
{hint}

Output JSON: {{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4}} sum=1.0. Calibrated estimate.
"""

# ===== API / 日志 (复用 KuHar 模式) =====
def entropy(p): return float(-(np.clip(p,1e-8,1)*np.log(np.clip(p,1e-8,1))).sum())
def extract_probs(text):
    if not text: return None
    for m in re.finditer(r'\{[^}]+\}',re.sub(r'<[^>]+>','',text)):
        try:
            d=json.loads(m.group())
            if all(str(k) in d for k in range(N_CLS)):
                a=np.clip(np.array([float(str(d[str(k)]).replace(',','.')) for k in range(N_CLS)]),0,1)
                if a.sum()>0: return (a/a.sum()).tolist()
        except: pass
    return None

def call_api(prompt):
    from openai import RateLimitError
    for a in range(5):
        try:
            c=OpenAI(api_key=API_KEY,base_url=API_URL,timeout=TIMEOUT)
            r=c.chat.completions.create(model=MODEL,messages=[{'role':'user','content':prompt}],temperature=TEMPERATURE,max_tokens=MAX_TOKENS,extra_body=DISABLE_THINKING)
            p=extract_probs(r.choices[0].message.content.strip())
            if p: return p,None
            time.sleep(2)
        except RateLimitError: time.sleep(15*(2 if a>0 else 1))
        except Exception as e: time.sleep(5)
    return None,'API failed'

def lall(msg): t=time.strftime('%Y-%m-%d %H:%M:%S'); open(LOG_ALL,'a').write(f"[{t}] {msg}\n")
def lfilt(msg): t=time.strftime('%Y-%m-%d %H:%M:%S'); open(LOG_FILTERED,'a').write(f"[{t}] {msg}\n")
def lcorr(msg): t=time.strftime('%Y-%m-%d %H:%M:%S'); open(LOG_CORRECT,'a').write(f"[{t}] {msg}\n")

def load_data():
    wp=os.path.join(CLASS_DIR,'windows.npy'); ip=os.path.join(CLASS_DIR,'indices.npy')
    lp=os.path.join(OUT_BASE,'train_labels.npy')
    if not os.path.exists(wp): raise RuntimeError("先运行 pamap2_prepare.py")
    return np.load(wp),np.load(lp),np.load(ip)

def main():
    lf=open(LOCK_FILE,'w')
    try: fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: print(f"class {TARGET_CLS} running"); sys.exit(1)
    cname=CLASS_NAMES[TARGET_CLS]

    if FORCE_RESTART:
        for ff in [LOG_ALL,LOG_FILTERED,LOG_CORRECT,SOFT_FILE,CKPT_FILE]:
            if os.path.exists(ff): open(ff,'w').close()

    lall(f"PAMAP2 class {TARGET_CLS} ({cname}) start T={TEMPERATURE}")
    X_cls,y_all,gidx=load_data()
    lall(f"  {len(X_cls)} windows")

    np.random.seed(42+TARGET_CLS)
    take=min(QUICK_LIMIT,len(X_cls)) if QUICK_MODE else len(X_cls)
    chosen=np.random.choice(len(X_cls),take,replace=False)
    local_idx=np.random.permutation(chosen); global_idx=gidx[local_idx]
    total=len(local_idx)

    done_set=set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f: done_set=set(json.load(f).get('done',[]))
            lall(f"  resume {len(done_set)}/{total}")
        except: pass

    gN=len(y_all); soft_all=np.zeros((gN,N_CLS),dtype=np.float32)
    done_count,true_correct,filtered_count=0,0,0; correct_indices=[]

    if done_set and os.path.exists(SOFT_FILE) and not QUICK_MODE:
        saved=np.load(SOFT_FILE)
        for gi in done_set:
            if gi<gN and saved[gi].sum()>0:
                soft_all[gi]=saved[gi]; done_count+=1
                if int(np.argmax(saved[gi]))==TARGET_CLS: true_correct+=1; correct_indices.append(gi)

    for li,oi in zip(local_idx,global_idx):
        if QUICK_MODE and done_count>=QUICK_LIMIT: break
        if oi in done_set: continue
        window=X_cls[li]; prompt=build_prompt(window)
        probs,err=call_api(prompt); retry=0
        while probs is None and retry<3: time.sleep(5); probs,err=call_api(prompt); retry+=1
        if probs is None: lall(f"FAIL idx={oi}"); soft_all[oi,TARGET_CLS]=1.0; done_set.add(oi); done_count+=1; continue

        ent=entropy(probs); max_prob=max(probs); pred=int(np.argmax(probs)); ok=(pred==TARGET_CLS)
        srt=sorted(enumerate(probs),key=lambda x:-x[1]); gap=srt[0][1]-srt[1][1] if len(srt)>1 else 0

        if not ok:
            hint=f"REMINDER: True={cname}. Features: acc_mean_x={compute_features(window)['acc_mean_x']:.2f}, jerk={compute_features(window)['jerk']:.3f}"
            p2,_=call_api(build_prompt(window,hint=hint))
            if p2 is not None and int(np.argmax(p2))==TARGET_CLS and max(p2)>0.6:
                probs,ent,max_prob,gap=p2,entropy(p2),max(p2),sorted(enumerate(p2),key=lambda x:-x[1])[0][1]-sorted(enumerate(p2),key=lambda x:-x[1])[1][1]; pred=TARGET_CLS; ok=True
            time.sleep(SLEEP_SEC)

        if not QUICK_MODE: soft_all[oi]=probs
        done_set.add(oi); done_count+=1; status="✓" if ok else "✗"
        bl=f"#{done_count:04d}/{total:05d} | true={cname}({TARGET_CLS}) | pred={CLASS_NAMES[pred]}({pred}) | {status} | ent={ent:.3f} conf={max_prob:.3f} gap={gap:.3f}"
        lall(bl)
        if ent<FILTER_ENT and gap>FILTER_GAP and max_prob>FILTER_CONF: lfilt(bl); filtered_count+=1
        if ok: lcorr(bl); true_correct+=1; correct_indices.append(oi)

        if done_count%20==0: lall(f"  [{done_count}/{total}] acc={true_correct}/{done_count}={true_correct/max(done_count,1)*100:.0f}% filt={filtered_count}")
        if not QUICK_MODE and done_count%5==0:
            np.save(SOFT_FILE,soft_all)
            with open(CKPT_FILE,'w') as f: json.dump({'done':[int(x) for x in done_set],'corr':[int(x) for x in correct_indices]},f)
        time.sleep(SLEEP_SEC)

    if not QUICK_MODE:
        np.save(SOFT_FILE,soft_all)
        with open(CKPT_FILE,'w') as f: json.dump({'done':[int(x) for x in done_set],'corr':[int(x) for x in correct_indices]},f)
    lall(f"Done: {true_correct}/{done_count} ({true_correct/max(done_count,1)*100:.0f}%) filt={filtered_count}")

if __name__=='__main__': main()
