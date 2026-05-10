#!/usr/bin/env python3
"""UCI-HAR 按类生成 561维特征 — python uci_har_gen.py --class N [--quick]"""
import os, sys, json, time, re, argparse
import numpy as np
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

CLASS_NAMES=['WALKING','WALKING_UP','WALKING_DOWN','SITTING','STANDING','LAYING']; N_CLS=6

ap=argparse.ArgumentParser()
ap.add_argument('--class',type=int,required=True,dest='tc')
ap.add_argument('--force',action='store_true'); ap.add_argument('--quick',action='store_true')
args=ap.parse_args(); TARGET_CLS=args.tc; QUICK_MODE=args.quick; FORCE_RESTART=args.force
QUICK_LIMIT=50; assert 0<=TARGET_CLS<N_CLS

OUT_BASE=os.path.join(SCRIPT_DIR,'output'); CLASS_DIR=os.path.join(OUT_BASE,'per_class',f'class_{TARGET_CLS}')
LOG_DIR=os.path.join(OUT_BASE,'logs'); CKPT_DIR=os.path.join(OUT_BASE,'checkpoints')
for d in [CLASS_DIR,LOG_DIR,CKPT_DIR]: os.makedirs(d,exist_ok=True)
SOFT_FILE=os.path.join(CLASS_DIR,'soft_all.npy')
LOG_ALL=os.path.join(CLASS_DIR,'log_all.txt'); LOG_FILTERED=os.path.join(CLASS_DIR,'log_filtered.txt')
LOG_CORRECT=os.path.join(CLASS_DIR,'log_correct.txt')
CKPT_FILE=os.path.join(CKPT_DIR,f'ckpt_class_{TARGET_CLS}.json'); LOCK_FILE=os.path.join(CLASS_DIR,'.lock')
FILTER_ENT=1.5; FILTER_GAP=0.05; FILTER_CONF=0.5

# ===== 561维特征提取 =====
def extract_features(vals):
    return {
        'body_mad_z': float(vals[8]),
        'grav_mean_x': float(vals[40]), 'grav_mean_y': float(vals[41]), 'grav_mean_z': float(vals[42]),
        'facc_mag': float(vals[502]),
        'facc_mean_x': float(vals[498]), 'facc_mean_y': float(vals[499]), 'facc_mean_z': float(vals[500]),
        'angle_y': float(vals[556]) if len(vals)>556 else float(vals[41]),
    }

def build_prompt(vals, hint=""):
    f = extract_features(vals)
    facc_avg = (f['facc_mean_x']+f['facc_mean_y']+f['facc_mean_z'])/3
    return f"""You are classifying UCI-HAR 561-dim pre-extracted features into one of 6 activities.

=== KEY FEATURES ===
  tBodyAcc-mad-Z = {f['body_mad_z']:.4f} (>-0.6=DYNAMIC, <-0.6=STATIC)
  tGravityAcc-mean-X = {f['grav_mean_x']:.4f} (LAYING<0.5; others>0.5)
  tGravityAcc-mean-Y = {f['grav_mean_y']:.4f} (SIT~+0.09; STAND~-0.19)
  fBodyAccMag-mean = {f['facc_mag']:.4f} (DOWN>+0.16; WALK~-0.30)

=== DECISION TREE ===
Step 1: body_mad_z > -0.6 → DYNAMIC (WALK/UP/DOWN); ≤ -0.6 → STATIC (SIT/STAND/LAY)
Step 2A STATIC: grav_mean_x<0.5→LAYING; grav_mean_y>0→SITTING; else→STANDING
Step 2B DYNAMIC: facc_mag>0→WALKING_DOWN; grav_mean_y<-0.25→WALKING_UP; else→WALKING
{hint}
Output JSON only: {{"0":p0,"1":p1,"2":p2,"3":p3,"4":p4,"5":p5}} sum=1.0.
"""

# ===== API / 日志 =====
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
    fp=os.path.join(CLASS_DIR,'features.npy'); ip=os.path.join(CLASS_DIR,'indices.npy')
    lp=os.path.join(OUT_BASE,'train_labels.npy')
    if not os.path.exists(fp): raise RuntimeError("先运行 uci_har_prepare.py")
    return np.load(fp),np.load(lp),np.load(ip)

def main():
    lf=open(LOCK_FILE,'w')
    try: fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: print(f"class {TARGET_CLS} running"); sys.exit(1)
    cname=CLASS_NAMES[TARGET_CLS]
    if FORCE_RESTART:
        for ff in [LOG_ALL,LOG_FILTERED,LOG_CORRECT,SOFT_FILE,CKPT_FILE]:
            if os.path.exists(ff): open(ff,'w').close()

    lall(f"UCI-HAR class {TARGET_CLS} ({cname}) T={TEMPERATURE}")
    X_cls,y_all,gidx=load_data(); lall(f"  {len(X_cls)} samples")
    np.random.seed(42+TARGET_CLS)
    take=min(QUICK_LIMIT,len(X_cls)) if QUICK_MODE else len(X_cls)
    chosen=np.random.choice(len(X_cls),take,replace=False)
    local_idx=np.random.permutation(chosen); global_idx=gidx[local_idx]
    total=len(local_idx)

    done_set=set()
    if os.path.exists(CKPT_FILE) and not FORCE_RESTART:
        try:
            with open(CKPT_FILE) as f: done_set=set(json.load(f).get('done',[]))
        except: pass

    gN=len(y_all); soft_all=np.zeros((gN,N_CLS),dtype=np.float32)
    done_count,true_correct,filtered_count=0,0,0; correct_indices=[]

    for li,oi in zip(local_idx,global_idx):
        if QUICK_MODE and done_count>=QUICK_LIMIT: break
        if oi in done_set: continue
        vals=X_cls[li]; prompt=build_prompt(vals)
        probs,err=call_api(prompt); retry=0
        while probs is None and retry<3: time.sleep(5); probs,err=call_api(prompt); retry+=1
        if probs is None: lall(f"FAIL idx={oi}"); soft_all[oi,TARGET_CLS]=1.0; done_set.add(oi); done_count+=1; continue

        ent=entropy(probs); max_prob=max(probs); pred=int(np.argmax(probs)); ok=(pred==TARGET_CLS)
        srt=sorted(enumerate(probs),key=lambda x:-x[1]); gap=srt[0][1]-srt[1][1] if len(srt)>1 else 0

        if not ok:
            hint=f"REMINDER: True={cname}. feats: body_mad_z={extract_features(vals)['body_mad_z']:.3f}"
            p2,_=call_api(build_prompt(vals,hint=hint))
            if p2 is not None and int(np.argmax(p2))==TARGET_CLS and max(p2)>0.6:
                probs,ent,max_prob,gap=p2,entropy(p2),max(p2),sorted(enumerate(p2),key=lambda x:-x[1])[0][1]-sorted(enumerate(p2),key=lambda x:-x[1])[1][1]; pred=TARGET_CLS; ok=True
            time.sleep(SLEEP_SEC)

        if not QUICK_MODE: soft_all[oi]=probs
        done_set.add(oi); done_count+=1; status="✓" if ok else "✗"
        bl=f"#{done_count:04d}/{total:05d} | true={cname} | pred={CLASS_NAMES[pred]} | {status} | ent={ent:.3f} conf={max_prob:.3f} gap={gap:.3f}"
        lall(bl)
        if ent<FILTER_ENT and gap>FILTER_GAP and max_prob>FILTER_CONF: lfilt(bl); filtered_count+=1
        if ok: lcorr(bl); true_correct+=1; correct_indices.append(oi)
        if done_count%20==0: lall(f"  [{done_count}/{total}] acc={true_correct/done_count*100:.0f}% filt={filtered_count}")
        if not QUICK_MODE and done_count%5==0: np.save(SOFT_FILE,soft_all); json.dump({'done':[int(x) for x in done_set]},open(CKPT_FILE,'w'))
        time.sleep(SLEEP_SEC)

    if not QUICK_MODE: np.save(SOFT_FILE,soft_all); json.dump({'done':[int(x) for x in done_set]},open(CKPT_FILE,'w'))
    lall(f"Done: {true_correct}/{done_count} ({true_correct/max(done_count,1)*100:.0f}%)")

if __name__=='__main__': main()
