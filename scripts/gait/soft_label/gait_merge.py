#!/usr/bin/env python3
"""合并 Gait 4类软标签 → A/B/C三版"""
import os, sys, json, re, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ['sit_on_bed','sit_on_chair','lying','ambulating']; N_CLS=4

ap=argparse.ArgumentParser(); ap.add_argument('--quick',action='store_true')
args=ap.parse_args(); QUICK_MODE=args.quick

GAIT_DIR=os.path.dirname(SCRIPT_DIR)
OUT_BASE=os.path.join(GAIT_DIR,'output')
PER_CLASS=os.path.join(OUT_BASE,'per_class')
SOFT_DIR=os.path.join(OUT_BASE,'soft_labels'); LOG_DIR=os.path.join(OUT_BASE,'logs')
os.makedirs(SOFT_DIR,exist_ok=True); os.makedirs(LOG_DIR,exist_ok=True)

FILTER_ENT=1.5; FILTER_GAP=0.05; FILTER_CONF=0.5

print("合并 Gait 软标签")
labels=np.load(os.path.join(OUT_BASE,'train_labels.npy'))
gN=len(labels)

soft_all=np.zeros((gN,N_CLS),dtype=np.float32); merged=0
for c in range(N_CLS):
    sf=os.path.join(PER_CLASS,f'class_{c}','soft_all.npy')
    if not os.path.exists(sf): print(f"  class {c} no file"); continue
    data=np.load(sf); cidx=np.where(labels==c)[0]
    for i in cidx:
        if data[i].sum()>0: soft_all[i]=data[i]; merged+=1
    print(f"  class {c} {CLASS_NAMES[c]:<15s}: ok")

np.save(os.path.join(SOFT_DIR,'gait_soft_all.npy'),soft_all)
print(f"A: gait_soft_all.npy ({merged}/{gN})")

soft_filt=soft_all.copy(); fc=0
for idx in range(gN):
    if soft_filt[idx].sum()==0: continue
    p=soft_filt[idx]; ent=-np.sum(np.clip(p,1e-8,1)*np.log(np.clip(p,1e-8,1)))
    srt=sorted(enumerate(p),key=lambda x:-x[1]); gap=srt[0][1]-srt[1][1] if len(srt)>1 else 0
    if not (ent<FILTER_ENT and gap>FILTER_GAP and p.max()>FILTER_CONF):
        soft_filt[idx]=0; soft_filt[idx,labels[idx]]=1.0; fc+=1
np.save(os.path.join(SOFT_DIR,'gait_soft_filtered.npy'),soft_filt)
print(f"B: gait_soft_filtered.npy (keep {gN-fc}/{gN})")

soft_corr=soft_all.copy(); wc=0
for idx in range(gN):
    if soft_corr[idx].sum()==0: continue
    if int(np.argmax(soft_corr[idx]))!=labels[idx]:
        soft_corr[idx]=0; soft_corr[idx,labels[idx]]=1.0; wc+=1
np.save(os.path.join(SOFT_DIR,'gait_soft_correct_only.npy'),soft_corr)
print(f"C: gait_soft_correct_only.npy (keep {gN-wc}/{gN})")

for lt,sp,dst in [('all','log_all.txt','all.log'),('filtered','log_filtered.txt','filtered.log'),('correct','log_correct.txt','correct.log')]:
    with open(os.path.join(LOG_DIR,dst),'w') as out:
        out.write(f"=== Gait {lt} ===\n\n")
        for c in range(N_CLS):
            src=os.path.join(PER_CLASS,f'class_{c}',sp)
            if os.path.exists(src):
                out.write(f"\n--- Class {c} ---\n")
                with open(src) as fi: out.write(fi.read())
    print(f"  {dst} merged")
print("Done!")
