#!/usr/bin/env python
"""TTT-256 mix vs Attn mix: SR vs perturbation difficulty, 每个 suite 一个子图。

复用 plot_ttt256_long_envgen.py 的 panel-b 风格(SR vs difficulty 折线) + 研究配色
+ Times-like serif。4 子图(spatial/object/goal/long), 横轴 difficulty 1-5,
两条线: TTT(实线) vs Attn(虚线)。SR = 该难度级 5 维合并成功率。

数据: task_id -> (success, difficulty)。
  TTT: VLANeXt_ablation_wm/ttt_chunk256_libero_mixed_clean/ 各 mix5dim/tttmix 目录 mp4。
  Attn: attention_libero_mixed_clean/ 各 attnmix 目录 mp4; long 三部分(remote/tail/resume)按 task_id 去重。
episode N -> 该目录 config task_ids[N-1] -> task_id。
保存 docs/ttt_vs_attn_difficulty.png。
"""
import os, json, glob, re, yaml
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
TTT_BASE = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_mixed_clean"
ATT_BASE = f"{REPO}/VLANeXt_ablation_wm/attention_libero_mixed_clean"
CLASS = f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
DIMS5 = {'Background Textures','Robot Initial States','Camera Viewpoints',
         'Sensor Noise','Light Conditions'}

# ---- font (STIXGeneral fallback, 同 lab 风格) ----
SERIF = "DejaVu Serif"
if any("STIXGeneral" in f.name for f in fm.fontManager.ttflist):
    SERIF = "STIXGeneral"
plt.rcParams["font.family"] = SERIF

C_TTT, C_ATT = "#C82423", "#2878B5"   # TTT 红, Attn 蓝

cls = json.load(open(CLASS))
# suite -> task_id(0-based) -> difficulty (仅 5 维)
tid2diff = {}
for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
    tid2diff[suite] = {t['id']-1: t['difficulty_level']
                       for t in cls[suite]
                       if t['category'] in DIMS5 and t['difficulty_level'] is not None}

def cfg_tids(name):
    return [int(x) for x in yaml.safe_load(open(f"{REPO}/config/{name}.yaml"))['eval']['task_ids']]

def mp4_succ(d):
    out={}
    for f in glob.glob(f"{d}/episode=*.mp4"):
        m=re.search(r"episode=(\d+)--success=(True|False)", os.path.basename(f))
        if m: out[int(m.group(1))]=(m.group(2)=="True")
    return out

def collect(base, jobs):
    """jobs: list of (dir_glob_token, cfg_name). return task_id->success(dedup)."""
    seen={}
    for tok, cfgname in jobs:
        ds = glob.glob(f"{base}/*{tok}_SR*") or glob.glob(f"{base}/*{tok}")
        ds=[d for d in ds if os.path.basename(d).split("_SR")[0].endswith(tok)]
        if not ds: continue
        try: tids=cfg_tids(cfgname)
        except FileNotFoundError: continue
        for ep,s in mp4_succ(ds[0]).items():
            if ep-1 < len(tids): seen[tids[ep-1]]=s
    return seen

# ---- TTT jobs ----
def ttt_jobs(suite):
    if suite=='libero_10':
        return [(f"envgen_tttmix_5dim_shard{s}", f"libero_plus_envgen_tttmix_5dim_shard{s}") for s in range(8)]
    return [(f"envgen_mix5dim_{suite}_shard{s}", f"libero_plus_envgen_mix5dim_{suite}_shard{s}") for s in range(8)]

# ---- Attn jobs ----
def att_jobs(suite):
    if suite=='libero_10':
        j=[]
        for s in range(8): j.append((f"envgen_attnmix_5dim_libero_10_shard{s}", f"libero_plus_envgen_attnmix_5dim_libero_10_shard{s}"))
        for s in range(8): j.append((f"envgen_attnmix_5dim_libero_10_tail_shard{s}", f"libero_plus_envgen_attnmix_5dim_libero_10_tail_shard{s}"))
        for s in range(8):
            for h in (0,1): j.append((f"envgen_attnmix_5dim_libero_10_resume{s}_h{h}", f"libero_plus_envgen_attnmix_5dim_libero_10_resume{s}_h{h}"))
        return j
    return [(f"envgen_attnmix_5dim_{suite}_shard{s}", f"libero_plus_envgen_attnmix_5dim_{suite}_shard{s}") for s in range(8)]

def sr_by_diff(seen, suite):
    """difficulty -> SR%."""
    agg=defaultdict(lambda:[0,0])
    dmap=tid2diff[suite]
    for tid,s in seen.items():
        lv=dmap.get(tid)
        if lv is None: continue
        agg[lv][1]+=1
        if s: agg[lv][0]+=1
    return {lv:(100*a[0]/a[1] if a[1] else None) for lv,a in agg.items()}

SUITES=[('libero_spatial','spatial'),('libero_object','object'),
        ('libero_goal','goal'),('libero_10','long')]
fig,axes=plt.subplots(1,4,figsize=(18,4.4),sharey=True)
LV=[1,2,3,4,5]
for ax,(skey,sname) in zip(axes,SUITES):
    t_sr=sr_by_diff(collect(TTT_BASE,ttt_jobs(skey)),skey)
    a_sr=sr_by_diff(collect(ATT_BASE,att_jobs(skey)),skey)
    for sr,c,lab,ls,mk in [(t_sr,C_TTT,'TTT-256','-','o'),(a_sr,C_ATT,'Attention','--','s')]:
        xs=[l for l in LV if sr.get(l) is not None]
        ys=[sr[l] for l in xs]
        ax.plot(xs,ys,marker=mk,color=c,lw=2.2,ms=7,ls=ls,label=lab,
                markeredgecolor="#3d3a33",markeredgewidth=0.6)
    ax.set_xticks(LV); ax.set_xlabel("perturbation difficulty level",fontsize=12)
    ax.set_title(sname,fontsize=13,color="#3d3a33")
    ax.set_ylim(0,103); ax.grid(True,alpha=0.3,lw=0.6)
    for sp in ax.spines.values(): sp.set_edgecolor("#b7b1a3")
axes[0].set_ylabel("success rate (%)",fontsize=12)
axes[0].legend(loc="lower left",fontsize=11,frameon=True,framealpha=0.92)
fig.suptitle("TTT-256 mix vs. Attention mix  |  SR vs. perturbation strength, per LIBERO suite (5-dim env-gen)",
             fontsize=14,color="#2d2a23",y=1.03)
plt.tight_layout()
out=f"{REPO}/docs/ttt_vs_attn_difficulty.png"
os.makedirs(f"{REPO}/docs",exist_ok=True)
plt.savefig(out,dpi=160,bbox_inches="tight")
print("saved",out,"| font:",SERIF)
