#!/usr/bin/env python3
"""聚合 attn-mix 四套件 env-gen 结果，与 TTT-256 mix 对照。

三套件(spatial/object/goal): 各 8 shard 不重叠，直接累加 category_stats。
long(libero_10): 三部分拼 + 去重:
  - 远端前半: ..._libero_10_shard{s}      (跑了 full[:跑到哪], 可能>128)
  - 本机 tail: ..._libero_10_tail_shard{s} (full[128:])
  - 本机 resume: ..._libero_10_resume{s}_h{0,1} (tail 剩余的再切)
  去重: 每个目录按 episode 顺序 -> task_ids[episode-1] -> task_id 唯一。
  同一 task_id 在多个目录出现时只取一次(优先级: resume > tail > remote，
  因为 resume/tail 是本机后补、远端是早先跑的；但只要 success 一致无所谓，
  取任一即可)。task_id -> category 由 task_classification.json 决定。
按 task_id 全局去重后，按 category 统计 success/total。
"""
import json, glob, re, os
from collections import defaultdict

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
BASE = f"{REPO}/VLANeXt_ablation_wm/attention_libero_mixed_clean"
CLASS = f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
DIMS = ['Camera', 'Robot', 'Light', 'Background', 'Noise']
CATMAP = {"Background Textures":"Background","Robot Initial States":"Robot",
          "Camera Viewpoints":"Camera","Language Instructions":"Language",
          "Sensor Noise":"Noise","Objects Layout":"Layout","Light Conditions":"Light"}

cls = json.load(open(CLASS))
# task_id (0-based) -> category, per suite
tid2cat = {}
for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
    tid2cat[suite] = {t['id']-1: CATMAP.get(t['category']) for t in cls[suite]}

def stats_json(d):
    j = glob.glob(f"{d}/category_stats_*.json")
    return json.load(open(j[0])) if j else None

def mp4_success_by_episode(d):
    """episode_num -> success(bool), from mp4 filenames."""
    out = {}
    for f in glob.glob(f"{d}/episode=*.mp4"):
        m = re.search(r"episode=(\d+)--success=(True|False)", os.path.basename(f))
        if m:
            out[int(m.group(1))] = (m.group(2) == "True")
    return out

# ---- 三套件: 直接累加 category_stats ----
def agg_simple(suite, pat):
    dim = defaultdict(lambda: [0,0])
    for s in range(8):
        ds = glob.glob(f"{BASE}/*{pat}_shard{s}_SR*") or glob.glob(f"{BASE}/*{pat}_shard{s}")
        sj = stats_json(ds[0])
        for d,v in sj['category_stats'].items():
            if d in DIMS:
                dim[d][0]+=v['success']; dim[d][1]+=v['total']
    return dim

# ---- long: 按 task_id 去重 ----
import yaml
def cfg_task_ids(cfgname):
    p = f"{REPO}/config/{cfgname}.yaml"
    return [int(x) for x in yaml.safe_load(open(p))['eval']['task_ids']]

def agg_long():
    # task_id -> success(bool). 优先级 resume > tail > remote (后处理覆盖)
    seen = {}
    # (dir_glob, cfg_name_fn) 顺序: 先 remote, 再 tail, 再 resume (后者覆盖前者)
    jobs = []
    for s in range(8):
        jobs.append((f"*envgen_attnmix_5dim_libero_10_shard{s}",
                     f"libero_plus_envgen_attnmix_5dim_libero_10_shard{s}"))
    for s in range(8):
        jobs.append((f"*envgen_attnmix_5dim_libero_10_tail_shard{s}",
                     f"libero_plus_envgen_attnmix_5dim_libero_10_tail_shard{s}"))
    for s in range(8):
        for h in (0,1):
            jobs.append((f"*envgen_attnmix_5dim_libero_10_resume{s}_h{h}",
                         f"libero_plus_envgen_attnmix_5dim_libero_10_resume{s}_h{h}"))
    for dglob, cfgname in jobs:
        ds = glob.glob(f"{BASE}/{dglob}_SR*") or glob.glob(f"{BASE}/{dglob}")
        # 精确匹配: 排除 basename 不以期望 token 结尾的(避免 shard0 误配 tail_shard0)
        key = dglob.lstrip("*")  # e.g. envgen_attnmix_5dim_libero_10_shard0
        ds = [d for d in ds if (os.path.basename(d).split("_SR")[0]).endswith(key)]
        if not ds: continue
        d = ds[0]
        try:
            tids = cfg_task_ids(cfgname)
        except FileNotFoundError:
            continue
        epi = mp4_success_by_episode(d)
        for ep, succ in epi.items():
            if ep-1 < len(tids):
                seen[tids[ep-1]] = succ
    dim = defaultdict(lambda: [0,0])
    for tid, succ in seen.items():
        cat = tid2cat['libero_10'].get(tid)
        if cat in DIMS:
            dim[cat][1]+=1
            if succ: dim[cat][0]+=1
    return dim, len(seen)

suites = {
    'spatial': agg_simple('spatial','mix5dim_libero_spatial' if False else 'envgen_attnmix_5dim_libero_spatial'),
    'object':  agg_simple('object','envgen_attnmix_5dim_libero_object'),
    'goal':    agg_simple('goal','envgen_attnmix_5dim_libero_goal'),
}
long_dim, long_n = agg_long()
suites['long'] = long_dim

# ---- 输出 ----
def fmt(dim):
    cells=[]; gs=gt=0; srs=[]
    for d in DIMS:
        s,t = dim.get(d,[0,0]); gs+=s; gt+=t
        if t: cells.append(f"{100*s/t:5.1f}"); srs.append(100*s/t)
        else: cells.append("  -  ")
    macro = sum(srs)/len(srs) if srs else 0
    return cells, macro, gs, gt

print("="*82)
print("ATTN-MIX  per-suite 5-dim env-gen")
print(f"{'suite':<9} "+" ".join(f"{d[:6]:>6}" for d in DIMS)+f"  {'Macro':>6} {'Wtd':>6}  (succ/tot)")
print("-"*82)
merged = defaultdict(lambda:[0,0])
for name in ['spatial','object','goal','long']:
    dim = suites[name]
    for d in DIMS:
        merged[d][0]+=dim.get(d,[0,0])[0]; merged[d][1]+=dim.get(d,[0,0])[1]
    cells,macro,gs,gt = fmt(dim)
    print(f"{name:<9} "+" ".join(f"{c:>6}" for c in cells)+f"  {macro:6.1f} {100*gs/gt if gt else 0:6.1f}  ({gs}/{gt})")
print("-"*82)
cells,macro,gs,gt = fmt(merged)
print(f"{'MERGED':<9} "+" ".join(f"{c:>6}" for c in cells)+f"  {macro:6.1f} {100*gs/gt if gt else 0:6.1f}  ({gs}/{gt})")
print(f"\nlong unique task_ids aggregated: {long_n}")
