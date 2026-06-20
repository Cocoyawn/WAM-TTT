#!/usr/bin/env python
"""TTT-256 (libero_long, wu5400) env-gen: 5-dim x difficulty L1-L5 breakdown.

Reads the 8 finished shard dirs (log.txt + category_stats task_ids), joins each
task to (dimension, difficulty_level) via task_classification.json[libero_10],
and prints: per-dim n + SR, and the dim x level SR matrix. Dumps a JSON.
"""
import os, re, json, glob
from collections import defaultdict

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
RUN = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_long_wu5400"
DIMS = ["Camera", "Light", "Background", "Noise", "Robot"]

TC = json.load(open(f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"))
L10 = TC["libero_10"]
CMAP = {"Background Textures": "Background", "Robot Initial States": "Robot",
        "Camera Viewpoints": "Camera", "Sensor Noise": "Noise",
        "Light Conditions": "Light"}
def cat_of(i):  return CMAP.get(L10[i]["category"]) if 0 <= i < len(L10) else None
def diff_of(i): return L10[i]["difficulty_level"] if 0 <= i < len(L10) else None

TASK_SR = re.compile(r"Current task success rate:\s*([01]\.0)")
def parse_succ(p):
    return [int(float(m.group(1)))
            for ln in open(p) for m in [TASK_SR.search(ln)] if m]

per = {}
for s in range(8):
    d = glob.glob(f"{RUN}/*envgen_tttlong_5dim_shard{s}_SR*")[0]
    cs = json.load(open(glob.glob(f"{d}/category_stats_*.json")[0]))
    for i, v in zip(cs["task_ids"], parse_succ(f"{d}/log.txt")):
        per.setdefault(i, v)

# per-dim totals
dim_tot = defaultdict(lambda: [0, 0])
# dim x level
cell = defaultdict(lambda: [0, 0])
for i, v in per.items():
    d, l = cat_of(i), diff_of(i)
    if d in DIMS:
        dim_tot[d][0] += v; dim_tot[d][1] += 1
        cell[(d, l)][0] += v; cell[(d, l)][1] += 1

print("\n=== per-dim n + SR ===")
gs = gt = 0
for d in DIMS:
    s, t = dim_tot[d]; gs += s; gt += t
    print(f"  {d:11s} n={t:4d}  SR={100*s/t:5.1f}%")
print(f"  {'Overall':11s} n={gt:4d}  SR={100*gs/gt:5.1f}%")

print("\n=== dim x difficulty (SR% | n) ===")
print(f"  {'dim':11s}" + "".join(f"{'L'+str(l):>12s}" for l in range(1, 6)))
matrix = {}
for d in DIMS:
    row = f"  {d:11s}"
    matrix[d] = {}
    for l in range(1, 6):
        s, t = cell[(d, l)]
        matrix[d][l] = {"success": s, "total": t,
                        "sr": (100 * s / t if t else None)}
        row += f"{(f'{100*s/t:.1f}|{t}' if t else '-|0'):>12s}"
    print(row)

out = {
    "model": "TTT-256 (long, wu5400)", "suite": "libero_10",
    "per_dim": {d: {"n": dim_tot[d][1], "sr": 100*dim_tot[d][0]/dim_tot[d][1]} for d in DIMS},
    "overall": {"n": gt, "sr": 100*gs/gt},
    "dim_by_level": matrix,
}
json.dump(out, open(f"{REPO}/docs/ttt256_long_envgen_bylevel.json", "w"), indent=2)
print(f"\nwrote docs/ttt256_long_envgen_bylevel.json")
