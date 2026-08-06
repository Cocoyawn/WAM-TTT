#!/usr/bin/env python
"""Print LIBERO-SPATIAL env-gen SR by (dimension, difficulty_level) for TTT-256 vs Attention.
Reuses scripts/aggregate_envgen.collect (the finished spatial logs). Same format as the object table.
"""
import os, sys, json
os.chdir('/mnt/afs-h200/yuyangcheng/workplace/VLANeXt')
sys.path.insert(0, 'scripts')
from aggregate_envgen import collect, cat_of, DIMS

TC = json.load(open('third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'))
SP = TC['libero_spatial']
def diff_of(i): return SP[i]['difficulty_level']

ttt256 = collect('ttt256', list(range(8)))
att    = collect('attn',   list(range(13)))
common = set(ttt256) & set(att)
print("aligned:", len(common))

def sr(m, dim, lvl):
    s = t = 0
    for i in common:
        if cat_of(i) == dim and diff_of(i) == lvl:
            s += m[i]; t += 1
    return (100*s/t if t else float('nan')), t

for name, m in [("TTT-256", ttt256), ("Attention", att)]:
    print(f"\n===== {name} (spatial) — SR%% by dim x level =====")
    print(f"{'Dim':12s} " + " ".join(f"L{l:>4d}" for l in [1,2,3,4,5]))
    for dim in DIMS:
        row = f"{dim:12s} "
        for l in [1,2,3,4,5]:
            v, t = sr(m, dim, l)
            row += f"{v:5.0f}" if t else "  -  "
            row += " "
        print(row)

# also the n per (dim,level)
print("\n===== n per (dim x level) =====")
print(f"{'Dim':12s} " + " ".join(f"L{l:>4d}" for l in [1,2,3,4,5]))
for dim in DIMS:
    row = f"{dim:12s} "
    for l in [1,2,3,4,5]:
        _, t = sr(ttt256, dim, l)
        row += f"{t:5d} "
    print(row)
