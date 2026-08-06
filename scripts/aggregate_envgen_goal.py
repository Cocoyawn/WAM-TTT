#!/usr/bin/env python
"""Aggregate libero_GOAL env-gen results (TTT-256, 5 dims), zero-shot.
Authoritative per-task_id dedup from ALL shard logs (handles split shards 0-5 + tail-chewers
20/21 + any reruns; first completion per task_id wins). Dimension assigned via libero_goal's
own task_classification. Prints table + renders bar chart PNG."""
import json, yaml, re, os, glob
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TC = json.load(open('third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'))
GOAL = TC['libero_goal']
CMAP = {"Background Textures":"Background","Robot Initial States":"Robot","Camera Viewpoints":"Camera",
        "Sensor Noise":"Noise","Light Conditions":"Light","Language Instructions":"Language","Objects Layout":"Layout"}
DIMS = ["Camera", "Light", "Background", "Noise", "Robot"]
SR = re.compile(r'Current task success rate:\s*([01]\.0)')

per = {}
for cfgp in glob.glob('config/libero_plus_envgen_goal256_shard*.yaml'):
    s = re.search(r'shard(\d+)', cfgp).group(1)
    ids = yaml.safe_load(open(cfgp))['eval']['task_ids']
    lg = f'logs/plus_envgen_goal256_shard{s}.log'
    if not os.path.exists(lg):
        continue
    vals = [int(float(m.group(1))) for m in SR.finditer(open(lg, errors='ignore').read())]
    for i, v in enumerate(vals):
        if i < len(ids):
            per.setdefault(ids[i], v)   # first completion wins (dedup overlap)

universe = set()
for s in range(6):
    universe |= set(yaml.safe_load(open(f'config/libero_plus_envgen_goal256_shard{s}.yaml'))['eval']['task_ids'])
print(f"unique done {len(per)}/{len(universe)}  remaining {len(universe)-len(per)}\n")

agg = defaultdict(lambda: [0, 0])
for tid, v in per.items():
    d = CMAP.get(GOAL[tid]['category']) if 0 <= tid < len(GOAL) else None
    if d in DIMS:
        agg[d][0] += v; agg[d][1] += 1

print(f"{'Dimension':12s}{'SR':>9s}{'(succ/total)':>16s}")
tot_s = tot_t = 0
rows = []
for k in DIMS:
    s, t = agg[k]
    tot_s += s; tot_t += t
    sr = 100*s/t if t else 0
    rows.append((k, sr, s, t))
    print(f"{k:12s}{sr:8.1f}%{f'({s}/{t})':>16s}")
overall = 100*tot_s/tot_t if tot_t else 0
print(f"{'TOTAL':12s}{overall:8.1f}%{f'({tot_s}/{tot_t})':>16s}")

# plot
fig, ax = plt.subplots(figsize=(9, 5.2))
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
for b, (k, sr, s, t) in zip(bars, rows):
    ax.text(b.get_x()+b.get_width()/2, sr+1.0, f"{sr:.1f}%\n{s}/{t}", ha="center", va="bottom", fontsize=9)
ax.axhline(overall, ls="--", color="#444", lw=1.2, zorder=2, label=f"overall {overall:.1f}% ({tot_s}/{tot_t})")
ax.set_ylabel("Success Rate (%)"); ax.set_ylim(0, 105)
ax.set_title("TTT-256 — LIBERO-goal env-gen (5 perturbation dims, zero-shot)")
ax.grid(axis="y", alpha=0.3, zorder=0); ax.legend(loc="lower right")
fig.tight_layout()
out = "VLANeXt_ablation_wm/ttt_chunk256_libero_goal/GOAL_ENVGEN_5DIM.png"
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
