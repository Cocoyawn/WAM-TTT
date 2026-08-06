#!/usr/bin/env python
"""Aggregate libero_OBJECT env-gen results (TTT-256, 5 dims) from the per-shard
category_stats_*.json that each finished shard writes (authoritative, dedup-safe).
Sums Camera/Light/Background/Noise/Robot across the 6 obj256 shards, prints a table,
and renders a bar chart PNG. (object has its OWN classification; 5-dim excludes
Language + Objects-Layout.)"""
import json, glob
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMS = ["Camera", "Light", "Background", "Noise", "Robot"]
BASE = "VLANeXt_ablation_wm/ttt_chunk256_libero_object"

agg = defaultdict(lambda: [0, 0])           # dim -> [success, total]
seen_tags = set()
files = sorted(glob.glob(f"{BASE}/*obj256_shard*_SR*/category_stats_envgen_obj256_shard*.json"))
for f in files:
    d = json.load(open(f))
    tag = d["shard_tag"]
    if tag in seen_tags:                     # dedup duplicate _SR dirs (reruns)
        continue
    seen_tags.add(tag)
    for dim, st in d["category_stats"].items():
        if dim in DIMS:
            agg[dim][0] += st["success"]; agg[dim][1] += st["total"]

print(f"shards aggregated: {len(seen_tags)}  ({sorted(seen_tags)})\n")
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

# ---- plot ----
fig, ax = plt.subplots(figsize=(9, 5.2))
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
for b, (k, sr, s, t) in zip(bars, rows):
    ax.text(b.get_x()+b.get_width()/2, sr+1.0, f"{sr:.1f}%\n{s}/{t}",
            ha="center", va="bottom", fontsize=9)
ax.axhline(overall, ls="--", color="#444", lw=1.2, zorder=2,
           label=f"overall {overall:.1f}% ({tot_s}/{tot_t})")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(0, 105)
ax.set_title("TTT-256 — LIBERO-object env-gen (5 perturbation dims, 1761 tasks)")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(loc="lower right")
fig.tight_layout()
out = "VLANeXt_ablation_wm/ttt_chunk256_libero_object/OBJ_ENVGEN_5DIM.png"
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
