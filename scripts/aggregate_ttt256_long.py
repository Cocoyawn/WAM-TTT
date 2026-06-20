#!/usr/bin/env python
"""Aggregate TTT-256 (trained on libero_10/long, warmup 5400) env-gen results.

Reads the 8 finished shard category_stats JSONs (non-overlapping round-robin
task_ids, so a straight sum is correct) and prints the 5-dim success table.
Also dumps docs/ttt256_long_envgen.json for the plotting step.
"""
import json, glob, os
from collections import defaultdict

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
RUN = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_long_wu5400"
DIMS = ["Camera", "Light", "Background", "Noise", "Robot"]

stats = defaultdict(lambda: [0, 0])  # dim -> [success, total]
shard_sr = {}
nshards = 0
for s in range(8):
    cands = glob.glob(
        f"{RUN}/*envgen_tttlong_5dim_shard{s}_SR*/category_stats_*.json")
    assert cands, f"no category_stats json for shard{s}"
    j = json.load(open(cands[0]))
    nshards += 1
    cs = j["category_stats"]
    for d in DIMS:
        stats[d][0] += cs[d]["success"]
        stats[d][1] += cs[d]["total"]
    shard_sr[s] = (j["total_successes"], j["total_episodes"])

# table
print(f"\nTTT-256 trained on libero_10(long), warmup 5400 | env-gen 5-dim | "
      f"{nshards}/8 shards\n")
print(f"{'Dimension':12s}{'Success':>10s}{'Total':>8s}{'SR %':>9s}")
print("-" * 39)
gs = gt = 0
for d in DIMS:
    s, t = stats[d]
    gs += s; gt += t
    print(f"{d:12s}{s:>10d}{t:>8d}{100*s/t:>8.1f}%")
print("-" * 39)
print(f"{'TOTAL':12s}{gs:>10d}{gt:>8d}{100*gs/gt:>8.1f}%")

# dump for plotting
out = {
    "model": "TTT-256 (long, wu5400)",
    "suite": "libero_10",
    "dims": {d: {"success": stats[d][0], "total": stats[d][1],
                 "sr": 100 * stats[d][0] / stats[d][1]} for d in DIMS},
    "total": {"success": gs, "total": gt, "sr": 100 * gs / gt},
    "per_shard_sr": {s: 100 * a / b for s, (a, b) in shard_sr.items()},
}
os.makedirs(f"{REPO}/docs", exist_ok=True)
json.dump(out, open(f"{REPO}/docs/ttt256_long_envgen.json", "w"), indent=2)
print(f"\nwrote docs/ttt256_long_envgen.json")
