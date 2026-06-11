#!/usr/bin/env python
"""Aggregate LIBERO-plus env-gen results: TTT-16 vs Attention, per dimension.

Robust against the three known pitfalls:
  1. mp4 filenames truncate long Camera task names -> DO NOT use filenames for dim.
  2. category_stats JSON double-counts across duplicate/overlapping shard dirs.
  3. logs are re-appended on restart -> map block-index to config task_id and keep
     FIRST completed value per task_id (a restart replays the same prefix).

Dimension is assigned deterministically by task_id via task_classification.json
(idx == task_id for libero_spatial; the 5 env-gen dims sum to 1627).
"""
import re, yaml, json
from collections import defaultdict

TC = json.load(open('third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'))
SP = TC['libero_spatial']
CMAP = {"Background Textures":"Background","Robot Initial States":"Robot",
        "Camera Viewpoints":"Camera","Language Instructions":"Language",
        "Sensor Noise":"Noise","Objects Layout":"Layout","Light Conditions":"Light"}
def cat_of(i): return CMAP.get(SP[i]['category']) if 0 <= i < len(SP) else None

TASK_SR = re.compile(r'Current task success rate:\s*([01]\.0)')
def parse_successes(logpath):
    out = []
    try:
        for ln in open(logpath):
            m = TASK_SR.search(ln)
            if m: out.append(int(float(m.group(1))))
    except FileNotFoundError:
        pass
    return out

def cfgids(prefix, s):
    return yaml.safe_load(open(f'config/libero_plus_envgen_{prefix}_shard{s}.yaml'))['eval']['task_ids']

def collect(prefix, shards):
    per = {}
    for s in shards:
        succ = parse_successes(f'logs/plus_envgen_{prefix}_shard{s}.log')
        ids = cfgids(prefix, s)
        for i, v in enumerate(succ):
            if i >= len(ids): break
            per.setdefault(ids[i], v)   # first occurrence wins (restart replays prefix)
    return per

DIMS = ['Camera', 'Light', 'Background', 'Noise', 'Robot']

def dim_table(per):
    cat = defaultdict(lambda: [0, 0])
    for tid, v in per.items():
        cat[cat_of(tid)][0] += v; cat[cat_of(tid)][1] += 1
    return cat

def main():
    ttt16  = collect('ttt16',  list(range(12)))
    att    = collect('attn',   list(range(13)))   # 0-4 original + 5-12 catch-up
    ttt256 = collect('ttt256', list(range(8)))

    models = [('TTT-16', ttt16), ('TTT-256', ttt256), ('Attention', att)]
    for name, m in models:
        print(f"{name} done: {len(m)}/1627")
    # align on task_ids present in ALL three
    common = set(ttt16)
    for _, m in models[1:]:
        common &= set(m)
    common = sorted(common)
    print(f"aligned (in all 3): {len(common)}\n")

    # per-dim per-model on the aligned set
    def dim_on(m):
        c = defaultdict(lambda: [0, 0])
        for i in common:
            c[cat_of(i)][0] += m[i]; c[cat_of(i)][1] += 1
        return c
    cols = [(name, dim_on(m)) for name, m in models]

    header = f"{'Dimension':12s}" + "".join(f"{name:>14s}" for name, _ in cols)
    print(header)
    tot = {name: [0, 0] for name, _ in cols}
    for k in DIMS:
        row = f"{k:12s}"
        for name, c in cols:
            s, t = c[k]
            if t:
                tot[name][0] += s; tot[name][1] += t
                row += f"{100*s/t:12.1f}% "
            else:
                row += f"{'-':>13s} "
        print(row)
    row = f"{'TOTAL':12s}"
    for name, _ in cols:
        s, t = tot[name]
        row += f"{100*s/t:12.1f}% " if t else f"{'-':>13s} "
    print(row)
    # key deltas
    g = lambda nm: (100*tot[nm][0]/tot[nm][1]) if tot[nm][1] else 0
    print(f"\nTTT-256 vs TTT-16:    {g('TTT-256')-g('TTT-16'):+.1f}pp   (chunk16->256 generalization cost)")
    print(f"TTT-256 vs Attention: {g('TTT-256')-g('Attention'):+.1f}pp")

if __name__ == '__main__':
    main()
