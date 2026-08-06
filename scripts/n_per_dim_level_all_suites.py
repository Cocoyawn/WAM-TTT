#!/usr/bin/env python
"""Print n-per-(dimension x difficulty_level) tables for all 4 suites' env-gen 5-dim universe.
Pure counts from task_classification.json — does NOT need any eval results.
The 5-dim env-gen universe = the 5 perturbation dims (Camera/Light/Background/Noise/Robot),
excluding Language + Objects-Layout (the 2 dims dropped from the env-gen mainline).
"""
import os, json
os.chdir('/mnt/afs-h200/yuyangcheng/workplace/VLANeXt')
TC = json.load(open('third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'))

CMAP = {"Camera Viewpoints":"Camera","Light Conditions":"Light","Background Textures":"Background",
        "Sensor Noise":"Noise","Robot Initial States":"Robot"}
DIMS = ["Camera","Light","Background","Noise","Robot"]
LV = [1,2,3,4,5]
SUITES = [("spatial","libero_spatial"),("object","libero_object"),
          ("goal","libero_goal"),("long","libero_10")]

for label, key in SUITES:
    if key not in TC:
        print(f"\n##### {label} ({key}): NOT in classification #####")
        continue
    tasks = TC[key]
    grid = {d:{l:0 for l in LV} for d in DIMS}
    total = 0
    for t in tasks:
        d = CMAP.get(t['category'])
        l = t.get('difficulty_level')
        if d and l in grid[d]:
            grid[d][l] += 1; total += 1
    print(f"\n##### {label} ({key}) — n per dim x level | 5-dim total = {total} #####")
    print(f"{'Dim':12s} " + " ".join(f"L{l:>4d}" for l in LV) + "   row")
    for d in DIMS:
        row_tot = sum(grid[d][l] for l in LV)
        print(f"{d:12s} " + " ".join(f"{grid[d][l]:5d}" for l in LV) + f"   {row_tot:5d}")
    col = [sum(grid[d][l] for d in DIMS) for l in LV]
    print(f"{'col total':12s} " + " ".join(f"{c:5d}" for c in col) + f"   {total:5d}")
