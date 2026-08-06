#!/usr/bin/env python
"""Robustness-vs-perturbation-strength curves for LIBERO-OBJECT: TTT-256 vs Attention.
Adapted from scripts/plot_severity_curves.py (same Morandi palette / serif style / 5-subplot
layout), but: object suite, only TTT-256 (warmup-aligned) + Attention, 2 lines.
Data: TTT-256 = dev cam/bg logs + /tmp/tttwu45_lnr_eval.json ; Attention = attncam20k + attnobj4d.
Saves docs/object_severity_curves.png
"""
import os, json, re, glob, yaml
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

os.chdir('/mnt/afs-h200/yuyangcheng/workplace/VLANeXt')

_otf = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"
if os.path.exists(_otf):
    fm.fontManager.addfont(_otf)
    SERIF = fm.FontProperties(fname=_otf).get_name()
else:
    SERIF = "DejaVu Serif"
plt.rcParams["font.family"] = SERIF

TC = json.load(open('third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'))
OB = TC['libero_object']
CMAP = {"Camera Viewpoints":"Camera","Light Conditions":"Light","Background Textures":"Background",
        "Sensor Noise":"Noise","Robot Initial States":"Robot"}
DIMS = ["Camera","Light","Background","Noise","Robot"]
def cat_of(i): return CMAP.get(OB[i]['category']) if 0 <= i < len(OB) else None
def diff_of(i): return OB[i]['difficulty_level']

SR = re.compile(r'Current task success rate:\s*([01]\.0)')
def collect(cfg_glob):
    per = {}
    for f in glob.glob(cfg_glob):
        ids = yaml.safe_load(open(f))['eval']['task_ids']
        lg = f"logs/plus_envgen_{os.path.basename(f)[len('libero_plus_envgen_'):-5]}.log"
        if not os.path.exists(lg): continue
        for i, v in enumerate([int(float(m.group(1))) for m in SR.finditer(open(lg, errors='ignore').read())]):
            if i < len(ids): per.setdefault(ids[i], v)
    return per

# TTT-256 (warmup-aligned): cam/bg from dev 3w_cambg logs + L/N/R from synced json
ttt256 = collect('config/libero_plus_envgen_tttwu45_3w_cambg_shard*.yaml')
ttt256.update({int(k): v for k, v in json.load(open('/tmp/tttwu45_lnr_eval.json')).items()})
# Attention: Camera from 2w (attncam20k) + 4 dims from 3w-final (attnobj4d)
att = collect('config/libero_plus_envgen_attncam20k_shard*.yaml')
att.update(collect('config/libero_plus_envgen_attnobj4d_shard*.yaml'))

common = set(ttt256) & set(att)
MODELS = [('TTT-256', ttt256, '#6b8cae', 's'),
          ('Attention', att, '#bb8e7d', '^')]

def sr(m, dim, lvl):
    s = t = 0
    for i in common:
        if cat_of(i) == dim and diff_of(i) == lvl:
            s += m[i]; t += 1
    return (100*s/t if t else np.nan), t

fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
LV = [1, 2, 3, 4, 5]
for ax, dim in zip(axes, DIMS):
    ns = [sr(ttt256, dim, l)[1] for l in LV]
    for name, m, col, mk in MODELS:
        ys = [sr(m, dim, l)[0] for l in LV]
        ax.plot(LV, ys, marker=mk, color=col, lw=2, markersize=7, label=name)
    ax.set_title(dim, fontsize=14, fontweight='bold', color="#3d3a33")
    ax.set_xlabel("perturbation difficulty level", fontsize=11)
    ax.set_xticks(LV)
    ax.set_xticklabels([f"L{l}\n(n={n})" for l, n in zip(LV, ns)], fontsize=9)
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3, lw=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#b7b1a3")
axes[0].set_ylabel("success rate (%)", fontsize=12)
axes[0].legend(loc='lower left', fontsize=10, frameon=True, framealpha=0.9)
plt.tight_layout()
out = "docs/object_severity_curves.png"
os.makedirs("docs", exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches='tight')
print("saved", out, "| aligned:", len(common), "| font:", SERIF)
