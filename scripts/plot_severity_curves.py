#!/usr/bin/env python
"""Robustness-vs-perturbation-strength curves: TTT-16 / TTT-256 / Attention.

Re-aggregates the FINISHED env-gen logs (no re-eval) by (dimension, difficulty_level)
using task_classification.json. Each dim becomes a line plot SR(level), exposing the
robustness SLOPE that the single-number aggregate hides. Saves docs/severity_curves.png.
"""
import os, sys, json
sys.path.insert(0, 'scripts')
from aggregate_envgen import collect, cat_of, DIMS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_otf = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"
if os.path.exists(_otf):
    fm.fontManager.addfont(_otf)
    SERIF = fm.FontProperties(fname=_otf).get_name()
else:
    SERIF = "DejaVu Serif"
plt.rcParams["font.family"] = SERIF

TC = json.load(open('third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'))
SPx = TC['libero_spatial']
def diff_of(i): return SPx[i]['difficulty_level']

ttt16  = collect('ttt16',  list(range(12)))
ttt64  = collect('ttt64',  list(range(8)))
ttt256 = collect('ttt256', list(range(8)))
att    = collect('attn',   list(range(13)))
common = set(ttt16) & set(ttt64) & set(ttt256) & set(att)
MODELS = [('TTT-16', ttt16, '#8a9a87', 'o'),
          ('TTT-64', ttt64, '#c2a86b', 'D'),
          ('TTT-256', ttt256, '#6b8cae', 's'),
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
    ns = [sr(ttt16, dim, l)[1] for l in LV]
    for name, m, col, mk in MODELS:
        ys = [sr(m, dim, l)[0] for l in LV]
        ax.plot(LV, ys, marker=mk, color=col, lw=2, markersize=7, label=name)
    ax.set_title(dim, fontsize=14, fontweight='bold', color="#3d3a33")
    ax.set_xlabel("perturbation difficulty level", fontsize=11)
    ax.set_xticks(LV)
    ax.set_xticklabels([f"L{l}\n(n={n})" for l, n in zip(LV, ns)], fontsize=9)
    ax.set_ylim(20, 103)
    ax.grid(True, alpha=0.3, lw=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#b7b1a3")
axes[0].set_ylabel("success rate (%)", fontsize=12)
axes[0].legend(loc='lower left', fontsize=10, frameon=True, framealpha=0.9)
plt.tight_layout()
out = "docs/severity_curves.png"
os.makedirs("docs", exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches='tight')
print("saved", out, "| aligned:", len(common), "| font:", SERIF)
