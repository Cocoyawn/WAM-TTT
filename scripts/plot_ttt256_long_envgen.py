#!/usr/bin/env python
"""Plot TTT-256 (trained on libero_10/long, wu5400) env-gen results.

Two panels, reusing the lab's research palette + Times-like serif:
  (a) per-dimension success-rate bars (the 5-dim table, visualized)
  (b) SR vs perturbation difficulty level, per dimension (robustness slope)

Reads the 8 finished shard dirs:
  - category_stats_*.json  -> per-dim totals (panel a)
  - log.txt + cfg task_ids -> per-task success, joined to difficulty (panel b)
No re-eval. Saves docs/ttt256_long_envgen.png.
"""
import os, re, json, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
RUN = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_long_wu5400"
DIMS = ["Camera", "Light", "Background", "Noise", "Robot"]
# research palette (user-specified)
PALETTE = {"Camera": "#2878B5", "Light": "#9AC9DB", "Background": "#F8AC8C",
           "Noise": "#C82423", "Robot": "#FF8884"}

_otf = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"
SERIF = "DejaVu Serif"
for _cand in ["/mnt/afs-h200/yuyangcheng/fonts/times.ttf", _otf]:
    if os.path.exists(_cand):
        try:
            fm.fontManager.addfont(_cand)
            SERIF = fm.FontProperties(fname=_cand).get_name()
            break
        except Exception:
            pass
else:
    # STIXGeneral: publication-grade Times New Roman clone bundled with matplotlib
    if any("STIXGeneral" in f.name for f in fm.fontManager.ttflist):
        SERIF = "STIXGeneral"
plt.rcParams["font.family"] = SERIF

TC = json.load(open(f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"))
L10 = TC["libero_10"]
CMAP = {"Background Textures": "Background", "Robot Initial States": "Robot",
        "Camera Viewpoints": "Camera", "Sensor Noise": "Noise",
        "Light Conditions": "Light"}
def cat_of(i):  return CMAP.get(L10[i]["category"]) if 0 <= i < len(L10) else None
def diff_of(i): return L10[i]["difficulty_level"] if 0 <= i < len(L10) else None

TASK_SR = re.compile(r"Current task success rate:\s*([01]\.0)")
def parse_succ(p):
    out = []
    for ln in open(p):
        m = TASK_SR.search(ln)
        if m: out.append(int(float(m.group(1))))
    return out

# ---- panel (a) data: sum category_stats over shards ----
dim_tot = defaultdict(lambda: [0, 0])
# ---- panel (b) data: per-task success joined to difficulty ----
per = {}  # task_id -> 0/1
for s in range(8):
    d = glob.glob(f"{RUN}/*envgen_tttlong_5dim_shard{s}_SR*")[0]
    cs = json.load(open(glob.glob(f"{d}/category_stats_*.json")[0]))
    for dim in DIMS:
        dim_tot[dim][0] += cs["category_stats"][dim]["success"]
        dim_tot[dim][1] += cs["category_stats"][dim]["total"]
    ids = cs["task_ids"]
    succ = parse_succ(f"{d}/log.txt")
    for i, v in zip(ids, succ):
        per.setdefault(i, v)

# difficulty buckets per dim
def sr_at(dim, lvl):
    s = t = 0
    for i, v in per.items():
        if cat_of(i) == dim and diff_of(i) == lvl:
            s += v; t += 1
    return (100 * s / t if t else None), t

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

# (a) per-dim bars
ax = axes[0]
srs = [100 * dim_tot[d][0] / dim_tot[d][1] for d in DIMS]
bars = ax.bar(DIMS, srs, color=[PALETTE[d] for d in DIMS],
              edgecolor="#3d3a33", linewidth=0.8, width=0.66, zorder=3)
gs = sum(dim_tot[d][0] for d in DIMS); gt = sum(dim_tot[d][1] for d in DIMS)
tot_sr = 100 * gs / gt
ax.axhline(tot_sr, ls="--", lw=1.3, color="#3d3a33", alpha=0.7, zorder=2,
           label=f"overall {tot_sr:.1f}%")
for b, v, d in zip(bars, srs, DIMS):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
            ha="center", va="bottom", fontsize=10.5, color="#3d3a33")
ax.set_ylim(0, 103)
ax.set_ylabel("success rate (%)", fontsize=12)
ax.set_title("(a) per-dimension success rate", fontsize=13, color="#3d3a33")
ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
ax.grid(True, axis="y", alpha=0.3, lw=0.6, zorder=0)
ax.tick_params(axis="x", labelsize=10.5)

# (b) SR vs difficulty
ax = axes[1]
LV = [1, 2, 3, 4, 5]
for d in DIMS:
    ys = [sr_at(d, l)[0] for l in LV]
    xs = [l for l, y in zip(LV, ys) if y is not None]
    yy = [y for y in ys if y is not None]
    ax.plot(xs, yy, marker="o", color=PALETTE[d], lw=2, markersize=6,
            label=d, markeredgecolor="#3d3a33", markeredgewidth=0.5)
ax.set_xticks(LV)
ax.set_xlabel("perturbation difficulty level", fontsize=12)
ax.set_ylim(0, 103)
ax.set_title("(b) robustness vs. perturbation strength", fontsize=13, color="#3d3a33")
ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3, lw=0.6)
for a in axes:
    for sp in a.spines.values():
        sp.set_edgecolor("#b7b1a3")

fig.suptitle("TTT-256 trained on LIBERO-long  |  LIBERO-plus env-gen (5 dims, 1824 tasks)",
             fontsize=14, color="#2d2a23", y=1.02)
plt.tight_layout()
out = f"{REPO}/docs/ttt256_long_envgen.png"
os.makedirs(f"{REPO}/docs", exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out, "| font:", SERIF, "| overall SR:", f"{tot_sr:.1f}%")
