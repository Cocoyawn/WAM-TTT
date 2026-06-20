#!/usr/bin/env python
"""Per-layer error accumulation: incremental infer_step vs full-recompute forward.

Teacher-forced, real mix checkpoint, 256-token seq, bf16 (29-layer generator,
[A,A,A,T] stack -> TTT at layers 3,7,11,15,19,23,27). Two panels:
  (a) absolute mean|Delta| accumulating with depth, TTT layers color-highlighted
  (b) relative error = mean|Delta| / per-layer activation scale
Reuses the lab research palette + Times-like serif (STIXGeneral).
Saves docs/incremental_layer_error.png.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
D = json.load(open(f"{REPO}/docs/incremental_layer_error.json"))
rows = D["layers"]
ttt_set = set(D["meta"]["ttt_layers"])

# research palette (user-specified)
C_ATTN = "#2878B5"   # attention layers
C_TTT  = "#C82423"   # TTT layers (the replaced mixer) -- stand out in red
C_REL  = "#F8AC8C"
C_LINE = "#9AC9DB"

# Times-like serif
SERIF = "DejaVu Serif"
_times = "/mnt/afs-h200/yuyangcheng/fonts/times.ttf"
if os.path.exists(_times):
    try:
        fm.fontManager.addfont(_times); SERIF = fm.FontProperties(fname=_times).get_name()
    except Exception:
        pass
if SERIF == "DejaVu Serif" and any("STIXGeneral" in f.name for f in fm.fontManager.ttflist):
    SERIF = "STIXGeneral"
plt.rcParams["font.family"] = SERIF

L      = [r["layer"] for r in rows]
mean_d = [r["mean_abs"] for r in rows]
rel    = [r["mean_abs"] / r["ref_scale"] * 100 for r in rows]
is_ttt = [r["layer"] in ttt_set for r in rows]
bar_c  = [C_TTT if t else C_ATTN for t in is_ttt]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.7))

# (a) absolute mean|Delta| -- bars, TTT in red
ax = axes[0]
ax.bar(L, mean_d, color=bar_c, edgecolor="#3d3a33", linewidth=0.5, width=0.8, zorder=3)
ax.plot(L, mean_d, color="#3d3a33", lw=1.0, alpha=0.5, zorder=4)
for r in rows:
    if r["layer"] in ttt_set:
        ax.axvline(r["layer"], color=C_TTT, ls=":", lw=0.8, alpha=0.35, zorder=1)
ax.set_xlabel("generator layer index", fontsize=12)
ax.set_ylabel(r"mean $|\Delta|$  (incremental $-$ full-recompute)", fontsize=11.5)
ax.set_title("(a) absolute error accumulates with depth", fontsize=13, color="#3d3a33")
ax.grid(True, axis="y", alpha=0.3, lw=0.6, zorder=0)

# (b) relative error %
ax = axes[1]
ax.bar(L, rel, color=bar_c, edgecolor="#3d3a33", linewidth=0.5, width=0.8, zorder=3)
ax.plot(L, rel, color="#3d3a33", lw=1.0, alpha=0.5, zorder=4)
ax.set_xlabel("generator layer index", fontsize=12)
ax.set_ylabel(r"relative error  $\mathrm{mean}|\Delta| / \mathrm{activation\ scale}$  (%)", fontsize=11.5)
ax.set_title("(b) relative error stays in round-off band (<2.2%)", fontsize=13, color="#3d3a33")
ax.grid(True, axis="y", alpha=0.3, lw=0.6, zorder=0)

legend = [Patch(facecolor=C_ATTN, edgecolor="#3d3a33", label="attention layer"),
          Patch(facecolor=C_TTT, edgecolor="#3d3a33", label="TTT layer (replaced mixer)")]
for ax in axes:
    ax.legend(handles=legend, loc="upper left", fontsize=10, frameon=True, framealpha=0.9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#b7b1a3")

fig.suptitle("Incremental decode vs. full-recompute: per-layer error growth "
             "(teacher-forced, real ckpt, 256 tokens, bf16)",
             fontsize=13.5, color="#2d2a23", y=1.02)
plt.tight_layout()
out = f"{REPO}/docs/incremental_layer_error.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out, "| font:", SERIF,
      f"| layer0 rel={rel[0]:.3f}% layer28 rel={rel[-1]:.3f}%")
