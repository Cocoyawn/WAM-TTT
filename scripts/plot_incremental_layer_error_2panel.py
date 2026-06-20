#!/usr/bin/env python
"""Per-layer relative error: teacher-forced vs free-running (incremental vs
full-recompute, real mix checkpoint, 256 tokens, bf16, 29-layer [A,A,A,T] gen).

Two panels side by side:
  (a) teacher-forced: same fixed tokens -> error is pure bf16 round-off, climbs
      gently 0.13% -> 2.1% with depth (stays in round-off band).
  (b) free-running: each path argmax-self-feeds -> token sequences diverge from
      ~token 9 -> per-layer hidden error SATURATES at ~75-83% from layer 0.
This contrast is the root cause: the incremental APPLY is correct (panel a), but
argmax self-feedback amplifies round-off into full divergence (panel b).
TTT layers (replaced mixer, idx 3/7/11/15/19/23/27) highlighted in red.
Reuses lab research palette + Times-like serif (STIXGeneral).
Saves docs/incremental_layer_error_2panel.png.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
TF = json.load(open(f"{REPO}/docs/incremental_layer_error.json"))
FR = json.load(open(f"{REPO}/docs/incremental_layer_error_freerun.json"))
TTT = set(TF["meta"]["ttt_layers"])

C_ATTN, C_TTT = "#2878B5", "#C82423"

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

# teacher-forced rel = mean_abs/ref_scale*100 ; free-run rel = rel_pct
L_tf = [r["layer"] for r in TF["layers"]]
rel_tf = [r["mean_abs"] / r["ref_scale"] * 100 for r in TF["layers"]]
L_fr = [r["layer"] for r in FR["layers"]]
rel_fr = [r["rel_pct"] for r in FR["layers"]]
colors = [C_TTT if i in TTT else C_ATTN for i in L_tf]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.7))

def panel(ax, L, rel, title, ylab):
    ax.bar(L, rel, color=colors, edgecolor="#3d3a33", linewidth=0.5, width=0.8, zorder=3)
    ax.plot(L, rel, color="#3d3a33", lw=1.0, alpha=0.5, zorder=4)
    ax.set_xlabel("generator layer index", fontsize=12)
    ax.set_ylabel(ylab, fontsize=11.5)
    ax.set_title(title, fontsize=13, color="#3d3a33")
    ax.grid(True, axis="y", alpha=0.3, lw=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#b7b1a3")

panel(axes[0], L_tf, rel_tf,
      "(a) teacher-forced: round-off only (peaks 2.1%)",
      r"relative error  $\mathrm{mean}|\Delta|/\mathrm{scale}$  (%)")
panel(axes[1], L_fr, rel_fr,
      f"(b) free-running: saturated divergence (token match {FR['meta']['token_match_pct']:.0f}%)",
      r"relative error  $\mathrm{mean}|\Delta|/\mathrm{scale}$  (%)")
axes[1].set_ylim(0, 100)

legend = [Patch(facecolor=C_ATTN, edgecolor="#3d3a33", label="attention layer"),
          Patch(facecolor=C_TTT, edgecolor="#3d3a33", label="TTT layer (replaced mixer)")]
axes[0].legend(handles=legend, loc="upper left", fontsize=10, frameon=True, framealpha=0.9)

fig.suptitle("Incremental vs. full-recompute per-layer error: round-off (teacher-forced) "
             "vs. argmax-feedback divergence (free-running)",
             fontsize=12.8, color="#2d2a23", y=1.02)
plt.tight_layout()
out = f"{REPO}/docs/incremental_layer_error_2panel.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out, "| font:", SERIF,
      f"| TF peak={max(rel_tf):.2f}% FR mean={sum(rel_fr)/len(rel_fr):.1f}%")
