#!/usr/bin/env python3
"""Attention-mask comparison: full causal vs TTT-16 vs TTT-256.
Morandi palette, Times-like serif (Nimbus Roman), visible grid mask, per-panel captions.
Saves docs/attn_mask_comparison.png."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ---- Times New Roman look via Nimbus Roman (metric-compatible) ----
_otf = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"
if os.path.exists(_otf):
    fm.fontManager.addfont(_otf)
    fm.fontManager.addfont("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf")
    SERIF = fm.FontProperties(fname=_otf).get_name()
else:
    SERIF = "DejaVu Serif"
plt.rcParams["font.family"] = SERIF
plt.rcParams["mathtext.fontset"] = "stix"

N = 64            # image tokens (scaled; real vision DiT = 256). 4 blocks of 16 = TTT-16 view.
BLK = 16          # TTT-16 block size; also the VLM-column width (= one staircase step)
NV = BLK          # VLM context width == chunk-16 step width (per request)

def full_causal(n):
    return np.tril(np.ones((n, n)))

def ttt_blockcausal(n, chunk):
    qb = (np.arange(n) // chunk)[:, None]; kb = (np.arange(n) // chunk)[None, :]
    return (kb < qb).astype(float)

def with_vlm(m):
    return np.hstack([m, np.full((m.shape[0], NV), 2.0)])

panels = [
    ("Full Causal Attention",
     with_vlm(full_causal(N)),
     "Each query attends to all earlier tokens\n"
     "via exact pairwise dot-products (lossless)."),
    ("TTT-16   (block = 16)",
     with_vlm(ttt_blockcausal(N, BLK)),
     "Block-causal fast-weight: a query reads only\n"
     "earlier blocks, summarized in a fast weight."),
    ("TTT-256   (block = 256)",
     with_vlm(ttt_blockcausal(N, N)),
     "Whole frame is one block: image tokens are\n"
     "mutually invisible and rely on VLM context."),
]

# Morandi palette: muted sage (visible), warm greige (masked), dusty terracotta (VLM)
cmap = ListedColormap(["#cdc9bd", "#8a9a87", "#bb8e7d"])   # 0 masked, 1 visible, 2 VLM
# imshow maps 0->masked,1->visible,2->VLM ; reorder so indices match codes
cmap = ListedColormap(["#d8d4c8", "#8a9a87", "#bb8e7d"])   # [masked, visible, VLM]

fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
GRID = "#ffffff"
for ax, (title, M, caption) in zip(axes, panels):
    ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="equal", interpolation="nearest")
    # visible grid lines between every cell
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color=GRID, lw=0.6)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(N - 0.5, color="#9c6b59", lw=1.8)         # image | VLM divider
    for s in ax.spines.values():
        s.set_edgecolor("#b7b1a3"); s.set_linewidth(1.0)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10, color="#3d3a33")
    ax.text(N + NV/2 - 0.5, -2.2, "VLM", color="#9c6b59", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    # caption below each panel
    ax.text(0.5, -0.13, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=11.5, color="#4a4740", linespacing=1.4)

axes[0].set_ylabel("query  position", fontsize=12, color="#4a4740")
for ax in axes:
    ax.set_xlabel("key  position", fontsize=12, color="#4a4740", labelpad=2)

legend = [Patch(facecolor="#8a9a87", label="visible"),
          Patch(facecolor="#d8d4c8", edgecolor="#b7b1a3", label="masked"),
          Patch(facecolor="#bb8e7d", label="VLM context (always visible)")]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.12, 1, 1])
out = "docs/attn_mask_comparison.png"
os.makedirs("docs", exist_ok=True)
plt.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out, "| font:", SERIF)
