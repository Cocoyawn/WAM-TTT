"""
科研风格图表: 从 docs/ttt_validation_data.json 出
  - 误差分布: 图 (docs/ttt_error_dist.png) + 表 (docs/ttt_error_table.md / .csv)
  - 速度对比: 图 (docs/ttt_speed.png) + 表 (docs/ttt_speed_table.md / .csv)

配色 (科研风): #2878B5 #9AC9DB #F8AC8C #C82423 #FF8884

Run:
  /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.plot_ttt_validation
"""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BLUE = "#2878B5"
C_LBLUE = "#9AC9DB"
C_ORANGE = "#F8AC8C"
C_RED = "#C82423"
C_PINK = "#FF8884"
DATA = "docs/ttt_validation_data.json"

# --- Times New Roman if available, else a Times-compatible serif ---
import matplotlib.font_manager as fm
import os as _os
_SERIF = "DejaVu Serif"
for _cand_path in ["/mnt/afs-h200/yuyangcheng/fonts/times.ttf",
                   "/mnt/afs-h200/yuyangcheng/fonts/LiberationSerif-Regular.ttf"]:
    if _os.path.exists(_cand_path):
        try:
            fm.fontManager.addfont(_cand_path)
            _SERIF = fm.FontProperties(fname=_cand_path).get_name()
            break
        except Exception:
            pass
else:
    # STIXGeneral is a publication-grade Times New Roman clone (IEEE/AIP standard)
    _avail = {f.name for f in fm.fontManager.ttflist}
    for _c in ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral"]:
        if _c in _avail:
            _SERIF = _c
            break

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [_SERIF, "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.titleweight": "bold", "axes.labelweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1, "axes.edgecolor": "#333333",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.dpi": 160, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
print(f"[plot] serif font in use: {_SERIF}")


def pct(a, p):
    return float(np.percentile(a, p))


def load():
    with open(DATA) as f:
        return json.load(f)


def error_figures(d):
    cor = d["correctness"]
    if not cor:
        print("WARN: no correctness data yet, skipping error figures")
        return
    rel_fp32 = np.array([c["rel_fp32"] for c in cor])
    rel_bc = np.array([c["rel_bf16_cuda"] for c in cor])
    rel_bt = np.array([c["rel_bf16_torch"] for c in cor])
    n = len(cor)

    # ---- table ----
    rows = []
    for name, arr in [("fp32 CUDA-vs-REF", rel_fp32),
                      ("bf16 CUDA-vs-truth", rel_bc),
                      ("bf16 torch-vs-truth", rel_bt)]:
        rows.append([name, f"{arr.max():.3e}", f"{arr.mean():.3e}",
                     f"{pct(arr,50):.3e}", f"{pct(arr,99):.3e}", f"{arr.std():.3e}"])
    with open("docs/ttt_error_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "max", "mean", "p50", "p99", "std"]); w.writerows(rows)
    with open("docs/ttt_error_table.md", "w") as f:
        f.write(f"# TTT-256 inference CUDA kernel — error distribution ({n} configs)\n\n")
        f.write("| metric | max | mean | p50 | p99 | std |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")

    # ---- figure: 3 panels ----
    from matplotlib.ticker import ScalarFormatter
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.subplots_adjust(wspace=0.28, top=0.80)

    def _sci(a):
        a.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))
        a.xaxis.get_offset_text().set_fontsize(9)

    # (a) fp32 CUDA-vs-REF histogram
    ax[0].hist(rel_fp32, bins=45, color=C_BLUE, alpha=0.88,
               edgecolor="white", linewidth=0.35)
    ax[0].axvline(rel_fp32.mean(), color=C_RED, lw=1.8, ls="--",
                  label=f"mean = {rel_fp32.mean():.2e}")
    ax[0].set_title("(a) fp32: CUDA vs. reference TTT layer")
    ax[0].set_xlabel("relative error"); ax[0].set_ylabel("count")
    ax[0].legend(frameon=False, loc="upper right")
    _sci(ax[0])

    # (b) bf16 CUDA vs torch, both vs fp32 truth — overlaid hist
    bins = np.linspace(min(rel_bc.min(), rel_bt.min()), max(rel_bc.max(), rel_bt.max()), 45)
    ax[1].hist(rel_bt, bins=bins, color=C_ORANGE, alpha=0.80, label="torch-TTT",
               edgecolor="white", linewidth=0.3)
    ax[1].hist(rel_bc, bins=bins, color=C_BLUE, alpha=0.68, label="CUDA-TTT (ours)",
               edgecolor="white", linewidth=0.3)
    ax[1].axvline(rel_bt.mean(), color=C_ORANGE, lw=1.8, ls="--")
    ax[1].axvline(rel_bc.mean(), color=C_BLUE, lw=1.8, ls="--")
    _ymax = ax[1].get_ylim()[1]
    ax[1].annotate(f"CUDA mean = {rel_bc.mean():.2e}", xy=(rel_bc.mean(), _ymax*0.38),
                   xytext=(rel_bc.mean() + (rel_bt.max()-rel_bc.mean())*0.30, _ymax*0.38),
                   ha="left", va="center", fontsize=9, color=C_BLUE,
                   arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))
    ax[1].annotate(f"torch mean = {rel_bt.mean():.2e}", xy=(rel_bt.mean(), _ymax*0.62),
                   xytext=(rel_bt.mean() + (rel_bt.max()-rel_bt.mean())*0.25, _ymax*0.62),
                   ha="left", va="center", fontsize=9, color="#C06A2B",
                   arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.0))
    ax[1].set_title("(b) bf16: error vs. fp32 ground truth\n(CUDA shifted left $\\Rightarrow$ more accurate)")
    ax[1].set_xlabel("relative error vs. fp32 truth"); ax[1].set_ylabel("count")
    ax[1].legend(frameon=False, loc="upper right")
    _sci(ax[1])

    # (c) box: the three distributions
    bp = ax[2].boxplot([rel_fp32, rel_bc, rel_bt], patch_artist=True, widths=0.62,
                       labels=["fp32\nCUDA-vs-REF", "bf16\nCUDA", "bf16\ntorch"],
                       medianprops=dict(color="black", lw=1.6),
                       whiskerprops=dict(color="#555555", lw=1.1),
                       capprops=dict(color="#555555", lw=1.1),
                       showfliers=True, flierprops=dict(marker="o", ms=2.2,
                       markerfacecolor="#999999", markeredgecolor="none", alpha=0.35))
    for patch, col in zip(bp["boxes"], [C_LBLUE, C_BLUE, C_ORANGE]):
        patch.set_facecolor(col); patch.set_alpha(0.88); patch.set_edgecolor("#333333"); patch.set_linewidth(1.0)
    ax[2].set_yscale("log")
    ax[2].set_title("(c) error distribution (log scale)")
    ax[2].set_ylabel("relative error")

    fig.suptitle(f"TTT-256 incremental inference — CUDA kernel correctness\n"
                 f"({n:,} configs: {d['meta'].get('total_seeds','?')} seeds × "
                 f"batch {d['meta']['batches']} × head-dim {d['meta']['head_dims']})",
                 fontsize=13.5, fontweight="bold", y=1.10)
    fig.savefig("docs/ttt_error_dist.png")
    plt.close(fig)
    print(f"wrote docs/ttt_error_dist.png + ttt_error_table.md/csv ({n} configs)")


def speed_figures(d):
    samples = d["speed_samples"]
    methods = ["attention", "torch_ttt", "cuda_ttt"]
    labels = {"attention": "attention", "torch_ttt": "torch-TTT-256", "cuda_ttt": "CUDA-TTT-256"}
    cols = {"attention": C_LBLUE, "torch_ttt": C_ORANGE, "cuda_ttt": C_BLUE}

    def agg(scene, key):
        # median across samples of each method's per-sample median
        out = {}
        for m in methods:
            vals = [s[scene][m]["median"] for s in samples if scene in s]
            out[m] = (float(np.median(vals)), float(np.std(vals)))
        return out

    A = agg("A_full", "median")
    B = agg("B_rollout", "median")

    # ---- table ----
    def wtable(path_md, path_csv):
        with open(path_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["scene", "method", "median_ms", "std_ms", "speedup_vs_attn"])
            for scene, agg_ in [("A_single_forward", A), ("B_256step_rollout", B)]:
                base = agg_["attention"][0]
                for m in methods:
                    med, sd = agg_[m]
                    w.writerow([scene, labels[m], f"{med:.3f}", f"{sd:.3f}", f"{base/med:.2f}"])
        with open(path_md, "w") as f:
            f.write(f"# TTT-256 single-layer inference speed ({len(samples)} samples)\n\n")
            for title, agg_ in [("Scene A: single full-prefix forward (L=256)", A),
                                ("Scene B: 256-step autoregressive rollout (deployment)", B)]:
                base = agg_["attention"][0]
                f.write(f"## {title}\n\n| method | median (ms) | std | speedup vs attn |\n|---|---|---|---|\n")
                for m in methods:
                    med, sd = agg_[m]
                    f.write(f"| {labels[m]} | {med:.3f} | {sd:.3f} | {base/med:.2f}x |\n")
                f.write("\n")
    wtable("docs/ttt_speed_table.md", "docs/ttt_speed_table.csv")

    # ---- figure: 2 bar panels (A linear /ms, B log) ----
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(wspace=0.26)
    scenes = [("A", A, "(a) single forward (L = 256)", False),
              ("B", B, "(b) 256-step AR rollout (deployment)", False)]
    for k, (scene, agg_, ttl, use_log) in enumerate(scenes):
        meds = [agg_[m][0] for m in methods]
        sds = [agg_[m][1] for m in methods]
        bars = ax[k].bar([labels[m] for m in methods], meds, yerr=sds, capsize=5,
                         color=[cols[m] for m in methods], alpha=0.92,
                         edgecolor="#333333", linewidth=0.9,
                         error_kw=dict(ecolor="#333333", lw=1.1, capthick=1.1))
        base = agg_["attention"][0]
        if use_log:
            ax[k].set_yscale("log")
            ax[k].set_ylabel("latency / ms (log scale)")
            ax[k].set_ylim(top=max(meds) * 3.2)
        else:
            ax[k].set_ylabel("latency / ms")
            # linear: leave generous headroom so labels clear the error bars
            ax[k].set_ylim(top=(max(m + s for m, s in zip(meds, sds))) * 1.32)
        ax[k].set_title(ttl)
        ytop = ax[k].get_ylim()[1]
        for b, m, sd in zip(bars, methods, sds):
            med = agg_[m][0]
            sp = base / med
            win = sp >= 1.0 and m != "attention"
            txt = f"{med:.2f} ms\n{sp:.2f}$\\times$"
            # place label above the bar AND its error bar, with a small gap
            if use_log:
                ypos = (med + sd) * 1.18
            else:
                ypos = med + sd + ytop * 0.04
            ax[k].text(b.get_x() + b.get_width()/2, ypos, txt,
                       ha="center", va="bottom", fontsize=10,
                       fontweight="bold" if win else "normal",
                       color=C_RED if win else "#222222")
        for lbl in ax[k].get_xticklabels():
            lbl.set_rotation(10)
        ax[k].grid(axis="x", visible=False)
    fig.suptitle("TTT-256 inference latency: attention vs. torch-TTT vs. CUDA-TTT (ours)",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.savefig("docs/ttt_speed.png")
    plt.close(fig)
    print(f"wrote docs/ttt_speed.png + ttt_speed_table.md/csv ({len(samples)} samples)")


if __name__ == "__main__":
    d = load()
    error_figures(d)
    speed_figures(d)
    print("all figures + tables written to docs/")
