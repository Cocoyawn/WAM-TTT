#!/usr/bin/env python
"""Render VLANeXt training loss curves as ASCII — no wandb / browser needed.

Reads loss from logs/*.log (each line has `loss=<float>`; DDP logs each step ~2x, auto-deduped
by sampling). Works while a run is still training (wandb hides history for running runs — this
is the immediate alternative). Supports overlaying multiple runs for divergence comparison.

Usage:
  python scripts/plot_loss.py                       # list available runs
  python scripts/plot_loss.py gdn_retrain           # one run
  python scripts/plot_loss.py gdn gdn_retrain       # compare (diverged vs fixed)
  python scripts/plot_loss.py --rows 30 ttt_chunk256 # taller plot
Run keys are matched as substrings of the log basename (logs/ablation_wm_<key>.log).
"""
import sys, os, re, glob

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
os.chdir(REPO)
LOGDIR = "logs"

def find_logs(key):
    # match logs/*<key>*.log, prefer ablation_wm_ prefix
    cands = sorted(glob.glob(f"{LOGDIR}/*{key}*.log"))
    return cands

def load_losses(logpath):
    vals = []
    try:
        for ln in open(logpath, errors="ignore"):
            m = re.search(r"loss=([0-9]+\.[0-9]+)", ln)
            if m: vals.append(float(m.group(1)))
    except FileNotFoundError:
        pass
    # DDP prints each step on both ranks -> collapse consecutive exact dups
    out = []
    for v in vals:
        if not out or out[-1] != v: out.append(v)
    return out

def sample(vals, n):
    if len(vals) <= n: return list(enumerate(vals))
    step = len(vals) / n
    return [(int(i*step), vals[int(i*step)]) for i in range(n)]

def plot_one(vals, label, rows, width=46, scale=None):
    """scale=(lo,hi) forces a shared y-axis so multiple curves are directly comparable.
    If None, auto-scales to this run's own min/max."""
    if not vals:
        print(f"[{label}] no loss data"); return
    pts = sample(vals, 24)
    if scale:
        lo, hi = scale
    else:
        lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    tag = "shared-scale" if scale else "auto-scale"
    print(f"\n=== {label} ===  ({len(vals)} pts, loss {vals[0]:.2f} -> {vals[-1]:.2f}, min {min(vals):.2f})  [{tag} {lo:.1f}-{hi:.1f}]")
    for idx, v in pts:
        bar = "#" * max(0, int((v - lo) / span * width))
        print(f"  {idx:>6}  {v:6.2f}  {bar}")

def plot_compare(series, rows, width=40):
    """series = [(label, vals)]. Output: (1) each full curve on a SHARED y-scale so they're
    directly comparable side by side, then (2) an overlaid A/B scatter for shape-at-a-glance."""
    allv = [v for _, vs in series for v in vs]
    if not allv:
        print("no data"); return
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0
    markers = "ABCDEFGH"

    # (1) each curve, full, same scale
    print("\n" + "="*60)
    print(f"INDIVIDUAL CURVES (shared y-scale {lo:.1f}-{hi:.1f}, bars directly comparable)")
    print("="*60)
    for label, vs in series:
        plot_one(vs, label, rows, scale=(lo, hi))

    # (2) overlaid scatter
    N = 26
    print("\n" + "="*60)
    print("OVERLAY  (left=low loss, right=high; top→bottom = progress 0→100%)")
    print("="*60)
    for i,(label, vs) in enumerate(series):
        print(f"  [{markers[i]}] {label}  (min {min(vs):.2f}, final {vs[-1]:.2f})")
    print()
    for r in range(N):
        frac = r / (N-1)
        line = [" "] * (width+1)
        for i,(label, vs) in enumerate(series):
            j = int(frac * (len(vs)-1))
            col = int((vs[j] - lo) / span * width)
            line[col] = markers[i]
        print(f"  {int(frac*100):3d}%  " + "".join(line))

def main():
    args = [a for a in sys.argv[1:]]
    rows = 24
    if "--rows" in args:
        k = args.index("--rows"); rows = int(args[k+1]); del args[k:k+2]

    if not args:
        print("Available training runs (logs/*.log with loss):")
        for lg in sorted(glob.glob(f"{LOGDIR}/*.log")):
            vs = load_losses(lg)
            if vs:
                print(f"  {os.path.basename(lg)[:-4]:35s} {len(vs)} pts, final loss {vs[-1]:.2f}")
        print("\nUsage: plot_loss.py <key> [<key2> ...]  (key = substring of log name)")
        return

    series = []
    for key in args:
        logs = find_logs(key)
        if not logs:
            print(f"[warn] no log matches '{key}'"); continue
        lg = logs[0]
        if len(logs) > 1:
            print(f"[note] '{key}' matched {len(logs)} logs, using {os.path.basename(lg)}")
        series.append((os.path.basename(lg)[:-4], load_losses(lg)))

    if len(series) == 1:
        plot_one(series[0][1], series[0][0], rows)
    elif len(series) > 1:
        plot_compare(series, rows)   # outputs each curve (shared scale) + overlay

if __name__ == "__main__":
    main()
