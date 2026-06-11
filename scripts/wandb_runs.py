#!/usr/bin/env python
"""VLANeXt wandb run index — scan training logs, print each run's wandb URL + live loss + status.

Read-only. Used by the `vlanext-wandb` skill (and runnable standalone). Dynamically discovers every
run from logs/*.log, so new training runs appear automatically with no hardcoding.
"""
import os, re, glob, subprocess

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
os.chdir(REPO)

def sh(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""

URL_RE = re.compile(r"https://wandb\.ai/[^\s]+/runs/[a-z0-9]+")

# friendly label + known notes per log basename
NOTES = {
    "ablation_wm_gdn.log":          ("GDN run-1",        "DIVERGED (loss→14.5), superseded"),
    "ablation_wm_gdn_retrain.log":  ("GDN retrain",      "lr 5e-5 fix, CURRENT"),
    "ablation_wm_ttt_actcausal.log":("TTT action-causal","causality probe"),
    "ablation_wm_ttt_chunk256.log": ("TTT-256 spatial",  "vision chunk256"),
    "ablation_wm_ttt_chunk64.log":  ("TTT-64 spatial",   "vision chunk64"),
    "ablation_wm_ttt_chunk4.log":   ("TTT-4 spatial",    "vision chunk4"),
}

def fresh(logpath):
    """A run is 'live' if its log was written in the last ~3 min (more reliable than ps,
    since run-1 and retrain share the same --config name)."""
    mt = sh(f"stat -c %Y {logpath}")
    now = sh("date +%s")
    try: return (int(now) - int(mt)) < 180
    except Exception: return False

def main():
    print("="*78)
    print("VLANeXt WANDB RUNS  (project: cocoyawn2035-tsinghua-university/VLANeXt_ablation_wm)")
    print("="*78)
    rows = []
    for log in sorted(glob.glob("logs/*.log")):
        url = ""
        try:
            for ln in open(log, errors="ignore"):
                m = URL_RE.search(ln)
                if m: url = m.group(0); break
        except Exception:
            pass
        if not url:
            continue
        base = os.path.basename(log)
        label, note = NOTES.get(base, (base.replace(".log",""), ""))
        loss = (re.findall(r"loss=([0-9.]+)", sh(f"grep -oE 'loss=[0-9.]+' {log} | tail -1")) or [None])[0]
        live = fresh(log)
        rows.append((label, url, loss, "LIVE" if live else "done", note))

    w = max((len(r[0]) for r in rows), default=10)
    for label, url, loss, st, note in rows:
        print(f"\n  {label:<{w}}  [{st}]  loss={loss}   {note}")
        print(f"  {'':<{w}}  {url}")
    print("\n(open a URL in browser for the live loss/lr/grad_norm curves)")

if __name__ == "__main__":
    main()
