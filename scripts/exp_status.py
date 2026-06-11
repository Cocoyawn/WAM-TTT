#!/usr/bin/env python
"""VLANeXt experiment status board — scan logs/checkpoints, print a unified progress table.

Used by the `vlanext-experiments` skill (and runnable standalone). Read-only: never launches
or mutates anything. Covers: chunk/suite training runs, env-gen eval shards, GDN retrain.
"""
import os, re, glob, subprocess, yaml
from collections import defaultdict

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
os.chdir(REPO)

def sh(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""

def alive(pat):
    return int(sh(f"ps -eo cmd | grep -F '{pat}' | grep -v grep | wc -l") or 0)

def last_loss(logpath):
    ls = re.findall(r"loss=([0-9.]+)", sh(f"grep -oE 'loss=[0-9.]+' {logpath} 2>/dev/null | tail -1"))
    return ls[-1] if ls else None

def eval_progress(prefix, nshards):
    """Return (done_tasks, total_tasks, running_shards) for an env-gen eval family."""
    done = tot = run = 0
    for s in range(nshards):
        cfg = f"config/libero_plus_envgen_{prefix}_shard{s}.yaml"
        log = f"logs/plus_envgen_{prefix}_shard{s}.log"
        if not os.path.exists(cfg): continue
        try: n_tot = len(yaml.safe_load(open(cfg))["eval"]["task_ids"])
        except Exception: n_tot = 0
        n_done = int(sh(f"grep -c 'Current total success rate' {log}") or 0)
        tot += n_tot; done += min(n_done, n_tot)
        if alive(f"{prefix}_shard{s}"): run += 1
    return done, tot, run

def main():
    print("="*70)
    print("VLANeXt EXPERIMENT STATUS")
    print("="*70)

    # --- disk + RAM ---
    disk = sh("df -h /mnt/afs-h200 | tail -1 | awk '{print $4\" free (\"$5\" used)\"}'")
    lim = sh("cat /sys/fs/cgroup/memory/memory.limit_in_bytes")
    use = sh("cat /sys/fs/cgroup/memory/memory.usage_in_bytes")
    ram = f"{int(use)/1e9:.0f}/{int(lim)/1e9:.0f} GB" if lim and use else "?"
    print(f"Disk: {disk}   |   Host RAM (cgroup): {ram}")
    gpu = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
    print("GPU:")
    for ln in gpu.splitlines(): print(f"  {ln}")

    # --- env-gen evals ---
    print("\n-- env-gen eval --")
    for prefix, n, label in [("ttt16",12,"TTT-16"), ("attn",13,"Attention"), ("ttt256",8,"TTT-256")]:
        d, t, r = eval_progress(prefix, n)
        if t == 0: continue
        status = "DONE" if (r==0 and d>=t) else f"{r} running"
        print(f"  {label:10s}: {d}/{t} tasks ({100*d/max(t,1):.0f}%)  [{status}]")

    # --- training runs ---
    print("\n-- training --")
    for log, label, pat in [
        ("logs/ablation_wm_gdn_retrain.log", "GDN retrain", "ablation_wm_gdn"),
        ("logs/ablation_wm_ttt_chunk256.log", "TTT-256 spatial", "chunk256"),
    ]:
        if not os.path.exists(log): continue
        loss = last_loss(log); a = alive(pat)
        step = sh(f"grep -oE 'step=[0-9]+|([0-9]+)/30000' {log} | tail -1")
        print(f"  {label:18s}: loss={loss}  {'ALIVE' if a else 'stopped'}  {step}")

    # cross-suite TTT-256 training dirs (HP tasks)
    for suite in ["object","goal","10","long"]:
        d = glob.glob(f"VLANeXt_ablation_wm/*ttt*chunk256*{suite}*") + glob.glob(f"VLANeXt_ablation_wm/*ttt256*{suite}*")
        if d:
            fin = any(os.path.exists(os.path.join(x,"checkpoint_final.pt")) for x in d)
            print(f"  TTT-256 {suite:8s}: {'DONE (final ckpt)' if fin else 'training/partial'}  {d[0]}")

    print("\n(see EXPERIMENT_PLAN_2days.md for full todo list)")

if __name__ == "__main__":
    main()
