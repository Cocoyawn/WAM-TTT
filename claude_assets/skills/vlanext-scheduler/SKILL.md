---
name: vlanext-scheduler
description: Manage the VLANeXt job scheduler — a concurrent, fault-tolerant GPU orchestrator (scripts/scheduler.py) for BOTH training and eval jobs. Use when the user asks about the scheduler/调度器/调度系统, "what's queued", "scheduler status", to start/stop/restart it, add/reorder queued trainings or evals, check why a job is waiting, or diagnose a failed/retried job. ALSO use whenever launching ANY new training or eval task — prefer adding it to the scheduler queue over ad-hoc nohup/tmux launches. Working dir /mnt/afs-h200/yuyangcheng/workplace/VLANeXt.
---

# VLANeXt job scheduler — the unified resource-management entrypoint

**This is the primary way to run trainings AND evals on this box.** When a new training or eval
task needs to run, prefer EXTENDING the scheduler queue over launching it ad-hoc. The scheduler is
the single place that knows GPU/RAM availability, so it prevents the OOM / port-collision / silent-skip
failures that ad-hoc launches kept causing. (User directive 2026-06-10: "this is our entrypoint for
managing the powerful system resources.")

A Python daemon that auto-launches queued jobs as resources free up — concurrent (packs multiple jobs
onto the 4 cards), fault-tolerant (detects crashes, retries), with a dual GPU+RAM resource gate.
Handles TWO job kinds:
- **train**: 2-GPU DDP torchrun; done = checkpoint_final.pt; ~85GB RAM/job.
- **eval**: N single-GPU sharded workers (env-gen); done = all shards produced `*_SR*` dirs;
  ~22GB RAM/worker, count auto-scales to available RAM (cap EVAL_MAX_WORKERS=8). Workers co-reside on
  cards (~10GB each) and ELASTICALLY GROW: when a training finishes and frees RAM, the next poll adds
  more eval workers automatically.

Repo: `/mnt/afs-h200/yuyangcheng/workplace/VLANeXt`
Daemon: `scripts/scheduler.py` ｜ Venv: `/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python`
State: `.claude/scheduler_state.json` ｜ Log: `logs/scheduler.log` ｜ Stop signal: `.scheduler_stop`
**Runs inside tmux session `vlanext-sched`** (user prefers tmux over nohup for background jobs).

## Commands

```bash
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
PY=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python

# STATUS — queue + per-job status (done/running/queued/failed), gpus, pid, retries
$PY scripts/scheduler.py --status

# IS IT ALIVE? (process + tmux session)
ps -eo pid,etimes,cmd | grep scheduler.py | grep -v grep
tmux ls | grep vlanext-sched

# WATCH live (attach to the tmux session; Ctrl-b d to detach)
tmux attach -t vlanext-sched
# ...or peek without attaching:
tmux capture-pane -t vlanext-sched -p | tail -20
tail -f logs/scheduler.log

# START in tmux (preferred over nohup — survives, reattachable, observable)
rm -f .scheduler_stop
tmux new-session -d -s vlanext-sched -c /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
tmux send-keys -t vlanext-sched "$PY scripts/scheduler.py 2>&1 | tee logs/scheduler.log" Enter

# STOP gracefully (running trainings KEEP going; only the watcher exits)
touch .scheduler_stop          # picked up within one poll (~2min)
# then kill the tmux session once it exits:
tmux kill-session -t vlanext-sched

# RESTART (safe — state file + checkpoint_final.pt make it idempotent; won't rerun done jobs)
touch .scheduler_stop; sleep 130
tmux kill-session -t vlanext-sched 2>/dev/null
rm -f .scheduler_stop
tmux new-session -d -s vlanext-sched -c /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
tmux send-keys -t vlanext-sched "$PY scripts/scheduler.py 2>&1 | tee logs/scheduler.log" Enter
```

NOTE: trainings themselves are launched by the scheduler via subprocess (not tmux); they survive
the scheduler restart because they're separate process groups. Only the watcher lives in tmux.

## How it works (logic)
Every `POLL_SEC` (120s): (1) reap running jobs — a job with checkpoint_final.pt → done; a
dead pid without final ckpt → retry (up to MAX_RETRIES=2) else `failed`; (2) compute GPUs
LEASED by our own running jobs; (3) **concurrently** launch as many queued experiments as fit:
each needs `GPUS_PER_JOB=2` cards with ≥`MIN_GPU_FREE_MIB=30000` free (not leased) AND
RAM ≥ `RAM_PER_JOB_GB+RAM_SAFETY_GB` (85+20=105 GB). Exits when all jobs are done/failed.
RAM (cgroup ~447GB) is the binding constraint — see [[vlanext-compute-resource-limits]].

## The queue (priority order, edit the `QUEUE` list in scripts/scheduler.py)
Entries are dicts tagged by `kind`. To **add a job**, append a dict and restart the daemon
(running jobs are untouched; restart re-reads the queue; done jobs are skipped via state + ckpt/_SR).

- **train**: `{"kind":"train", "tag":..., "cfg":"<config-basename>", "save_dir":"<ablation-subdir>"}`
  — needs a ready `config/<cfg>.yaml` (remember `save_interval: 10000`). `save_dir` = train.py's
  project.name split on '_', parts[3:] joined.
- **eval**: `{"kind":"eval", "tag":..., "prefix":"<X>", "nshards":8, "result_dir":"<ablation-subdir>", "shard_glob":"envgen_<X>_shard"}`
  — needs `config/libero_plus_envgen_<X>_shard{0..nshards-1}.yaml` (generate by cloning an existing
  shard set, swapping the ckpt path + shard_tag; see how ttt64/ttt256 configs were made). done = all
  shards produced `*_SR*` dirs under `VLANeXt_ablation_wm/<result_dir>/`.

**Workflow rule (user directive):** when a NEW training or eval task is written, the default is to add
it to this QUEUE — that's the entrypoint for managing GPU/RAM. Only launch ad-hoc (tmux/nohup) for
quick one-offs that don't compete for scarce resources. If the scheduler doesn't yet support a job
shape, EXTEND scheduler.py (as eval support was added 2026-06-10) rather than bypassing it.

To **reorder/reprioritize**: move dicts in QUEUE and restart. To **drop** a job (e.g. a negative-result
run): remove its dict AND delete its stale entry from .claude/scheduler_state.json before restart, else
the reaper may try to manage a job no longer in the queue.

## Reading status
- `running` + `gpus`/`pid`: live; check loss via `vlanext-plot-loss` skill or `grep loss= logs/<config>.log`.
- `queued`: waiting for the resource gate — check `--status` retries and `logs/scheduler.log` for the gate values.
- `failed`: exhausted retries — inspect `logs/<config>.log` for `OutOfMemoryError|ChildFailedError|Traceback`,
  fix (config / resources), then restart the daemon to requeue it.
- `STALL` line in the log = nothing could start for 1h (resources stuck); investigate what's holding GPU/RAM.

## Tuning knobs (top of scripts/scheduler.py)
MIN_GPU_FREE_MIB, RAM_PER_JOB_GB, RAM_SAFETY_GB, GPUS_PER_JOB, POLL_SEC, MAX_RETRIES, STALL_WARN_SEC.
Change for differently-sized jobs (e.g. a 1-GPU eval would use GPUS_PER_JOB=1, smaller RAM).

## Related skills
- `vlanext-experiments` — overall experiment status board / launch one-off.
- `vlanext-plot-loss` — plot loss of a running queued job.
- `vlanext-wandb` — wandb links for finished runs.
