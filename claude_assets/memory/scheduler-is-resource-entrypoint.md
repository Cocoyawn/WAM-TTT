---
name: scheduler-is-resource-entrypoint
description: The VLANeXt job scheduler is the default entrypoint for running any training/eval; extend it rather than launching ad-hoc
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 20209452-ac23-402f-b63a-11a0121f5282
---

When a new training or evaluation task needs to run on the H200 box, the default is to add it to the
**VLANeXt job scheduler** (`scripts/scheduler.py`, managed by the `vlanext-scheduler` skill) — NOT to
launch it ad-hoc with nohup/tmux.

**Why** (user directive 2026-06-10): "this is our entrypoint for managing the powerful system resources."
The scheduler is the single place that tracks GPU free-memory + cgroup RAM, so routing all jobs through
it prevents the failures ad-hoc launches kept causing: cgroup-OOM on full cards, torchrun master_port
collisions, and silently-skipped experiments. It packs jobs concurrently and retries crashes.

**How to apply:**
- Adding a job = append a dict to the `QUEUE` list in scheduler.py and restart the tmux daemon
  (`vlanext-sched`). Two kinds: `train` (2-GPU DDP) and `eval` (N sharded single-GPU workers,
  RAM-elastic). See the `vlanext-scheduler` skill for exact dict shapes.
- If the scheduler doesn't yet support a new job SHAPE (e.g. eval support was added 2026-06-10,
  a 1-GPU job, a multi-node job), **EXTEND scheduler.py** to handle that kind — don't bypass it.
- Reserve ad-hoc tmux/nohup launches for trivial one-offs that don't compete for scarce GPU/RAM.

Related: [[prefer-tmux-over-nohup]] (the scheduler itself runs in tmux), [[vlanext-compute-resource-limits]]
(the GPU/RAM limits the scheduler enforces).
