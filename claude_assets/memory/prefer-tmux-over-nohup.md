---
name: prefer-tmux-over-nohup
description: "For background/long-running jobs, launch inside a tmux session rather than nohup"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 20209452-ac23-402f-b63a-11a0121f5282
---

When starting background or long-running jobs (training schedulers, daemons, watchers, long evals),
prefer launching inside a **tmux session** over `nohup ... &`.

**Why:** tmux sessions are reattachable, observable live, and survive parent/terminal exit cleanly;
nohup jobs are fire-and-forget — hard to inspect, easy to lose track of, and their output only lives
in a redirected file.

**How to apply:**
```bash
tmux new-session -d -s <name> -c <workdir>
tmux send-keys -t <name> "<command> 2>&1 | tee <logfile>" Enter
# observe:  tmux attach -t <name>   (Ctrl-b d to detach)  |  tmux capture-pane -t <name> -p
# stop:     tmux kill-session -t <name>
```
Use a descriptive session name. `tee` keeps a log file too so output is greppable without attaching.

Applied to: the VLANeXt training scheduler runs in tmux session `vlanext-sched` (see the
`vlanext-scheduler` skill). Related: [[vlanext-compute-resource-limits]].
