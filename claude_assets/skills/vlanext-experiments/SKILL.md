---
name: vlanext-experiments
description: Manage VLANeXt ablation experiments — read the todo list, show current progress of all running/done training & env-gen eval jobs, and launch new experiments. Use when the user asks about VLANeXt experiment status, "what's running", "show progress", "launch/start a training or eval", TTT/GDN/chunk/env-gen experiments, or wants the experiment todo list. Working dir /mnt/afs-h200/yuyangcheng/workplace/VLANeXt.
---

# VLANeXt Experiments

Orchestrate the VLANeXt TTT/GDN ablation experiment suite: **read todos → show progress → launch experiments**.

Repo: `/mnt/afs-h200/yuyangcheng/workplace/VLANeXt`
Venv: `/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python`
Background memory: `vlanext-ttt-experiment-status`, `vlanext-compute-resource-limits`, `vlanext-gla-causal-plan`, `vlanext-envgen-ttt-vs-attn-results`.

## Modes (pick based on the user's request)

### 1. Show status / progress  (default — "进度", "what's running", "status")
Run the read-only status board:
```bash
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python scripts/exp_status.py
```
It prints disk/RAM/GPU + every env-gen eval family (TTT-16/Attention/TTT-256) + training runs (GDN, cross-suite).
Then summarize in a table and call out anything important (divergence, completion, disk/RAM near limit).

### 2. Show todo list  ("待办", "todo", "what's next")
Read the master list (two sources, keep them in sync):
- `EXPERIMENT_PLAN_2days.md` — full plan, the "🔴 高优先级" section is current high-pri (TTT-256 cross-suite).
- memory `vlanext-ttt-experiment-status` "Pending todos (rewritten 2026-06-10)".
Present DONE / IN-PROGRESS / HIGH-PRIORITY / OTHER. Surface the open decision points.

### 3. Launch an experiment  ("启动", "跑", "train", "eval")
**Always check resources first** (`exp_status.py` shows disk/RAM/GPU) — see `vlanext-compute-resource-limits`:
- Disk: each WM ckpt ~20GB; **set `save_interval: 10000`** in any new training config (default 2000 fills disk).
- RAM (cgroup ~447GB) is the real limit: ≤ 8 env-gen workers + 2 trainings. Stagger same-GPU launches ~40s.
- GPU: training = **2-GPU DDP** (`torchrun --nproc_per_node=2`); env-gen eval = sharded (8-way), chunk256 ckpt.
- **Every launch needs `export PYTHONPATH=$REPO`** (training) or `PYTHONPATH=$REPO/third_party/LIBERO-plus:$REPO` + cd into LIBERO-plus (eval) — else `ModuleNotFoundError`.

Existing launch scripts to reuse / copy:
| Goal | Script |
|---|---|
| GDN retrain (2-GPU) | `scripts/run_gdn_retrain.sh` |
| TTT-256 env-gen (8-way) | `scripts/run_ttt256_envgen.sh` |
| Attention env-gen catch-up | `scripts/run_attn_envgen_catchup.sh` |
| Aggregate env-gen → per-dim table | `scripts/aggregate_envgen.py` |

**TTT-256 cross-suite training (current high-pri HP-1/2/3):** copy `config/ablation_wm_ttt_chunk256.yaml`,
change `task_suite_name` (→ `libero_object_no_noops` / `libero_goal_no_noops` / `libero_10_no_noops`),
`project.name`, output subdir, and `save_interval: 10000`. Data at `~/data/LIBERO_modified/`. Launch with a
2-GPU torchrun like `run_gdn_retrain.sh` (swap the `--config`). ~6h/run.

After launching: wait for model load (~3min), verify no error in the log, confirm GPU/RAM moved, estimate ETA
from actual s/task, and report back. Never claim "launched" without verifying the process is alive and progressing.

### 4. Aggregate results  ("出表", "对比", "aggregate")
```bash
/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python scripts/aggregate_envgen.py
```
Handles the 3 pitfalls (truncated mp4 names / dup category_stats / restart-appended logs). Extend its
`collect()` calls to add a new model family (e.g. ttt256) before aggregating 3-way.

## Guardrails
- Read-only status/todo modes never mutate anything — safe to run anytime.
- Before deleting checkpoints to free disk, follow the cleanup playbook in `vlanext-compute-resource-limits`
  (keep final+30000; optimizer `state/` dirs are the biggest safe win). Confirm with user before mass deletion.
- Do not exceed the RAM-bound parallelism (8 eval workers + 2 trainings). Check `exp_status.py` first.
