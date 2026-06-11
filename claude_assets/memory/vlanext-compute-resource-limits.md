---
name: vlanext-compute-resource-limits
description: "Compute resource hard limits (disk + CPU-RAM cgroup + GPU) that gate VLANeXt training/eval jobs"
metadata:
  node_type: memory
  type: project
  originSessionId: 096b4910-03a6-4e31-a8a3-426c2bb29e23
---

Hard resource limits on the H200 box that MUST be checked before launching any VLANeXt training/eval.
These have ALREADY caused silent failures (disk-full blocking training, cgroup OOM killing eval workers).

## DISK — /mnt/afs-h200 is a 20T quarkfs, shared, fills FAST (learned 2026-06-10 the hard way)
- **A single WM checkpoint = ~20GB** (Qwen3-VL-2B backbone + experts). save_interval=2000 → 15 ckpts = ~300GB
  PER RUN. This is what filled the disk to 100% and blocked GDN training entirely.
- **RULE: set save_interval=10000** (3 intermediate + final, ~80GB/run), NOT the default 2000. The actcausal
  run left 16×19GB=302GB; GDN-run-1 + chunk runs each 76GB.
- **Disk-full = SILENT training failure**: `OSError: [Errno 28] No space left on device` on checkpoint save /
  even pytest __pycache__. Always `df -h /mnt/afs-h200` BEFORE launching; if <100GB free, clean first.
- **quarkfs reclaim is DELAYED**: after `rm`, `df` can still show old usage for tens of seconds to minutes.
  Verify with an actual write test (`python -c "open('x','wb').write(b'0'*1e9)"`), not just df.
- **Cleanup playbook** (frees ~1TB, done 2026-06-10): (a) intermediate training ckpts — keep final+30000,
  delete 10000/20000 etc; (b) optimizer STATE dirs (e.g. Little-WAM training/outputs/*/checkpoints/state/,
  12GB each, only needed to RESUME a stopped run) — biggest win, ~600GB; (c) diverged-run dirs entirely.
  NOTE Little-WAM is a SEPARATE project (Wan world-model, train stopped 2026-06-04) — its 938GB→64GB cleanup
  was user-authorized; don't re-delete what's already gone. workplace/data is a symlink to ~/data (don't double-count).
- Snapshot 2026-06-10 after cleanup: 2.3T free (89% used). Big consumers: ~/data 585G, VLANeXt_ablation_wm 152G,
  VLANeXt_final_libero 110G (KEEP — final results), Little-WAM 64G.

## CPU-RAM (cgroup) — the REAL bottleneck for env-gen eval, NOT GPU (see [[vlanext-ttt-experiment-status]])
- **cgroup memory.limit = 416GB** (`/sys/fs/cgroup/memory/memory.limit_in_bytes`; `free -g` LIES, shows machine 1TB).
  NOTE: observed value drifted to ~446GB on 2026-06-10 — always re-read the cgroup file, don't hardcode.
- libero-plus env-gen eval is CPU-render bound (OSMesa single-thread): each worker ~1 core + ~10GB GPU +
  **~22GB host RSS**, renders serially. So GPU mem is NEVER the limit for eval parallelism — host RAM is.
- **Exceeding cgroup = SILENT OOM-kill** (workers vanish, no traceback; check `dmesg | grep -i "cgroup out of memory"`).
- **Safe: 8 eval workers (~176GB) + 2 trainings (~176GB) ≈ 352GB, ~64GB headroom.** 12 workers overflows → killed.
  Budget: `max_workers ≈ (cgroup_limit − train_RSS) / 22`. Stagger same-GPU launches ~30-45s (sim loads spike+kill).

## GPU — H200 80GB/card, 4 cards. Rarely the limit.
- Each eval worker ~10GB; each 2-GPU-DDP WM training ~41GB/card. "Available GPU" = can still FIT a job
  (training can SHARE a card with eval). Target: keep all 4 cards >50GB used when work is queued.
- STANDING RULES (from [[vlanext-ttt-experiment-status]]): training = 2-GPU DDP (torchrun --nproc_per_node=2,
  batch_size in config is TOTAL across GPUs); env-gen eval = 4-GPU sharded; vision TTT chunk_size=256.

## Launch gotchas that waste a cycle each
- **PYTHONPATH**: training (scripts/train.py) AND eval (libero_plus_bench_eval.py) both die without it:
  training → `ModuleNotFoundError: No module named 'src'` (need `export PYTHONPATH=$REPO`);
  eval → `No module named 'libero'` (need `PYTHONPATH=$REPO/third_party/LIBERO-plus:$REPO` AND cd into LIBERO-plus).
- **max_steps vs warmup_steps live in DIFFERENT config sections**: train.py reads `data.max_steps`
  (train.py:579/611/614) but `train.warmup_steps` (train.py:578). Setting `train.max_steps` is a SILENT no-op
  (default-template has max_steps under `data`, not `train`). Bug caught 2026-06-10: a cross-suite config had
  long's 36k written to train.max_steps → training would've run only 30k from the stale data.max_steps.
  When scripting config edits, set `cfg['data']['max_steps']` + `cfg['train']['warmup_steps']`.
- env vars every run needs: TORCHDYNAMO_DISABLE=1, (eval) MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg.

Related: [[vlanext-ttt-experiment-status]] (standing rules), [[vlanext-gla-causal-plan]] (GDN run gated on disk).
