---
name: vlanext-ttt-experiment-status
description: "VLANeXt TTT (LaCT fast-weight) ablation — current progress, running jobs, and pending todos"
metadata: 
  node_type: memory
  type: project
  originSessionId: 096b4910-03a6-4e31-a8a3-426c2bb29e23
---

VLANeXt ablation: replace cross-attention with TTT (LaCT fast-weight SwiGLU MLP) in
BOTH action expert (bidirectional) and vision expert (causal), [A,A,A,T] interleave
(every 4th of 29 layers), "Method B" semantic-equal replacement (TTT update reads
VLM hidden state exactly as the attention KV did). World-model mode (future_image_loss=1.0).
Repo: /mnt/afs-h200/yuyangcheng/workplace/VLANeXt. Isolated venv: /mnt/afs-h200/yuyangcheng/venvs/fla_triton32.

## Results so far (LIBERO-spatial base, 10 task ×50, world-model)
- **TTT chunk16 (16 blocks): 97.80%** (489/500) — trained 30k + evaluated.
- **Attention baseline (spatial_planB_wu4500): 97.60%** (488/500).
- Conclusion: TTT is a clean drop-in replacement for cross-attention, no accuracy loss.

## Speed/memory (controlled, see [[ttt-chunk-size-train-speed]])
TTT is NOT faster/lighter: ~1.29× slower per step (compile ON), memory equal (33.4 vs 33.3 GB).
Earlier "TTT faster" was a 1-GPU-vs-2-GPU wall-clock confusion. TTT's value = accuracy-parity, not speed.

## Video-gen quality verified (2026-06-07): low loss = better IN-DIST future-frame gen (generalization TBD at scale)
Q: TTT 训练 loss 比 attention 低很多 → 视频生成更好吗？ Built scripts/{cache_wm_eval_samples,eval_wm_quality,
summarize_wm_quality}.py + run_wm_quality_eval.sh. Shared cached eval set (same inputs both ckpts),
30k-final TTT-16 vs attention. Layer A = ISOLATED teacher-forced image-token CE + top-1 acc;
Layer B = free-running greedy AR predict_image → Emu3.5 VQ decode → PSNR/SSIM/LPIPS + GT|pred grids.
WM predicts a SINGLE future frame (horizon t+8), not a video sequence. Results (spatial=in-dist, object/goal=OOD):
- **spatial: TTT wins everywhere** — loss_img 2.05 vs 3.10, tok-acc .68 vs .45; PSNR 21.56 vs 20.19,
  SSIM .790 vs .750, LPIPS .130 vs .185. Low-loss advantage is REAL (lives in vision branch, not action —
  action SR already tied) AND transfers to free-running gen (not a teacher-forcing artifact).
- **OOD: TTT advantage gone** (object slightly worse, goal small lead); grids show BOTH models hallucinate
  the spatial scene on OOD inputs. BUT do NOT conclude "TTT doesn't generalize" — only 3 suites, single-suite
  training, small n. User says generalization claim needs SCALE; large-scale LIBERO-plus env-gen eval running
  on separate compute → revisit when done (see env-gen todos below).
- **Firm conclusion: TTT lower loss ⇒ better IN-DISTRIBUTION future-frame gen. OOD generalization = TBD at scale.**
- Output: VLANeXt_ablation_wm/wm_eval_cache/WM_QUALITY_SUMMARY.md, grids/{suite}_{ttt,attn}.png (GT|pred),
  grids/CMP_{suite}.png (GT|TTT|Attn 3-way, same sample per row).
- Gotchas: (1) attention ckpt saved DDP-wrapped (module. prefix) — loader MUST strip + fail-loud on mismatch,
  else silently runs RANDOM weights (uniform loss=ln(131072)=11.79, acc=0). (2) TTT fast-weight kernels are
  @torch.compile(dynamic=True); AR var-length → recompile storm (256× slow, GPU~15%); set EVAL_WM_NO_COMPILE=1
  (torch._dynamo.config.disable BEFORE importing model).

## Snapshot (2026-06-09 — chunk training DONE, env-gen ~90%)
- chunk64/256 TRAINING FINISHED (checkpoint_final.pt present), base-spatial eval DONE.
- GPU0/1 = env-gen eval workers (~9.6GB each, NOT training anymore). GPU2/3 saturated (~99%).
- NOTE non-VLANeXt job also sharing GPU0/1: scripts/eval_rollout_official.py (Wan2.2-TI2V-5B video model) — unrelated.

### Chunk-size scan COMPLETE (vision-expert TTT block SIZE; #blocks=256/size). LIBERO-spatial base SR:
| chunk_size | #blocks | base SR |
|---|---|---|
| 16  | 16 | **97.80%** (489/500) |
| 64  | 4  | **97.80%** (489/500) |
| 256 | 1  | **97.40%** (487/500) |
- Conclusion: 16→1 blocks ≈ no SR loss (256 only -0.4pp). VALIDATES standing rule "vision default chunk256"
  (fastest deploy 1.8×, SR essentially flat). chunk4 NOT run (tiny-chunk = 3.6× slower for no benefit).
- Cron c3a11114 (session-only :08/:38, auto 4-shard base eval on checkpoint_final.pt) did its job → EVAL_SUMMARY.txt
  in each chunk dir. May still be active — delete if no longer needed.

### LIBERO-plus environment-generalization eval (RUNNING, ~90%) — UPGRADED to 12-shard parallel
Env-gen = dims Camera/Light/Background/Noise/Robot-init, ~1627 spatial tasks. Split changed from 2-shard → **12-shard**
(per RAM-bound parallelism rule). Per-shard result dir suffixed _SR<pct> when done; category_stats_*.json for merge.
- **TTT-16: 9/12 shards DONE, 3 running** (shard0 ~539/814≈66%, shard1 ~524/813≈64%, shard2 ~238/246≈97% nearly done).
  Partial done-shard dims so far: Light 97.60% (447/458), Noise 97.26% (320/329). NOTE shard0/1 are the big 814/813
  catch-all shards; shard2-11 are small per-dim slices (~60-246 tasks). Final per-dim table needs all 12 merged.
- **Attention baseline env-gen: NOW RUNNING (was pending)** — shard0 DONE SR 97.55%, shard1-4 running (204 tasks each,
  ~60-70%, ~5h in). Configs libero_plus_envgen_attn_shard{0..4}.yaml.
- Aggregation helper: glob VLANeXt_ablation_wm/{ttt,attention}_libero_spatial/*_SR*/category_stats_*.json, sum
  category_stats[dim].{success,total} dedup by shard_tag.

## Pending todos (rewritten 2026-06-10 — current master list)
DONE: ✅ chunk 16/4/1-block SR curve; ✅ TTT-16 & attention env-gen full 1627 → per-dim table (TTT +5.6pp,
see [[vlanext-envgen-ttt-vs-attn-results]]); ✅ disk cleanup + resource limits ([[vlanext-compute-resource-limits]]).

IN PROGRESS:
- A. **TTT-256 env-gen eval** (8-way, full 1627, chunk256 ckpt) — RUNNING GPU2/3, ~152s/task, ~9h ETA.
  Cron c144ee5f hourly checks; on completion auto-builds TTT-16/TTT-256/Attention 3-way per-dim table.
- B. **GDN retrain** (lr 1e-4→5e-5 after run-1 diverged) — RUNNING GPU0/1. Watch step 5k-15k for rebound.
  If diverges again → plan B = add SWA fallback (match Kairos) and/or make action GDN causal (see [[vlanext-gla-causal-plan]]).

HIGH PRIORITY (新增 2026-06-10, full spec in EXPERIMENT_PLAN_2days.md "🔴 高优先级"):
- **HP. TTT-256 cross-suite training** — DECISIONS LOCKED 2026-06-10: object/goal=30k steps warmup 1500;
  long(libero_10)=**36k=30k×1.2, warmup 1800=1500×1.2** (proportional scale-up, follows final_libero 1.2× precedent).
  HP-5 (per-suite env-gen) DROPPED for now (prioritize env-gen mainline). Configs generated+verified:
  ablation_wm_ttt_chunk256_libero_{object,goal,long}.yaml (save_interval=10000).
  - **CONFIG GOTCHA**: train.py reads `data.max_steps` (NOT train.max_steps!) + `train.warmup_steps` —
    the two live in DIFFERENT sections. Setting train.max_steps is a silent no-op (caught a bug: long was
    stuck at 30k until moved to data.max_steps). See [[vlanext-compute-resource-limits]].
  - **AUTO-LAUNCH**: scripts/autostart_hp_trainings.sh running in bg — polls 5min, starts one 2-GPU DDP run
    when ≥2 GPUs <40GB AND RAM ≥100GB free; queues object→goal→long. Waiting now (RAM full). Fires when
    TTT-256 env-gen (~5h) or GDN (~6h) frees resources.
  - HP-4: after all 3 train → 4-suite base eval → spatial/object/goal/long × TTT-256 table (vs attention/TTT-16).

OTHER PENDING:
- C. GLA/GDN + action causal-TTT probe full table (gated on B's outcome; code+configs ready, [[vlanext-gla-causal-plan]]).
- D. (Open) noWM+TTT fair train for train-inference-consistent comparison.

SWA FALLBACK EXPERIMENT FAMILY (2026-06-10, code DONE — generic fallback_mixer works for ANY main mixer):
The SWA code (build_swa_causal_mask + fallback_mixer arg, see [[vlanext-gla-causal-plan]]) is generic, so both
[SWA,SWA,SWA,GDN] and [SWA,SWA,SWA,TTT] need NO new code — only configs. Goal: does SWA fallback (Kairos-style,
window 64) beat full-attn fallback for each linear mixer?
- **S1. SWA+GDN train** — config/ablation_wm_swa_gdn.yaml (lr 5e-5 GDN-fix hparams). Tests the divergence-fix
  hypothesis: does SWA fallback stabilize GDN better than [full-attn,GDN]? Gated on G1 (plain-GDN base eval).
- **S2. SWA+TTT train** — config/ablation_wm_swa_ttt.yaml (TTT default lr 1e-4, converges to ~2.17). Symmetric
  control: does SWA fallback help/hurt the already-good TTT? Compare vs plain [A,A,A,TTT] (=ttt_chunk256, SR known).
- **S3. eval + compare**: [A,A,A,TTT] vs [SWA,SWA,SWA,TTT] vs [A,A,A,GDN] vs [SWA,SWA,SWA,GDN] — base SR + loss.
  Answers: (a) is SWA fallback universally better, (b) does it close the GDN-vs-TTT gap (GDN stuck ~5.6 vs TTT ~2.1).
Both configs vision chunk256, window 64, save_interval 10000. Queue after GPUs free (autostart-eligible).

Memory pref: [[no-claude-attribution]].

## STANDING RULES (set 2026-06-08, apply to all future runs)
1. **All future LIBERO-plus (env-gen) eval MUST use 4-GPU acceleration** (it's slow; ~2 days on 2 GPU → ~1 day on 4).
   Use scripts/envgen_accel_4gpu.sh pattern (split task list across 4 shards/GPUs).
2. **All future TTT runs use vision generator_ttt_chunk_size=256** (1 block/frame, fastest deploy, SR no drop).
   This applies to the VISION DiT (256 image tokens). The ACTION expert keeps its own small chunk
   (8 action tokens; e.g. causal-probe uses chunk=2) — the "256" rule does NOT apply to action.
3. **All training launched on 2 GPUs (DDP)** — set 2026-06-08. Use distributed=true / torchrun 2-proc.
   batch_size in config is TOTAL across GPUs (auto-divided). Prior single-GPU launches are the exception, not the norm.
   "Available GPU" = a GPU that can still FIT another job in memory (each H200 = 80GB, each eval uses only
   ~9.7GB, each WM training ~25-48GB) — NOT a fully-idle GPU. So training can SHARE GPUs with running eval.
4. **Server utilization target (set 2026-06-08): keep ALL 4 GPUs >50GB memory used.**
   H200 = 80GB/card; each eval ~10GB, each WM training (2-GPU DDP) ~41GB/card. Pack jobs so every card
   carries enough work (eval shards + training) to stay >50GB. Idle/underused cards = wasted compute;
   launch the next ready experiment (training or eval shard) to fill any card under ~50GB, as long as
   headroom remains (<80GB). This operationalizes the "available GPU = can still fit a job" rule.
5. **THE REAL BOTTLENECK IS CPU RAM (cgroup), NOT GPU (learned 2026-06-08 the hard way).**
   - libero-plus env-gen eval is **CPU-render bound** (OSMesa, single-threaded): each worker uses only
     ~1.1 CPU core + ~10GB GPU + **~22GB RSS host RAM**, and renders serially. So GPU memory is NEVER the
     limit for adding eval parallelism — host RAM is. Throughput scales with #workers, not per-worker speed.
   - **Hard cap: the container cgroup memory.limit_in_bytes = 416GB** (NOT the machine's 1TB — `free -g`
     lies; always check `/sys/fs/cgroup/memory/memory.limit_in_bytes`). Exceeding it triggers SILENT
     cgroup OOM-kills (no Python traceback; workers just vanish mid-run). Check `dmesg | grep -i "cgroup out of memory"`.
   - **Safe env-gen parallelism = 8 eval workers** (8×22≈176GB) **+ 2 WM trainings** (≈176GB) ≈ 352GB,
     leaving ~64GB headroom under 416GB. 12 workers (264GB)+training OVERFLOWS → got killed. Do NOT exceed
     ~8 concurrent eval workers while 2 trainings run. Budget by RAM: `(416 - train_RSS) / 22 ≈ max workers`.
   - Same-GPU worker launches must be **staggered ~30-45s** (sleep between nohup) — simultaneous model loads
     spike both host RAM and GPU transient peak and silently kill a sibling on that card.
   - Driver written: scripts/envgen_fullsend_8x.sh (8 accel workers over unreached tails, dedup by task name).
     12-way config files (shard8-11) exist but MUST stay unused under current cgroup limit.
