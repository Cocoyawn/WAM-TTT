---
name: vlanext-gla-causal-plan
description: VLANeXt next-2-day plan — GLA/GatedDeltaNet mixer integration + action causality ablation
metadata: 
  node_type: memory
  type: project
  originSessionId: 096b4910-03a6-4e31-a8a3-426c2bb29e23
---

Next-phase VLANeXt plan (full doc: /mnt/afs-h200/yuyangcheng/workplace/VLANeXt/EXPERIMENT_PLAN_2days.md).
Builds on [[vlanext-ttt-experiment-status]] and [[ttt-chunk-size-train-speed]].

**Goal:** decide whether action DiT should become a causal structure (Gated DeltaNet). Strategy:
use a cheap causal-TTT probe first, then commit to GDN only if causality doesn't hurt action success.

**Reference:** KairosDiT (kairos-sensenova/kairos/modules/dits/kairos_dit.py:632-790, 1029-1038) uses
GatedDeltaNet as a token mixer on the `(i+1)%4==0` [A,A,A,Linear] pattern — SAME interleave idea as our
VLANeXt [A,A,A,T]. KairosDiT injects context via a SEPARATE cross-attn (not by stuffing VLM into the
linear-attn KV); our TTT method-B stuffs VLM into the update KV. For ablation comparability prefer
method-B injection for GLA/GDN too (only swap the token-mixing operator, keep adaLN/residual/MLP/VLM-injection).

**Facts:** GatedDeltaNet already vendored (src/models/fla/layers/gated_deltanet.py, test passing), ~6·d²/layer,
mode='chunk', inherently causal (can't do action's bidirectional). GLA is same fla family (gla.py in kairos),
also causal. Current action DiT is bidirectional (ttt_causal=False, 8 action tokens); vision DiT causal (256 tokens).
Mixer dispatch: _mixer_at() + mixer_type enum {attention, ttt} → extend with {gla, gdn}.

**Day1:** finish chunk64/256 eval (16/4/1-block SR curve); implement GLA mixer (vendor gla.py, method-B inject,
unit test); add policy_ttt_causal switch for action causal-TTT probe.
**Day2:** train action bidirectional-TTT vs causal-TTT (THE causality probe — if causal hurts SR, GDN on action
is unjustified); train GLA mixer; conditionally train GatedDeltaNet; aggregate big table
{attention/TTT-bidir/TTT-causal/GLA/GDN} × {SR, per-dim generalization, speed, params}.

**Decisions:** (2) VLM injection LOCKED to method-B (KV-concat, same as TTT — only swap token-mixing operator,
keep adaLN/residual/MLP/VLM-injection identical across attention/TTT/GLA/GDN for max comparability). Pending:
(1) chunk-eval GPU freeing (pause env-gen vs 2-GPU slow vs wait). (3) GDN full rollout gated on causality-probe result.

Stage-0 (in progress, part of this plan): chunk64 + chunk256 training (GPU0/1, ~6h left) feed Day-1.1 SR curve;
TTT-16 LIBERO-plus env-gen eval (GPU2/3, 1627 tasks) is the TTT generalization baseline for Day-2.4 table.

Memory pref: [[no-claude-attribution]].

## CODE READY (2026-06-07, written ahead, all tests passing — just launch when GPUs free)
- New mixers `gdn` (GatedDeltaNet) + `gla` (GLA, gla.py NOT yet vendored) wired into MoEBlock (policies.py) and
  MoEGeneratorBlock (generator.py) via method-B. New file src/models/linear_attn_mixer.py: LinearAttnMixer
  prepends projected VLM ctx to input seq, runs causal fla recurrence over [ctx;x], slices off ctx → keeps x
  outputs. Same forward(x,info,ctx)->(out,None) as TTT; block dispatch adds `elif mixer_type in (gla,gdn)`.
- Action causal-TTT probe: added policy_ttt_causal + policy_ttt_chunk_size, threaded through 3 action MoE classes,
  VLANeXt.__init__ (4 action_head sites), scripts/train.py, eval loader VLANeXt_utils.py.
- Tests PASS: src/models/test_linear_attn_mixer.py (attention/ttt/gdn shape/finite/grad + causality + method-B);
  full-model gdn e2e (construct 2754.9M, train loss 2.09, backward finite, predict_action ok); regression
  (attention default unchanged, [A,A,A,X] interleave correct).
- Configs ready: config/ablation_wm_gdn.yaml (both experts gdn), config/ablation_wm_ttt_actcausal.yaml
  (action TTT causal chunk_size 2). Launch like chunk runs (venv, TORCHDYNAMO_DISABLE=1, wandb).
- Remaining: vendor gla.py closure (optional; gdn is main route; GLA has triton-ops vendor risk, deferred to user OK).

## GDN RUN #1 DIVERGED + ARCHITECTURE CLARIFICATION (2026-06-10)
- **First GDN train (lr 1e-4) DIVERGED**: loss 13→4.25 (learning) then REBOUNDED to ~14.5 and stuck → bad
  checkpoint_final. (TTT chunk256 converged to 2.17, actcausal to 2.07 — GDN is the only diverging run.)
  Deleted the diverged gdn_libero_spatial dir. Retraining with lr 1e-4→**5e-5**, warmup 1500→**3000**,
  max_grad_norm 1.0→**0.5**, save_interval 2000→**10000** (script scripts/run_gdn_retrain.sh, 2-GPU torchrun
  GPU0/1; LAUNCH GOTCHA: must `export PYTHONPATH=$REPO` or train.py dies `ModuleNotFoundError: No module named 'src'`).
- **CORRECTION to earlier wording**: the diverged GDN is **[A,A,A,GDN]** = full-attention × 3 + GDN every 4th
  layer (policies.py:17-24 `_mixer_at`: `(layer_idx+1)%mix_every_n==0 ? gdn : "attention"`, mix_every_n=4).
  It is **NOT pure-GDN-every-layer**, and NOT "GDN with no fallback". GDN layers ARE interleaved with full
  attention (~7/29 layers are GDN in each expert). So divergence is NOT caused by "all layers GDN / no fallback".
- **Kairos comparison (verified in code)**: Kairos uses **[SWA,SWA,SWA,GDN]** — the fallback layers are
  SLIDING-WINDOW attention (window=3 frames, flash-attn native window_size; dilated_lengths=[1,1,4,1]), NOT full
  attention. kairos_dit.py:1029-1048 (stack), :388 (window), :681-712 (GDN layer). No "DSWA"/decayed-window
  exists in Kairos (the `w_swa` var at :406 is attention-sink, default off attend_k0=False). So the ONLY
  structural diff vs Kairos is fallback = SWA (theirs) vs full-attn (ours) — both HAVE a fallback.
- **Our GDN already has full stabilization** (vendored upstream FLA, nothing stripped): short_conv(k=4,silu)
  ON (linear_attn_mixer.py:71 use_short_conv=True), gated RMSNorm output, in-kernel QK-L2norm, Mamba-style
  A_log/dt_bias init (no-WD), sigmoid beta. So divergence is NOT a missing-short_conv issue.
- **Likely real causes** (since fallback+stabilization both present): (1) **action expert runs GDN BIDIRECTIONAL**
  but GDN is inherently causal — Kairos only uses GDN in the causal vision DiT, never on action. (2) method-B
  VLM-injection (prepend ctx to seq, causal recurrence over [ctx;x]) may amplify instability. (3) plain lr/hparam
  (testing now via lr 5e-5 retrain). Plan B if still diverges: add SWA-fallback variant (match Kairos) and/or
  make action GDN causal. TEST GAP: test_linear_attn_mixer.py covers only shape/finite/grad/causality — NO
  numerical-stability or convergence test, which is why divergence wasn't caught pre-train.

Disk/RAM limits that gate all this: [[vlanext-compute-resource-limits]].

## SWA+GDN CODE DONE (2026-06-10, all tests pass) — [SWA,SWA,SWA,GDN] à la Kairos
Built the SWA-fallback variant (plan B from above) so the GDN-layer FALLBACK is sliding-window attention
instead of full attention — matching Kairos's [SWA,SWA,SWA,GDN] stack (ours was [full-attn,full-attn,full-attn,GDN]).
- **Files (5)**: generator.py — new `build_swa_causal_mask(T_img,T_ctx,window,...)` (causal AND windowed for
  image→image; VLM ctx columns always 0 = globally visible, preserves method-B), `MoEGeneratorBlock` accepts
  mixer_type='swa' (reuses nn.MultiheadAttention + the window mask in forward), `_mixer_at` gains
  `fallback_mixer` arg (default 'attention' = unchanged [A,A,A,X]; 'swa' = [SWA,SWA,SWA,GDN]). VLANeXt.py +
  train.py thread `generator_fallback_mixer` + `generator_swa_window_size`.
- **Adaptations vs Kairos** (ours ≠ theirs): Kairos window=3 FRAMES (multi-frame seq); ours is single-frame
  256 image tokens → window in TOKENS, default **64**. CAUSAL sliding window (query i sees [i-W+1 .. i]), matching
  vision DiT autoregression — NOT Kairos's bidirectional window. SWA only on VISION (action expert = 8 tokens,
  windowing pointless; keeps attention fallback). VLM ctx never windowed.
- **Tests**: src/models/test_swa_gdn.py — 5 CPU (mask semantics, wide-window==causal, _mixer_at, block fwd+grad,
  causality+window isolation) + 1 GPU e2e (real GDN triton fwd+backward on [SWA,SWA,SWA,GDN]). ALL PASS.
  Regression: attention / [A,A,A,GDN] / [A,A,A,TTT] unchanged. GDN triton kernel is GPU-ONLY (CPU e2e skips).
- **Config**: config/ablation_wm_swa_gdn.yaml — vision [SWA,SWA,SWA,GDN], window 64, chunk256, inherits GDN
  divergence-fix hparams (lr 5e-5, warmup 3000, grad_clip 0.5, save_interval 10000). Ready to launch when GPU frees.
- **Task chain**: G1 GDN base eval (after GDN train ~done) → G2 ✅ SWA+GDN code → G3 SWA+GDN train → G4 eval+compare.
