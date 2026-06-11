---
name: vlanext-ttt-latency
description: VLANeXt TTT vs attention — training step time and deployment inference latency (measured)
metadata: 
  node_type: memory
  type: project
  originSessionId: 096b4910-03a6-4e31-a8a3-426c2bb29e23
---

Measured latency for VLANeXt world-model (Qwen3-VL-2B backbone, [A,A,A,X] mixer interleave,
soft cond, 5-step action denoise). Method: warmup then ITERS-average with torch.cuda.synchronize()
before+after; @torch.no_grad for inference, train()+fwd+backward+opt.step for training.
Scripts: scripts/bench_ttt_vs_attn.py (train), scripts/bench_chunk_infer_latency.py (deploy).

## Deployment inference latency — predict_action (world-model: 256-step vision AR + 5-step action denoise), B=1
Same shell, only mixer/chunk swapped (chunk swept in-place on the 7 vision-TTT layers, weights identical):

| config | latency | vs attention | fw-updates/rollout |
|---|---|---|---|
| **softmax attention** | **4868 ms** | 1.00x (ref, FASTEST) | — |
| ttt chunk=16 (16 blocks) | 19762 ms | 0.25x (4.1x slower) | ~2176 |
| ttt chunk=64 (4 blocks)  | 10822 ms | 0.45x (2.2x slower) | ~640 |
| ttt chunk=128 (2 blocks) | 9349 ms  | 0.52x (1.9x slower) | ~384 |
| ttt chunk=256 (1 block)  | 8717 ms  | 0.56x (1.8x slower) | ~256 |

**Key:** user was RIGHT that bigger chunk speeds up inference. At deploy the vision DiT runs a 256-step
AR loop; each step the causal TTT op does ceil(t/chunk) fast-weight updates (each a Newton-Schulz). Larger
chunk → fewer updates summed over the rollout (2176→256, ~8.5x fewer) → 2.3x faster end-to-end (16→256).
BUT even chunk=256 (1 update/step) is still 1.8x slower than attention — the residual gap is the
Newton-Schulz + apply cost intrinsic to TTT, which chunk cannot remove. Attention is fastest at deploy.
(My earlier claim "AR seq too short to chunk" was WRONG — each step re-chunks the current length-t seq.)

## Training step time (fwd+backward+opt.step), controlled benchmark, batch=8, world-model ON, grad-ckpt ON
| | compile OFF | compile ON | peak mem |
|---|---|---|---|
| attention | 871 ms/step (1.15 it/s) | 799 ms/step (1.25 it/s) | 33.3 GB |
| ttt (default chunk) | 1352 ms/step (0.74 it/s) | 1030 ms/step (0.97 it/s) | 33.4 GB |
| TTT vs attention | 1.55x slower | **1.29x slower** | equal |

Real 30k-step runs (batch16, grad-ckpt, 1 GPU): chunk64=1.42 s/it, chunk256=1.36 s/it (bigger chunk
slightly faster in training too, but small effect — vision-TTT is only 7/29 layers). NOTE: a same-condition
attention training wall-clock is NOT yet measured (prior attention runs used multi-GPU/other configs — not
comparable; this was a past mistake). Controlled benchmark (1.29x slower) is the trustworthy train comparison.

## Bottom line
TTT is slower than attention in BOTH train (1.29x) and deploy (1.8–4.1x), memory equal. Bigger chunk mainly
rescues DEPLOY latency (4.1x→1.8x slower), helps training little. TTT's value is capability/generalization
(pending env-gen results), NOT speed. See [[ttt-chunk-size-train-speed]], [[vlanext-ttt-experiment-status]].
