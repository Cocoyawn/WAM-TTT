---
name: ttt-chunk-size-train-speed
description: VLANeXt vision-expert TTT chunk_size vs training step time (LaCT block-causal ablation)
metadata: 
  node_type: memory
  type: project
  originSessionId: 096b4910-03a6-4e31-a8a3-426c2bb29e23
---

VLANeXt world-model ablation (libero_spatial, 30k steps, 1 GPU H200, bf16, sdpa,
grad-checkpoint ON, batch 16). Vision expert processes a fixed **256 image tokens**
per frame; the causal TTT operator splits them into blocks of `generator_ttt_chunk_size`
and does one apply-then-update (incl. Newton-Schulz) per block. So **#blocks = 256 / chunk_size**.
Smaller chunk_size ⇒ more serial blocks ⇒ slower; larger chunk_size ⇒ LaCT "large chunk" ⇒ fewer updates ⇒ faster.

Measured training speed (early steps, same config except chunk_size):

| chunk_size | #blocks | speed | est. 30k total |
|---|---|---|---|
| 4   | 64 | ~4.9 s/it  | ~40 h |
| 16  | 16 | ~1.37 s/it | ~11.4 h (original baseline) |
| 64  | 4  | ~1.43 s/it | ~11.4 h |
| 256 | 1  | ~1.30 s/it | ~10.8 h (one fast-weight update per frame, most LaCT-extreme) |

**Why:** chunk=4 is 3.6× slower than chunk=64 purely from 64 serial Python-loop
block updates (each a small matmul + Newton-Schulz orthogonalization), GPU 100% util
but starved on serial small ops. chunk 16→64 barely changes wall-clock because the
vision-TTT layers (7 of 29, every 4th via [A,A,A,T]) are a small fraction of total
step cost — bottleneck is backbone forward + 256-token vision gen loss.

**How to apply:** for LaCT-faithful fast training pick large chunk_size (64 = 4 blocks,
or 256 = 1 update/frame). Avoid tiny chunk_size (4) unless specifically ablating
fine-grained causality — it costs ~3.6× wall-clock for little benefit.

**Terminology pin (user confusion source):** `chunk_size` = block SIZE (tokens per block),
NOT block count. "4 个块/4 blocks" = chunk_size **64**. "贴近 LaCT 大块" = LARGER chunk_size.
Related: [[no-claude-attribution]].
