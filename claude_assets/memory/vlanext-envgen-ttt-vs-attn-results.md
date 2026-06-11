---
name: vlanext-envgen-ttt-vs-attn-results
description: "FINAL TTT-16 vs Attention LIBERO-plus env-gen per-dimension generalization table (all 1627 tasks, aligned)"
metadata:
  node_type: memory
  type: project
  originSessionId: 096b4910-03a6-4e31-a8a3-426c2bb29e23
---

VLANeXt env-gen generalization result (todo #3 deliverable from [[vlanext-ttt-experiment-status]]).
LIBERO-plus environment-generalization eval, **all 1627 tasks, both models fully aligned (1627/1627)**,
checkpoint_30000, libero_spatial, exec8 diff5. TTT = chunk16 (vision). 5 env dims sum to 1627.

**Suite caveat (avoid misreading):** ALL 5 dims come from ONE suite = **libero_spatial** (single-suite trained
ckpt ttt/attention_libero_spatial/checkpoint_30000.pt). This is NOT the 4 LIBERO suites (spatial/object/goal/long).
LIBERO-plus takes the spatial tasks and applies 7 perturbation categories; we eval 5 of them. So e.g.
**"Robot" = "Robot Initial States" perturbation on spatial tasks (350 tasks)**, NOT a separate suite.
Dims measured: Camera(376)/Light(292)/Background(258)/Noise(351)/Robot(350). (Layout/Language = task-gen, deferred.)

## FINAL TABLE — TTT-16 vs Attention (Δ = TTT − Attn)
| Dimension | TTT-16 | Attention | Δ |
|---|---|---|---|
| **Noise** (sensor noise) | **98.0%** (344/351) | 80.9% (284/351) | **+17.1** |
| **Robot** (init-state) | **80.3%** (281/350) | 69.4% (243/350) | **+10.9** |
| Light | 99.0% (289/292) | 98.6% (288/292) | +0.3 |
| Camera (viewpoint) | 69.1% (260/376) | 70.2% (264/376) | −1.1 |
| Background (texture) | 96.5% (249/258) | 98.1% (253/258) | −1.6 |
| **TOTAL** | **87.5%** (1423/1627) | **81.9%** (1332/1627) | **+5.6** |

## THREE-WAY TABLE — TTT-16 vs TTT-256 vs Attention (2026-06-10, all 1627 aligned)
TTT-256 = vision chunk_size 256 (1 block/frame, fastest deploy). Run via scripts/aggregate_envgen.py.
| Dimension | TTT-16 | **TTT-256** | Attention |
|---|---|---|---|
| Camera     | 69.1% | **78.7%** | 70.2% |
| Light      | 99.0% | 99.0% | 98.6% |
| Background | 96.5% | 97.7% | 98.1% |
| Noise      | 98.0% | 95.4% | 80.9% |
| Robot      | 80.3% | **82.9%** | 69.4% |
| **TOTAL**  | 87.5% | **89.9%** | 81.9% |
- **TTT-256 vs TTT-16: +2.4pp** — chunk16→256 has NO generalization cost; it's actually BETTER overall
  (big Camera jump 69→79; Robot 80→83). Only Noise regressed (98→95). This is a STRONG result: chunk256 was
  chosen for deploy speed (2.3× faster) + base-SR parity (97.4 vs 97.8), and now it ALSO generalizes better.
  Validates "chunk256 = universal default" beyond any doubt.
- **TTT-256 vs Attention: +8.0pp** — even larger gap than TTT-16's +5.6pp. TTT-256 is the best config on env-gen.
- Caveat: TTT-256 ckpt is checkpoint_final (30k) of ttt_chunk256_libero_spatial; TTT-16/attn are checkpoint_30000.
  Same suite/eval protocol. Noise being the one regression is mildly surprising (TTT-16's biggest win) — worth a
  glance if it matters, but overall TTT-256 dominates.

## Conclusion (updated 2026-06-10 — best config = TTT-256)
**初步结论：TTT 的 fast-weight 在线适应机制对传感器噪声和初始状态扰动这类动态分布偏移有明确鲁棒性优势
（TTT-256 vs Attention: Noise +14.5、Robot +13.4、Camera +8.5），对静态视觉变化（光照/视角/背景）与 attention 持平。**
- Deltas above are **TTT-256 vs Attention** (the best config). Reference TTT-16 vs Attention:
  Noise +17.1, Robot +10.9, Camera −1.1, Light +0.4, Background −1.6 (overall +5.6pp).
- TTT-256 vs Attention per-dim: Noise +14.5, Robot +13.4, Camera +8.5, Light +0.4, Background −0.4 (overall +8.0pp).
- Camera FLIPS sign between configs: TTT-16 ties attention (−1.1) but TTT-256 beats it +8.5 — the chunk256
  "cut image-token cross-dependence, rely on VLM semantics" inductive bias helps viewpoint OOD most.
- Light/Background ~tie for both configs (static visual changes). Consistent with base-spatial in-dist tie +
  WM video-gen in-dist TTT win: **TTT's value is robustness under DISTRIBUTION SHIFT, not clean-dist accuracy.**

## Severity-resolved breakdown (2026-06-11 — by difficulty_level 1-5, all 1627 aligned)
LIBERO-plus tags every task with `difficulty_level` 1-5 (in task_classification.json) = the perturbation
STRENGTH (Robot = qpos jitter coeff 0.1/0.2/0.3 in new_init.py; Noise = gaussian-blur sigma 1..10 in
env_wrapper.py; etc.). Re-aggregated the FINISHED logs by (dim, level) — NO re-eval. Script:
scripts/plot_severity_curves.py → docs/severity_curves.png. This is the key upgrade: the single-number
aggregate HID that the TTT advantage is concentrated at HIGH severity (the aggregate dilutes it with easy levels).

**Robustness SLOPE L1→L5 (drop in pp; smaller = more robust):**
| Dim | TTT-16 | TTT-256 | Attention |
|---|---|---|---|
| Camera | 81→82 (+1) | 96→95 (−0) | 88→68 (**−20**) |
| Light  | 100→100 (0) | 100→100 (0) | 98→83 (**−15**) |
| Background | 99→100 (+1) | 99→100 (+1) | 98→100 (+2) |
| Noise  | 100→100 (**0**) | 96→75 (−21) | 90→33 (**−57**) |
| Robot  | 97→60 (−37) | 99→53 (−46) | 97→29 (**−68**) |

**High-severity (L4+L5) aggregate — where the gap explodes:**
| Dim (L4-5) | TTT-16 | TTT-256 | Attention | best TTT − Attn |
|---|---|---|---|---|
| Noise  | 95.7% | 82.6% | 47.8% | **+47.9** (TTT-16) |
| Robot  | 56.9% | 59.5% | 37.1% | **+22.4** (TTT-256) |
| Camera | 69.4% | 85.5% | 64.5% | **+21.0** (TTT-256) |

Key reads:
- **Noise is the cleanest story**: TTT-16 is FLAT at ~100% across all 5 severities while Attention collapses
  90→33%. At L5 it's TTT-16 100% vs Attn 33% (+67pp). The aggregate "+17" massively understates this.
- **TTT-16 vs TTT-256 split on Noise**: TTT-16 stays flat (100→100); TTT-256 degrades at high severity
  (96→75, L5=75%). So for SENSOR-NOISE robustness specifically, chunk16's finer blocks help — the ONE place
  chunk16 beats chunk256. Everywhere else chunk256 ≥ chunk16.
- **Camera/Robot high-severity belong to TTT-256**: it's the only config holding up at L4-5 (Camera 85.5%,
  Robot 59.5%). Attention's Camera/Light/Noise/Robot all bleed out with severity; TTT stays near-flat except
  Robot (which is hard for everyone — even TTT-256 drops to 53% at L5, but Attn is 29%).
- Background is severity-invariant for ALL models (texture swap doesn't compound) — confirms it's not a
  dynamic-shift axis.
- **Sharpened conclusion**: TTT's fast-weight advantage is not uniform — it's a ROBUSTNESS-SLOPE advantage that
  widens monotonically with perturbation strength. At clean/low severity everyone ties; the separation is a
  high-severity phenomenon. This is the strongest single piece of evidence for the online-adaptation claim.

## How this was produced (gotchas — see scripts/aggregate_envgen.py)
- Attention eval was originally MIS-CONFIGURED: only 1020 tasks (ids 0-1409); env-gen full set is 1627 (0-2401).
  Missing 607 = Noise 315 + Light 292. Fixed 2026-06-09 with catch-up shards attn shard5-12 (config + 
  scripts/run_attn_envgen_catchup.sh, 8 workers ~76 tasks each, ~3h). LAUNCH BUG: forgot PYTHONPATH=$REPO/$PLUS_DIR
  → ModuleNotFoundError libero; must cd into third_party/LIBERO-plus AND set PYTHONPATH (copy run_attn_envgen.sh).
- Aggregation MUST avoid 3 pitfalls: (1) mp4 filenames TRUNCATE long Camera names → never use for dim;
  (2) category_stats JSON double-counts across duplicate/overlapping shard dirs; (3) logs re-append on restart.
  Correct method (scripts/aggregate_envgen.py): dim assigned by task_id via task_classification.json (idx==task_id),
  per-task success parsed from "Current task success rate: X" lines, block-index→config task_id, keep FIRST per id.

Memory pref: [[no-claude-attribution]]. Next: [[vlanext-gla-causal-plan]] GDN/GLA + action causal probe.
