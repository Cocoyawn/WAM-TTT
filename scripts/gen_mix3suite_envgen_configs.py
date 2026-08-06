#!/usr/bin/env python3
"""Generate mix-checkpoint env-gen 5-dim eval shard configs for the THREE suites
the other machine is NOT covering: libero_spatial / libero_object / libero_goal.

Purpose: the official LIBERO-plus leaderboard aggregates each perturbation
dimension ACROSS ALL FOUR suites. The other machine evaluates the mix checkpoint
on libero_10 only; this fills spatial/object/goal so the four-suite pool can be
assembled. Same mix checkpoint, independent shard_tag + result dirs (zero
conflict with the running libero_10 shards).

5 dims (EXCLUDES Language Instructions + Objects Layout, matching the libero_10 run):
  Camera Viewpoints, Light Conditions, Background Textures, Sensor Noise,
  Robot Initial States.
task_ids = classification.json id - 1 (0-based), round-robin into NSHARD shards.
"""
import json, yaml, os

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
CLASS = f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
MIX_CKPT = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_mixed_clean/checkpoint_final.pt"
DIMS = ('Camera Viewpoints', 'Light Conditions', 'Background Textures',
        'Sensor Noise', 'Robot Initial States')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal')
NSHARD = 8  # 4 free GPUs x 2 workers/card = 8-way (env-gen is host-RAM bound, not GPU)

d = json.load(open(CLASS))
total = 0
for suite in SUITES:
    ids = sorted(t['id'] - 1 for t in d[suite] if t['category'] in DIMS)
    assert ids, f"no env-gen task ids for {suite}"
    shards = [ids[i::NSHARD] for i in range(NSHARD)]
    for s in range(NSHARD):
        cfg = {
            'data': {'augmentation': {'center_crop': True, 'center_crop_ratio': 0.9}},
            'model': {'diffusion_steps': 5, 'scheduler_type': 'flow_match',
                      'attn_implementation': 'sdpa'},
            'eval': {
                'finetuned_checkpoint': MIX_CKPT,
                'task_suite_name': suite,
                'task_ids': shards[s],
                'shard_tag': f'envgen_mix5dim_{suite}_shard{s}',
                'num_steps_wait': 10, 'num_steps_execute': 8,
                'num_trials_per_task': 1, 'seed': 42, 'image_size': 256,
                'resume_episodes': 0, 'resume_successes': 0,
                # CUDA fused TTT kernel + O(n) incremental decode. CUDA Graph is
                # disabled in code (ttt.py _use_infer_graph default False) because
                # it caused illegal-memory-access in long eval rollouts; the eager
                # fused-kernel incremental path is correct (verified SR-restored
                # 2026-06-19) and still O(n).
                'ttt_use_cuda_kernel': True,
                'use_incremental_gen': True,
            },
        }
        p = f"{REPO}/config/libero_plus_envgen_mix5dim_{suite}_shard{s}.yaml"
        yaml.safe_dump(cfg, open(p, 'w'), default_flow_style=False, sort_keys=False)
        print(f"wrote {os.path.basename(p)}: {len(shards[s])} tasks")
    n = sum(len(x) for x in shards)
    total += n
    print(f"  {suite}: {n} tasks across {NSHARD} shards\n")
print(f"TOTAL: {total} tasks (3 suites x {NSHARD} shards = {3 * NSHARD} configs)")
