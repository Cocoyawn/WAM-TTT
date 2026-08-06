#!/usr/bin/env python3
"""Generate attn-mix-checkpoint env-gen 5-dim eval shard configs for ALL FOUR
suites (spatial / object / goal / libero_10), mirroring the TTT-256 mix eval
EXACTLY so the A/B is strict.

The only differences vs the TTT configs:
  - finetuned_checkpoint -> attention_libero_mixed_clean/checkpoint_final.pt
  - shard_tag prefix -> envgen_attnmix_5dim_<suite>  (zero dir collision w/ TTT)
  - NO ttt_use_cuda_kernel / use_incremental_gen (attention model has no TTT path)

Same 5 dims, same task_id round-robin into NSHARD=8 shards, same eval knobs
(diffusion_steps=5, exec=8, wait=10, 1 trial/task, seed=42, 256px, center-crop 0.9).
"""
import json, yaml, os

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
CLASS = f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
ATTN_CKPT = f"{REPO}/VLANeXt_ablation_wm/attention_libero_mixed_clean/checkpoint_final.pt"
DIMS = ('Camera Viewpoints', 'Light Conditions', 'Background Textures',
        'Sensor Noise', 'Robot Initial States')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
NSHARD = 8

assert os.path.exists(ATTN_CKPT), f"attn ckpt missing: {ATTN_CKPT}"
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
                'finetuned_checkpoint': ATTN_CKPT,
                'task_suite_name': suite,
                'task_ids': shards[s],
                'shard_tag': f'envgen_attnmix_5dim_{suite}_shard{s}',
                'num_steps_wait': 10, 'num_steps_execute': 8,
                'num_trials_per_task': 1, 'seed': 42, 'image_size': 256,
                'resume_episodes': 0, 'resume_successes': 0,
                # attention model: no TTT kernel / incremental path
                'ttt_use_cuda_kernel': False,
                'use_incremental_gen': False,
            },
        }
        p = f"{REPO}/config/libero_plus_envgen_attnmix_5dim_{suite}_shard{s}.yaml"
        yaml.safe_dump(cfg, open(p, 'w'), default_flow_style=False, sort_keys=False)
    n = sum(len(x) for x in shards)
    total += n
    print(f"{suite}: {n} tasks across {NSHARD} shards")
print(f"TOTAL: {total} tasks (4 suites x {NSHARD} = {4*NSHARD} configs)")
