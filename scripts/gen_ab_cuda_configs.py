#!/usr/bin/env python3
"""Build two tiny A/B configs (torch vs CUDA+incremental) on the SAME task set,
to verify the CUDA kernel path matches torch end-to-end in real LIBERO sim and
to measure the real speedup. Uses the mix checkpoint, libero_spatial, a small
set spanning all 5 dims. Independent shard_tag/result dirs -> no conflict with
the running spatial 8-way job.
"""
import json, yaml, os
from collections import defaultdict

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
CLASS = f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
MIX = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_mixed_clean/checkpoint_final.pt"
DIMS = ('Camera Viewpoints', 'Light Conditions', 'Background Textures',
        'Sensor Noise', 'Robot Initial States')
PER_DIM = 4  # 4 tasks per dim x 5 dims = 20 tasks

d = json.load(open(CLASS))
by_dim = defaultdict(list)
for t in d['libero_spatial']:
    if t['category'] in DIMS:
        by_dim[t['category']].append(t['id'] - 1)  # 0-based
ids = []
for dim in DIMS:
    ids += sorted(by_dim[dim])[:PER_DIM]
ids = sorted(ids)
print(f"A/B task set: {len(ids)} tasks across {len(DIMS)} dims -> {ids}")

base = dict(
    data={'augmentation': {'center_crop': True, 'center_crop_ratio': 0.9}},
    model={'diffusion_steps': 5, 'scheduler_type': 'flow_match', 'attn_implementation': 'sdpa'},
)
for tag, cuda, inc in [('abtorch', False, False), ('abcuda', True, True)]:
    cfg = dict(base)
    cfg['eval'] = {
        'finetuned_checkpoint': MIX,
        'task_suite_name': 'libero_spatial',
        'task_ids': ids,
        'shard_tag': f'envgen_{tag}',
        'num_steps_wait': 10, 'num_steps_execute': 8, 'num_trials_per_task': 1,
        'seed': 42, 'image_size': 256, 'resume_episodes': 0, 'resume_successes': 0,
        'ttt_use_cuda_kernel': cuda, 'use_incremental_gen': inc,
    }
    p = f"{REPO}/config/libero_plus_envgen_{tag}.yaml"
    yaml.safe_dump(cfg, open(p, 'w'), default_flow_style=False, sort_keys=False)
    print(f"wrote {os.path.basename(p)}: cuda={cuda} incremental={inc}")
