#!/usr/bin/env python3
"""Generate goal env-gen 5-dim eval shard configs (6-way), mirroring the object setup.

5 perturbation dims: Camera Viewpoints, Light Conditions, Background Textures,
Sensor Noise, Robot Initial States (EXCLUDES Language Instructions + Objects Layout).
task_ids = classification.json libero_goal id - 1 (0-based), round-robin into 6 shards.
"""
import json, yaml, os

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
CLASS = f"{REPO}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
CKPT = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_goal/checkpoint_final.pt"
DIMS = ('Camera Viewpoints', 'Light Conditions', 'Background Textures',
        'Sensor Noise', 'Robot Initial States')
NSHARD = 6

d = json.load(open(CLASS))
ids = sorted(t['id'] - 1 for t in d['libero_goal'] if t['category'] in DIMS)
assert ids, "no goal env-gen task ids found"

shards = [ids[i::NSHARD] for i in range(NSHARD)]  # round-robin
for s in range(NSHARD):
    cfg = {
        'data': {'augmentation': {'center_crop': True, 'center_crop_ratio': 0.9}},
        'model': {'diffusion_steps': 5, 'scheduler_type': 'flow_match', 'attn_implementation': 'sdpa'},
        'eval': {
            'finetuned_checkpoint': CKPT,
            'task_suite_name': 'libero_goal',
            'task_ids': shards[s],
            'shard_tag': f'envgen_goal256_shard{s}',
            'num_steps_wait': 10, 'num_steps_execute': 8, 'num_trials_per_task': 1,
            'seed': 42, 'image_size': 256, 'resume_episodes': 0, 'resume_successes': 0,
        },
    }
    p = f"{REPO}/config/libero_plus_envgen_goal256_shard{s}.yaml"
    yaml.safe_dump(cfg, open(p, 'w'), default_flow_style=False, sort_keys=False)
    print(f"wrote {os.path.basename(p)}: {len(shards[s])} tasks")
print(f"total: {sum(len(x) for x in shards)} tasks across {NSHARD} shards")
