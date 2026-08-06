#!/usr/bin/env python3
"""协作加速 long(libero_10)套件: 远端已在跑 8 个 shard 的前半,本机接管尾部。

切法: 每个远端 shard 的 task_ids 列表取 [SPLIT:] 作为本机的"tail"分片,
写成独立配置 + 独立结果目录(shard_tag 带 _tail),两机零重叠。
远端那边需手动在 ~SPLIT 处停掉(或让它自然跑过头也无妨——聚合时按 task 去重)。

聚合时: 远端 shard 结果目录取 task_ids[:SPLIT], 本机 tail 目录取 task_ids[SPLIT:],
合并即完整 1824。
"""
import yaml, os
REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
ATTN_CKPT = f"{REPO}/VLANeXt_ablation_wm/attention_libero_mixed_clean/checkpoint_final.pt"
SPLIT = 128  # 远端已跑 ~110, 留 18 缓冲

for s in range(8):
    src = f"{REPO}/config/libero_plus_envgen_attnmix_5dim_libero_10_shard{s}.yaml"
    c = yaml.safe_load(open(src))
    full = [int(x) for x in c['eval']['task_ids']]
    tail = full[SPLIT:]
    c['eval']['task_ids'] = tail
    c['eval']['shard_tag'] = f'envgen_attnmix_5dim_libero_10_tail_shard{s}'
    p = f"{REPO}/config/libero_plus_envgen_attnmix_5dim_libero_10_tail_shard{s}.yaml"
    yaml.safe_dump(c, open(p, 'w'), default_flow_style=False, sort_keys=False)
    print(f"tail_shard{s}: {len(tail)} tasks (task_ids[{SPLIT}:])  -> {os.path.basename(p)}")
print(f"\n本机 tail 合计 = {8*0 + sum(len(yaml.safe_load(open(f'{REPO}/config/libero_plus_envgen_attnmix_5dim_libero_10_shard{s}.yaml'))['eval']['task_ids'][SPLIT:]) for s in range(8))} tasks")
print(f"远端需在每 shard 跑到 task #{SPLIT} 处停(现在 ~110, 还差 ~18)")
