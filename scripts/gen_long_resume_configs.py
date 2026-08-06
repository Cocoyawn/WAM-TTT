#!/usr/bin/env python3
"""把本机 long_tail 8 个 shard 的剩余 task 重切成 16 份(每 shard 剩余对半),
起 16 worker(每卡 4)加速收尾。已完成 task 的 mp4 已落盘,不会丢。

对每个 tail_shard{s}:
  - 读其 config 的 task_ids(= full[128:], 共100个)
  - 读 log 的 done 数 -> 剩余 = task_ids[done:]
  - 对半切成 _r0 / _r1 两个新配置(disjoint)
共 8*2 = 16 个配置, shard_tag 带 _resume{s}_h{0,1}, 独立结果目录。

聚合时: 原 tail_shard{s} 目录(前 done 个) + resume{s}_h0 + resume{s}_h1 = 完整 [128:]。
"""
import yaml, os, re

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
LOGD = f"{REPO}/logs"

def done_count(s):
    p = f"{LOGD}/plus_envgen_attnmix_5dim_libero_10_tail_shard{s}.log"
    return open(p, errors="ignore").read().count("Current task success rate:")

total_remain = 0
for s in range(8):
    src = f"{REPO}/config/libero_plus_envgen_attnmix_5dim_libero_10_tail_shard{s}.yaml"
    c = yaml.safe_load(open(src))
    ids = [int(x) for x in c['eval']['task_ids']]
    done = done_count(s)
    remain = ids[done:]                      # 还没跑的
    half = len(remain) // 2
    parts = {0: remain[:half], 1: remain[half:]}
    for h, sub in parts.items():
        if not sub:
            continue
        cc = yaml.safe_load(open(src))       # fresh copy
        cc['eval']['task_ids'] = sub
        cc['eval']['shard_tag'] = f'envgen_attnmix_5dim_libero_10_resume{s}_h{h}'
        p = f"{REPO}/config/libero_plus_envgen_attnmix_5dim_libero_10_resume{s}_h{h}.yaml"
        yaml.safe_dump(cc, open(p, 'w'), default_flow_style=False, sort_keys=False)
    total_remain += len(remain)
    print(f"shard{s}: done={done}, remain={len(remain)} -> h0={len(parts[0])} h1={len(parts[1])}")
print(f"\n总剩余 {total_remain} tasks, 切成 16 worker (每卡4)")
