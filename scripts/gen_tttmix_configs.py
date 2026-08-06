"""Generate 8 TTT-mix env-gen shard configs from the tttlong templates.
Only changes: finetuned_checkpoint -> mix ckpt; shard_tag -> tttmix.
task_suite_name (libero_10) + task_ids stay identical (evaluate mix on long suite).
"""
import re

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
MIX_CKPT = f"{REPO}/VLANeXt_ablation_wm/ttt_chunk256_libero_mixed_clean/checkpoint_final.pt"

for s in range(8):
    src = f"{REPO}/config/libero_plus_envgen_tttlong_5dim_shard{s}.yaml"
    dst = f"{REPO}/config/libero_plus_envgen_tttmix_5dim_shard{s}.yaml"
    with open(src) as f:
        txt = f.read()
    txt = re.sub(r"finetuned_checkpoint:.*", f"finetuned_checkpoint: {MIX_CKPT}", txt)
    txt = re.sub(r"shard_tag:.*", f"shard_tag: envgen_tttmix_5dim_shard{s}", txt)
    with open(dst, "w") as f:
        f.write(txt)
    print(f"wrote {dst}")
print("DONE: 8 tttmix shard configs (libero_10 suite)")
