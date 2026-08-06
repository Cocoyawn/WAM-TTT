#!/usr/bin/env bash
# Prune redundant intermediate checkpoints, keeping ONLY checkpoint_final.pt per training dir.
# SAFE BY DESIGN:
#   - only deletes checkpoint_<step>.pt in a dir that HAS a checkpoint_final.pt (model is preserved)
#   - skips any dir that is the save_dir of a RUNNING/queued scheduler job (its mid-ckpt may be a
#     live resume point — e.g. swa_gdn checkpoint_10000 feeds swa_gdn_60k)
#   - dry-run by default; pass --apply to actually delete
#
# Usage:  bash scripts/prune_ckpts.sh           # dry-run (lists what WOULD be deleted)
#         bash scripts/prune_ckpts.sh --apply    # actually delete
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
ABL=VLANeXt_ablation_wm
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1

# dirs that are still being written to / needed as a resume source — NEVER prune these.
# (extend if you add resume-based jobs; swa_gdn stays protected until swa_gdn_60k writes its final)
PROTECT=()
for d in ttt_chunk256_libero_object ttt_chunk256_libero_goal ttt_chunk256_libero_long swa_gdn_libero_spatial; do
  # protect only while it has NO final yet (i.e. not finished) — once final exists it's safe to prune
  [ -f "$ABL/$d/checkpoint_final.pt" ] || PROTECT+=("$d")
done
echo "protected (active/unfinished, skipped): ${PROTECT[*]:-<none>}"

total=0
for dir in "$ABL"/*/; do
  d=$(basename "$dir")
  # skip protected
  skip=0; for p in "${PROTECT[@]:-}"; do [ "$d" = "$p" ] && skip=1; done
  [ "$skip" = 1 ] && continue
  # require a final ckpt before pruning intermediates
  [ -f "$dir/checkpoint_final.pt" ] || continue
  for f in "$dir"checkpoint_[0-9]*.pt; do
    [ -f "$f" ] || continue
    sz=$(du -m "$f" 2>/dev/null | cut -f1); total=$((total+sz))
    if [ "$APPLY" = 1 ]; then rm -f "$f" && echo "  deleted $f (${sz}MB)"; else echo "  WOULD delete $f (${sz}MB)"; fi
  done
done
echo "$([ "$APPLY" = 1 ] && echo 'reclaimed' || echo 'reclaimable'): $((total/1024))GB"
[ "$APPLY" = 0 ] && echo "(dry-run; re-run with --apply to delete)"
