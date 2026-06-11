#!/usr/bin/env python3
"""Cache a fixed set of world-model eval samples to disk.

Why: the LIBERO dataset is an IterableDataset with a non-seeded numpy shuffle
buffer (src/datasets/libero_act_old.py), so two eval runs never see the same
inputs. We dump a fixed batch ONCE here, then evaluate both the TTT and the
attention checkpoint on the identical cached tensors — apples-to-apples.

Reuses the exact training-time preprocessing by importing the real collator
(DataCollatorForVLANeXt) and dataset (LiberoAct) from the training stack, so
future_image normalization, Qwen chat template, proprio, etc. all match.

Output: VLANeXt_ablation_wm/wm_eval_cache/{suite}.pt holding a dict with
pixel_values / input_ids / attention_mask / image_grid_thw (if present) /
proprioception / future_images / instruction, for N samples (collated).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
sys.path.insert(0, REPO)

from scripts.train import DataCollatorForVLANeXt  # noqa: E402
from src.datasets.libero_act_old import LiberoAct  # noqa: E402

DATA_ROOT = "/mnt/afs-h200/yuyangcheng/data/LIBERO_modified"
LMM_PATH = "/mnt/afs-h200/yuyangcheng/models/Qwen3-VL-2B-Instruct"
CACHE_DIR = os.path.join(REPO, "VLANeXt_ablation_wm", "wm_eval_cache")

# Match the world-model training config (config/ablation_wm_ttt.yaml).
HISTORY_LEN = 8
FUTURE_LEN = 8
VIEW_MODE = "multi"
INPUT_MODALITY = "image"


def collate_keys(inputs, gt_actions, proprio, hist_actions, future_images):
    """Pull the tensors we need out of the collator's tuple output into a dict."""
    out = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "future_images": future_images,
        "proprioception": proprio,
    }
    for k in ("image_grid_thw", "video_grid_thw", "token_type_ids", "pixel_values_videos"):
        if k in inputs:
            out[k] = inputs[k]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64, help="samples per suite")
    ap.add_argument(
        "--buffer",
        type=int,
        default=2000,
        help="shuffle buffer for cross-trajectory diversity. We cache once to "
        "disk so non-determinism is harmless; both ckpts read the same file.",
    )
    ap.add_argument(
        "--suites",
        nargs="+",
        default=[
            "libero_spatial_no_noops",
            "libero_object_no_noops",
            "libero_goal_no_noops",
        ],
    )
    ap.add_argument("--out_dir", default=CACHE_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[cache] loading processor from {LMM_PATH}")
    processor = AutoProcessor.from_pretrained(LMM_PATH, trust_remote_code=True)

    # eval = clean frames: augmentation OFF.
    collator = DataCollatorForVLANeXt(
        processor=processor,
        use_proprio_input_vlm=True,
        use_action_input_policy=False,
        input_modality=INPUT_MODALITY,
        view_mode=VIEW_MODE,
        augmentation={"enabled": False},
        load_future_image=True,
    )

    for suite in args.suites:
        data_path = os.path.join(DATA_ROOT, suite, "1.0.0")
        if not os.path.isdir(data_path):
            print(f"[cache] SKIP {suite}: {data_path} not found")
            continue
        print(f"[cache] {suite}: streaming {args.n} samples (shuffle off) ...")
        ds = LiberoAct(
            data_path=data_path,
            dataset_name=suite.replace("_no_noops", ""),
            history_len=HISTORY_LEN,
            future_len=FUTURE_LEN,
            full_sequence=True,
            input_modality=INPUT_MODALITY,
            view_mode=VIEW_MODE,
            load_future_image=True,
            future_image_mode="horizon",
            buffer_size=args.buffer,  # mix across trajectories for scene diversity
        )
        # batch_size=n so a single collated batch is exactly our eval set.
        loader = DataLoader(ds, batch_size=args.n, collate_fn=collator, num_workers=0)
        batch = next(iter(loader))
        inputs, gt_actions, proprio, hist_actions, future_images = batch
        cached = collate_keys(inputs, gt_actions, proprio, hist_actions, future_images)
        n_got = cached["future_images"].shape[0]
        out_path = os.path.join(args.out_dir, f"{suite}.pt")
        torch.save(cached, out_path)
        print(
            f"[cache] {suite}: saved {n_got} samples -> {out_path} "
            f"(future_images {tuple(cached['future_images'].shape)}, "
            f"input_ids {tuple(cached['input_ids'].shape)})"
        )


if __name__ == "__main__":
    main()
