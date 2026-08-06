#!/usr/bin/env python3
"""Temporal open-loop prediction video: walk ONE episode's consecutive timesteps,
predict the future frame at each, stitch into a continuous video.

Per-shard worker: processes timesteps where (i % num_shards == shard), writes
frames/frame_{i:05d}.png = [GT | VQ-recon | pred] side by side. A separate stitch
step (in the launcher) globs all frames in order into an mp4.

Deterministic ordering: buffer_size=1, num_workers=0, fixed seed -> the dataloader
yields episode-0 timesteps t=0,1,2,... in order. Augmentation is applied with a
FIXED seed every frame (same crop) so the input distribution matches training
(avoids the aug-OFF OOD collapse) while staying temporally consistent.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
sys.path.insert(0, REPO)
import scripts.eval_wm_quality as wmq  # noqa: E402 (dynamo-disable guard)
from scripts.eval_wm_quality import load_model, _to_uint8  # noqa: E402
from scripts.gen_molmoact_wm_compare import (  # noqa: E402
    _grid_hw, generate_one_grid, vq_recon, _resize_to,
)
from scripts.train import DataCollatorForVLANeXt, load_config  # noqa: E402
from src.datasets.molmoact_droid_act import MolmoActDroidAct  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

CFG = os.path.join(REPO, "config", "ablation_wm_ttt_chunk256_molmoact_droid.yaml")
AUG_SEED = 7  # fixed -> identical crop every frame (temporally consistent, in-distribution)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-frames", type=int, default=0, help="0 = full episode")
    ap.add_argument("--episode-pos", type=int, default=0, help="which episode (list position)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--aug", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    model, _ = load_model(args.ckpt, device)

    cfg = load_config(CFG); d = cfg["data"]
    torch.manual_seed(1234); np.random.seed(1234)   # deterministic episode iteration
    ds = MolmoActDroidAct(data_root=d["data_root"], dataset_name="molmoact_droid",
        history_len=d["history_len"], future_len=d["future_len"], full_sequence=True,
        input_modality="image", view_mode="multi", load_future_image=True,
        future_image_mode=cfg["model"].get("future_image_mode","horizon"), buffer_size=1)
    # Pin to ONE episode and iterate its FULL length (start -> task end, not a fragment).
    # Build BOTH _episodes and _data_table (the lazy __iter__ init builds both together;
    # building only the episode index leaves _data_table unset -> 0 frames read).
    ds._episodes = ds._build_episode_index()
    ds._data_table = ds._build_data_file_table()
    ep = ds._episodes[args.episode_pos]
    ds._episodes = [ep]
    n_total = args.n_frames if args.n_frames > 0 else int(ep["length"])
    print(f"[shard{args.shard}] episode pos={args.episode_pos} idx={ep['episode_index']} "
          f"length={ep['length']} -> {n_total} frames", flush=True)
    coll = DataCollatorForVLANeXt(processor=model.processor, use_proprio_input_vlm=True,
        use_action_input_policy=False, input_modality="image", view_mode="multi", fps=15.0,
        augmentation=(d["augmentation"] if args.aug else None), load_future_image=True)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=_fixed_aug_collate(coll))

    h = w = None
    it = iter(loader)
    for i in range(n_total):
        try:
            inp, _, prop, _, fimg = next(it)
        except StopIteration:
            print(f"[shard{args.shard}] dataset exhausted at frame {i}"); break
        if i % args.num_shards != args.shard:
            continue
        batch = {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"],
                 "pixel_values": inp["pixel_values"], "image_grid_thw": inp["image_grid_thw"],
                 "proprioception": prop, "future_images": fimg}
        for k in batch: batch[k] = batch[k].to(device)
        if h is None:
            h, w = _grid_hw(model, batch["future_images"])
        s = {k: (batch[k][:1] if k != "image_grid_thw" else batch[k]) for k in batch}
        # image_grid_thw is per-view; single sample so pass through
        pred = generate_one_grid(model, _slice1(batch), device, h, w)
        recon = vq_recon(model, batch["future_images"][0])
        gt = batch["future_images"][0]
        sep = np.full((256, 2, 3), 255, np.uint8)
        row = np.concatenate([_resize_to(_to_uint8(gt)), sep,
                              _resize_to(_to_uint8(recon)), sep,
                              _resize_to(_to_uint8(pred))], axis=1)
        import imageio.v2 as iio
        iio.imwrite(os.path.join(args.outdir, f"frame_{i:05d}.png"), row)
        print(f"[shard{args.shard}] frame {i} done", flush=True)


def _slice1(batch):
    """single-sample dict already; reuse as-is for generate_one_grid."""
    return {k: v for k, v in batch.items()}


def _fixed_aug_collate(coll):
    """Wrap the collator so augmentation uses the SAME RNG seed every call
    -> identical crop/jitter each frame -> temporally consistent video."""
    def _c(samples):
        np.random.seed(AUG_SEED)
        return coll(samples)
    return _c


if __name__ == "__main__":
    main()
