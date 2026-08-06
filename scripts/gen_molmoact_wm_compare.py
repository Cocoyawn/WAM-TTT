#!/usr/bin/env python3
"""RoboLab / MolmoAct2-DROID world-model future-frame visualization.

Feeds REAL molmoact_droid samples into the (already-tested, LIBERO-agnostic)
generation+VQ-decode pipeline from scripts/eval_wm_quality.py and dumps a
GT | predicted comparison grid for a given checkpoint (e.g. the TTT 60k run).

Run (idle GPU, fla_triton32 venv):
  TORCHDYNAMO_DISABLE=1 EVAL_WM_NO_COMPILE=1 PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=4 \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python \
    scripts/gen_molmoact_wm_compare.py \
    --ckpt VLANeXt_robolab_ttt/chunk256_molmoact_droid/checkpoint_final.pt \
    --tag ttt60k --n 8 --out docs/robolab_ttt_wm_pred.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
sys.path.insert(0, REPO)

# eval_wm_quality sets the EVAL_WM_NO_COMPILE dynamo guard at import time, BEFORE
# the TTT compiled kernels are pulled in. Import it first and reuse its helpers.
import torch.nn.functional as F  # noqa: E402
import scripts.eval_wm_quality as wmq  # noqa: E402
from scripts.eval_wm_quality import (  # noqa: E402
    load_model, slice_sample, _to_uint8,
)


@torch.no_grad()
def _grid_hw(model, future_images):
    """Real VQ token grid (h, w) for these frames — NOT the LIBERO-hardcoded 16x16."""
    fimg = future_images[:1].to(device=model.vq_model.device, dtype=model.vq_model.dtype)
    tok = model.vq_model.encode(fimg)[2][2]
    if tok.dim() == 3:
        return tok.shape[1], tok.shape[2]
    n = tok.view(1, -1).shape[1]
    # DROID 180x320 -> 11x20; fall back to a near-square factorization otherwise.
    return (11, 20) if n == 220 else (int(n ** 0.5), int(n ** 0.5))


@torch.no_grad()
def generate_one_grid(model, s, device, h, w):
    """Greedy AR over the CORRECT number of tokens (h*w), decoded on the (h,w) grid."""
    _, hidden = model.get_vlm_condition(
        s["input_ids"], s["attention_mask"], proprioception=s["proprioception"],
        pixel_values=s["pixel_values"], image_grid_thw=s["image_grid_thw"],
    )
    ntok = h * w
    curr = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(ntok):
        logits, _ = model.generator(curr, hidden)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr = torch.cat([curr, nxt], dim=1)
    gen = curr[:, 1:]
    dec = model.vq_model.decode_code(gen, shape=(1, h, w))  # [-1,1]
    return dec[0]


@torch.no_grad()
def vq_recon(model, future_images_i):
    """Token upper-bound: encode GT -> decode. Shows what the tokens CAN represent."""
    fimg = future_images_i.unsqueeze(0).to(device=model.vq_model.device, dtype=model.vq_model.dtype)
    tok = model.vq_model.encode(fimg)[2][2].view(1, -1)
    h, w = _grid_hw(model, future_images_i.unsqueeze(0))
    return model.vq_model.decode_code(tok, shape=(1, h, w))[0]
from scripts.train import DataCollatorForVLANeXt, load_config  # noqa: E402
from src.datasets.molmoact_droid_act import MolmoActDroidAct  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

CFG = os.path.join(REPO, "config", "ablation_wm_ttt_chunk256_molmoact_droid.yaml")


def build_batch(model, n, device, seed=1234, aug=False):
    """Pull one molmoact_droid batch and reshape it into the cache-format dict
    that slice_sample/generate_one expect. aug=True matches the TRAINING input
    distribution (RandomResizedCrop + color jitter); aug=False = clean inputs."""
    cfg = load_config(CFG)
    d = cfg["data"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = MolmoActDroidAct(
        data_root=d["data_root"],
        dataset_name="molmoact_droid",
        history_len=d["history_len"],
        future_len=d["future_len"],
        full_sequence=bool(d.get("full_sequence", False)),
        input_modality="image",
        view_mode="multi",
        load_future_image=True,
        future_image_mode=cfg["model"].get("future_image_mode", "horizon"),
        buffer_size=d.get("buffer_size", 256),
    )
    # aug=True reproduces training's input distribution; aug=False = clean.
    collator = DataCollatorForVLANeXt(
        processor=model.processor,
        use_proprio_input_vlm=True,
        use_action_input_policy=False,
        input_modality="image",
        view_mode="multi",
        fps=15.0,
        augmentation=(d["augmentation"] if aug else None),
        load_future_image=True,
    )
    loader = DataLoader(ds, batch_size=n, num_workers=4, collate_fn=collator)
    inputs, _gt_actions, proprio, _hist, future_images = next(iter(loader))

    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "proprioception": proprio,
        "future_images": future_images,
    }
    return batch


def _resize_to(u8_hwc, size=256):
    if u8_hwc.shape[0] == size and u8_hwc.shape[1] == size:
        return u8_hwc
    from PIL import Image
    return np.asarray(Image.fromarray(u8_hwc).resize((size, size), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="ttt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "robolab_ttt_wm_pred.png"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--aug", action="store_true",
                    help="match TRAINING input distribution (RandomResizedCrop+jitter); "
                         "default off = clean inputs (OOD to a model trained with aug)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, config = load_model(args.ckpt, device)
    step = config and None  # step is printed by load_model already

    print(f"[data] building molmoact_droid batch (n={args.n}, aug={args.aug}) ...")
    batch = build_batch(model, args.n, device, seed=args.seed, aug=args.aug)
    n = batch["input_ids"].shape[0]
    print(f"[data] got {n} samples")

    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    h, w = _grid_hw(model, batch["future_images"])
    print(f"[grid] VQ token grid = {h}x{w} ({h*w} tokens)")

    # teacher-forced image-token CE + top1 acc on the SAME batch (cheap, no AR).
    tf_mean, tf_std, tf_acc = wmq.eval_loss(model, batch, n, device)
    print(f"[TF] {args.tag}: loss_img={tf_mean:.4f}+/-{tf_std:.4f} token_acc={tf_acc:.4f} "
          f"(random CE={np.log(model.vq_codebook_size):.2f})")

    rows, psnrs, ssims = [], [], []
    for i in range(n):
        s = slice_sample(batch, i, device)
        pred = generate_one_grid(model, s, device, h, w)   # (3,Hpix,Wpix) in [-1,1]
        recon = vq_recon(model, s["future_images"][0])     # token upper bound
        gt = s["future_images"][0]
        pred_u8 = _resize_to(_to_uint8(pred))
        recon_u8 = _resize_to(_to_uint8(recon))
        gt_u8 = _resize_to(_to_uint8(gt))
        psnrs.append(psnr_fn(gt_u8, pred_u8, data_range=255))
        ssims.append(ssim_fn(gt_u8, pred_u8, data_range=255, channel_axis=2))
        # GT | VQ-recon (token ceiling) | model prediction
        sep = np.full((gt_u8.shape[0], 2, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([gt_u8, sep, recon_u8, sep, pred_u8], axis=1))
        print(f"  [{i}] psnr={psnrs[-1]:.2f} ssim={ssims[-1]:.4f}")

    # White separator rows between samples.
    hsep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    stacked = rows[0]
    for r in rows[1:]:
        stacked = np.concatenate([stacked, hsep, r], axis=0)

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.imwrite(args.out, stacked)
    print(f"\n[done] {args.tag}: n={n}  grid={h}x{w}  cols = GT | VQ-recon | {args.tag}-pred  -> {args.out}")
    print(f"[done] psnr_mean={np.mean(psnrs):.2f}  ssim_mean={np.mean(ssims):.4f}")


if __name__ == "__main__":
    main()
