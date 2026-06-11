#!/usr/bin/env python3
"""Evaluate world-model (future-frame) generation quality: TTT vs attention.

Answers the real question behind "TTT 的训练 loss 低很多": does that translate
into better *generated video*, and is it generalization or just in-dist fitting?

Two layers, run on a SHARED cached eval set (see cache_wm_eval_samples.py):

  loss : teacher-forced image-token cross-entropy ISOLATED from the action/dct
         terms, plus top-1 token accuracy. Cheap. Decomposes the training-loss
         gap and checks whether it survives OOD (object/goal).

  gen  : free-running greedy AR generation via model.predict_image -> VQ decode
         -> pixels, scored vs the GT future frame with PSNR / SSIM (+ LPIPS if
         installed). This is what "video quality" actually means at inference.
         Also dumps GT|pred comparison grids for visual judgment.

Model is rebuilt from the config embedded in each checkpoint, so structure
matches exactly regardless of which ablation produced it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# The TTT fast-weight kernels (src/models/ttt.py) are @torch.compile(dynamic=True).
# During free-running AR generation the sequence length grows 1->256, so dynamo
# recompiles at every length and thrashes (GPU starves at ~15% util, 256x slower).
# Eager is numerically identical and avoids the recompile storm — force it off
# globally BEFORE the model (and its compiled kernels) are imported.
if os.environ.get("EVAL_WM_NO_COMPILE", "1") == "1":
    import torch._dynamo
    torch._dynamo.config.disable = True

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
sys.path.insert(0, REPO)

from src.models.VLANeXt import VLANeXt  # noqa: E402

CACHE_DIR = os.path.join(REPO, "VLANeXt_ablation_wm", "wm_eval_cache")
NUM_IMG_TOKENS = 256  # 16x16 latent grid from the Emu3.5 VQ (256x256 / 16x)


# --------------------------------------------------------------------------- #
# model build (mirrors scripts/train.py model construction)
# --------------------------------------------------------------------------- #
def build_model_from_config(config, device):
    m = config["model"]
    d = config["data"]
    model = VLANeXt(
        lmm_path=m["lmm_path"],
        vision_encoder_path=m.get("vision_encoder_path", "google/siglip2-base-patch16-256"),
        action_dim=m["action_dim"],
        num_actions=d["future_len"],
        num_queries=m["num_queries"],
        num_history=d["history_len"],
        loss_type=m.get("loss_type", "diffusion"),
        future_image_loss_weight=float(m.get("future_image_loss_weight", 0.0)),
        num_train_timesteps=m.get("num_train_timesteps", 1000),
        num_inference_timesteps=m.get("num_inference_timesteps", 10),
        scheduler_type=m["scheduler_type"],
        condition_type=m.get("condition_type", "loose"),
        policy_hidden_size=m["policy_hidden_size"],
        policy_depth=m["policy_depth"],
        policy_num_heads=m["policy_num_heads"],
        policy_mlp_ratio=m["policy_mlp_ratio"],
        policy_mixer_type=m.get("policy_mixer_type", "attention"),
        policy_mix_every_n=m.get("policy_mix_every_n", 4),
        use_proprio_input_vlm=m.get("use_proprio_input_vlm", True),
        use_action_input_policy=m.get("use_action_input_policy", False),
        use_transformer_proprio_projector=m["use_transformer_proprio_projector"],
        projector_depth=m["projector_depth"],
        projector_num_heads=m["projector_num_heads"],
        use_transformer_connector=m["use_transformer_connector"],
        connector_depth=m["connector_depth"],
        connector_num_heads=m["connector_num_heads"],
        backbone_mode=m.get("backbone_mode", "finetune"),
        gradient_checkpointing=False,
        num_bins=m.get("num_bins", 256),
        generator_hidden_size=m.get("generator_hidden_size", 768),
        generator_depth=m.get("generator_depth", 12),
        generator_num_heads=m.get("generator_num_heads", 12),
        generator_mlp_ratio=m.get("generator_mlp_ratio", 4.0),
        generator_mixer_type=m.get("generator_mixer_type", "attention"),
        generator_mix_every_n=m.get("generator_mix_every_n", 4),
        generator_ttt_chunk_size=m.get("generator_ttt_chunk_size", 16),
        action_vqvae=m.get("action_vqvae", None),
        dct_loss_weight=m.get("dct_loss_weight", 0.1),
        dct_low_freq_weight=m.get("dct_low_freq_weight", 1.0),
        dct_high_freq_weight=m.get("dct_high_freq_weight", 1.0),
        dct_freq_split=m.get("dct_freq_split", 0.125),
        dct_similarity_type=m.get("dct_similarity_type", "mae"),
        attn_implementation=m.get("attn_implementation", "sdpa"),
    ).to(device, dtype=torch.bfloat16)
    return model


def load_model(ckpt_path, device):
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    mixer = config["model"].get("generator_mixer_type", "attention")
    chunk = config["model"].get("generator_ttt_chunk_size", "-")
    print(f"[load] generator_mixer_type={mixer} chunk_size={chunk} step={ckpt.get('step')}")
    model = build_model_from_config(config, device)
    state = ckpt["model_state_dict"]
    # Checkpoints saved under DDP carry a "module." prefix on every key; strip it
    # so the keys line up with the unwrapped model. (The attention ckpt was saved
    # DDP-wrapped, the TTT one was not — handle both.)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # vq_model is loaded fresh from pretrained (frozen) and may not be in the ckpt;
    # anything else missing/unexpected means a real load failure -> abort loudly.
    miss_real = [k for k in missing if not k.startswith("vq_model.")]
    unexp_real = [k for k in unexpected if not k.startswith("vq_model.")]
    if miss_real or unexp_real:
        raise RuntimeError(
            f"state_dict mismatch loading {ckpt_path}: "
            f"{len(miss_real)} missing (e.g. {miss_real[:4]}), "
            f"{len(unexp_real)} unexpected (e.g. {unexp_real[:4]})")
    model.eval()
    return model, config


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_cache(suite, device, limit=None):
    path = os.path.join(CACHE_DIR, f"{suite}.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = d["future_images"].shape[0]
    if limit is not None:
        n = min(n, limit)
    batch = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            batch[k] = v
    return batch, n


def slice_sample(batch, i, device):
    """Pull sample i. Qwen vision tensors (pixel_values/image_grid_thw) are
    flattened across the batch with 2 views per sample, so index by view pairs."""
    out = {}
    out["input_ids"] = batch["input_ids"][i : i + 1].to(device)
    out["attention_mask"] = batch["attention_mask"][i : i + 1].to(device)
    out["proprioception"] = batch["proprioception"][i : i + 1].to(device)
    out["future_images"] = batch["future_images"][i : i + 1].to(device)
    # image_grid_thw: (B*views, 3); pixel_values: (sum tokens, dim)
    grid = batch["image_grid_thw"]
    views = grid.shape[0] // batch["input_ids"].shape[0]
    g_lo, g_hi = i * views, (i + 1) * views
    out["image_grid_thw"] = grid[g_lo:g_hi].to(device)
    tok = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    starts = np.cumsum([0] + tok)
    p_lo, p_hi = int(starts[g_lo]), int(starts[g_hi])
    out["pixel_values"] = batch["pixel_values"][p_lo:p_hi].to(device)
    return out


# --------------------------------------------------------------------------- #
# Layer A: isolated teacher-forced image loss + token acc
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_loss(model, batch, n, device):
    losses, accs = [], []
    for i in range(n):
        s = slice_sample(batch, i, device)
        _, hidden = model.get_vlm_condition(
            s["input_ids"], s["attention_mask"],
            proprioception=s["proprioception"],
            pixel_values=s["pixel_values"].to(model.vq_model.dtype) if False else s["pixel_values"],
            image_grid_thw=s["image_grid_thw"],
        )
        fimg = s["future_images"].to(device=model.vq_model.device, dtype=model.vq_model.dtype)
        _, _, (_, _, token_ids) = model.vq_model.encode(fimg)
        B = fimg.shape[0]
        token_ids = token_ids.view(B, -1)
        sos = torch.zeros((B, 1), dtype=token_ids.dtype, device=token_ids.device)
        gen_input = torch.cat([sos, token_ids[:, :-1]], dim=1)
        logits, _ = model.generator(gen_input, hidden)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vq_codebook_size).float(), token_ids.reshape(-1)
        )
        acc = (logits.argmax(-1) == token_ids).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())
    return float(np.mean(losses)), float(np.std(losses)), float(np.mean(accs))


# --------------------------------------------------------------------------- #
# Layer B: free-running generation -> pixels -> PSNR/SSIM(+LPIPS)
# --------------------------------------------------------------------------- #
def _to_uint8(img_chw_m11):
    """[-1,1] CHW float -> HWC uint8 [0,255]."""
    x = ((img_chw_m11.float().clamp(-1, 1) + 1) / 2 * 255).round().clamp(0, 255)
    return x.permute(1, 2, 0).to(torch.uint8).cpu().numpy()


@torch.no_grad()
def generate_one(model, s, device):
    _, hidden = model.get_vlm_condition(
        s["input_ids"], s["attention_mask"],
        proprioception=s["proprioception"],
        pixel_values=s["pixel_values"],
        image_grid_thw=s["image_grid_thw"],
    )
    curr = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(NUM_IMG_TOKENS):
        logits, _ = model.generator(curr, hidden)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr = torch.cat([curr, nxt], dim=1)
    gen_tokens = curr[:, 1:]
    H = int(gen_tokens.shape[1] ** 0.5)
    dec = model.vq_model.decode_code(gen_tokens, shape=(1, H, H))  # [-1,1], (1,3,256,256)
    return dec[0]


@torch.no_grad()
def eval_gen(model, batch, n, device, grid_path=None, n_grid=8):
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    lpips_model = _maybe_lpips(device)
    psnrs, ssims, lpipss = [], [], []
    grid_rows = []
    for i in range(n):
        s = slice_sample(batch, i, device)
        pred = generate_one(model, s, device)
        gt = s["future_images"][0]
        pred_u8 = _to_uint8(pred)
        gt_u8 = _to_uint8(gt)
        psnrs.append(psnr_fn(gt_u8, pred_u8, data_range=255))
        ssims.append(ssim_fn(gt_u8, pred_u8, data_range=255, channel_axis=2))
        if lpips_model is not None:
            lp = lpips_model(
                pred.float().clamp(-1, 1).unsqueeze(0),
                gt.float().clamp(-1, 1).unsqueeze(0),
            ).item()
            lpipss.append(lp)
        if grid_path is not None and i < n_grid:
            grid_rows.append(np.concatenate([gt_u8, pred_u8], axis=1))  # GT | pred
    if grid_path is not None and grid_rows:
        import imageio.v2 as imageio
        imageio.imwrite(grid_path, np.concatenate(grid_rows, axis=0))
        print(f"[gen] grid -> {grid_path}")
    res = {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
    }
    if lpipss:
        res["lpips_mean"] = float(np.mean(lpipss))
        res["lpips_std"] = float(np.std(lpipss))
    return res


def _maybe_lpips(device):
    try:
        import lpips  # type: ignore
        net = lpips.LPIPS(net="alex").to(device).eval()
        print("[gen] LPIPS available (alex)")
        return net
    except Exception as e:  # noqa: BLE001
        print(f"[gen] LPIPS unavailable, skipping ({type(e).__name__})")
        return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["loss", "gen"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="short model tag, e.g. ttt / attn")
    ap.add_argument("--suites", nargs="+", default=[
        "libero_spatial_no_noops", "libero_object_no_noops", "libero_goal_no_noops"])
    ap.add_argument("--limit", type=int, default=None, help="cap samples/suite")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, config = load_model(args.ckpt, device)

    results = {}
    for suite in args.suites:
        path = os.path.join(CACHE_DIR, f"{suite}.pt")
        if not os.path.exists(path):
            print(f"[skip] no cache for {suite}")
            continue
        batch, n = load_cache(suite, device, limit=args.limit)
        if args.mode == "loss":
            mean, std, acc = eval_loss(model, batch, n, device)
            results[suite] = {"n": n, "loss_img_mean": mean, "loss_img_std": std,
                              "token_acc": acc}
            print(f"[loss] {args.tag} {suite}: n={n} loss_img={mean:.4f}±{std:.4f} "
                  f"token_acc={acc:.4f}")
        else:
            os.makedirs(os.path.join(CACHE_DIR, "grids"), exist_ok=True)
            gp = os.path.join(CACHE_DIR, "grids", f"{suite}_{args.tag}.png")
            res = eval_gen(model, batch, n, device, grid_path=gp)
            res["n"] = n
            results[suite] = res
            extra = f" lpips={res['lpips_mean']:.4f}" if "lpips_mean" in res else ""
            print(f"[gen] {args.tag} {suite}: n={n} psnr={res['psnr_mean']:.2f} "
                  f"ssim={res['ssim_mean']:.4f}{extra}")

    out_json = args.out_json or os.path.join(
        CACHE_DIR, f"result_{args.mode}_{args.tag}.json")
    with open(out_json, "w") as f:
        json.dump({"tag": args.tag, "ckpt": args.ckpt, "mode": args.mode,
                   "results": results}, f, indent=2)
    print(f"[done] -> {out_json}")


if __name__ == "__main__":
    main()
