"""Real-condition image-generation quality compare: image DiT, 3 paths.

Uses a REAL LIBERO observation -> Qwen VLM -> gen_context, then generates the 256
VQ image tokens THREE ways and decodes each to a 256x256 image via the Emu3.5 VQ
decoder, plus the GT future-image tokens as reference:
  torch : full-recompute, ttt_use_cuda_kernel=False  (baseline)
  K     : full-recompute, ttt_use_cuda_kernel=True    (CUDA fused kernel)
  I     : generate_incremental, ttt_use_cuda_kernel=True (incremental O(n))
Same gen_context for all three -> only the image-DiT operator/decode path differs.

Saves docs/image_quality_compare.png (torch | K | I side by side) and prints
pairwise token-match % + pixel PSNR/L1.

Run (clean-ish GPU, sim needs OSMesa):
  TORCHDYNAMO_DISABLE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg \
  PYTHONPATH="$REPO/third_party/LIBERO-plus:$REPO" CUDA_VISIBLE_DEVICES=<g> \
  python scripts/gen_image_quality_compare.py
"""
import os, sys, types, math
import numpy as np
import torch
from PIL import Image

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
CFG_PATH = f"{REPO}/config/libero_plus_envgen_mix5dim_libero_spatial_shard0.yaml"
OUT = f"{REPO}/docs/image_quality_compare.png"
NIMG = 256
GRID = 16  # 256 tokens -> 16x16

sys.path.insert(0, REPO)
from omegaconf import OmegaConf
from src.evaluation.libero_bench.VLANeXt_utils import get_vla, get_processor, get_vla_action
from src.evaluation.libero_bench.libero_utils import (
    get_libero_env, get_libero_image, quat2axisangle, get_libero_dummy_action)
from libero.libero import benchmark


@torch.no_grad()
def build_gen_context(model, processor, observation, task_desc):
    """Reuse get_vla_action's obs->inputs construction, but intercept the model
    inputs at get_vlm_condition and abort before the (expensive) rest of
    predict_action. Then run get_vlm_condition ourselves to get gen_context."""
    captured = {}
    orig = model.get_vlm_condition

    class _Stop(Exception):
        pass

    def spy(input_ids, attention_mask, **kw):
        captured["input_ids"] = input_ids
        captured["attention_mask"] = attention_mask
        captured["kw"] = kw
        raise _Stop()  # abort predict_action right after we have the inputs

    model.get_vlm_condition = spy
    try:
        get_vla_action({}, model, processor, observation, task_desc)
    except _Stop:
        pass
    finally:
        model.get_vlm_condition = orig
    if "input_ids" not in captured:
        raise RuntimeError("failed to capture VLM inputs")
    _, hidden_states = orig(captured["input_ids"], captured["attention_mask"], **captured["kw"])
    return hidden_states


@torch.no_grad()
def gen_tokens_full(model, gen_context, use_cuda):
    gen = model.generator
    for blk in gen.blocks:
        if getattr(blk, "mixer_type", None) == "ttt":
            blk.attn.use_cuda_kernel = use_cuda
    B = gen_context[0].shape[0]
    curr = torch.zeros((B, 1), dtype=torch.long, device=gen_context[0].device)
    for _ in range(NIMG):
        lg, _ = gen(curr, gen_context)
        curr = torch.cat([curr, torch.argmax(lg[:, -1:], dim=-1)], dim=1)
    return curr[:, 1:]


@torch.no_grad()
def gen_tokens_incr(model, gen_context, use_cuda):
    gen = model.generator
    for blk in gen.blocks:
        if getattr(blk, "mixer_type", None) == "ttt":
            blk.attn.use_cuda_kernel = use_cuda
    ids, _ = gen.generate_incremental(gen_context, NIMG)
    return ids


@torch.no_grad()
def decode_img(model, tokens):
    vq = model.vq_model
    t = tokens.view(-1).to(vq.device)
    dec = vq.decode_code(t, shape=(1, GRID, GRID))  # (1,3,H,W) in [-1,1] typ.
    img = dec.float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
    return ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)


def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 99.0 if mse < 1e-6 else 20 * math.log10(255.0 / math.sqrt(mse))


if __name__ == "__main__":
    cfg = OmegaConf.load(CFG_PATH)
    print("loading model (Qwen + generator + vq)...")
    model = get_vla(cfg)
    processor = get_processor(cfg)
    model.eval()

    # one real LIBERO observation
    bm = benchmark.get_benchmark_dict()[cfg.eval.task_suite_name]()
    task = bm.get_task(0)
    init_states = bm.get_task_init_states(0)
    env, task_desc = get_libero_env(task, "vlanext", resolution=256)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(10):  # settle
        obs, _, _, _ = env.step(get_libero_dummy_action("vlanext"))
    img = get_libero_image(obs, 256, center_crop=True, center_crop_ratio=0.9,
                           obs_key="agentview_image")
    gripper = np.clip(1 - (np.mean(np.abs(obs["robot0_gripper_qpos"])) / 0.04), 0.0, 1.0)
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [gripper]))
    observation = {"full_image": img, "full_image_wrist": img, "image_history": [img],
                   "image_history_wrist": [], "state_history": [state], "action_history": []}
    env.close()
    print("got real obs, task:", task_desc)

    gen_context = build_gen_context(model, processor, observation, task_desc)
    print(f"gen_context: {len(gen_context)} layers, shape {tuple(gen_context[0].shape)}")

    ids_torch = gen_tokens_full(model, gen_context, use_cuda=False)
    ids_K = gen_tokens_full(model, gen_context, use_cuda=True)
    ids_I = gen_tokens_incr(model, gen_context, use_cuda=True)

    m_TK = float((ids_torch == ids_K).float().mean() * 100)
    m_TI = float((ids_torch == ids_I).float().mean() * 100)
    print(f"\ntoken match: torch-vs-K(cuda full)={m_TK:.1f}%   torch-vs-I(incr)={m_TI:.1f}%")

    im_torch = decode_img(model, ids_torch)
    im_K = decode_img(model, ids_K)
    im_I = decode_img(model, ids_I)
    print(f"pixel PSNR vs torch:  K={psnr(im_torch, im_K):.2f} dB   I={psnr(im_torch, im_I):.2f} dB")
    print(f"pixel L1 vs torch:    K={np.abs(im_torch.astype(int)-im_K).mean():.2f}   "
          f"I={np.abs(im_torch.astype(int)-im_I).mean():.2f}")

    # side-by-side panel: real-obs | torch | K | I
    labels = ["real obs (input)", "torch (full)", "CUDA-kernel (full)", "incremental"]
    panels = [np.array(Image.fromarray(img).resize((256, 256))), im_torch, im_K, im_I]
    gap = 8
    W = 256 * 4 + gap * 3
    canvas = np.full((256, W, 3), 255, np.uint8)
    x = 0
    for p in panels:
        canvas[:, x:x + 256] = p
        x += 256 + gap
    Image.fromarray(canvas).save(OUT)
    print("saved", OUT)
