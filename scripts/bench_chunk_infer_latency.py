"""
Deployment-inference latency vs vision-DiT TTT chunk_size.

Tests the user's hypothesis: at deployment the world-model vision DiT runs a
256-step autoregressive loop; in each step the CAUSAL TTT operator does
ceil(seq_len / chunk_size) fast-weight updates (each a Newton-Schulz). Larger
chunk_size => fewer updates per step => potentially faster predict_action.

We build ONE world-model VLANeXt with mixer_type='ttt' and, for each chunk_size,
swap the chunk_size on every vision-expert TTT layer in place (weights identical;
only the operator's block granularity changes), then time predict_action.

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
    venv/bin/python -m scripts.bench_chunk_infer_latency
"""
import time
import torch
from PIL import Image
from src.models.VLANeXt import VLANeXt
from src.models.ttt import FastWeightGluMLPMultihead

QWEN = "/mnt/afs-h200/yuyangcheng/models/Qwen3-VL-2B-Instruct"
DEV, DT = "cuda", torch.bfloat16
WARMUP, ITERS = 2, 5
CHUNKS = [16, 64, 128, 256]


def build(mixer="ttt"):
    torch.manual_seed(0)
    return VLANeXt(
        lmm_path=QWEN, action_dim=7, num_actions=8, num_queries=16,
        loss_type="diffusion", condition_type="soft", scheduler_type="flow_match",
        num_inference_timesteps=5,
        future_image_loss_weight=1.0,                       # world-model ON (runs vision AR)
        policy_depth=29, policy_num_heads=16,
        policy_mixer_type=mixer, policy_mix_every_n=4,
        generator_depth=29, generator_num_heads=12,
        generator_mixer_type=mixer, generator_mix_every_n=4, generator_ttt_chunk_size=16,
        use_proprio_input_vlm=True, use_transformer_proprio_projector=False,
        backbone_mode="finetune", gradient_checkpointing=False,
        action_vqvae={"enabled": False}, attn_implementation="sdpa",
    ).to(DEV, DT).eval()


def set_vision_chunk(model, cs):
    """Set chunk_size on every causal TTT layer in the vision generator."""
    n = 0
    for m in model.generator.modules():
        if isinstance(m, FastWeightGluMLPMultihead) and m.causal:
            m.chunk_size = cs
            n += 1
    return n


def make_inputs(model, B=1):
    proc = model.processor
    img = Image.new("RGB", (256, 256), "red")
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "pick up the black bowl and place it on the plate"},
    ]}] for _ in range(B)]
    texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    inp = proc(text=texts, images=[img]*B, padding=True, return_tensors="pt")
    inp = {k: v.to(DEV) for k, v in inp.items()}
    if "pixel_values" in inp:
        inp["pixel_values"] = inp["pixel_values"].to(DT)
    valid = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    fwd = {k: v for k, v in inp.items() if k in valid}
    proprio = torch.randn(B, 1, 7, device=DEV, dtype=DT)
    return fwd, proprio


@torch.no_grad()
def timed(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize()
    return (time.time() - t) / ITERS


if __name__ == "__main__":
    print("=== predict_action deployment latency: softmax-ATTENTION vs TTT chunk-size (world-model AR 256 steps) ===")
    print(f"(B=1, {ITERS} iters after {WARMUP} warmup)\n")

    # --- baseline: softmax attention (same shell, mixer=attention) ---
    amodel = build(mixer="attention")
    afwd, aproprio = make_inputs(amodel, B=1)
    attn_s = timed(lambda: amodel.predict_action(proprioception=aproprio, **afwd))
    print(f"attention      |  baseline           | {attn_s*1000:8.1f} ms/call | 1.00x (ref)")
    del amodel; torch.cuda.empty_cache()

    # --- TTT chunk-size sweep ---
    model = build(mixer="ttt")
    fwd, proprio = make_inputs(model, B=1)
    for cs in CHUNKS:
        nlayers = set_vision_chunk(model, cs)
        s = timed(lambda: model.predict_action(proprioception=proprio, **fwd))
        upd = sum((t + cs - 1)//cs for t in range(1, 257))
        print(f"ttt chunk={cs:4d} | {nlayers} vision-TTT layers | {s*1000:8.1f} ms/call "
              f"| {attn_s/s:4.2f}x vs attention | ~{upd} fw-updates/AR-rollout")
    del model; torch.cuda.empty_cache()
