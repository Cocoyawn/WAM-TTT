"""
Deployment (inference) latency + scaling benchmark: TTT vs softmax-attention.

Measures the REAL deployment call: model.predict_action(...) (diffusion 5-step
denoise, no grad), which is what a robot policy actually runs each control step.

Also isolates per-expert latency:
  - action DiT  : the action head alone (diffusion denoise over 8 action tokens)
  - vision DiT  : the image generator alone (autoregressive — only used if the
                  deployment runs world-model image prediction; action policy
                  does NOT call it, so we time it separately)

Scaling: batch sizes 1,2,4,8.

Run:
  PYTHONPATH=$PWD venv/bin/python -m scripts.bench_infer_latency
"""
import time
import torch
from PIL import Image
from src.models.VLANeXt import VLANeXt

QWEN = "/mnt/afs-h200/yuyangcheng/models/Qwen3-VL-2B-Instruct"
DEV, DT = "cuda", torch.bfloat16
WARMUP, ITERS = 3, 10
BATCHES = [1, 2, 4, 8]


def build(mixer, world_model=False):
    torch.manual_seed(0)
    return VLANeXt(
        lmm_path=QWEN, action_dim=7, num_actions=8, num_queries=16,
        loss_type="diffusion", condition_type="soft", scheduler_type="flow_match",
        num_inference_timesteps=5,
        future_image_loss_weight=(1.0 if world_model else 0.0),
        policy_depth=29, policy_num_heads=16,
        policy_mixer_type=mixer, policy_mix_every_n=4,
        generator_depth=29, generator_num_heads=12,
        generator_mixer_type=mixer, generator_mix_every_n=4, generator_ttt_chunk_size=16,
        use_proprio_input_vlm=True, use_transformer_proprio_projector=False,
        backbone_mode="finetune", gradient_checkpointing=False,
        action_vqvae={"enabled": False}, attn_implementation="sdpa",
    ).to(DEV, DT).eval()


def make_inputs(model, B):
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
    return (time.time() - t) / ITERS * 1000  # ms


@torch.no_grad()
def bench_predict_action(mixer):
    """Full deployment call: predict_action. With world_model=True (our real
    config: soft + future_image_loss), this ALSO runs the vision DiT as a
    256-step autoregressive generation before the action head."""
    model = build(mixer, world_model=True)   # real deploy config
    print(f"  --- {mixer}: model.predict_action (full deploy, WORLD-MODEL on) ---")
    for B in BATCHES:
        fwd, proprio = make_inputs(model, B)
        ms = timed(lambda: model.predict_action(proprioception=proprio, **fwd))
        print(f"    B={B}: {ms:8.1f} ms/call | {ms/B:6.1f} ms/sample")
    del model; torch.cuda.empty_cache()


@torch.no_grad()
def bench_experts_isolated(mixer):
    """Isolate action DiT and vision DiT latency (exclude backbone)."""
    model = build(mixer, world_model=True)  # need generator for vision DiT
    print(f"  --- {mixer}: isolated experts (backbone excluded) ---")
    # fake vlm hidden states list (depth+1 layers), real dims
    H = model.hidden_size
    for B in BATCHES:
        vlm_states = [torch.randn(B, 300, H, device=DEV, dtype=DT) for _ in range(30)]
        # action DiT: diffusion denoise 5 steps over 8 action tokens (soft cond uses hidden states)
        def action_call():
            act = torch.randn(B, 8, 7, device=DEV, dtype=DT)
            model.noise_scheduler.set_timesteps(5)
            for t in model.noise_scheduler.timesteps:
                ts = torch.full((B,), t, device=DEV)
                out = model.action_head(act, ts, vlm_states)
                act = model.noise_scheduler.step(out, t, act).prev_sample.to(DT)
            return act
        ms_a = timed(action_call)
        # vision DiT: one generator forward over 256 image tokens
        def vision_call():
            ids = torch.randint(0, model.vq_codebook_size, (B, 256), device=DEV)
            return model.generator(ids, vlm_states)
        ms_v = timed(vision_call)
        print(f"    B={B}: action DiT (5-step denoise)={ms_a:8.1f} ms | vision DiT (256 tok fwd)={ms_v:8.1f} ms")
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=== DEPLOYMENT INFERENCE LATENCY (predict_action, diffusion 5-step) ===")
    for mixer in ["attention", "ttt"]:
        bench_predict_action(mixer)
    print("\n=== ISOLATED EXPERT LATENCY (action DiT vs vision DiT) ===")
    for mixer in ["attention", "ttt"]:
        bench_experts_isolated(mixer)
