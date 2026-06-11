"""
Controlled real-scenario speed benchmark: TTT vs softmax-attention.

Same GPU, same batch, same world-model setting (future_image_loss), same
torch.compile setting, same everything EXCEPT policy/generator mixer_type.
Measures real training step time (forward + backward), which is the true
"real usage" cost.

Run:
  TORCHDYNAMO_DISABLE=0 PYTHONPATH=$PWD venv/bin/python -m scripts.bench_ttt_vs_attn
"""
import time
import torch
from PIL import Image
from src.models.VLANeXt import VLANeXt

QWEN = "/mnt/afs-h200/yuyangcheng/models/Qwen3-VL-2B-Instruct"
DEV, DT = "cuda", torch.bfloat16
B = 8          # real training batch (matches config batch_size=16 split; use 8 for one card)
WARMUP, ITERS = 5, 20


def build(mixer):
    torch.manual_seed(0)
    return VLANeXt(
        lmm_path=QWEN, action_dim=7, num_actions=8, num_queries=16,
        loss_type="diffusion", condition_type="soft", scheduler_type="flow_match",
        future_image_loss_weight=1.0,          # WORLD MODEL on (real scenario)
        policy_depth=29, policy_num_heads=16,
        policy_mixer_type=mixer, policy_mix_every_n=4,
        generator_depth=29, generator_num_heads=12,
        generator_mixer_type=mixer, generator_mix_every_n=4, generator_ttt_chunk_size=16,
        use_proprio_input_vlm=True, use_transformer_proprio_projector=False,
        backbone_mode="finetune", gradient_checkpointing=True,   # real training setting
        action_vqvae={"enabled": False}, attn_implementation="sdpa",
    ).to(DEV, DT)


def make_batch(model):
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
    # world-model needs future_images: (B, 3, H, W) normalized to [-1,1] (matches collator)
    future = torch.rand(B, 3, 256, 256) * 2.0 - 1.0
    actions = torch.randn(B, 8, 7, device=DEV, dtype=DT)
    proprio = torch.randn(B, 1, 7, device=DEV, dtype=DT)
    return fwd, actions, proprio, future


def bench(mixer):
    model = build(mixer)
    model.train()
    fwd, actions, proprio, future = make_batch(model)
    future = future.to(DEV, DT)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(actions=actions, proprioception=proprio, future_images=future, **fwd)
        loss.backward()
        opt.step()
        return float(loss)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t = time.time()
    last = 0.0
    for _ in range(ITERS):
        last = step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / ITERS
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return dt, mem, last


if __name__ == "__main__":
    print(f"batch={B}, world_model=ON, gradient_checkpointing=ON, attn=sdpa, "
          f"compile={'OFF' if __import__('os').environ.get('TORCHDYNAMO_DISABLE')=='1' else 'ON'}")
    for mixer in ["attention", "ttt"]:
        dt, mem, loss = bench(mixer)
        print(f"[{mixer:9s}] {dt*1000:8.1f} ms/step | {1/dt:5.2f} it/s | peak {mem:5.1f} GB | loss {loss:.3f}")
