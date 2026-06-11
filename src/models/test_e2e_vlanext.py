"""
End-to-end smoke test for the full VLANeXt model (Qwen3-VL-2B backbone + action
expert), to confirm the isolated venv supports training + inference.

Run (inside the isolated venv with torch>=2.4 / triton>=3.2), from repo root:
    cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
    TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" \
      /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python \
      -m src.models.test_e2e_vlanext

Checks: construct -> train forward -> backward -> predict_action.
Uses future_image_loss_weight=0 to avoid the (weight-less) Emu3.5 tokenizer,
and attn_implementation="eager" to avoid a flash-attn dependency.
"""

import torch
from PIL import Image

from src.models.VLANeXt import VLANeXt

QWEN_PATH = "/mnt/afs-h200/yuyangcheng/models/Qwen3-VL-2B-Instruct"
DEV, DT = "cuda", torch.bfloat16


def build_model(policy_mixer_type="attention"):
    print(f"constructing VLANeXt (Qwen3-VL-2B backbone, policy_mixer_type={policy_mixer_type})...")
    model = VLANeXt(
        lmm_path=QWEN_PATH,
        action_dim=7, num_actions=8, num_queries=16,
        loss_type="diffusion", condition_type="soft", scheduler_type="flow_match",
        future_image_loss_weight=0.0,
        policy_depth=29, policy_num_heads=16,
        policy_mixer_type=policy_mixer_type, policy_mix_every_n=4,
        use_proprio_input_vlm=True, use_transformer_proprio_projector=False,
        backbone_mode="finetune", gradient_checkpointing=False,
        action_vqvae={"enabled": False},
        attn_implementation="eager",
    ).to(DEV, DT)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[ok] CONSTRUCT params(M)={n:.1f}")
    return model


def make_batch(model, B=2):
    proc = model.processor
    img = Image.new("RGB", (256, 256), "red")
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "pick up the cup"},
    ]}] for _ in range(B)]
    texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in msgs]
    inp = proc(text=texts, images=[img] * B, padding=True, return_tensors="pt")
    inp = {k: v.to(DEV) for k, v in inp.items()}
    if "pixel_values" in inp:
        inp["pixel_values"] = inp["pixel_values"].to(DT)
    valid = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    return {k: v for k, v in inp.items() if k in valid}


def run_one(policy_mixer_type):
    model = build_model(policy_mixer_type=policy_mixer_type)
    fwd = make_batch(model)
    B = fwd["input_ids"].shape[0]
    actions = torch.randn(B, 8, 7, device=DEV, dtype=DT)
    proprio = torch.randn(B, 1, 7, device=DEV, dtype=DT)

    model.train()
    loss = model(actions=actions, proprioception=proprio, **fwd)
    assert torch.isfinite(loss), "non-finite train loss"
    print(f"[ok] TRAIN forward loss={float(loss):.4f}")
    loss.backward()
    print("[ok] BACKWARD")

    model.eval()
    with torch.no_grad():
        act = model.predict_action(proprioception=proprio, **fwd)
    assert act.shape == (B, 8, 7), act.shape
    assert torch.isfinite(act).all(), "non-finite action"
    print(f"[ok] PREDICT_ACTION shape={tuple(act.shape)}")
    del model
    torch.cuda.empty_cache()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"torch={torch.__version__}")
    print("\n===== baseline: policy_mixer_type='attention' =====")
    run_one("attention")
    print("\n===== ablation: policy_mixer_type='ttt' =====")
    run_one("ttt")
    print("\nEnd-to-end VLANeXt smoke test passed (attention + ttt).")


if __name__ == "__main__":
    main()
