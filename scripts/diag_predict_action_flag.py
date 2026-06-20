"""Decisive: compare predict_action's FINAL action with use_incremental_gen
True vs False, on the SAME real LIBERO observation. Everything else identical.

If actions match -> incremental is fine end-to-end (SR=0 must be elsewhere).
If actions differ a lot -> the incremental branch in predict_action produces bad
actions even with good image tokens -> that's the SR=0 cause; we then inspect why
(the diffusion loop consumes gen_hidden_states; maybe a dtype/shape/scale issue
specific to the incremental hidden vs the full-recompute hidden).
"""
import os, sys
import numpy as np
import torch
from PIL import Image

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
CFG_PATH = f"{REPO}/config/libero_plus_envgen_mix5dim_libero_spatial_shard0.yaml"
sys.path.insert(0, REPO)
from omegaconf import OmegaConf
from src.evaluation.libero_bench.VLANeXt_utils import get_vla, get_processor, get_vla_action
from src.evaluation.libero_bench.libero_utils import (
    get_libero_env, get_libero_image, quat2axisangle, get_libero_dummy_action)
from libero.libero import benchmark


@torch.no_grad()
def predict_with_flag(model, captured, flag):
    """Call predict_action with use_incremental_gen=flag, fixing the rng so the
    diffusion init noise is identical across the two calls."""
    model.use_incremental_gen = flag
    torch.manual_seed(12345)  # same diffusion init noise both times
    return model.predict_action(**captured)


if __name__ == "__main__":
    cfg = OmegaConf.load(CFG_PATH)
    print("loading model...")
    model = get_vla(cfg)
    processor = get_processor(cfg)
    model.eval()

    bm = benchmark.get_benchmark_dict()[cfg.eval.task_suite_name]()
    task = bm.get_task(0)
    init_states = bm.get_task_init_states(0)
    env, task_desc = get_libero_env(task, "vlanext", resolution=256)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("vlanext"))
    img = get_libero_image(obs, 256, center_crop=True, center_crop_ratio=0.9,
                           obs_key="agentview_image")
    gripper = np.clip(1 - (np.mean(np.abs(obs["robot0_gripper_qpos"])) / 0.04), 0.0, 1.0)
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [gripper]))
    observation = {"full_image": img, "full_image_wrist": img, "image_history": [img],
                   "image_history_wrist": [], "state_history": [state], "action_history": []}
    env.close()

    # capture the exact predict_action kwargs that get_vla_action builds
    captured = {}
    orig = model.predict_action

    def spy(**kw):
        captured.update(kw)
        return torch.zeros(1, 1, 7, device=next(model.parameters()).device)

    model.predict_action = spy
    try:
        get_vla_action({}, model, processor, observation, task_desc)
    finally:
        model.predict_action = orig
    print("captured predict_action kwargs:", sorted(captured.keys()))

    aF = predict_with_flag(model, captured, flag=False).float().cpu()
    aI = predict_with_flag(model, captured, flag=True).float().cpu()

    print(f"\naction shape: {tuple(aF.shape)}")
    print(f"action FULL  [:1]: {aF.flatten()[:7].tolist()}")
    print(f"action INCR  [:1]: {aI.flatten()[:7].tolist()}")
    d = (aF - aI).abs()
    print(f"\n|aFULL - aINCR| mean={float(d.mean()):.4e} max={float(d.max()):.4e}")
    print(f"|aFULL| mean={float(aF.abs().mean()):.4e}  |aINCR| mean={float(aI.abs().mean()):.4e}")
    print(f"any NaN/Inf in INCR: {bool(torch.isnan(aI).any() or torch.isinf(aI).any())}")
