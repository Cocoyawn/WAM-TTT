"""VLANeXt inference server for RoboLab eval (ZMQ REQ/REP + msgpack).

RoboLab runs in its own Isaac-Sim uv venv (Python 3.11); VLANeXt runs in the
fla_triton32 venv (torch 2.6 / triton 3.2). They cannot share a process, so the
model is served here and RoboLab connects via policies/vlanext/client.py.

Wire protocol (mirrors policies/gr00t/client.py's self-contained msgpack codec, so
no openpi dependency is needed on either side):

  request  = {
      "exterior_image": uint8 HWC ndarray,   # over-shoulder camera
      "wrist_image":    uint8 HWC ndarray,    # wrist camera
      "joint_position": float32 (7,) ndarray, # arm joints, rad
      "gripper_position": float32 (1,) ndarray,  # [0,1], 0=open 1=closed
      "prompt": str,
      "reset": bool,                          # optional, clears history
  }
  response = {"actions": float32 (horizon, 8) ndarray}   # 7 joint (rad) + gripper [0,1]

Actions are produced by the model in normalized [-1, 1] space and de-normalized
here with the MolmoAct2-DROID per-dim stats before being returned, so the client
can command RoboLab's DroidJointPositionActionCfg directly (7 joints in rad +
BinaryJointPositionZeroToOneAction gripper in [0, 1]).
"""

import io
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import zmq
import msgpack
import cv2

DROID_IMAGE_HEIGHT = 180
DROID_IMAGE_WIDTH = 320


def _resize_droid_image(image):
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    if image.shape[:2] == (DROID_IMAGE_HEIGHT, DROID_IMAGE_WIDTH):
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(cv2.resize(
        image,
        (DROID_IMAGE_WIDTH, DROID_IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    ))


# --------------------------------------------------------------------------
# msgpack ndarray codec (identical to policies/gr00t/client.py _MsgSerializer)
# --------------------------------------------------------------------------
def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def to_bytes(data):
    return msgpack.packb(data, default=_encode)


def from_bytes(data):
    return msgpack.unpackb(data, object_hook=_decode, raw=False)


class _Cfg:
    """Minimal dot-access cfg for get_vla/get_processor (mirrors libero eval DictConfig)."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) else v)


def build_cfg(checkpoint, image_size, diffusion_steps, use_incremental_gen, ttt_cuda):
    return _Cfg({
        "eval": {
            "finetuned_checkpoint": checkpoint,
            "image_size": image_size,
            "diffusion_steps": diffusion_steps,
            "use_incremental_gen": use_incremental_gen,
            "ttt_use_cuda_kernel": ttt_cuda,
        },
        "model": {
            "attn_implementation": "sdpa",
            "diffusion_steps": diffusion_steps,
        },
    })


def validate_model_contract(model):
    """Fail fast when a checkpoint does not match the RoboLab DROID adapter."""
    data_cfg = model.train_config["data"]
    model_cfg = model.train_config.get("model", {})
    view_mode = data_cfg.get("view_mode", "single")
    input_modality = data_cfg.get("input_modality", "image")
    dataset_name = data_cfg.get("dataset_name", "")
    train_fps = float(data_cfg.get(
        "fps", 15.0 if dataset_name in ("droid", "molmoact_droid") else 20.0
    ))
    future_len = int(data_cfg.get("future_len", getattr(model, "num_actions", 0)))
    action_dim = int(model_cfg.get("action_dim", getattr(model, "action_dim", 0)))
    video_action_ratio = int(model_cfg.get("video_action_ratio", 1))

    if dataset_name not in ("droid", "molmoact_droid"):
        raise ValueError(f"Expected a DROID checkpoint, got dataset_name={dataset_name!r}.")
    if view_mode != "multi":
        raise ValueError(
            f"RoboLab VLANeXt client sends exterior+wrist views, but checkpoint "
            f"was trained with view_mode={view_mode!r}."
        )
    if input_modality != "image":
        raise ValueError(
            f"RoboLab VLANeXt adapter supplies current images, but checkpoint "
            f"was trained with input_modality={input_modality!r}."
        )
    if train_fps != 15.0:
        raise ValueError(f"RoboLab DROID runs at 15 Hz, but checkpoint fps={train_fps:g}.")
    if future_len != 8:
        raise ValueError(
            f"This RoboLab DROID adapter is configured for 8-step chunks, "
            f"but checkpoint future_len={future_len}."
        )
    if action_dim != 8:
        raise ValueError(f"Expected 7 joints + gripper (action_dim=8), got {action_dim}.")
    if video_action_ratio < 1 or future_len % video_action_ratio != 0:
        raise ValueError(
            f"future_len={future_len} must be divisible by "
            f"video_action_ratio={video_action_ratio}."
        )

    return {
        "dataset_name": dataset_name,
        "view_mode": view_mode,
        "input_modality": input_modality,
        "fps": train_fps,
        "future_len": future_len,
        "action_dim": action_dim,
        "video_action_ratio": video_action_ratio,
        "qworld_inference_frames": 1 + future_len // video_action_ratio,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="VLANeXt .pt checkpoint")
    p.add_argument(
        "--code-root",
        default="/mnt/afs-h200/yuyangcheng/workplace/VLANeXt-kairos",
        help="VLANeXt code root used to load the checkpoint (defaults to Kairos training repo)",
    )
    p.add_argument("--data-root", default="/mnt/afs-h200/NTU_slab/draven/data/MolmoAct2-DROID",
                   help="MolmoAct2-DROID root (for action denorm stats)")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--image-size", type=int, default=256,
                   help="Eval image size (Qwen processor resizes; matches train resolution policy)")
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--use-incremental-gen", action="store_true",
                   help="O(n) incremental image-token decode in predict_action")
    p.add_argument("--ttt-cuda", action="store_true", help="Use TTT CUDA kernel at eval")
    args = p.parse_args()

    # Delay model/Triton imports until after argument parsing so `--help` and
    # contract tooling work in CPU-only environments.
    code_root = str(Path(args.code_root).resolve())
    if code_root in sys.path:
        sys.path.remove(code_root)
    sys.path.insert(0, code_root)
    from src.evaluation.libero_bench.VLANeXt_utils import (
        get_vla as get_model,
        get_processor,
        get_vla_action,
    )
    from src.datasets.molmoact_droid_act import get_molmoact_action_stats

    cfg = build_cfg(args.checkpoint, args.image_size, args.diffusion_steps,
                    args.use_incremental_gen, args.ttt_cuda)

    print(f"[serve] loading model from {args.checkpoint}")
    model = get_model(cfg)
    # The VLANeXt model builds its own processor internally; get_vla_action prefers
    # model.processor. Reuse it instead of calling get_processor (which would re-read
    # the full multi-GB checkpoint from disk a second time just for the lmm_path).
    processor = getattr(model, "processor", None)
    if processor is None:
        processor = get_processor(cfg)
    contract = validate_model_contract(model)
    view_mode = contract["view_mode"]
    print(f"[serve] view_mode={view_mode} num_history={getattr(model, 'num_history', 0)} "
          f"input_modality={contract['input_modality']} train_fps={contract['fps']:g} "
          f"future_len={contract['future_len']} "
          f"action_dim={model.action_dim} input_size={DROID_IMAGE_HEIGHT}x{DROID_IMAGE_WIDTH}")

    # De-normalization stats: normalized [-1,1] -> physical units (rad / [0,1] gripper).
    amin, amax = get_molmoact_action_stats(args.data_root)  # (8,), (8,)
    amin = amin.astype(np.float32)
    amax = amax.astype(np.float32)
    print(f"[serve] denorm stats loaded: action dim={amin.shape[0]}")

    def denorm(chunk):
        # chunk: (horizon, 8) in [-1, 1]. Inverse of dataset _normalize.
        return ((chunk + 1.0) / 2.0) * (amax - amin) + amin

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[serve] listening on tcp://0.0.0.0:{args.port}")

    while True:
        msg = sock.recv()
        try:
            req = from_bytes(msg)

            if req.get("reset", False):
                # No server-side session state to clear (history is sent each call),
                # but acknowledge so the client can flush its chunk cache.
                sock.send(to_bytes({"ok": True}))
                continue

            # Keep the server robust to old clients that still send RoboLab's
            # native 720x1280 frames. The policy always receives the same
            # 180x320 per-view size used by MolmoAct2-DROID training.
            ext = _resize_droid_image(req["exterior_image"])
            wrist = _resize_droid_image(req["wrist_image"])
            joint = np.asarray(req["joint_position"], dtype=np.float32).reshape(-1)
            grip = np.asarray(req["gripper_position"], dtype=np.float32).reshape(-1)
            prompt = str(req.get("prompt", ""))

            # Proprioception state = normalized [joint(7), gripper(1)] (8-dim), matching
            # the training observation.state layout. Normalize with the same stats.
            state = np.concatenate([joint[:7], grip[:1]]).astype(np.float32)
            state_norm = np.clip(2.0 * (state - amin) / np.where(amax - amin == 0, 1.0, amax - amin) - 1.0,
                                 -1.0, 1.0)

            # Single-step obs (history is rebuilt from the single frame inside
            # get_vla_action via _take_last padding when no history is provided).
            obs = {
                "full_image": ext,
                "full_image_wrist": wrist if view_mode == "multi" else ext,
                "image_history": [ext],
                "image_history_wrist": [wrist] if view_mode == "multi" else [],
                "state_history": [state_norm],
                "action_history": [],
            }

            chunk_norm = get_vla_action(cfg, model, processor, obs, prompt)  # (horizon, 8)
            if chunk_norm.ndim == 1:
                chunk_norm = chunk_norm[None, :]
            chunk = denorm(chunk_norm.astype(np.float32))

            sock.send(to_bytes({"actions": chunk.astype(np.float32)}))

        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(to_bytes({"error": str(e)}))


if __name__ == "__main__":
    main()
