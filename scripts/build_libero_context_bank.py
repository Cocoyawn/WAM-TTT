#!/usr/bin/env python
"""Extract a small successful-trajectory context bank from LIBERO TFDS data.

The bank is used by ``libero_plus_bench_eval.py`` for the same-task and
other-task context probe.  It contains RGB frames only; actions are excluded
so the probe cannot leak the demonstration trajectory's action labels.

Example:
  python scripts/build_libero_context_bank.py \
    --data-root /mnt/afs-h200/NTU_slab/draven/data/LIBERO_modified \
    --suite libero_spatial --frames 4 --max-per-instruction 2 \
    --out-dir /mnt/afs-h200/yuyangcheng/workplace/VLANeXt/context_banks/libero_spatial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _decode_instruction(value) -> str:
    value = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def build(args) -> dict:
    import numpy as np
    import tensorflow_datasets as tfds

    data_path = Path(args.data_root) / args.suite / args.version
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(builder_dir=str(data_path))
    read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
    ds = builder.as_dataset(split="train", shuffle_files=False, read_config=read_config)

    per_instruction: dict[str, int] = {}
    entries: list[dict] = []
    seen_trajectories = 0
    skipped_unsuccessful = 0

    for traj_id, traj_data in enumerate(ds):
        if args.max_trajectories > 0 and seen_trajectories >= args.max_trajectories:
            break
        seen_trajectories += 1
        try:
            steps = next(iter(traj_data["steps"].batch(args.max_steps)))
            rewards = steps["reward"].numpy()
            if len(rewards) == 0 or float(rewards[-1]) < 0.5:
                skipped_unsuccessful += 1
                continue

            instruction = _decode_instruction(steps["language_instruction"][0])
            instruction_key = " ".join(instruction.lower().split())
            if per_instruction.get(instruction_key, 0) >= args.max_per_instruction:
                continue

            images = steps["observation"]["image"].numpy()
            if images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
                raise ValueError(f"invalid exterior image shape: {images.shape}")
            wrist = None
            if "wrist_image" in steps["observation"]:
                wrist = steps["observation"]["wrist_image"].numpy()
                if wrist.shape != images.shape:
                    raise ValueError(f"wrist shape {wrist.shape} != image shape {images.shape}")

            frame_ids = np.linspace(0, len(images) - 1, args.frames, dtype=np.int64)
            sampled = images[frame_ids].astype(np.uint8, copy=False)
            sampled_wrist = wrist[frame_ids].astype(np.uint8, copy=False) if wrist is not None else None

            entry_id = len(entries)
            npz_path = out_dir / f"entry_{entry_id:05d}.npz"
            if sampled_wrist is None:
                np.savez_compressed(npz_path, frames=sampled)
            else:
                np.savez_compressed(npz_path, frames=sampled, wrist=sampled_wrist)

            entries.append({
                "instruction": instruction,
                "suite": args.suite,
                "trajectory_id": int(traj_id),
                "frame_indices": [int(x) for x in frame_ids],
                "path": str(npz_path.resolve()),
            })
            per_instruction[instruction_key] = per_instruction.get(instruction_key, 0) + 1

            if len(entries) % 10 == 0:
                print(f"[bank] entries={len(entries)} trajectories={seen_trajectories}", flush=True)
            if args.max_entries > 0 and len(entries) >= args.max_entries:
                break
        except Exception as exc:
            print(f"[bank] skip trajectory {traj_id}: {exc}", flush=True)

    if not entries:
        raise RuntimeError("No successful trajectory entries were extracted")

    manifest = {
        "format": "vlanext-libero-context-v1",
        "suite": args.suite,
        "version": args.version,
        "frames_per_entry": args.frames,
        "data_root": str(data_path),
        "entries": entries,
        "stats": {
            "entries": len(entries),
            "instructions": len(per_instruction),
            "successful_trajectories_skipped": skipped_unsuccessful,
            "min_entries_per_instruction": min(per_instruction.values()),
            "max_entries_per_instruction": max(per_instruction.values()),
        },
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["stats"], indent=2))
    print(f"[bank] wrote {manifest_path}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--max-per-instruction", type=int, default=2)
    ap.add_argument("--max-trajectories", type=int, default=0, help="0 = all")
    ap.add_argument("--max-entries", type=int, default=0, help="0 = all")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.frames <= 0 or args.max_per_instruction <= 0:
        ap.error("--frames and --max-per-instruction must be positive")
    build(args)


if __name__ == "__main__":
    main()
