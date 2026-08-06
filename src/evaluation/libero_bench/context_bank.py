"""Small, deterministic context-bank utilities for LIBERO context probes.

The bank stores only sampled RGB frames from successful demonstration
trajectories.  It is deliberately independent of the model and simulator so
that the same bank can be reused by open-loop and closed-loop probes.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


def normalize_instruction(text: str) -> str:
    """Normalize task text for matching TFDS demonstrations to LIBERO tasks."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        manifest = json.load(f)
    if manifest.get("format") != "vlanext-libero-context-v1":
        raise ValueError(f"Unsupported context-bank format in {path}")
    if not isinstance(manifest.get("entries"), list):
        raise ValueError(f"Context bank has no entries: {path}")
    return manifest


def _entries_for(manifest: dict, instruction: str) -> list[dict]:
    key = normalize_instruction(instruction)
    return [e for e in manifest["entries"]
            if normalize_instruction(e.get("instruction", "")) == key]


def select_entry(
    manifest: dict,
    instruction: str,
    mode: str,
    *,
    seed: int = 42,
    exclude_path: str | None = None,
) -> dict | None:
    """Select a deterministic demonstration entry.

    ``same_task`` requires an exact normalized instruction match.  ``other_task``
    requires a different instruction.  Returning ``None`` is intentional when
    the requested control cannot be constructed; callers should fail loudly.
    """
    if mode == "none":
        return None
    if mode not in {"same_task", "other_task"}:
        raise ValueError(f"Unknown context mode: {mode}")

    key = normalize_instruction(instruction)
    if mode == "same_task":
        candidates = [e for e in manifest["entries"]
                      if normalize_instruction(e.get("instruction", "")) == key]
    else:
        candidates = [e for e in manifest["entries"]
                      if normalize_instruction(e.get("instruction", "")) != key]

    if exclude_path:
        candidates = [e for e in candidates if e.get("path") != exclude_path]
    if not candidates:
        return None

    # Stable selection makes paired conditions reproducible across workers.
    candidates = sorted(candidates, key=lambda e: str(e.get("path", "")))
    return candidates[random.Random(seed).randrange(len(candidates))]


def load_frames(entry: dict, *, max_frames: int | None = None):
    """Load RGB frames from one bank entry, with optional deterministic truncation."""
    import numpy as np

    path = Path(entry["path"])
    with np.load(path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"], dtype=np.uint8)
        wrist = np.asarray(data["wrist"], dtype=np.uint8) if "wrist" in data else None
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Bad context frames shape {frames.shape} in {path}")
    if wrist is not None and wrist.shape != frames.shape:
        raise ValueError(f"Wrist shape {wrist.shape} != frames shape {frames.shape} in {path}")
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
        if wrist is not None:
            wrist = wrist[:max_frames]
    return [frame for frame in frames], ([frame for frame in wrist] if wrist is not None else None)


def preprocess_context_frames(
    frames,
    image_size: int | tuple[int, int],
    *,
    center_crop: bool = False,
    center_crop_ratio: float = 1.0,
):
    """Match eval crop/resize for TFDS frames without simulator-only rotation.

    Simulator observations need the 180-degree correction in
    ``get_libero_image``. TFDS frames are already in training orientation, so
    applying that correction here would make the context inconsistent.
    """
    from PIL import Image
    import numpy as np

    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    resampling = getattr(Image, "Resampling", Image)
    out = []
    for frame in frames:
        arr = np.asarray(frame, dtype=np.uint8)
        if center_crop:
            h, w = arr.shape[:2]
            side = max(1, int(round(min(h, w) * float(center_crop_ratio))))
            top = max(0, (h - side) // 2)
            left = max(0, (w - side) // 2)
            arr = arr[top : top + side, left : left + side]
        out.append(np.asarray(Image.fromarray(arr).resize(size, resampling.LANCZOS), dtype=np.uint8))
    return out


def prepend_context(current, context, max_context_frames: int):
    """Return ``context + current`` without modifying either input list."""
    if max_context_frames <= 0 or not context:
        return list(current)
    return list(context[-max_context_frames:]) + list(current)


def compose_multiview_video_inputs(
    current_exterior,
    current_wrist,
    context_exterior,
    context_wrist,
    max_context_frames: int,
):
    """Compose separate exterior/wrist videos while preserving suffix order."""
    return (
        prepend_context(current_exterior, context_exterior, max_context_frames),
        prepend_context(current_wrist, context_wrist, max_context_frames),
    )


def compose_multiview_image_inputs(
    current_exterior,
    current_wrist,
    context_exterior,
    context_wrist,
    max_context_frames: int,
):
    """Compose Qwen image inputs: all exterior context, wrist context, then current pair."""
    exterior = list(context_exterior[-max_context_frames:]) if max_context_frames > 0 else []
    wrist = list(context_wrist[-max_context_frames:]) if max_context_frames > 0 else []
    return exterior + wrist + [current_exterior, current_wrist]


def manifest_summary(manifest: dict) -> dict:
    counts: dict[str, int] = {}
    for entry in manifest["entries"]:
        key = normalize_instruction(entry.get("instruction", ""))
        counts[key] = counts.get(key, 0) + 1
    return {
        "entries": len(manifest["entries"]),
        "instructions": len(counts),
        "min_entries_per_instruction": min(counts.values()) if counts else 0,
        "max_entries_per_instruction": max(counts.values()) if counts else 0,
    }
