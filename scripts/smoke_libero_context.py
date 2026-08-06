#!/usr/bin/env python
"""Dependency-light smoke test for the LIBERO context probe.

This test intentionally does not import torch, MuJoCo, LIBERO, or TFDS.  It
checks the deterministic bank selection and the exact frame ordering consumed
by the model-side input adapter.  The real GPU command should run this before
the simulator evaluation.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluation.libero_bench.context_bank import (
    load_manifest,
    manifest_summary,
    normalize_instruction,
    compose_multiview_image_inputs,
    compose_multiview_video_inputs,
    prepend_context,
    select_entry,
)


def main() -> None:
    entries = [
        {"instruction": "Put the red mug on the plate", "path": "/tmp/demo_a.npz"},
        {"instruction": "Put the red mug on the plate", "path": "/tmp/demo_b.npz"},
        {"instruction": "Open the drawer", "path": "/tmp/demo_c.npz"},
    ]
    manifest = {"format": "vlanext-libero-context-v1", "entries": entries}

    assert normalize_instruction("  Put   the red mug on the plate ") == entries[0]["instruction"].lower()
    assert manifest_summary(manifest)["instructions"] == 2

    same_a = select_entry(manifest, "put the red mug on the plate", "same_task", seed=7)
    same_b = select_entry(manifest, "put the red mug on the plate", "same_task", seed=7)
    assert same_a is not None and same_a == same_b
    assert normalize_instruction(same_a["instruction"]) == "put the red mug on the plate"

    other = select_entry(manifest, "put the red mug on the plate", "other_task", seed=7)
    assert other is not None
    assert normalize_instruction(other["instruction"]) != "put the red mug on the plate"
    assert select_entry(manifest, "missing task", "same_task") is None

    current = ["cur0", "cur1", "cur2"]
    context = ["ctx0", "ctx1", "ctx2", "ctx3"]
    combined = prepend_context(current, context, 2)
    assert combined == ["ctx2", "ctx3", "cur0", "cur1", "cur2"]
    assert prepend_context(current, context, 0) == current
    assert current == ["cur0", "cur1", "cur2"]

    ext, wrist = compose_multiview_video_inputs(
        ["cur_ext"], ["cur_wrist"], ["ctx_ext"], ["ctx_wrist"], 1
    )
    assert ext == ["ctx_ext", "cur_ext"]
    assert wrist == ["ctx_wrist", "cur_wrist"]
    images = compose_multiview_image_inputs(
        "cur_ext", "cur_wrist", ["ctx_ext"], ["ctx_wrist"], 1
    )
    assert images == ["ctx_ext", "ctx_wrist", "cur_ext", "cur_wrist"]

    # Also verify the manifest file contract used by the evaluator.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        path.write_text(json.dumps(manifest))
        loaded = load_manifest(path)
        assert loaded["entries"] == entries

    print("PASS: LIBERO context-bank selection and frame-order smoke")


if __name__ == "__main__":
    main()
