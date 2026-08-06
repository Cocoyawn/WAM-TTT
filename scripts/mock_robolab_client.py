"""Mock RoboLab client for serve_vlanext.py — validates the real-checkpoint server
inference path end to end (load -> predict -> denorm -> wire) without needing Isaac.

Sends one request with the exact schema policies/vlanext/client.py uses and asserts
the response is a finite (horizon, 8) action chunk in plausible physical ranges.
"""
import io
import sys

import numpy as np
import zmq
import msgpack


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


def to_bytes(d):
    return msgpack.packb(d, default=_encode)


def from_bytes(b):
    return msgpack.unpackb(b, object_hook=_decode, raw=False)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5556
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 600000)  # first call compiles TTT triton kernels; allow minutes
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://127.0.0.1:{port}")

    # Realistic-ish DROID inputs: over-shoulder + wrist 180x320x3, 7 joints + gripper.
    rng = np.random.RandomState(0)
    ext = rng.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    wrist = rng.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    joint = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.7], dtype=np.float32)
    grip = np.array([0.0], dtype=np.float32)

    req = {
        "exterior_image": ext,
        "wrist_image": wrist,
        "joint_position": joint,
        "gripper_position": grip,
        "prompt": "pick up the banana and put it in the bowl",
    }

    print(f"[mock] sending request to tcp://127.0.0.1:{port} ...")
    sock.send(to_bytes(req))
    resp = from_bytes(sock.recv())

    if "error" in resp:
        print(f"[mock] SERVER ERROR: {resp['error']}")
        sys.exit(1)

    actions = resp["actions"]
    print(f"[mock] got actions shape={actions.shape} dtype={actions.dtype}")
    assert actions.ndim == 2 and actions.shape[1] == 8, f"bad shape {actions.shape}"
    assert np.isfinite(actions).all(), "non-finite actions"

    joints = actions[:, :7]
    gripper = actions[:, 7]
    print(f"[mock] joint range  [{joints.min():.3f}, {joints.max():.3f}] (rad, expect ~[-3, 3])")
    print(f"[mock] gripper range [{gripper.min():.3f}, {gripper.max():.3f}] (expect ~[0, 1])")
    assert np.abs(joints).max() < 6.5, "joints out of plausible rad range"

    # second call to confirm server stays alive / repeatable
    sock.send(to_bytes(req))
    resp2 = from_bytes(sock.recv())
    assert "actions" in resp2 and resp2["actions"].shape == actions.shape
    print("[mock] second call OK — server stable")
    print("[mock] PASS: real-checkpoint server inference path validated")


if __name__ == "__main__":
    main()
