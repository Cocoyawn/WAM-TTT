"""MolmoAct2-DROID (LeRobot v3.0) dataset for VLANeXt training.

Parallels :class:`src.datasets.droid_act.DroidAct` (same output sample keys and
windowing logic) but reads the LeRobot v3.0 on-disk format instead of RLDS/tfds:

  data/chunk-XXX/file-YYY.parquet   tabular rows (action[8], observation.state[8],
                                    language_instruction, episode_index, index, ...)
  videos/<cam>/chunk-XXX/file-YYY.mp4   AV1-encoded video, all episodes concatenated
  meta/episodes/**/*.parquet        per-episode index: data row range
                                    (dataset_from_index..dataset_to_index) + per-camera
                                    (file_index, from_timestamp, to_timestamp)
  meta/stats.json                   per-feature min/max (used for [-1, 1] normalization)
  meta/info.json                    feature schema + fps

Action space (MolmoAct2-DROID, absolute joint-pose control):
  action = [joint_0..joint_6 (rad), gripper (0=open .. 1=closed)]  -> 8-dim
  observation.state has the same 8-dim layout.

Both action and proprioception are normalized to [-1, 1] per-dimension using the
dataset stats. The inverse mapping is applied at eval time when commanding RoboLab's
DroidJointPositionActionCfg (7 joints in rad + BinaryJointPositionZeroToOneAction
gripper in [0, 1]).

Video decoding uses imageio-ffmpeg (bundled ffmpeg v7 with libdav1d software AV1
decode); cv2's OpenCV ffmpeg build cannot decode AV1 in this environment.
"""

import os
import gc
import glob
import json
import ctypes

import numpy as np
import torch
from torch.utils.data import IterableDataset

import imageio


# Force glibc to return freed memory to the OS (Linux only). Mirrors DroidAct so
# long-running iterable datasets don't balloon RSS across many episodes.
try:
    _LIBC = ctypes.CDLL("libc.so.6")

    def _malloc_trim():
        _LIBC.malloc_trim(0)
except Exception:
    def _malloc_trim():
        pass


# 8-dim action / state layout: 7 joint angles + gripper.
ACTION_DIM = 8
_CAM_EXT = "observation.images.exterior_1_left"
_CAM_WRIST = "observation.images.wrist_left"


def _load_stats(data_root):
    """Return (min, max) float32 arrays of shape (8,) for the `action` feature."""
    with open(os.path.join(data_root, "meta", "stats.json")) as f:
        stats = json.load(f)
    amin = np.asarray(stats["action"]["min"], dtype=np.float32)
    amax = np.asarray(stats["action"]["max"], dtype=np.float32)
    return amin, amax


def get_molmoact_action_stats(data_root):
    """Public helper (mirrors libero_act.get_libero_action_stats) for eval-time denorm."""
    return _load_stats(data_root)


class MolmoActDroidAct(IterableDataset):
    def __init__(
        self,
        data_root,
        dataset_name="molmoact_droid",
        length=None,
        history_len=8,
        future_len=8,
        full_sequence=False,
        input_modality="image",
        view_mode="multi",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=30000,
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.buffer_size = buffer_size

        self.action_min, self.action_max = _load_stats(data_root)
        denom = self.action_max - self.action_min
        self.action_denominator = np.where(denom == 0, 1.0, denom).astype(np.float32)

        with open(os.path.join(data_root, "meta", "info.json")) as f:
            info = json.load(f)
        self.fps = float(info.get("fps", 15))

        self._episodes = None  # lazily built per-process in __iter__

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_episode_index(self):
        """Read meta/episodes/**/*.parquet into a list of episode descriptors."""
        import pandas as pd

        ep_files = sorted(glob.glob(
            os.path.join(self.data_root, "meta", "episodes", "**", "*.parquet"),
            recursive=True,
        ))
        episodes = []
        for ef in ep_files:
            df = pd.read_parquet(ef)
            for _, r in df.iterrows():
                episodes.append({
                    "episode_index": int(r["episode_index"]),
                    "length": int(r["length"]),
                    "data_from": int(r["dataset_from_index"]),
                    "data_to": int(r["dataset_to_index"]),
                    "ext_file": int(r[f"videos/{_CAM_EXT}/file_index"]),
                    "ext_chunk": int(r[f"videos/{_CAM_EXT}/chunk_index"]),
                    "ext_from_ts": float(r[f"videos/{_CAM_EXT}/from_timestamp"]),
                    "wrist_file": int(r[f"videos/{_CAM_WRIST}/file_index"]),
                    "wrist_chunk": int(r[f"videos/{_CAM_WRIST}/chunk_index"]),
                    "wrist_from_ts": float(r[f"videos/{_CAM_WRIST}/from_timestamp"]),
                })
        episodes.sort(key=lambda e: e["episode_index"])
        if self.length is not None:
            episodes = episodes[: self.length]
        return episodes

    # ------------------------------------------------------------------
    # Data parquet access: map a global row range to the right data file.
    # ------------------------------------------------------------------
    def _build_data_file_table(self):
        """Map each data parquet file to its global `index` range for fast lookup."""
        import pandas as pd
        import pyarrow.parquet as pq

        data_files = sorted(glob.glob(
            os.path.join(self.data_root, "data", "**", "*.parquet"), recursive=True,
        ))
        table = []
        for df_path in data_files:
            # Read only the `index` column footer-cheaply to get min/max.
            col = pq.read_table(df_path, columns=["index"])["index"].to_numpy()
            table.append((int(col.min()), int(col.max()), df_path))
        table.sort()
        return table

    def _read_episode_rows(self, ep):
        """Return (actions[N,8] normalized, proprio[N,8] normalized, instruction str)."""
        import pandas as pd

        lo, hi = ep["data_from"], ep["data_to"]  # [lo, hi) global index range
        # An episode's rows live within a single data file (LeRobot v3 invariant:
        # data_from/to come from one (chunk,file)). Find the file covering `lo`.
        for fmin, fmax, path in self._data_table:
            if fmin <= lo <= fmax:
                df = pd.read_parquet(path)
                break
        else:
            raise RuntimeError(f"No data parquet covers global index {lo}")

        sub = df[(df["index"] >= lo) & (df["index"] < hi)]
        sub = sub.sort_values("frame_index")

        actions = np.stack(sub["action"].to_numpy()).astype(np.float32)          # (N, 8)
        proprio = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)  # (N, 8)

        instr = sub["language_instruction"].iloc[0]
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")
        instr = str(instr)

        actions = self._normalize(actions)
        proprio = self._normalize(proprio)
        return actions, proprio, instr

    def _normalize(self, x):
        """Per-dim affine map [min, max] -> [-1, 1], then clip."""
        x = 2.0 * (x - self.action_min) / self.action_denominator - 1.0
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Video decoding (AV1 via imageio-ffmpeg software decode)
    # ------------------------------------------------------------------
    def _decode_clip(self, cam, file_index, chunk_index, from_ts, n_frames):
        path = os.path.join(
            self.data_root, "videos", cam,
            f"chunk-{chunk_index:03d}", f"file-{file_index:03d}.mp4",
        )
        # `-ss` before input does fast keyframe seek; decode exactly n_frames after.
        reader = imageio.get_reader(
            path, format="FFMPEG",
            input_params=["-ss", f"{from_ts:.6f}"],
            output_params=["-frames:v", str(n_frames)],
        )
        frames = []
        try:
            for i, fr in enumerate(reader):
                frames.append(np.asarray(fr, dtype=np.uint8))
                if i + 1 >= n_frames:
                    break
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f"Decoded 0 frames from {path} @ ts={from_ts}")
        arr = np.stack(frames, axis=0)  # (n, H, W, 3)
        # Pad (rare short-read) by edge-repeating the last frame.
        if arr.shape[0] < n_frames:
            pad = np.repeat(arr[-1:], n_frames - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        return arr

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self):
        if self._episodes is None:
            self._episodes = self._build_episode_index()
            self._data_table = self._build_data_file_table()

        episodes = self._episodes

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        my_episodes = episodes[shard_index::total_shards]

        shuffle_buffer = []
        BUFFER_SIZE = self.buffer_size
        history_len, future_len = self.history_len, self.future_len

        traj_id = -1
        for ep in my_episodes:
            traj_id += 1
            try:
                actions_np, proprio_np, instruction = self._read_episode_rows(ep)
                traj_len = actions_np.shape[0]
                if traj_len < 2:
                    continue

                images_np = self._decode_clip(
                    _CAM_EXT, ep["ext_file"], ep["ext_chunk"], ep["ext_from_ts"], traj_len)

                wrist_np = None
                if self.view_mode == "multi":
                    wrist_np = self._decode_clip(
                        _CAM_WRIST, ep["wrist_file"], ep["wrist_chunk"], ep["wrist_from_ts"], traj_len)

                # Guard against image/label length drift: clamp to the common length.
                n = min(traj_len, images_np.shape[0],
                        wrist_np.shape[0] if wrist_np is not None else traj_len)
                if n < traj_len:
                    actions_np, proprio_np = actions_np[:n], proprio_np[:n]
                    images_np = images_np[:n]
                    if wrist_np is not None:
                        wrist_np = wrist_np[:n]
                    traj_len = n

                if self.full_sequence:
                    sample_indices = np.arange(traj_len)
                else:
                    num_samples = max(1, int(traj_len / (self.fps * 5)))
                    num_samples = min(num_samples, traj_len)
                    sample_indices = np.random.choice(traj_len, size=num_samples, replace=False)

                for t in sample_indices:
                    start_hist_obs = t - history_len + 1
                    hist_indices_obs = np.clip(np.arange(start_hist_obs, t + 1), 0, traj_len - 1)

                    hist_indices_act = np.arange(t - history_len, t)
                    fut_indices = np.arange(t, t + future_len)

                    hist_imgs = images_np[hist_indices_obs]
                    hist_imgs_wrist = wrist_np[hist_indices_obs] if wrist_np is not None else None
                    hist_proprio = torch.from_numpy(proprio_np[hist_indices_obs])

                    hist_actions = np.zeros((history_len, ACTION_DIM), dtype=np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        vi = np.clip(hist_indices_act[valid_mask], 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[vi]
                    hist_actions = torch.from_numpy(hist_actions)

                    fut_acts_np = np.zeros((future_len, ACTION_DIM), dtype=np.float32)
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        fut_acts_np[valid_mask_fut] = actions_np[fut_indices[valid_mask_fut]]
                    fut_acts = torch.from_numpy(fut_acts_np)

                    sample = {
                        "proprioception": hist_proprio,
                        "history_actions": hist_actions,
                        "future_actions": fut_acts,
                        "instruction": instruction,
                    }

                    if self.load_future_image:
                        if self.future_image_mode == "last":
                            target_idx = traj_len - 1
                        else:
                            target_idx = min(t + future_len, traj_len - 1)
                        sample["future_image"] = images_np[target_idx].copy()

                    if self.input_modality == "video":
                        sample["video"] = hist_imgs
                        if self.view_mode == "multi":
                            sample["video_wrist"] = hist_imgs_wrist
                    elif self.input_modality == "image":
                        sample["image"] = images_np[t].copy()
                        if self.view_mode == "multi":
                            sample["image_wrist"] = (
                                wrist_np[t].copy() if wrist_np is not None else images_np[t].copy())
                    else:
                        raise ValueError(f"Unknown input_modality: {self.input_modality}")

                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= BUFFER_SIZE:
                        idx = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

                del images_np, actions_np, proprio_np
                if wrist_np is not None:
                    del wrist_np

            except Exception as e:
                print(f"[Warn] Skipping episode {ep.get('episode_index', traj_id)}: {e}")
                continue
            finally:
                # `gc` may already be torn down during interpreter shutdown.
                if traj_id % 50 == 0 and gc is not None:
                    gc.collect()
                    _malloc_trim()

        np.random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/mnt/afs-h200/NTU_slab/draven/data/MolmoAct2-DROID")
    parser.add_argument("--num", type=int, default=3)
    args = parser.parse_args()

    ds = MolmoActDroidAct(
        data_root=args.data_root,
        history_len=8, future_len=8,
        full_sequence=False, input_modality="image", view_mode="multi",
        buffer_size=1,
    )
    it = iter(ds)
    for i in range(args.num):
        s = next(it)
        print(f"--- sample {i} ---")
        print("  instruction:", repr(s["instruction"])[:80])
        print("  image:", s["image"].shape, s["image"].dtype)
        print("  image_wrist:", s["image_wrist"].shape)
        print("  proprioception:", tuple(s["proprioception"].shape), s["proprioception"].dtype)
        print("  history_actions:", tuple(s["history_actions"].shape))
        print("  future_actions:", tuple(s["future_actions"].shape),
              "range [%.3f, %.3f]" % (s["future_actions"].min(), s["future_actions"].max()))
