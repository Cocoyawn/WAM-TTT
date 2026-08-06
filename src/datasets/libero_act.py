import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], 'GPU')

action_min_spatial = [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475]
action_max_spatial = [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375]

action_min_object = [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943]
action_max_object = [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174]

action_min_goal =  [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.2871428430080414]
action_max_goal = [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375]

action_min_10 = [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574]
action_max_10 = [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375]

action_min_mixed = [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.3675000071525574]
action_max_mixed = [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375]


def get_libero_normalization_suite_name(task_suite_name, configured_suite_name=None):
    if configured_suite_name:
        return configured_suite_name
    return task_suite_name


def get_libero_action_stats(suite_name):
    """Return action min/max stats for a LIBERO normalization suite."""
    stats_suite_name = suite_name

    if 'mixed' in stats_suite_name:
        action_min, action_max = action_min_mixed, action_max_mixed
    elif 'spatial' in stats_suite_name:
        action_min, action_max = action_min_spatial, action_max_spatial
    elif 'object' in stats_suite_name:
        action_min, action_max = action_min_object, action_max_object
    elif 'goal' in stats_suite_name:
        action_min, action_max = action_min_goal, action_max_goal
    elif '10' in stats_suite_name:
        action_min, action_max = action_min_10, action_max_10
    else:
        raise ValueError(f"Unknown LIBERO stats suite '{stats_suite_name}'")

    return np.array(action_min, dtype=np.float32), np.array(action_max, dtype=np.float32)


class LiberoAct(IterableDataset):
    def __init__(
        self,
        data_path,
        dataset_name='libero',
        normalization_suite_name=None,
        length=None,
        history_len=15,
        future_len=15,
        full_sequence=True,
        input_modality="video",
        view_mode="single",
        buffer_size=10000,
        sampling_rate=0.1,
        allow_end_padding=True,
        load_future_image=False,
        future_image_mode="horizon",
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.allow_end_padding = allow_end_padding
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode

        self.normalization_suite_name = get_libero_normalization_suite_name(
            dataset_name, normalization_suite_name
        )
        self.action_min, self.action_max = get_libero_action_stats(self.normalization_suite_name)

        self.action_denominator = self.action_max - self.action_min
        self.action_denominator = np.where(self.action_denominator == 0, 1.0, self.action_denominator)

        zero_pose = np.zeros(6, dtype=np.float32)
        self.normalized_zero_pose = 2.0 * (zero_pose - self.action_min) / self.action_denominator - 1.0
        self.normalized_zero_pose = np.clip(self.normalized_zero_pose, -1.0, 1.0)

    def __iter__(self):
        builder = tfds.builder_from_directory(builder_dir=self.data_path)
        
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)
        
        if self.length is not None:
            ds = ds.take(self.length)

        shuffle_buffer = []
        BUFFER_SIZE = self.buffer_size

        main_key = "image"
        wrist_key = "wrist_image"

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        ds_iterator = ds.shard(num_shards=total_shards, index=shard_index)

        for traj_id, traj_data in enumerate(ds_iterator):
            try:
                traj_batch = next(iter(traj_data['steps'].batch(2000)))

                if traj_batch['reward'][-1].numpy() != 1:
                    continue

                traj_len = traj_batch['action'].shape[0]

                obs = traj_batch['observation']
                images_np = obs[main_key].numpy()
                if images_np.dtype != np.uint8:
                    images_np = (images_np * 255).astype(np.uint8)
                
                wrist_np = None
                if self.view_mode == "multi":
                    if wrist_key in obs:
                        wrist_np = obs[wrist_key].numpy()
                        if wrist_np.dtype != np.uint8:
                            wrist_np = (wrist_np * 255).astype(np.uint8)
                    else:
                        wrist_np = images_np

                # Process Proprioception: 6D Pose + 1D Normalized Gripper
                raw_state = traj_batch['observation']['state'].numpy().astype(np.float32)
                
                # [Proprio Gripper]
                # Raw: 2D gripper fingers width (qpos), approx range [0, 0.04]. 0.04 = Open, 0 = Closed.
                # Processed: 1.0 - (width / 0.04).
                # Result: Range [0, 1]. 0 = Open, 1 = Closed.
                gripper_qpos = raw_state[:, 6:8]
                gripper_state = 1.0 - (np.mean(np.abs(gripper_qpos), axis=1, keepdims=True) / 0.04)
                gripper_state = np.clip(gripper_state, 0.0, 1.0)
                
                proprio_np = np.concatenate([raw_state[:, :6], gripper_state], axis=1)

                # Process Actions: Normalize delta pose and gripper to [-1, 1]
                raw_actions = traj_batch['action'].numpy().astype(np.float32)
                delta_pose = raw_actions[:, :6]

                delta_pose = 2.0 * (delta_pose - self.action_min) / self.action_denominator - 1.0
                delta_pose = np.clip(delta_pose, -1.0, 1.0)

                # [Action Gripper]
                # Raw: 1D signal in [-1, 1]. -1 = Open, 1 = Closed.
                # Processed: Clipped to ensure bounds.
                # Result: Range [-1, 1]. -1 = Open, 1 = Closed.
                gripper_action = raw_actions[:, 6:7]
                gripper_action = np.clip(gripper_action, -1.0, 1.0)

                actions_np = np.concatenate([delta_pose, gripper_action], axis=1)

                # Padding values: normalized zero delta_pose + edge gripper state.
                # Gripper is edge-padded (first/last action) since it's binary (-1/1)
                # and 0.0 has no meaningful interpretation.
                pad_action_hist = np.concatenate([self.normalized_zero_pose, actions_np[0:1, 6]])   # gripper: first
                pad_action_fut  = np.concatenate([self.normalized_zero_pose, actions_np[-1:, 6]])  # gripper: last
                
                instruction = traj_batch['language_instruction'][0].numpy().decode('utf-8')

                if self.full_sequence:
                    sample_indices = np.arange(traj_len)
                else:
                    num_samples = max(1, int(traj_len * self.sampling_rate))
                    sample_indices = np.random.choice(traj_len, size=num_samples, replace=False)

                if not self.allow_end_padding:
                    sample_indices = sample_indices[sample_indices + self.future_len <= traj_len]

                for t in sample_indices:
                    start_hist_obs = t - self.history_len + 1
                    hist_indices_obs = np.arange(start_hist_obs, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)
                    
                    start_hist_act = t - self.history_len
                    hist_indices_act = np.arange(start_hist_act, t)
                    
                    end_fut = t + self.future_len
                    fut_indices = np.arange(t, end_fut)

                    hist_imgs = images_np[hist_indices_obs]
                    hist_imgs_wrist = wrist_np[hist_indices_obs] if wrist_np is not None else None
                    hist_proprio = torch.from_numpy(proprio_np[hist_indices_obs])
                    
                    hist_actions = np.tile(pad_action_hist, (self.history_len, 1)).astype(np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        valid_indices = hist_indices_act[valid_mask]
                        valid_indices = np.clip(valid_indices, 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[valid_indices]
                    hist_actions = torch.from_numpy(hist_actions)

                    fut_acts_np = np.tile(pad_action_fut, (self.future_len, 1)).astype(np.float32)
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        valid_indices_fut = fut_indices[valid_mask_fut]
                        fut_acts_np[valid_mask_fut] = actions_np[valid_indices_fut]
                    fut_acts = torch.from_numpy(fut_acts_np)

                    sample = {
                        'proprioception': hist_proprio,
                        'history_actions': hist_actions,
                        'future_actions': fut_acts,
                        'instruction': instruction,
                    }

                    if self.load_future_image:
                        if self.future_image_mode == "last":
                            target_idx = traj_len - 1
                        else:
                            target_idx = min(t + self.future_len, traj_len - 1)
                        sample['future_image'] = images_np[target_idx]

                    if self.input_modality == "video":
                        sample['video'] = hist_imgs
                        if self.view_mode == "multi":
                            sample['video_wrist'] = hist_imgs_wrist if hist_imgs_wrist is not None else hist_imgs
                    elif self.input_modality == "image":
                        sample['image'] = images_np[t]
                        if self.view_mode == "multi":
                            sample['image_wrist'] = wrist_np[t] if wrist_np is not None else images_np[t]
                    else:
                        raise ValueError(f"Unknown input_modality: {self.input_modality}")

                    shuffle_buffer.append(sample)
                    
                    if len(shuffle_buffer) >= BUFFER_SIZE:
                        idx = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

            except Exception as e:
                print(f"[Warn] Skipping trajectory {traj_id} due to error: {e}")
                continue

        np.random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample

class LiberoMixedAct(IterableDataset):
    """Mix of all four LIBERO suites (spatial, object, goal, 10) for joint training.

    All sub-datasets are constructed with ``dataset_name="libero_mixed"`` so every
    suite is normalized with the shared ``action_min_mixed`` / ``action_max_mixed``
    stats (union across suites). This keeps a single consistent [-1, 1] target
    space across the mixed batch.
    At iteration time, samples are drawn uniformly at random from whichever
    sub-iterators are still active; when a sub-iterator is exhausted it is dropped
    and the remaining iterators continue.
    """

    SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

    def __init__(
        self,
        data_root,
        version="1.0.0",
        suites=None,
        length=None,
        history_len=15,
        future_len=15,
        full_sequence=True,
        input_modality="video",
        view_mode="single",
        buffer_size=10000,
        sampling_rate=0.1,
        allow_end_padding=True,
        load_future_image=False,
        future_image_mode="horizon",
    ):
        super().__init__()
        self.suites = tuple(suites) if suites is not None else self.SUITES
        self._sub_datasets = [
            LiberoAct(
                data_path=os.path.join(data_root, suite, version),
                dataset_name="libero_mixed",
                normalization_suite_name="libero_mixed",
                length=length,
                history_len=history_len,
                future_len=future_len,
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                buffer_size=buffer_size,
                sampling_rate=sampling_rate,
                allow_end_padding=allow_end_padding,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
            )
            for suite in self.suites
        ]

    def __iter__(self):
        iterators = [iter(ds) for ds in self._sub_datasets]
        active = list(range(len(iterators)))
        while active:
            i = active[np.random.randint(len(active))]
            try:
                yield next(iterators[i])
            except StopIteration:
                active.remove(i)


def collate_fn(batch):
    return batch

if __name__ == "__main__":
    """
    Fast stats: count how many training samples LiberoAct would yield for each suite.
    Also computes min/max statistics for the first 6 dimensions of actions.
    """
    from tqdm import tqdm

    # Configuration
    BASE_DIR = "/data/NTU_slab/draven/data/LIBERO_modified"
    SUITES = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ]
    VERSION = "1.0.0"

    print(f"Scanning Libero datasets in {BASE_DIR}...")

    mixed_total_trajs = 0
    mixed_success_trajs = 0
    mixed_total_samples = 0
    mixed_act_min = np.full(6, np.inf)
    mixed_act_max = np.full(6, -np.inf)

    for suite_name in SUITES:
        data_path = os.path.join(BASE_DIR, suite_name, VERSION)
        if not os.path.exists(data_path):
            continue

        builder = tfds.builder_from_directory(builder_dir=data_path)
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)

        total_files = builder.info.splits['train'].num_examples

        total_trajs = 0
        success_trajs = 0
        total_samples = 0

        act_min = np.full(6, np.inf)
        act_max = np.full(6, -np.inf)

        print(f"\nProcessing {suite_name} ({total_files} trajectories)...")
        pbar = tqdm(enumerate(ds), total=total_files, unit="traj", desc=suite_name)

        for traj_id, traj_data in pbar:
            total_trajs += 1
            try:
                traj_batch = next(iter(traj_data['steps'].batch(2000)))
                if traj_batch['reward'][-1].numpy() != 1:
                    continue

                success_trajs += 1
                traj_len = int(traj_batch['action'].shape[0])

                # Global Action Stats
                actions = traj_batch['action'].numpy()[:, :6]
                current_min = np.min(actions, axis=0)
                current_max = np.max(actions, axis=0)
                act_min = np.minimum(act_min, current_min)
                act_max = np.maximum(act_max, current_max)

                total_samples += traj_len
                pbar.set_postfix({"Succ": success_trajs, "Samples": total_samples})

            except Exception as e:
                continue

        mixed_total_trajs += total_trajs
        mixed_success_trajs += success_trajs
        mixed_total_samples += total_samples
        mixed_act_min = np.minimum(mixed_act_min, act_min)
        mixed_act_max = np.maximum(mixed_act_max, act_max)

        print(f"--- {suite_name} Stats ---")
        print(f"Total Trajectories: {total_trajs}")
        print(f"Successful Trajs:   {success_trajs}")
        print(f"Avg Samples/Succ:   {total_samples / success_trajs:.4f}" if success_trajs > 0 else "")
        print(f"action_min = {act_min.tolist()}")
        print(f"action_max = {act_max.tolist()}")

    print(f"\n--- libero_mixed Stats (union of {', '.join(SUITES)}) ---")
    print(f"Total Trajectories: {mixed_total_trajs}")
    print(f"Successful Trajs:   {mixed_success_trajs}")
    print(f"Avg Samples/Succ:   {mixed_total_samples / mixed_success_trajs:.4f}" if mixed_success_trajs > 0 else "")
    print(f"action_min_mixed = {mixed_act_min.tolist()}")
    print(f"action_max_mixed = {mixed_act_max.tolist()}")
