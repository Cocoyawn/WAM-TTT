"""Offline TTT-checkpoint loss-split evaluation.

For each saved checkpoint of a VLANeXt training run, load it, run forward on a
fixed 512-sample train batch with augmentation disabled, and dump the three loss
components separately (action / dct / image) to a CSV.

Usage:
    python scripts/eval_loss_split.py \
        --runs long allttt 1to1 \
        --device cuda:1 \
        --num-batches 32 --batch-size 16

Output layout (docs/ttt_loss_split/):
    {run}.csv                  one row per ckpt: step, total, action, dct, img
    {run}.png                  3 curves on a single axes
    combined.png               3 subplots, one per run
"""
import argparse
import csv
import os
import re
import sys
import time
from glob import glob

import numpy as np
import torch
import yaml

# Make `scripts.train` importable as a sibling module.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.datasets.molmoact_droid_act import MolmoActDroidAct  # noqa: E402
from src.models.VLANeXt import VLANeXt  # noqa: E402
from scripts.train import DataCollatorForVLANeXt, set_seed  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


# ----- Runs and configs -----------------------------------------------------
# All "1/4 TTT main-line" runs (mix_every_n=4) form a resume chain:
#   `early` (0 -> 60k) -> `cont60k` resumes from 60k (+20k more) -> optionally
#   `long` is a separate fresh cosine cycle starting from a pretrained ckpt.
# `allttt` (mix_every_n=1) and `1to1` (mix_every_n=2) are independent arches.
# Each entry is (run_dir, config_path, step_offset). step_offset is added to
# every ckpt's saved step to convert a relative-to-run step into an ABSOLUTE
# training step across the whole 1/4-TTT main line.
RUN_TABLE = {
    "long":   ("VLANeXt_robolab_ttt/chunk256_molmoact_droid_long",
               "config/ablation_wm_ttt_chunk256_molmoact_droid_long.yaml",     0),
    "allttt": ("VLANeXt_robolab_ttt/chunk256_molmoact_droid_allttt",
               "config/ablation_wm_ttt_chunk256_molmoact_droid_allttt.yaml",   0),
    "1to1":   ("VLANeXt_robolab_ttt/chunk256_molmoact_droid_1to1",
               "config/ablation_wm_ttt_chunk256_molmoact_droid_1to1.yaml",     0),
    "early":   ("VLANeXt_robolab_ttt/chunk256_molmoact_droid",
                "config/ablation_wm_ttt_chunk256_molmoact_droid.yaml",         0),
    "cont60k": ("VLANeXt_robolab_ttt/chunk256_molmoact_droid_cont60k",
                "config/ablation_wm_ttt_chunk256_molmoact_droid_cont60k.yaml", 60000),
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config, device):
    """Mirror scripts/train.py:375-424 (VLANeXt branch only)."""
    future_image_loss_weight = float(config['model'].get('future_image_loss_weight', 0.0))
    model = VLANeXt(
        lmm_path=config['model']['lmm_path'],
        vision_encoder_path=config['model'].get('vision_encoder_path', "google/siglip2-base-patch16-256"),
        action_dim=config['model']['action_dim'],
        num_actions=config['data']['future_len'],
        num_queries=config['model']['num_queries'],
        num_history=config['data']['history_len'],
        loss_type=config['model'].get('loss_type', 'diffusion'),
        future_image_loss_weight=future_image_loss_weight,
        num_train_timesteps=config['model'].get('num_train_timesteps', 1000),
        num_inference_timesteps=config['model'].get('num_inference_timesteps', 10),
        scheduler_type=config['model']['scheduler_type'],
        condition_type=config['model'].get('condition_type', 'loose'),
        policy_hidden_size=config['model']['policy_hidden_size'],
        policy_depth=config['model']['policy_depth'],
        policy_num_heads=config['model']['policy_num_heads'],
        policy_mlp_ratio=config['model']['policy_mlp_ratio'],
        policy_mixer_type=config['model'].get('policy_mixer_type', 'attention'),
        policy_mix_every_n=config['model'].get('policy_mix_every_n', 4),
        policy_ttt_causal=config['model'].get('policy_ttt_causal', False),
        policy_ttt_chunk_size=config['model'].get('policy_ttt_chunk_size', 64),
        use_proprio_input_vlm=config['model'].get('use_proprio_input_vlm', True),
        use_action_input_policy=config['model'].get('use_action_input_policy', True),
        use_transformer_proprio_projector=config['model']['use_transformer_proprio_projector'],
        projector_depth=config['model']['projector_depth'],
        projector_num_heads=config['model']['projector_num_heads'],
        use_transformer_connector=config['model']['use_transformer_connector'],
        connector_depth=config['model']['connector_depth'],
        connector_num_heads=config['model']['connector_num_heads'],
        backbone_mode=config['model'].get('backbone_mode', 'finetune'),
        gradient_checkpointing=config['model'].get('gradient_checkpointing', False),
        num_bins=config['model'].get('num_bins', 256),
        generator_hidden_size=config['model'].get('generator_hidden_size', 768),
        generator_depth=config['model'].get('generator_depth', 12),
        generator_num_heads=config['model'].get('generator_num_heads', 12),
        generator_mlp_ratio=config['model'].get('generator_mlp_ratio', 4.0),
        generator_mixer_type=config['model'].get('generator_mixer_type', 'attention'),
        generator_mix_every_n=config['model'].get('generator_mix_every_n', 4),
        generator_ttt_chunk_size=config['model'].get('generator_ttt_chunk_size', 16),
        generator_fallback_mixer=config['model'].get('generator_fallback_mixer', 'attention'),
        generator_swa_window_size=config['model'].get('generator_swa_window_size', 64),
        action_vqvae=config['model'].get('action_vqvae', None),
        dct_loss_weight=config['model'].get('dct_loss_weight', 0.1),
        dct_low_freq_weight=config['model'].get('dct_low_freq_weight', 1.0),
        dct_high_freq_weight=config['model'].get('dct_high_freq_weight', 1.0),
        dct_freq_split=config['model'].get('dct_freq_split', 0.125),
        dct_similarity_type=config['model'].get('dct_similarity_type', 'mae'),
        attn_implementation=config['model'].get('attn_implementation', 'flash_attention_2'),
    ).to(device, dtype=torch.bfloat16)
    return model


def build_dataloader(config, model, batch_size, buffer_size, num_workers):
    """Eval dataloader: aug off, tiny shuffle buffer, fewer workers."""
    future_image_loss_weight = float(config['model'].get('future_image_loss_weight', 0.0))
    load_future_image = future_image_loss_weight > 0
    future_image_mode = config['model'].get('future_image_mode', 'horizon')
    input_modality = config['data'].get('input_modality', 'video')
    view_mode = config['data'].get('view_mode', 'single')
    full_sequence = bool(config['data'].get('full_sequence', False))

    collator = DataCollatorForVLANeXt(
        processor=model.processor,
        use_proprio_input_vlm=config['model'].get('use_proprio_input_vlm', True),
        use_action_input_policy=config['model'].get('use_action_input_policy', True),
        input_modality=input_modality,
        view_mode=view_mode,
        fps=15.0,  # molmoact_droid
        augmentation={"enabled": False},
        load_future_image=load_future_image,
    )

    assert config['data'].get('dataset_name') == 'molmoact_droid', \
        "this eval script is wired for molmoact_droid only"
    ds = MolmoActDroidAct(
        data_root=config['data']['data_root'],
        dataset_name='molmoact_droid',
        history_len=config['data']['history_len'],
        future_len=config['data']['future_len'],
        full_sequence=full_sequence,
        input_modality=input_modality,
        view_mode=view_mode,
        load_future_image=load_future_image,
        future_image_mode=future_image_mode,
        buffer_size=buffer_size,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collator)


def cache_batches(dl, n_batches, device):
    """Iterate the dataloader once, materialize n_batches on `device` (bf16), then reuse for every ckpt."""
    cached = []
    it = iter(dl)
    for _ in range(n_batches):
        batch = next(it)
        inputs, gt_actions, proprio, hist_actions, future_images = batch
        model_inputs = {k: v.to(device) for k, v in inputs.items()}
        for k in ('pixel_values', 'pixel_values_videos'):
            if k in model_inputs:
                model_inputs[k] = model_inputs[k].to(dtype=torch.bfloat16)
        gt_actions = gt_actions.to(device, dtype=torch.bfloat16)
        if proprio is not None:
            proprio = proprio.to(device, dtype=torch.bfloat16)
        if hist_actions is not None:
            hist_actions = hist_actions.to(device, dtype=torch.bfloat16)
        if future_images is not None:
            future_images = future_images.to(device, dtype=torch.bfloat16)
        valid = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
                 "image_grid_thw", "video_grid_thw", "token_type_ids"}
        forward_args = {k: v for k, v in model_inputs.items() if k in valid}
        cached.append((gt_actions, proprio, hist_actions, future_images, forward_args))
    return cached


def load_ckpt(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck['model_state_dict']
    if next(iter(sd)).startswith('module.'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            f"state_dict mismatch loading {path}: "
            f"missing={len(missing)} unexpected={len(unexpected)} -- "
            f"model arch doesn't match ckpt arch (likely wrong mix_every_n).")
    step = int(ck.get('step', 0))
    return step, missing, unexpected


@torch.no_grad()
def eval_ckpt(model, cached, n_diffusion_runs=1):
    """Average parts dict over the cached batches.

    For diffusion loss_type, each forward samples a fresh noise/timestep, so we
    optionally re-run a batch n_diffusion_runs>1 times to denoise the curve. 1
    is enough at 512 samples.
    """
    accum = {"loss_total": 0.0, "loss_action": 0.0, "loss_dct": 0.0, "loss_img": 0.0}
    counts = {"loss_total": 0, "loss_action": 0, "loss_dct": 0, "loss_img": 0}
    for (gt_actions, proprio, hist_actions, future_images, forward_args) in cached:
        for _ in range(n_diffusion_runs):
            loss, parts = model(
                actions=gt_actions,
                proprioception=proprio,
                history_actions=hist_actions,
                future_images=future_images,
                return_parts=True,
                **forward_args,
            )
            accum["loss_total"] += float(loss.item()); counts["loss_total"] += 1
            for k in ("loss_action", "loss_dct", "loss_img"):
                v = parts.get(k)
                if v is not None:
                    accum[k] += float(v.item()); counts[k] += 1
    return {k: (accum[k] / counts[k] if counts[k] > 0 else float('nan'))
            for k in accum}


def list_ckpts(run_dir, max_steps=None):
    """All checkpoints under run_dir. Numbered ckpts (`checkpoint_<N>.pt`) use
    N as the step; `checkpoint_final.pt` uses `max_steps` if provided, else
    skipped (no meaningful x-coordinate).
    """
    pat = re.compile(r'^checkpoint_(\d+)\.pt$')
    files = []
    for p in glob(os.path.join(run_dir, 'checkpoint_*.pt')):
        base = os.path.basename(p)
        m = pat.match(base)
        if m:
            files.append((int(m.group(1)), p))
        elif base == 'checkpoint_final.pt' and max_steps is not None:
            files.append((int(max_steps), p))
    files.sort()
    return files


def write_csv(rows, path):
    fields = ["step", "loss_total", "loss_action", "loss_dct", "loss_img"]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


# ----- Plotting -------------------------------------------------------------
# Colors: user-requested scientific palette (deep blue / light blue / peach /
# deep red / light red). We pick the two darker tones as the primary loss
# curves (action = blue, image = red — the two big components), and the
# lighter blue as dct (an action-side auxiliary regulariser, so same family
# as action, lighter to visually recede).
COLORS = {
    "loss_action": "#2878B5",  # deep blue
    "loss_dct":    "#9AC9DB",  # light blue (same family as action)
    "loss_img":    "#C82423",  # deep red
}
LABELS = {
    "loss_action": "action (MSE)",
    "loss_dct":    "dct",
    "loss_img":    "image (CE)",
}


def _apply_rc():
    """Set matplotlib rcParams to a Times-like serif look."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    rcParams['font.family']    = 'serif'
    rcParams['font.serif']     = ['Times New Roman', 'Liberation Serif',
                                  'DejaVu Serif', 'serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.titlesize']   = 13
    rcParams['axes.labelsize']   = 12
    rcParams['xtick.labelsize']  = 10
    rcParams['ytick.labelsize']  = 10
    rcParams['legend.fontsize']  = 10
    rcParams['axes.linewidth']   = 0.9
    rcParams['xtick.direction']  = 'in'
    rcParams['ytick.direction']  = 'in'
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4
    rcParams['xtick.minor.size'] = 2
    rcParams['ytick.minor.size'] = 2
    rcParams['axes.grid']        = True
    rcParams['grid.linestyle']   = ':'
    rcParams['grid.alpha']       = 0.45
    rcParams['grid.linewidth']   = 0.6


def _annotate(ax, xs, ys, color, offset_y_pts, fmt):
    """Annotate each (x, y) marker with its numeric value slightly offset."""
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), xy=(x, y), xytext=(0, offset_y_pts),
                    textcoords='offset points', ha='center',
                    va=('bottom' if offset_y_pts > 0 else 'top'),
                    fontsize=8, color=color)


# Where to place the numeric label for each series, relative to its marker.
# image (top curve, large values) -> label BELOW so it never clips the top axis.
# action (middle) -> label ABOVE its marker.
# dct    (bottom, smallest) -> label BELOW its marker, so action-above + dct-below
#   never collide (action ~0.04 sits above dct ~0.02 in log-space).
_ANNOT = {
    "loss_action": (10,  "{:.3f}"),
    "loss_dct":    (-12, "{:.3f}"),
    "loss_img":    (-14, "{:.2f}"),
}


def plot_run(rows, run_name, out_png):
    _apply_rc()
    import matplotlib.pyplot as plt
    steps = [r['step'] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key in ('loss_action', 'loss_dct', 'loss_img'):
        ys = [r[key] for r in rows]
        ax.plot(steps, ys, marker='s', markersize=5.5, markerfacecolor='white',
                markeredgewidth=1.4, color=COLORS[key], label=LABELS[key],
                linewidth=1.7)
        off, fmt = _ANNOT[key]
        _annotate(ax, steps, ys, COLORS[key], off, fmt)
    ax.set_xlabel('training step')
    ax.set_ylabel('loss')
    ax.set_yscale('log')
    ax.set_ylim(6e-3, 40)  # extra headroom for the bottom dct labels + top image labels
    ax.set_title(run_name)
    ax.legend(loc='center right', frameon=True, framealpha=0.92,
              edgecolor='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_combined(all_rows, out_png, suptitle=None):
    _apply_rc()
    import matplotlib.pyplot as plt
    runs = list(all_rows.keys())
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        rows = all_rows[run]
        steps = [r['step'] for r in rows]
        for key in ('loss_action', 'loss_dct', 'loss_img'):
            ys = [r[key] for r in rows]
            ax.plot(steps, ys, marker='s', markersize=5.5, markerfacecolor='white',
                    markeredgewidth=1.4, color=COLORS[key], label=LABELS[key],
                    linewidth=1.7)
            off, fmt = _ANNOT[key]
            _annotate(ax, steps, ys, COLORS[key], off, fmt)
        ax.set_xlabel('training step')
        ax.set_yscale('log')
        ax.set_ylim(6e-3, 40)
        ax.set_title(run)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
    axes[0].set_ylabel('loss')
    axes[-1].legend(loc='center right', frameon=True, framealpha=0.92,
                    edgecolor='#888888')
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['long', 'allttt', '1to1'])
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-batches', type=int, default=32)
    ap.add_argument('--buffer-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true', help='only 1 batch, only 1 ckpt per run')
    ap.add_argument('--ckpts', nargs='+', default=None,
                    help='absolute paths; overrides --runs ckpt discovery')
    ap.add_argument('--out-dir', default='docs/ttt_loss_split')
    args = ap.parse_args()

    if args.smoke:
        args.num_batches = 1

    out_dir = os.path.join(_REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    # All 3 runs share the SAME data config (dataset_name, history_len, future_len,
    # input_modality, lmm_path, etc.) -- but DIFFER in model arch (policy/generator
    # mix_every_n is 4/2/1 across long/1to1/allttt -> different #TTT layers ->
    # different state_dict keys). So we cache batches once with the first run's
    # processor, then REBUILD the model per run from its own config.
    first = args.runs[0]
    cfg0 = load_config(os.path.join(_REPO, RUN_TABLE[first][1]))

    print(f"[build] initial model from {RUN_TABLE[first][1]} (for processor + first run)", flush=True)
    set_seed(args.seed)
    model = build_model(cfg0, device)
    model.eval()

    print(f"[data] buffer_size={args.buffer_size}, batch={args.batch_size}, batches={args.num_batches}", flush=True)
    set_seed(args.seed)
    dl = build_dataloader(cfg0, model, args.batch_size, args.buffer_size, args.num_workers)
    t0 = time.time()
    cached = cache_batches(dl, args.num_batches, device)
    print(f"[data] cached {len(cached)} batches ({len(cached) * args.batch_size} samples) in {time.time()-t0:.1f}s", flush=True)
    # The processor lives inside the model; we don't need it after cache_batches
    # (collator already turned text+pixels into tensors). The dataloader workers
    # have copied the dataset state, so we can drop dl too.
    del dl

    cur_run = first  # initial model already built from `first`'s config
    all_rows = {}
    for run in args.runs:
        # Rebuild model per run if needed (different mix_every_n -> different
        # state_dict layout). First run reuses the initial model built above.
        if run != cur_run:
            print(f"[build] free previous model, rebuilding for run={run}", flush=True)
            del model
            torch.cuda.empty_cache()
            cfg = load_config(os.path.join(_REPO, RUN_TABLE[run][1]))
            model = build_model(cfg, device)
            model.eval()
            cur_run = run

        run_dir = os.path.join(_REPO, RUN_TABLE[run][0])
        run_cfg = load_config(os.path.join(_REPO, RUN_TABLE[run][1]))
        step_offset = RUN_TABLE[run][2]
        max_steps = int(run_cfg['data'].get('max_steps', 0)) or None
        raw_ckpts = list_ckpts(run_dir, max_steps=max_steps)
        ckpts = [(raw_step + step_offset, path) for raw_step, path in raw_ckpts]
        if args.smoke:
            ckpts = ckpts[-1:]
        if not ckpts:
            print(f"[skip] no numbered ckpts under {run_dir}")
            continue

        print(f"\n[run={run}] {len(ckpts)} ckpts in {run_dir}", flush=True)
        rows = []
        for step, path in ckpts:
            t = time.time()
            torch.cuda.empty_cache()
            sk_step, missing, unexpected = load_ckpt(model, path, device)
            print(f"  [load] step={step} (ckpt.step={sk_step}) missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            model.eval()
            mean = eval_ckpt(model, cached)
            row = {"step": step, **mean}
            rows.append(row)
            print(f"  [eval] step={step} total={mean['loss_total']:.4f} action={mean['loss_action']:.4f} "
                  f"dct={mean['loss_dct']:.4f} img={mean['loss_img']:.4f}  ({time.time()-t:.1f}s)", flush=True)

        csv_path = os.path.join(out_dir, f"{run}.csv")
        write_csv(rows, csv_path)
        print(f"[csv] {csv_path}", flush=True)

        png_path = os.path.join(out_dir, f"{run}.png")
        plot_run(rows, run, png_path)
        print(f"[png] {png_path}", flush=True)

        all_rows[run] = rows

    # Stitch the 1/4-TTT main-line: early + cont60k share an arch and form a
    # continuous resume chain (cont60k's step already includes +60000 offset).
    mainline_rows = []
    for r in ("early", "cont60k"):
        if r in all_rows:
            mainline_rows.extend(all_rows[r])
    if mainline_rows:
        mainline_rows.sort(key=lambda r: r['step'])
        csv_path = os.path.join(out_dir, 'mainline_quarter.csv')
        write_csv(mainline_rows, csv_path)
        print(f"[csv] {csv_path}  (early + cont60k stitched)", flush=True)
        png_path = os.path.join(out_dir, 'mainline_quarter.png')
        plot_run(mainline_rows, 'mainline 1/4 TTT (early + cont60k)', png_path)
        print(f"[png] {png_path}", flush=True)
        all_rows['mainline_quarter'] = mainline_rows

    if len(all_rows) > 1:
        combined = os.path.join(out_dir, 'combined.png')
        plot_combined(all_rows, combined)
        print(f"[png] {combined}", flush=True)


if __name__ == "__main__":
    main()
