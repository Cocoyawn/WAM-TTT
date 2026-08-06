# LIBERO Context Probe

This probe starts from the second context condition in the instance-irrelevant
context plan: a successful demonstration from a different instance of the same
task. It does not toggle TTT updates and does not retrain a checkpoint.

## Conditions

| Condition | Added visual prefix | Purpose |
|---|---|---|
| `none` | none | paired reference |
| `same_task` | 2 RGB frames from another successful trajectory with the same instruction | test structured same-task context |
| `other_task` | 2 RGB frames from a successful trajectory with a different instruction | task-irrelevant control |

The current observation history, task IDs, trial seeds, checkpoint, diffusion
steps, and number of context frames are identical across conditions. The bank
contains RGB frames only; action labels are never passed to the model. The
context frames are prepended to the current visual input, so the current
observation remains the suffix seen by the policy. For the current mixed-clean
checkpoint this means two context frames from each of the exterior and wrist
views, followed by the current exterior and wrist images.

## Two-stage workflow

1. Run `scripts/smoke_libero_context.py` before touching the simulator.
2. Run `scripts/run_libero_context_probe.sh` on 4 spatial tasks and one trial.

The bank builder requires the existing TFDS environment. The simulator requires
the usual LIBERO-plus environment with `MUJOCO_GL=osmesa`. Set `CKPT` and
`PYTHON_BIN` explicitly on a new machine.

## Reading the result

The first result is the paired success-rate change:

`same_task - none` versus `other_task - none`.

A positive same-task effect with a weak or negative other-task effect supports
useful structured context. No effect in both conditions means the current
checkpoint is not using this prompt context, and should not be interpreted as
evidence for or against persistent TTT. A positive effect in both conditions
requires a token-count or generic visual-prefix control before claiming task
structure is useful.
