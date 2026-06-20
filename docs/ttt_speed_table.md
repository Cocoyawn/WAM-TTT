# TTT-256 single-layer inference speed (10 samples)

## Scene A: single full-prefix forward (L=256)

| method | median (ms) | std | speedup vs attn |
|---|---|---|---|
| attention | 0.104 | 0.145 | 1.00x |
| torch-TTT-256 | 1.903 | 0.752 | 0.05x |
| CUDA-TTT-256 | 1.308 | 0.385 | 0.08x |

## Scene B: 256-step autoregressive rollout (deployment)

| method | median (ms) | std | speedup vs attn |
|---|---|---|---|
| attention | 90.502 | 2.099 | 1.00x |
| torch-TTT-256 | 516.238 | 8.470 | 0.18x |
| CUDA-TTT-256 | 14.082 | 1.136 | 6.43x |

