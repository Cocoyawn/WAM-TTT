# Memory Index

- [No Claude attribution](no-claude-attribution.md) — never add Claude/Anthropic co-author trailer to commits or PRs
- [Prefer tmux over nohup](prefer-tmux-over-nohup.md) — background/long-running jobs go in a tmux session, not nohup
- [Report in Beijing time](report-in-beijing-time.md) — server runs UTC; convert all reported clock times/ETAs to Beijing (UTC+8)
- [Scheduler is resource entrypoint](scheduler-is-resource-entrypoint.md) — run new trainings/evals via scripts/scheduler.py queue (extend it, don't launch ad-hoc); see vlanext-scheduler skill
- [TTT chunk_size train speed](ttt-chunk-size-train-speed.md) — VLANeXt vision-expert TTT chunk_size↔step-time; chunk_size=block SIZE not count, 256/size=#blocks
- [VLANeXt TTT experiment status](vlanext-ttt-experiment-status.md) — TTT ablation: chunk 16/4/1-block SR curve DONE, env-gen eval ~90% (TTT 9/12 + attn baseline running), pending todos
- [VLANeXt GLA/causal plan](vlanext-gla-causal-plan.md) — next-2-day plan: GLA/GatedDeltaNet mixer + action causality probe (KairosDiT ref)
- [VLANeXt TTT latency](vlanext-ttt-latency.md) — measured train (1.29x slower) + deploy inference latency (attn fastest; bigger chunk 4.1x→1.8x); methods
- [VLANeXt env-gen TTT vs Attn results](vlanext-envgen-ttt-vs-attn-results.md) — FINAL per-dim generalization table (1627 tasks): TTT +5.6pp overall; Noise +17, Robot +11; Camera/Bg ~tie
- [VLANeXt compute resource limits](vlanext-compute-resource-limits.md) — disk(20T quarkfs, 20GB/ckpt, save_interval=10000) + cgroup-RAM(416GB, 8 eval workers) + GPU limits; cleanup playbook; PYTHONPATH gotchas
