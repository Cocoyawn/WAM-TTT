---
name: report-in-beijing-time
description: "When reporting times/ETAs to the user, convert server UTC to Beijing time (UTC+8)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 20209452-ac23-402f-b63a-11a0121f5282
---

The H200/A800 box runs in **UTC** (confirmed `date` = UTC, timezone +0000). The user is in Beijing
and wants all reported clock times / ETAs expressed in **Beijing time = UTC + 8 hours**.

**Why** (user directive 2026-06-11): the user reads progress reports in their local timezone; raw UTC
timestamps from logs (`logs/scheduler.log`, training tqdm, eval logs) and `date` are 8h behind Beijing
and caused me to report wrong wall-clock ETAs (e.g. "07:45" when it was really 15:45 Beijing).

**How to apply:**
- Any time I quote a completion time, deadline, or "expected at HH:MM", add 8h to the server/UTC time
  and label it 北京时间 (or just state Beijing time).
- Log timestamps stay UTC on disk — only the human-facing report is converted. Elapsed durations
  ("~3.2h remaining") are timezone-independent and need no conversion; only absolute clock times do.
- Quick check: `TZ='Asia/Shanghai' date` prints Beijing time directly.

Related: [[vlanext-ttt-experiment-status]], [[scheduler-is-resource-entrypoint]].
