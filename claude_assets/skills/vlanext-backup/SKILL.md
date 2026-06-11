---
name: vlanext-backup
description: Back up the VLANeXt repo (code, configs, docs, logs) plus Claude memory & skills to GitHub (Cocoyawn/WAM-TTT). Use whenever the user says "备份到github", "backup to github", "推送备份", "备份代码", or asks to snapshot/commit the project to the remote. Excludes all data/checkpoints (~395G). Commits with the user's own git identity — NEVER adds Claude attribution. Working dir /mnt/afs-h200/yuyangcheng/workplace/VLANeXt.
---

# VLANeXt → GitHub backup

One command snapshots the **code + research artifacts** (not the 395G of data) to
`https://github.com/Cocoyawn/WAM-TTT.git`. The repo at `/mnt/afs-h200/yuyangcheng/workplace/VLANeXt`
is a git repo on branch `main` tracking that remote.

## Run it

```bash
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
bash scripts/backup_to_github.sh                       # default timestamped commit msg
bash scripts/backup_to_github.sh "msg describing what changed"
```

The script: (1) refreshes `claude_assets/{memory,skills}/` from the out-of-repo
`~/.claude` locations so they get versioned; (2) `git add -A`; (3) **aborts if any staged file
> 90MB** (GitHub limit); (4) commits with the user's identity (Cocoyawn <cocoyawn2035@gmail.com>, **no Claude trailer**);
(5) pushes via the newest live VSCode credential socket.

## What's included vs excluded (.gitignore)

| Included (~85M) | Excluded |
|---|---|
| `src/` `scripts/` `config/` (model code, 90 configs) | `VLANeXt_ablation_wm/` `VLANeXt_final_libero/` (395G ckpts) |
| `*.md` design/results docs | `third_party/` (LIBERO-plus etc., own upstream git) |
| `docs/*.png` (result figures) | `docs/static/` + all `*.mp4` (demo videos) |
| `logs/` (training/eval logs) | `wandb/`, `*.pt/.ckpt/.hdf5/.npz`, `__pycache__` |
| `claude_assets/memory` + `claude_assets/skills` | `.claude/scheduler_state.json` |

## Critical constraints (why the script is shaped this way)
- **Identity = Cocoyawn `<cocoyawn2035@gmail.com>`** ([[github-commit-identity]]): the script sets
  `git config user.name/user.email` to this on every run (the box's default was wrongly `Shirk6`).
  **NO Claude attribution** ([[no-claude-attribution]]) — never add a `Co-Authored-By: Claude` trailer.
- **Push flakes two ways, both handled by forcing HTTP/1.1 + retrying:**
  1. *Stale credential socket* — `credential.helper` points at a VSCode IPC socket; the handle in the
     shell env (`VSCODE_GIT_IPC_HANDLE`) is often a **dead** socket → `ECONNREFUSED` / "Authentication
     failed". Fix: pick the NEWEST `/tmp/vscode-git-*.sock` and bypass the stale helper.
  2. *HTTP/2 framing errors / 443 timeouts* — the link to github.com is unstable; HTTP/2 multiplexes
     everything onto one connection, so any blip kills the whole push with `Error in the HTTP2 framing
     layer`. Fix: force `http.version=HTTP/1.1` (stateless, per-request, far more resilient) — set
     globally AND per-push — and retry up to 5×. If it still fails the script exits non-zero; run
     `git push` from an interactive terminal (or `! git push` in this session).
- **Size guard**: a >90MB file aborts the commit — add it to `.gitignore` and retry. Logs are the
  biggest included items (~70M total); if they balloon, consider excluding `logs/` too.
- Memory/skills live OUTSIDE the repo (`~/.claude/...`); they're only captured because the script
  copies them into `claude_assets/`. Editing them later requires a re-run to back up.

## After running
Report the commit line + the pushed short-sha, and confirm the author is Cocoyawn (no Claude attribution).
Related: [[scheduler-is-resource-entrypoint]], [[no-claude-attribution]].
