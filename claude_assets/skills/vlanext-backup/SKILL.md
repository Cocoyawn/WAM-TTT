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
> 90MB** (GitHub limit); (4) commits with the user's identity (Shirk6, **no Claude trailer**);
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
- **NO Claude attribution** ([[no-claude-attribution]]): commits use `git config user.name`=Shirk6.
  Never add a `Co-Authored-By: Claude` trailer here.
- **Push auth is finicky**: the git `credential.helper` points at a VSCode IPC socket. The handle in
  the shell's env (`VSCODE_GIT_IPC_HANDLE`) is frequently a **stale/dead** socket → `ECONNREFUSED` /
  "Authentication failed". The script works around this by selecting the NEWEST `/tmp/vscode-git-*.sock`
  and bypassing the stale `credential.helper`. If push still fails, the user may need to run the push
  themselves in their interactive terminal: `git push -u origin main` (or `! git push` in this session).
- **Size guard**: a >90MB file aborts the commit — add it to `.gitignore` and retry. Logs are the
  biggest included items (~70M total); if they balloon, consider excluding `logs/` too.
- Memory/skills live OUTSIDE the repo (`~/.claude/...`); they're only captured because the script
  copies them into `claude_assets/`. Editing them later requires a re-run to back up.

## After running
Report the commit line + remote head short-sha, and confirm "no Claude attribution" in the author.
Related: [[scheduler-is-resource-entrypoint]], [[no-claude-attribution]].
