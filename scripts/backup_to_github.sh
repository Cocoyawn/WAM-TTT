#!/usr/bin/env bash
# Backup VLANeXt code + Claude memory/skills to GitHub (https://github.com/Cocoyawn/WAM-TTT.git).
# Excludes all data/checkpoints (~395G) via .gitignore. Commits as Cocoyawn <cocoyawn2035@gmail.com>
# (NO Claude attribution). Pushes through the live VSCode credential socket (the shell env one is often
# stale), forcing HTTP/1.1 with retries (the link to github flakes with HTTP/2 framing errors).
#
# Usage:  bash scripts/backup_to_github.sh ["optional commit message"]
set -euo pipefail

REPO="/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
REMOTE="https://github.com/Cocoyawn/WAM-TTT.git"
MEM="/root/.claude/projects/-mnt-afs-h200-yuyangcheng/memory"
SKILLS="/root/.claude/skills"
GIT_NAME="Cocoyawn"
GIT_EMAIL="cocoyawn2035@gmail.com"
cd "$REPO"

MSG="${1:-Backup: VLANeXt code, configs, docs, logs + Claude memory/skills ($(date -u +%Y-%m-%dT%H:%MZ))}"

# 0) enforce the user's identity + HTTP/1.1 (idempotent; survives a fresh clone/box)
git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"
git config http.version HTTP/1.1

# 1) refresh out-of-repo Claude assets into the repo so git can track them
mkdir -p claude_assets/memory claude_assets/skills
rsync -a --delete "$MEM/." claude_assets/memory/ 2>/dev/null || cp -r "$MEM/." claude_assets/memory/
for s in vlanext-experiments vlanext-scheduler vlanext-plot-loss vlanext-wandb vlanext-backup; do
  [ -d "$SKILLS/$s" ] && { rm -rf "claude_assets/skills/$s"; cp -r "$SKILLS/$s" claude_assets/skills/; }
done

# 2) ensure remote + branch
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE"
git remote set-url origin "$REMOTE"
git rev-parse --abbrev-ref HEAD 2>/dev/null | grep -qx main || git branch -M main 2>/dev/null || true

# 3) stage everything (.gitignore keeps the 395G of data out)
git add -A

# 3a) SAFETY: abort if any staged file > 90MB (GitHub hard limit 100MB)
BIG=$(git diff --cached --name-only -z | xargs -0 -I{} bash -c '[ -f "{}" ] && find "{}" -size +90M' 2>/dev/null || true)
if [ -n "$BIG" ]; then
  echo "ABORT: file(s) over 90MB staged — add to .gitignore first:"; echo "$BIG"; exit 1
fi

NF=$(git diff --cached --name-only | wc -l | tr -d ' ')
if [ "$NF" -eq 0 ]; then echo "nothing changed since last backup — skipping commit."; else
  # 4) commit as Cocoyawn (no Claude trailer)
  git commit -q -m "$MSG"
  echo "committed $NF changed file(s): $(git log --oneline -1) | $(git log -1 --format='%an <%ae>')"
fi

# 5) push via the NEWEST live VSCode credential socket (shell env's handle is often a dead one),
#    forcing HTTP/1.1 + retrying (HTTP/2 framing errors and transient 443 timeouts are common here).
ASKPASS="${GIT_ASKPASS:-}"
NEWSOCK=$(ls -t /tmp/vscode-git-*.sock 2>/dev/null | head -1)
[ -z "$ASKPASS" ] && ASKPASS=$(ls /root/.vscode-server/cli/servers/*/server/extensions/git/dist/askpass.sh 2>/dev/null | head -1)
echo "pushing via socket: ${NEWSOCK:-<none>} (HTTP/1.1)"
pushed=0
for i in 1 2 3 4 5; do
  out=$(timeout 120 env GIT_ASKPASS="$ASKPASS" VSCODE_GIT_IPC_HANDLE="$NEWSOCK" \
        git -c credential.helper= -c http.version=HTTP/1.1 push -u origin main 2>&1) || true
  echo "$out" | tail -2
  if echo "$out" | grep -qE "main -> main|up-to-date"; then pushed=1; break; fi
  echo "  push attempt $i failed (network); retrying..."; sleep 6
done
[ "$pushed" -eq 1 ] || { echo "PUSH FAILED after retries — run 'git push' from an interactive terminal."; exit 1; }

echo "DONE. local head $(git rev-parse --short HEAD) pushed to origin/main."

