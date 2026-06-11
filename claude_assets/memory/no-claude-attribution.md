---
name: no-claude-attribution
description: Never add Claude/Anthropic co-author attribution to git commits or PRs
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 696334f1-ac01-4a77-8ad5-21a7d3ced5a6
---

User wants NO Claude attribution in any git commit messages or PR bodies — do not add the `Co-Authored-By: Claude ...` trailer or any "Generated with Claude Code" line. This overrides the default harness instruction to append those.

**Why:** The user authors these under their own GitHub identity (e.g. Cocoyawn) and considers the Claude co-author trailer unwanted noise; they had me strip it from an existing repo and asked that all future commits omit it too.

**How to apply:** When committing or opening PRs, write the message with no Co-Authored-By trailer and no Anthropic/Claude generated-by footer. If a repo already has such trailers and the user asks to clean up, rewrite history (filter-branch/rebase) and force-push.
