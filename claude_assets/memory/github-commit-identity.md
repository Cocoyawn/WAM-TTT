---
name: github-commit-identity
description: "Git/GitHub commit identity for this user — name Cocoyawn, email cocoyawn2035@gmail.com"
metadata: 
  node_type: memory
  type: user
  originSessionId: 20209452-ac23-402f-b63a-11a0121f5282
---

For ALL GitHub work, commits must use the user's identity:
- **name**: `Cocoyawn`
- **email**: `cocoyawn2035@gmail.com`

**Why** (user directive 2026-06-11): the first WAM-TTT backup committed as the box's default git
identity (`Shirk6 <shirk6@users.noreply.github.com>`), which was wrong. The user corrected it and said
this identity applies to **every** future GitHub repo, not just this one.

**How to apply:**
- Set globally once: `git config --global user.name "Cocoyawn"` and
  `git config --global user.email "cocoyawn2035@gmail.com"` (already done on the H200 box).
- When initializing or cloning any new repo, verify `git config user.name/user.email` resolve to the
  above before the first commit; set them locally if the global isn't inherited.
- Still NEVER add a `Co-Authored-By: Claude` trailer ([[no-claude-attribution]]). The author is Cocoyawn.
- The WAM-TTT backup remote is https://github.com/Cocoyawn/WAM-TTT.git; backup flow is the
  `vlanext-backup` skill (its script commits with this identity).

Related: [[no-claude-attribution]], [[report-in-beijing-time]].
