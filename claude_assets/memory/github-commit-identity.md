---
name: github-commit-identity
description: "Git/GitHub commit identity — name Cocoyawn, email 180568729+Cocoyawn@users.noreply.github.com (the ONLY email tied to the Cocoyawn GitHub account)"
metadata: 
  node_type: memory
  type: user
  originSessionId: 20209452-ac23-402f-b63a-11a0121f5282
---

For ALL GitHub work, commits must use the user's identity:
- **name**: `Cocoyawn`
- **email**: `180568729+Cocoyawn@users.noreply.github.com`

This is the GitHub-provided no-reply address for the **Cocoyawn** account
(numeric user id 180568729). It is the ONLY email that makes commits show up as
the user's contribution on GitHub.

**Why** (corrected 2026-06-20): TWO earlier emails were both WRONG and caused
contributions to be attributed to a STRANGER (GitHub user `shirk6`) instead of
the user, leaving the user's own contribution graph empty:
- `Shirk6 <shirk6@users.noreply.github.com>` — box default; the noreply belongs
  to a stranger's account.
- `Cocoyawn <cocoyawn2035@gmail.com>` — a gmail NOT verified on the Cocoyawn
  GitHub account, so GitHub could not associate it with the user either.
On 2026-06-20 the entire WAM-TTT history (13 commits) was rewritten via
filter-branch to the correct no-reply email and force-pushed; tree hashes
unchanged (content untouched, only author/committer email).

**How to apply:**
- Set globally once: `git config --global user.name "Cocoyawn"` and
  `git config --global user.email "180568729+Cocoyawn@users.noreply.github.com"`
  (done on the H200 box 2026-06-20).
- When initializing or cloning any new repo, verify `git config user.email`
  resolves to the no-reply above before the first commit.
- To find the no-reply for any account: `curl -s https://api.github.com/users/<login>`
  gives the numeric `id`; the address is `<id>+<login>@users.noreply.github.com`.
- Still NEVER add a `Co-Authored-By: Claude` trailer ([[no-claude-attribution]]).
- The WAM-TTT backup remote is git@github.com:Cocoyawn/WAM-TTT.git; push uses the
  SSH key /mnt/afs-h200/yuyangcheng/.ssh/id_rsa (the ghfast.top https URL fails on push).

Related: [[no-claude-attribution]], [[report-in-beijing-time]].
