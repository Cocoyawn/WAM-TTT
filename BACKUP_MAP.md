# VLANeXt-TTT backup map

Audit date: 2026-08-06 (UTC)

## Primary code repository

- GitHub: `https://github.com/Cocoyawn/WAM-TTT`
- The original local `HEAD` (`2711db7`) was already on `main`.
- Backup commit: `24a10397b114e174187bcc04d0ee1a09058ef8c7`
- Security cleanup commit: `b48d59a2f538d05f1d0337b188d8cc75a02b8420`
- The backup commit adds 337 reviewed files (about 76 MB): TTT source changes,
  experiment configurations, safe helper scripts, and lightweight evaluation
  tables/curves.

Download:

```bash
git clone https://github.com/Cocoyawn/WAM-TTT.git
```

## Hugging Face repositories already containing relevant artifacts

| Local/working-tree scope | Hugging Face repository | Type | Status |
|---|---|---|---|
| LIBERO VLANeXt TTT/ablation checkpoints | `Cocoyawn32/VLANeXt-ablation-ckpt` | public dataset | Existing public backup; includes TTT, attention/GDN comparison checkpoints and three WM evaluation caches |
| DROID/Kairos comparison artifacts | `Cocoyawn32/vlanext-robolab-droid-eval` | public model | Existing backup, recorded for reference only; outside the VLANeXt-TTT scope |
| Kairos auxiliary model states | `Cocoyawn32/VLANeXt-kairos-ckpt` | public model | Existing auxiliary backup; not the primary VLANeXt-TTT deliverable |
| Earlier local sanitized bundle | `Cocoyawn32/local-workspace-backup-20260806` | private dataset | Existing private backup; contains the sanitized code archive, manifest, and selected Kairos asset |

Examples:

```bash
hf download Cocoyawn32/VLANeXt-ablation-ckpt --repo-type dataset --local-dir ./VLANeXt_ablation_ckpt
hf download Cocoyawn32/vlanext-robolab-droid-eval --repo-type model --local-dir ./vlanext_robolab_droid_eval
hf download Cocoyawn32/VLANeXt-kairos-ckpt --repo-type model --local-dir ./VLANeXt_kairos_ckpt
```

## Scope note

DROID TTT checkpoints under `workplace/VLANeXt-kairos_ckpts` belong to the
VLANeXt-Kairos project and are intentionally out of scope for this backup map.

## Deliberately excluded from the public GitHub backup

- Training data, caches, logs, checkpoints, model weights, rollout videos, and
  the NVIDIA driver installer.
- About 30 local untracked launch scripts containing hard-coded W&B credentials.
- `.env`/secret files, SSH material, proxy/environment files, and Git metadata.

The current public tree was scanned for common HF/GitHub/W&B/AWS/OpenAI token
and private-key patterns; no matches were found in the included files after
`b48d59a2`. The old W&B value existed in earlier public history; this cleanup
did not rewrite history, so that credential must be revoked/rotated. This is
not a proof that the local workspace contains no secrets; the excluded local
files still require credential rotation and cleanup.
