#!/usr/bin/env python3
"""Aggregate eval_wm_quality result_*.json into one markdown comparison table.

Reads result_{loss,gen}_{tag}.json from the cache dir and emits
WM_QUALITY_SUMMARY.md: per-suite TTT-vs-attention table for
loss_img / token_acc / PSNR / SSIM / LPIPS, with the TTT-minus-attn delta.
"""
from __future__ import annotations

import argparse
import json
import os

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
CACHE_DIR = os.path.join(REPO, "VLANeXt_ablation_wm", "wm_eval_cache")

SUITE_LABEL = {
    "libero_spatial_no_noops": "spatial (in-dist)",
    "libero_object_no_noops": "object (OOD)",
    "libero_goal_no_noops": "goal (OOD)",
}


def load(tag, mode):
    p = os.path.join(CACHE_DIR, f"result_{mode}_{tag}.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))["results"]


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def dfmt(v, nd=4):
    """Signed delta formatter."""
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttt_tag", default="ttt")
    ap.add_argument("--attn_tag", default="attn")
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "WM_QUALITY_SUMMARY.md"))
    args = ap.parse_args()

    ttt_loss, attn_loss = load(args.ttt_tag, "loss"), load(args.attn_tag, "loss")
    ttt_gen, attn_gen = load(args.ttt_tag, "gen"), load(args.attn_tag, "gen")
    suites = list(dict.fromkeys(
        list(ttt_loss) + list(attn_loss) + list(ttt_gen) + list(attn_gen)))

    lines = []
    lines.append("# 世界模型视频生成质量：TTT-16 vs Attention（30k final）\n")
    lines.append("共享缓存 eval 集（两模型同一批输入）。loss=teacher-forced 图像 token "
                 "交叉熵（隔离自 action/dct 项）+ token top-1 acc；"
                 "gen=free-running greedy AR → VQ decode → 像素，对比 GT 未来帧。\n")
    lines.append("Δ = TTT − Attn。loss_img 越低越好，其余越高越好"
                 "（LPIPS 越低越好）。\n")

    # --- Layer A: teacher-forced loss ---
    lines.append("\n## Layer A — teacher-forced 图像 token loss / accuracy\n")
    lines.append("| suite | loss_img TTT | loss_img Attn | Δloss | tok-acc TTT | tok-acc Attn | Δacc |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in suites:
        a, b = ttt_loss.get(s, {}), attn_loss.get(s, {})
        if not a and not b:
            continue
        lt, lb = a.get("loss_img_mean"), b.get("loss_img_mean")
        at, ab = a.get("token_acc"), b.get("token_acc")
        dl = (lt - lb) if (lt is not None and lb is not None) else None
        da = (at - ab) if (at is not None and ab is not None) else None
        lines.append(
            f"| {SUITE_LABEL.get(s, s)} | {fmt(lt)} | {fmt(lb)} | "
            f"{dfmt(dl)} | {fmt(at)} | {fmt(ab)} | {dfmt(da)} |")

    # --- Layer B: free-running generation ---
    lines.append("\n## Layer B — free-running 生成像素质量\n")
    has_lpips = any("lpips_mean" in v for v in list(ttt_gen.values()) + list(attn_gen.values()))
    hdr = "| suite | PSNR TTT | PSNR Attn | ΔPSNR | SSIM TTT | SSIM Attn | ΔSSIM |"
    sep = "|---|---|---|---|---|---|---|"
    if has_lpips:
        hdr += " LPIPS TTT | LPIPS Attn | ΔLPIPS |"
        sep += "---|---|---|"
    lines.append(hdr)
    lines.append(sep)
    for s in suites:
        a, b = ttt_gen.get(s, {}), attn_gen.get(s, {})
        if not a and not b:
            continue
        row = [SUITE_LABEL.get(s, s)]
        for key, nd in (("psnr_mean", 2), ("ssim_mean", 4)):
            ta, tb = a.get(key), b.get(key)
            d = (ta - tb) if (ta is not None and tb is not None) else None
            row += [fmt(ta, nd), fmt(tb, nd), dfmt(d, nd)]
        if has_lpips:
            ta, tb = a.get("lpips_mean"), b.get("lpips_mean")
            d = (ta - tb) if (ta is not None and tb is not None) else None
            row += [fmt(ta), fmt(tb), dfmt(d)]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## 判读\n")
    lines.append("- **loss_img 在 spatial 上 TTT 更低** → 训练 loss 优势确实落在视频项。")
    lines.append("- **OOD(object/goal) 上 Δ 保持/扩大** → 真泛化更好；**Δ 消失/反转** → 只是 in-dist 拟合更狠。")
    lines.append("- **PSNR/SSIM↑ 且 LPIPS↓** → 自回归生成的视频质量真的更好；"
                 "**指标打平但 loss 低** → teacher-forcing 假象，低 loss 未转化为更好视频。")
    lines.append("- 对比图：`grids/{suite}_{tag}.png`（每行 GT | pred）。\n")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] -> {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
