#!/usr/bin/env python3
"""Compose GT | TTT | attn 3-column temporal comparison video from two per-frame dirs.
Each source frame = GT|VQ-recon|pred (772 wide). Usage:
  compose_temporal_cmp.py <ttt_dir> <attn_dir> <out_mp4> [fps]
"""
import os, sys, glob, subprocess, numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

ttt_dir, attn_dir, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
fps = int(sys.argv[4]) if len(sys.argv) > 4 else 12
P = 256; GT0 = 0; PRED0 = 2*(P+2)
N = min(len(glob.glob(f"{ttt_dir}/frame_*.png")), len(glob.glob(f"{attn_dir}/frame_*.png")))
if N == 0:
    sys.exit(f"[compose] no frames in {ttt_dir} / {attn_dir}")

def label(text, w, h=26, bg=(20,20,20), fg=(235,235,235)):
    im = Image.new("RGB", (w, h), bg); d = ImageDraw.Draw(im)
    try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception: f = ImageFont.load_default()
    tw = d.textlength(text, font=f); d.text(((w-tw)//2,(h-15)//2), text, fill=fg, font=f)
    return np.asarray(im)

titles = [("GT (real future)",(20,20,20)), ("TTT-140k",(15,55,30)), ("attn-120k",(70,20,20))]
strip = [label(t, P, bg=bg) for t, bg in titles]; hh = strip[0].shape[0]
gc = np.full((P,8,3),255,np.uint8); hgc = np.full((hh,8,3),255,np.uint8)
head = np.concatenate([strip[0],hgc,strip[1],hgc,strip[2]], axis=1)
fdir = out_mp4 + ".frames"; os.makedirs(fdir, exist_ok=True)
for i in range(N):
    ta = iio.imread(f"{ttt_dir}/frame_{i:05d}.png"); aa = iio.imread(f"{attn_dir}/frame_{i:05d}.png")
    body = np.concatenate([ta[:,GT0:GT0+P], gc, ta[:,PRED0:PRED0+P], gc, aa[:,PRED0:PRED0+P]], axis=1)
    frame = np.concatenate([head, body], axis=0)
    foot = label(f"RoboLab/DROID open-loop future prediction  -  frame {i+1}/{N}", frame.shape[1], h=22, bg=(0,0,0), fg=(200,200,200))
    iio.imwrite(f"{fdir}/f{i:05d}.png", np.concatenate([frame, foot], axis=0))
subprocess.run(["ffmpeg","-y","-framerate",str(fps),"-i",f"{fdir}/f%05d.png",
                "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",out_mp4],
               check=True, capture_output=True)
subprocess.run(["rm","-rf",fdir])
print(f"[compose] {N} frames -> {out_mp4}")
