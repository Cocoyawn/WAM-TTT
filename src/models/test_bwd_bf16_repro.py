import os, torch
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
from src.models import ttt_cuda

dev = "cuda"
ext = ttt_cuda._load_extension()
B, L, d, dh, T = 4, 128, 16, 16, 8
g = torch.Generator(device=dev).manual_seed(1)
bf = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16)
f32 = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float32) * 0.02 + 0.001
w0 = bf(B, d, dh); w1 = bf(B, dh, d); w2 = bf(B, d, dh)
q = bf(B, L, d); k = bf(B, L, d); v = bf(B, L, d)
lr0 = f32(B, L, 1); lr1 = f32(B, L, 1); lr2 = f32(B, L, 1)
g_out = bf(B, L, d); g_w0 = bf(B, d, dh); g_w1 = bf(B, dh, d); g_w2 = bf(B, d, dh)

# with VLM ctx (the real path)
vk = bf(B, T, d); vv = bf(B, T, d)
vl0 = f32(B, T, 1); vl1 = f32(B, T, 1); vl2 = f32(B, T, 1)

for cs in (256, 64):
    try:
        res = ext.causal_ttt_backward(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, cs,
            g_out, g_w0, g_w1, g_w2, vk, vv, vl0, vl1, vl2)
        print(f"cs={cs} OK -> {len(res)} grads, dtypes:", [str(r.dtype).split('.')[-1] for r in res[:6]])
    except RuntimeError as e:
        print(f"cs={cs} CRASH: {e}")

# numerical check vs torch reference (autograd), bf16 -> relative error
from src.models.ttt import causal_block_fast_weight_swish_glu as ref_op
print("\n-- bf16 numerical parity vs torch reference --")
for cs in (256, 64):
    leaves = [t.detach().clone().requires_grad_(True) for t in
              (w0, w1, w2, q, k, v, vk, vv)]
    lw0, lw1, lw2, lq, lk, lv, lvk, lvv = leaves
    out = ref_op(lw0, lw1, lw2, lq, lk, lv, lr0, lr1, lr2, chunk_size=cs,
                 muon_update_steps=0, vlm_k=lvk, vlm_v=lvv,
                 vlm_lr0=vl0, vlm_lr1=vl1, vlm_lr2=vl2)
    torch.autograd.backward([out[0], out[1], out[2], out[3]],
                            [g_out, g_w0, g_w1, g_w2])
    res = ext.causal_ttt_backward(w0, w1, w2, q, k, v, lr0, lr1, lr2, cs,
                                  g_out, g_w0, g_w1, g_w2, vk, vv, vl0, vl1, vl2)
    cu = dict(zip(("w0","w1","w2","q","k","v"), res[:6]))
    ref = dict(zip(("w0","w1","w2","q","k","v"), [t.grad for t in leaves[:6]]))
    mx = 0.0; worst = ""
    for kk in ("w0","w1","w2","q","k","v"):
        e = (ref[kk].float()-cu[kk].float()).abs().max().item()
        mag = ref[kk].float().abs().max().item()+1e-9
        if e/mag > mx: mx, worst = e/mag, kk
    print(f"cs={cs}: max rel grad err={mx:.2e} (worst={worst}) {'OK' if mx<5e-2 else 'BAD'}")
print("DONE")

