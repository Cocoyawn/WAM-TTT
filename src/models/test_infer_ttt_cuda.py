import os, time, torch
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
from src.models.generator import ImageGeneratorTransformer

dev = "cuda"
torch.manual_seed(0)
B, L = 1, 256
# inference path: forward-only under no_grad, exactly what predict_action does.
gen = ImageGeneratorTransformer(
    vocab_size=1024, vlm_hidden_size=512, hidden_size=768,
    depth=29, num_heads=12, mixer_type="ttt", mix_every_n=4, ttt_chunk_size=256,
    ttt_use_cuda_kernel=False).to(dev, torch.bfloat16).eval()
ids = torch.randint(0, 1024, (B, L), device=dev)
vlm = [torch.randn(B, 16, 512, device=dev, dtype=torch.bfloat16) for _ in range(29)]


def toggle(flag):
    n = 0
    for b in gen.blocks:
        if getattr(b, "mixer_type", None) == "ttt":
            b.attn.use_cuda_kernel = flag and b.attn.muon_update_steps == 0
            n += 1
    return n


@torch.no_grad()
def run():
    return gen(ids, vlm)[0]


for use_cuda in (False, True):
    nl = toggle(use_cuda)
    out = run()  # functional: no crash, finite
    fin = torch.isfinite(out).all().item()
    # quick wall-clock of the generator forward only (NOTE: shared card -> polluted)
    for _ in range(2):
        run()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    ms = (time.time() - t) / 5 * 1000
    print(f"use_cuda={use_cuda} | ttt_layers={nl} out{tuple(out.shape)} finite={fin} "
          f"| gen-fwd {ms:7.1f} ms (POLLUTED shared card, not a paper number)")

# parity of the two inference outputs (must match within bf16)
toggle(False); o0 = run().float()
toggle(True);  o1 = run().float()
rel = (o0 - o1).abs().max().item() / (o0.abs().max().item() + 1e-9)
print(f"infer-path CUDA-vs-torch out rel-err = {rel:.2e} (want < 3e-2)")
print("INFER-PATH OK" if (fin and rel < 3e-2) else "INFER-PATH FAIL")
