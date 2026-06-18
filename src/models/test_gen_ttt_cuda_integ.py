import os, torch
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
from src.models.generator import ImageGeneratorTransformer

dev = "cuda"
torch.manual_seed(0)
# real-ish small generator: depth=8, mix_every_n=4 -> TTT at layers 4,8
for use_cuda in (False, True):
    gen = ImageGeneratorTransformer(
        vocab_size=1024, vlm_hidden_size=512, hidden_size=768,
        depth=8, num_heads=12, mixer_type="ttt", mix_every_n=4, ttt_chunk_size=256,
        ttt_use_cuda_kernel=use_cuda).to(dev, torch.bfloat16)
    B, L = 2, 256
    ids = torch.randint(0, 1024, (B, L), device=dev)
    vlm = [torch.randn(B, 16, 512, device=dev, dtype=torch.bfloat16) for _ in range(8)]
    logits, hs = gen(ids, vlm)
    loss = logits.float().mean()
    loss.backward()
    gnorm = sum(p.grad.float().norm().item() for p in gen.parameters() if p.grad is not None)
    ttt_blocks = [b for b in gen.blocks if b.mixer_type == "ttt"]
    flag = ttt_blocks[0].attn.use_cuda_kernel
    fin = torch.isfinite(loss).item() and all(
        torch.isfinite(p.grad).all().item() for p in gen.parameters() if p.grad is not None)
    print(f"use_cuda={use_cuda} | logits{tuple(logits.shape)} loss={loss.item():.4f} "
          f"grad_norm_sum={gnorm:.1f} ttt_blocks={len(ttt_blocks)} "
          f"use_cuda_kernel={flag} all_finite={fin}")
print("INTEG OK")
