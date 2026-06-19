"""
严肃验证: chunk_size=256 下 TTT 的两个 update 各自是否有效。
回答用户的核心担忧 "我的 test-time update 是不是白做了"。

区分两个 update:
  A = VLM 预更新 (快权重被 VLM context 更新, 得到 w_vlm) —— TTT 的 test-time training 主体
  B = image-token causal 更新 (图像 token 用自己 k/v 更新快权重给后续 token)

实验:
  E1: A 是否有效? 扰动 VLM context -> 输出应大幅变化 (delta >> 0)
  E2: B 是否有效? 扰动某 image token p 的 k/v -> 其他位置输出变不变?
      chunk=256: 预期 delta=0 (B 被丢); chunk<256: 预期 delta>0 (B 生效)
  E3: chunk=256 TTT 是否 == "只做 A, 完全跳过 B"?
      用一个 reference: 手动只算 apply(q, w_vlm), 完全不做 image update.
      若逐位相同 -> 证实 chunk=256 下 B 确实无效。
  E4: 对照 chunk=16/64, B 在小 chunk 下确实生效 (输出依赖前缀 image token)。

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<x> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_ttt_update_validity
"""
import torch
from src.models.ttt import causal_block_fast_weight_swish_glu, _vlm_preupdate

dev = "cuda"


def mk(B, L, d, dh, T, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    rk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float32)
    pr = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float32) * 0.02 + 0.001
    return dict(
        w0=rk(B, d, dh), w1=rk(B, dh, d), w2=rk(B, d, dh),
        q=rk(B, L, d), k=rk(B, L, d), v=rk(B, L, d),
        lr0=pr(B, L, 1), lr1=pr(B, L, 1), lr2=pr(B, L, 1),
        vk=rk(B, T, d), vv=rk(B, T, d), vl0=pr(B, T, 1), vl1=pr(B, T, 1), vl2=pr(B, T, 1))


def run(t, cs):
    return causal_block_fast_weight_swish_glu(
        t["w0"].clone(), t["w1"].clone(), t["w2"].clone(),
        t["q"], t["k"], t["v"], t["lr0"], t["lr1"], t["lr2"],
        chunk_size=cs, muon_update_steps=0,
        vlm_k=t["vk"], vlm_v=t["vv"], vlm_lr0=t["vl0"], vlm_lr1=t["vl1"], vlm_lr2=t["vl2"])[0]


def main():
    if not torch.cuda.is_available():
        print("SKIP"); return 0
    B, L, d, dh, T = 2, 256, 64, 64, 16
    t = mk(B, L, d, dh, T, seed=1)

    print("=== E1: VLM 预更新(A) 是否有效? ===")
    o0 = run(t, 256)
    t_v = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in t.items()}
    t_v["vk"] = t_v["vk"] + 3.0; t_v["vv"] = t_v["vv"] + 3.0
    o_v = run(t_v, 256)
    d_vlm = (o_v - o0).abs().max().item()
    print(f"  扰动 VLM context -> 输出 max delta = {d_vlm:.3e}")
    print(f"  => A {'有效 (TTT test-time training 主体在工作)' if d_vlm > 1e-2 else '可疑!'}")

    print("\n=== E2: image-token 更新(B) 在 chunk=256 是否有效? ===")
    for cs in (256, 64, 16):
        t2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in t.items()}
        p = 50
        t2["k"][:, p] += 5.0; t2["v"][:, p] += 5.0   # 扰动 image token p 的 k/v
        oc = run(t, cs); op = run(t2, cs)
        # 看 p 之后的位置 (B 若生效, p 的 update 会影响 > p 的 apply)
        after = (oc[:, p+1:] - op[:, p+1:]).abs().max().item()
        print(f"  chunk={cs:3d}: 扰动 token{p} 的 k/v -> 位置>{p} 的输出 max delta = {after:.3e}"
              f"  ({'B 生效' if after > 1e-4 else 'B 被丢弃(无效)'})")

    print("\n=== E3: chunk=256 TTT == 只做 A 跳过 B? (铁证) ===")
    # reference: 手动只做 VLM 预更新得到 w_vlm, 然后 apply(q, w_vlm), 完全不做 image update
    import torch.nn.functional as F
    from einops import rearrange
    w0, w1, w2 = t["w0"].clone(), t["w1"].clone(), t["w2"].clone()
    w0v, w1v, w2v = _vlm_preupdate(w0, w1, w2, t["vk"], t["vv"], t["vl0"], t["vl1"], t["vl2"])
    o_apply_only = (F.silu(t["q"] @ w0v) * (t["q"] @ w2v)) @ w1v
    o_full = run(t, 256)
    d_e3 = (o_full - o_apply_only).abs().max().item()
    rel = d_e3 / (o_full.abs().max().item() + 1e-9)
    print(f"  full TTT(chunk256) vs 只apply(q,w_vlm)不做image-update:")
    print(f"  abs delta = {d_e3:.3e}, rel = {rel:.3e}")
    print(f"  => {'逐位相同 -> 证实 chunk=256 下 image-update(B) 确实被丢' if rel < 1e-5 else '不同 -> B 在 chunk256 下仍有效!'}")

    print("\n=== 结论 ===")
    print(f"  A(VLM条件化, test-time training主体): {'有效' if d_vlm>1e-2 else '可疑'}")
    print(f"  B(image-token互相更新) 在 chunk=256: 见 E2/E3")
    return 0


if __name__ == "__main__":
    main()
