// CUDA-accelerated causal TTT (LaCT fast-weight SwiGLU) forward for VLANeXt.
//
// Strategy: the GEMMs (k@w0, k@w2, v@w1^T, q@w0, q@w2, o@w1, and the three
// grad outer-products k^T@..) are batched [B, *, *] matmuls -- we leave those
// to ATen bmm (cuBLAS). The custom CUDA kernels fuse the launch-storm of tiny
// elementwise / reduction ops:
//   - silu_glu:     hidden = silu(gate) * up
//   - silu_bwd_glu: the dgate/dhidden chain (silu_backprop) for the grad path
//   - frob_normalize_add: w <- w + grad/||grad||_F ; then weight-norm rescale
//
// Parity contract (muon_update_steps == 0): zeropower_via_newtonschulz5 with
// steps=0 reduces to grad / (||grad||_F + 1e-7) per [d,d] head matrix. The
// weight-norm step then rescales each column back to the detached init norm.

#include <torch/extension.h>
#include <vector>

// ---- declarations of kernels defined in ttt_fused.cu ----
namespace ttt_cuda {

// hidden = silu(gate) * up        (all [N, D] contiguous, bf16/fp16/fp32)
torch::Tensor silu_glu(const torch::Tensor& gate, const torch::Tensor& up);

// Given dhidden, gate, up: returns (dgate_before_act, dhidden_before_mul)
// dhidden_before_mul = dhidden * silu(gate)
// dgate_before_act   = silu_backprop(dhidden * up, gate)
std::vector<torch::Tensor> silu_bwd_glu(
    const torch::Tensor& dhidden,
    const torch::Tensor& gate,
    const torch::Tensor& up);

// w_new = (w + grad/(||grad||_F + 1e-7)); then column-normalize to w_init_norm.
// grad is normalized per batch-matrix over dims (1,2). norm_dim selects the
// weight-norm reduction dim (1 for these [B,d,dh] layouts, matching torch).
torch::Tensor frob_norm_update(
    const torch::Tensor& w,
    const torch::Tensor& grad,
    const torch::Tensor& w_init_norm,
    int64_t norm_dim);

// backward primitives
std::vector<torch::Tensor> silu_derivs(const torch::Tensor& x);   // returns {silu', silu''}
torch::Tensor frobnorm_bwd(const torch::Tensor& gy, const torch::Tensor& x, double eps);
torch::Tensor weightnorm_bwd(const torch::Tensor& gy, const torch::Tensor& w_pre,
                             const torch::Tensor& wn_target, double eps);

}  // namespace ttt_cuda

// ---- helpers (host-side, ATen) ----
static inline torch::Tensor frob_normalize(const torch::Tensor& g) {
    // grad / (||grad||_F + 1e-7), Frobenius over last two dims, per batch.
    auto nrm = g.flatten(1).norm(2, /*dim=*/1, /*keepdim=*/true).unsqueeze(-1);
    return g / (nrm + 1e-7);
}

// One fast-weight update step (apply-then-update uses this for the update half).
// Mutates w0,w1,w2 in place (returns new tensors). steps==0 parity path.
static void fw_update(
    torch::Tensor& w0, torch::Tensor& w1, torch::Tensor& w2,
    const torch::Tensor& ki, const torch::Tensor& vi,
    const torch::Tensor& lr0i, const torch::Tensor& lr1i, const torch::Tensor& lr2i,
    const torch::Tensor& w0_norm, const torch::Tensor& w1_norm, const torch::Tensor& w2_norm) {

    auto gate = ki.bmm(w0);              // [B, l, dh]
    auto up   = ki.bmm(w2);              // [B, l, dh]
    auto hidden = ttt_cuda::silu_glu(gate, up);

    auto dhidden = vi.bmm(w1.transpose(-1, -2));   // [B, l, dh]
    auto chain = ttt_cuda::silu_bwd_glu(dhidden, gate, up);
    auto dgate_before_act   = chain[0];            // [B, l, dh]
    auto dhidden_before_mul = chain[1];            // [B, l, dh]

    // grads (Frobenius-normalized, steps==0)
    auto w1_grad = frob_normalize(
        (hidden * lr1i).to(vi.dtype()).transpose(-1, -2).bmm(vi));      // [B, dh, d]
    auto w0_grad = frob_normalize(
        (ki * lr0i).to(dgate_before_act.dtype()).transpose(-1, -2).bmm(dgate_before_act)); // [B, d, dh]
    auto w2_grad = frob_normalize(
        (ki * lr2i).to(dhidden_before_mul.dtype()).transpose(-1, -2).bmm(dhidden_before_mul)); // [B, d, dh]

    w1 = ttt_cuda::frob_norm_update(w1, w1_grad, w1_norm, /*norm_dim=*/1);
    w0 = ttt_cuda::frob_norm_update(w0, w0_grad, w0_norm, /*norm_dim=*/1);
    w2 = ttt_cuda::frob_norm_update(w2, w2_grad, w2_norm, /*norm_dim=*/1);
}

// output_i = (silu(qi@w0) * (qi@w2)) @ w1
static torch::Tensor fw_apply(
    const torch::Tensor& qi, const torch::Tensor& w0,
    const torch::Tensor& w1, const torch::Tensor& w2) {
    auto gate = qi.bmm(w0);
    auto up   = qi.bmm(w2);
    auto h = ttt_cuda::silu_glu(gate, up);
    return h.bmm(w1);
}

std::vector<torch::Tensor> causal_ttt_forward(
    torch::Tensor w0, torch::Tensor w1, torch::Tensor w2,
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor lr0, torch::Tensor lr1, torch::Tensor lr2,
    int64_t chunk_size,
    c10::optional<torch::Tensor> vlm_k,
    c10::optional<torch::Tensor> vlm_v,
    c10::optional<torch::Tensor> vlm_lr0,
    c10::optional<torch::Tensor> vlm_lr1,
    c10::optional<torch::Tensor> vlm_lr2) {

    TORCH_CHECK(q.is_cuda(), "causal_ttt_forward: inputs must be CUDA tensors");

    // detached init column norms (weight-norm targets), dim=1 like torch ref
    auto w0_norm = w0.detach().norm(2, /*dim=*/1, /*keepdim=*/true);
    auto w1_norm = w1.detach().norm(2, /*dim=*/1, /*keepdim=*/true);
    auto w2_norm = w2.detach().norm(2, /*dim=*/1, /*keepdim=*/true);

    // ---- global (non-causal) VLM pre-update: makes VLM fully visible ----
    if (vlm_k.has_value()) {
        fw_update(w0, w1, w2,
                  vlm_k.value(), vlm_v.value(),
                  vlm_lr0.value(), vlm_lr1.value(), vlm_lr2.value(),
                  w0_norm, w1_norm, w2_norm);
    }

    const int64_t L = q.size(1);
    std::vector<torch::Tensor> outs;
    for (int64_t s = 0; s < L; s += chunk_size) {
        int64_t e = std::min(s + chunk_size, L);
        using torch::indexing::Slice;

        // apply current fast weights to this chunk's query (apply-then-update)
        auto qi = q.index({Slice(), Slice(s, e), Slice()});
        outs.push_back(fw_apply(qi, w0, w1, w2));

        // then update with this chunk's (k, v)
        auto ki = k.index({Slice(), Slice(s, e), Slice()});
        auto vi = v.index({Slice(), Slice(s, e), Slice()});
        auto l0 = lr0.index({Slice(), Slice(s, e), Slice()});
        auto l1 = lr1.index({Slice(), Slice(s, e), Slice()});
        auto l2 = lr2.index({Slice(), Slice(s, e), Slice()});
        fw_update(w0, w1, w2, ki, vi, l0, l1, l2, w0_norm, w1_norm, w2_norm);
    }

    auto output = torch::cat(outs, /*dim=*/1);
    return {output, w0, w1, w2};
}

// ===================== BACKWARD orchestration (Plan A) =====================
//
// Mirrors the proven torch manual backward (ttt_manual_backward.py):
// checkpoint-style BPTT. Forward pass recomputes + saves the ENTRY weights of
// each chunk (and the pre-update entry); reverse pass recomputes each chunk's
// intermediates from its entry weights and applies the vjp chain. GEMMs via
// ATen bmm; silu derivatives + the two normalize-vjps via custom kernels.

namespace {
using torch::indexing::Slice;
}  // namespace

// silu(x) helper via existing kernel (silu_glu(x, ones)=silu(x))
static inline torch::Tensor silu_only(const torch::Tensor& x) {
    return ttt_cuda::silu_glu(x, torch::ones_like(x));
}

std::vector<torch::Tensor> causal_ttt_backward(
    torch::Tensor w0, torch::Tensor w1, torch::Tensor w2,
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor lr0, torch::Tensor lr1, torch::Tensor lr2,
    int64_t chunk_size,
    torch::Tensor g_out, torch::Tensor g_w0n, torch::Tensor g_w1n, torch::Tensor g_w2n,
    c10::optional<torch::Tensor> vlm_k,
    c10::optional<torch::Tensor> vlm_v,
    c10::optional<torch::Tensor> vlm_lr0,
    c10::optional<torch::Tensor> vlm_lr1,
    c10::optional<torch::Tensor> vlm_lr2) {

    TORCH_CHECK(q.is_cuda(), "causal_ttt_backward: inputs must be CUDA");
    const double FEPS = 1e-7, WEPS = 1e-5;
    auto w0n_t = w0.norm(2, 1, true), w1n_t = w1.norm(2, 1, true), w2n_t = w2.norm(2, 1, true);
    bool has_vlm = vlm_k.has_value();

    const int64_t L = q.size(1);
    std::vector<int64_t> starts;
    for (int64_t s = 0; s < L; s += chunk_size) starts.push_back(s);

    // ---- forward pass: save ENTRY weights of each chunk (and post-vlm entry) ----
    std::vector<std::array<torch::Tensor, 3>> entry;  // weights entering each chunk
    auto cw0 = w0, cw1 = w1, cw2 = w2;
    std::array<torch::Tensor, 3> pre_entry = {w0, w1, w2};  // weights entering vlm pre-update
    if (has_vlm) {
        fw_update(cw0, cw1, cw2, vlm_k.value(), vlm_v.value(),
                  vlm_lr0.value(), vlm_lr1.value(), vlm_lr2.value(), w0n_t, w1n_t, w2n_t);
    }
    for (size_t ci = 0; ci < starts.size(); ++ci) {
        int64_t s = starts[ci], e = std::min(s + chunk_size, L);
        entry.push_back({cw0, cw1, cw2});
        auto ki = k.index({Slice(), Slice(s, e), Slice()});
        auto vi = v.index({Slice(), Slice(s, e), Slice()});
        auto l0 = lr0.index({Slice(), Slice(s, e), Slice()});
        auto l1 = lr1.index({Slice(), Slice(s, e), Slice()});
        auto l2 = lr2.index({Slice(), Slice(s, e), Slice()});
        fw_update(cw0, cw1, cw2, ki, vi, l0, l1, l2, w0n_t, w1n_t, w2n_t);
    }

    // ---- accumulators ----
    auto g_q = torch::zeros_like(q), g_k = torch::zeros_like(k), g_v = torch::zeros_like(v);
    auto g_lr0 = torch::zeros_like(lr0), g_lr1 = torch::zeros_like(lr1), g_lr2 = torch::zeros_like(lr2);
    auto gw0 = g_w0n.clone(), gw1 = g_w1n.clone(), gw2 = g_w2n.clone();

    // chunk-level vjp closure (also used for vlm pre-update with do_apply=false)
    auto chunk_vjp = [&](const torch::Tensor& W0, const torch::Tensor& W1, const torch::Tensor& W2,
                         const torch::Tensor& qi, const torch::Tensor& ki, const torch::Tensor& vi,
                         const torch::Tensor& l0, const torch::Tensor& l1, const torch::Tensor& l2,
                         const torch::Tensor& g_oi, bool do_apply,
                         torch::Tensor& out_gw0, torch::Tensor& out_gw1, torch::Tensor& out_gw2,
                         torch::Tensor& out_gq, torch::Tensor& out_gk, torch::Tensor& out_gv,
                         torch::Tensor& out_gl0, torch::Tensor& out_gl1, torch::Tensor& out_gl2) {
        // recompute forward intermediates
        auto gate = ki.bmm(W0), up = ki.bmm(W2);
        auto sd_g = ttt_cuda::silu_derivs(gate);  // silu'(gate), silu''(gate)
        auto sg = silu_only(gate);
        auto hidden = sg * up;
        auto dhidden = vi.bmm(W1.transpose(-1, -2));
        auto dhid_bm = dhidden * sg;
        auto dgate = dhidden * up;
        auto mm = sd_g[0];
        auto dgba = dgate * mm;
        // lr is fp32; A* promote to fp32 -> cast back to operand dtype before bmm
        // (matches the reference forward's `.to(vi.dtype())`).
        auto A0 = (ki * l0).to(dgba.dtype());
        auto A1 = (hidden * l1).to(vi.dtype());
        auto A2 = (ki * l2).to(dhid_bm.dtype());
        auto fn1_in = A1.transpose(-1, -2).bmm(vi);
        auto fn0_in = A0.transpose(-1, -2).bmm(dgba);
        auto fn2_in = A2.transpose(-1, -2).bmm(dhid_bm);
        auto rn1 = frob_normalize(fn1_in), rn0 = frob_normalize(fn0_in), rn2 = frob_normalize(fn2_in);
        auto w0_pre = W0 + rn0, w1_pre = W1 + rn1, w2_pre = W2 + rn2;

        // ---- vjp: weightnorm ----
        auto g_w0_pre = ttt_cuda::weightnorm_bwd(out_gw0, w0_pre, w0n_t, WEPS);
        auto g_w1_pre = ttt_cuda::weightnorm_bwd(out_gw1, w1_pre, w1n_t, WEPS);
        auto g_w2_pre = ttt_cuda::weightnorm_bwd(out_gw2, w2_pre, w2n_t, WEPS);
        // w_pre = w_old + raw
        auto g_w0_old = g_w0_pre.clone(), g_w1_old = g_w1_pre.clone(), g_w2_old = g_w2_pre.clone();
        auto g_rn0 = g_w0_pre, g_rn1 = g_w1_pre, g_rn2 = g_w2_pre;
        // frobnorm vjp
        auto g_fn0 = ttt_cuda::frobnorm_bwd(g_rn0, fn0_in, FEPS);
        auto g_fn1 = ttt_cuda::frobnorm_bwd(g_rn1, fn1_in, FEPS);
        auto g_fn2 = ttt_cuda::frobnorm_bwd(g_rn2, fn2_in, FEPS);

        // raw_w1 = A1^T @ vi
        auto g_A1 = vi.bmm(g_fn1.transpose(-1, -2));
        auto g_vi = A1.bmm(g_fn1);
        // lr is fp32; cast lr-scaled grads back to activation dtype for later bmm
        auto g_hidden = (g_A1 * l1).to(up.dtype());
        out_gl1.index({Slice(), Slice(), Slice()}) += (g_A1 * hidden).sum(-1, true);
        // raw_w0 = A0^T @ dgba
        auto g_A0 = dgba.bmm(g_fn0.transpose(-1, -2));
        auto g_dgba = A0.bmm(g_fn0);
        auto g_ki = (g_A0 * l0).to(ki.dtype());
        out_gl0.index({Slice(), Slice(), Slice()}) += (g_A0 * ki).sum(-1, true);
        // raw_w2 = A2^T @ dhid_bm
        auto g_A2 = dhid_bm.bmm(g_fn2.transpose(-1, -2));
        auto g_dhid_bm = A2.bmm(g_fn2);
        g_ki = g_ki + (g_A2 * l2).to(ki.dtype());
        out_gl2.index({Slice(), Slice(), Slice()}) += (g_A2 * ki).sum(-1, true);

        // dgba = dgate * m ; m=silu'(gate)
        auto g_dgate = g_dgba * mm;
        auto g_gate = g_dgba * dgate * sd_g[1];  // * silu''(gate)
        // dgate = dhidden*up
        auto g_dhidden = g_dgate * up;
        auto g_up = g_dgate * dhidden;
        // dhid_bm = dhidden*sg
        g_dhidden = g_dhidden + g_dhid_bm * sg;
        auto g_sg = g_dhid_bm * dhidden;
        // dhidden = vi @ w1^T
        g_vi = g_vi + g_dhidden.bmm(W1);
        g_w1_old = g_w1_old + g_dhidden.transpose(-1, -2).bmm(vi);
        // hidden = sg*up
        g_sg = g_sg + g_hidden * up;
        g_up = g_up + g_hidden * sg;
        // up = ki@w2
        g_ki = g_ki + g_up.bmm(W2.transpose(-1, -2));
        g_w2_old = g_w2_old + ki.transpose(-1, -2).bmm(g_up);
        // sg = silu(gate)
        g_gate = g_gate + g_sg * sd_g[0];
        // gate = ki@w0
        g_ki = g_ki + g_gate.bmm(W0.transpose(-1, -2));
        g_w0_old = g_w0_old + ki.transpose(-1, -2).bmm(g_gate);

        // accumulate k/v grads (cast to slice dtype: g_ki/g_vi may be fp32 due to lr)
        out_gk.index({Slice(), Slice(), Slice()}) += g_ki.to(out_gk.dtype());
        out_gv.index({Slice(), Slice(), Slice()}) += g_vi.to(out_gv.dtype());

        // ---- apply path ----
        if (do_apply) {
            auto gate_q = qi.bmm(W0), up_q = qi.bmm(W2);
            auto sq = silu_only(gate_q);
            auto h_q = sq * up_q;
            auto g_h_q = g_oi.bmm(W1.transpose(-1, -2));
            g_w1_old = g_w1_old + h_q.transpose(-1, -2).bmm(g_oi);
            auto g_sq = g_h_q * up_q;
            auto g_up_q = g_h_q * sq;
            auto g_qi = g_up_q.bmm(W2.transpose(-1, -2));
            g_w2_old = g_w2_old + qi.transpose(-1, -2).bmm(g_up_q);
            auto sd_q = ttt_cuda::silu_derivs(gate_q);
            auto g_gate_q = g_sq * sd_q[0];
            g_qi = g_qi + g_gate_q.bmm(W0.transpose(-1, -2));
            g_w0_old = g_w0_old + qi.transpose(-1, -2).bmm(g_gate_q);
            out_gq.index({Slice(), Slice(), Slice()}) += g_qi.to(out_gq.dtype());
        }

        out_gw0 = g_w0_old; out_gw1 = g_w1_old; out_gw2 = g_w2_old;
    };

    // ---- reverse over chunks ----
    for (int64_t idx = (int64_t)starts.size() - 1; idx >= 0; --idx) {
        int64_t s = starts[idx], e = std::min(s + chunk_size, L);
        auto W0 = entry[idx][0], W1 = entry[idx][1], W2 = entry[idx][2];
        auto qi = q.index({Slice(), Slice(s, e), Slice()});
        auto ki = k.index({Slice(), Slice(s, e), Slice()});
        auto vi = v.index({Slice(), Slice(s, e), Slice()});
        auto l0 = lr0.index({Slice(), Slice(s, e), Slice()});
        auto l1 = lr1.index({Slice(), Slice(s, e), Slice()});
        auto l2 = lr2.index({Slice(), Slice(s, e), Slice()});
        auto g_oi = g_out.index({Slice(), Slice(s, e), Slice()});
        auto gk_slice = g_k.index({Slice(), Slice(s, e), Slice()});
        auto gv_slice = g_v.index({Slice(), Slice(s, e), Slice()});
        auto gq_slice = g_q.index({Slice(), Slice(s, e), Slice()});
        auto gl0_slice = g_lr0.index({Slice(), Slice(s, e), Slice()});
        auto gl1_slice = g_lr1.index({Slice(), Slice(s, e), Slice()});
        auto gl2_slice = g_lr2.index({Slice(), Slice(s, e), Slice()});
        chunk_vjp(W0, W1, W2, qi, ki, vi, l0, l1, l2, g_oi, /*do_apply=*/true,
                  gw0, gw1, gw2, gq_slice, gk_slice, gv_slice, gl0_slice, gl1_slice, gl2_slice);
    }

    std::vector<torch::Tensor> result = {gw0, gw1, gw2, g_q, g_k, g_v, g_lr0, g_lr1, g_lr2};

    // ---- vlm pre-update (update-only) ----
    if (has_vlm) {
        auto dummy_q = torch::Tensor();
        auto g_vk = torch::zeros_like(vlm_k.value());
        auto g_vv = torch::zeros_like(vlm_v.value());
        auto g_vl0 = torch::zeros_like(vlm_lr0.value());
        auto g_vl1 = torch::zeros_like(vlm_lr1.value());
        auto g_vl2 = torch::zeros_like(vlm_lr2.value());
        auto g_oi_dummy = torch::Tensor();
        chunk_vjp(pre_entry[0], pre_entry[1], pre_entry[2], dummy_q,
                  vlm_k.value(), vlm_v.value(), vlm_lr0.value(), vlm_lr1.value(), vlm_lr2.value(),
                  g_oi_dummy, /*do_apply=*/false,
                  gw0, gw1, gw2, /*gq*/g_vk, g_vk, g_vv, g_vl0, g_vl1, g_vl2);
        // after pre-update vjp, gw0..gw2 are grads wrt the ORIGINAL w0/w1/w2
        result[0] = gw0; result[1] = gw1; result[2] = gw2;
        result.push_back(g_vk); result.push_back(g_vv);
        result.push_back(g_vl0); result.push_back(g_vl1); result.push_back(g_vl2);
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_ttt_forward", &causal_ttt_forward,
          "Causal block fast-weight SwiGLU TTT forward (CUDA)");
    m.def("causal_ttt_backward", &causal_ttt_backward,
          "Causal block fast-weight SwiGLU TTT backward (CUDA, Plan A)");
    // expose backward primitives for unit testing
    m.def("silu_derivs", &ttt_cuda::silu_derivs, "silu' and silu''");
    m.def("frobnorm_bwd", &ttt_cuda::frobnorm_bwd, "Frobenius-normalize vjp");
    m.def("weightnorm_bwd", &ttt_cuda::weightnorm_bwd, "weight-norm (per-col) vjp");
}
