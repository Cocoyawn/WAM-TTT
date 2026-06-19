// Fused elementwise/reduction CUDA kernels for the TTT fast-weight path.
//
// These replace the per-op kernel-launch storm in the eager torch reference:
//   silu_glu       : hidden = silu(gate) * up
//   silu_bwd_glu   : dgate_before_act = silu_backprop(dhidden*up, gate)
//                    dhidden_before_mul = dhidden * silu(gate)
//   frob_norm_update: w_new = w + grad/(||grad||_F+1e-7); then column weight-norm
//
// All compute is done in fp32 internally for parity with the torch ops (which
// upcast in silu/normalize), then cast back to the tensor dtype.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace ttt_cuda {

template <typename scalar_t>
__device__ __forceinline__ float to_f(scalar_t x) { return static_cast<float>(x); }

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}
__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// hidden = silu(gate) * up
template <typename scalar_t>
__global__ void silu_glu_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = to_f(gate[i]);
    float u = to_f(up[i]);
    out[i] = static_cast<scalar_t>(silu_f(g) * u);
}

// dgate_before_act = silu_backprop(dhidden*up, gate)
//                  = (dhidden*up) * sigma * (1 + gate*(1-sigma)), sigma=sigmoid(gate)
// dhidden_before_mul = dhidden * silu(gate)
template <typename scalar_t>
__global__ void silu_bwd_glu_kernel(
    const scalar_t* __restrict__ dhidden,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ dgate_before_act,
    scalar_t* __restrict__ dhidden_before_mul,
    int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dh = to_f(dhidden[i]);
    float g  = to_f(gate[i]);
    float u  = to_f(up[i]);
    float sig = sigmoid_f(g);
    float silu = g * sig;
    float dgate = dh * u;  // gradient wrt the silu output
    float dx = dgate * sig * (1.0f + g * (1.0f - sig));
    dgate_before_act[i]   = static_cast<scalar_t>(dx);
    dhidden_before_mul[i] = static_cast<scalar_t>(dh * silu);
}

torch::Tensor silu_glu(const torch::Tensor& gate, const torch::Tensor& up) {
    auto g = gate.contiguous();
    auto u = up.contiguous();
    auto out = torch::empty_like(g);
    int64_t n = g.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    const c10::cuda::CUDAGuard guard(g.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, g.scalar_type(), "silu_glu", [&] {
            silu_glu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), n);
        });
    return out;
}

std::vector<torch::Tensor> silu_bwd_glu(
    const torch::Tensor& dhidden, const torch::Tensor& gate, const torch::Tensor& up) {
    auto dh = dhidden.contiguous();
    auto g = gate.contiguous();
    auto u = up.contiguous();
    auto dgate = torch::empty_like(g);
    auto dhid  = torch::empty_like(g);
    int64_t n = g.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    const c10::cuda::CUDAGuard guard(g.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, g.scalar_type(), "silu_bwd_glu", [&] {
            silu_bwd_glu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                dh.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
                dgate.data_ptr<scalar_t>(), dhid.data_ptr<scalar_t>(), n);
        });
    return {dgate, dhid};
}

// w_new = w + grad; then per-column weight-norm rescale:
//   col_norm = ||w_new[:, j]||_2 over norm_dim;  w_new = w_new / (col_norm+1e-5) * w_init_norm
// NOTE: `grad` is ALREADY Frobenius-normalized host-side (frob_normalize, the
// NS-steps==0 parity path), so the kernel must NOT normalize it again.
// Layout: w is [B, A, C]. norm_dim==1 reduces over A (matching torch dim=1).
// One block per (batch b, column c); block reduces over the A axis.
template <typename scalar_t>
__global__ void frob_norm_update_kernel(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ w_init_norm,  // [B, 1, C]
    scalar_t* __restrict__ out,
    int64_t B, int64_t A, int64_t C) {
    // grid.x = B * C ; each block handles one (b, c) column, reduces over A.
    int64_t bc = blockIdx.x;
    int64_t b = bc / C;
    int64_t c = bc % C;
    if (b >= B) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // first pass: w_upd[a] = w + grad; accumulate sum of squares over A
    float partial = 0.0f;
    for (int64_t a = tid; a < A; a += blockDim.x) {
        int64_t idx = (b * A + a) * C + c;
        float wv = to_f(w[idx]) + to_f(grad[idx]);
        partial += wv * wv;
    }
    sdata[tid] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float col_norm = sqrtf(sdata[0]) + 1e-5f;
    float target = to_f(w_init_norm[b * C + c]);  // [B,1,C] -> b*C + c
    float scale = target / col_norm;

    // second pass: write normalized + rescaled column
    for (int64_t a = tid; a < A; a += blockDim.x) {
        int64_t idx = (b * A + a) * C + c;
        float wv = to_f(w[idx]) + to_f(grad[idx]);
        out[idx] = static_cast<scalar_t>(wv * scale);
    }
}

torch::Tensor frob_norm_update(
    const torch::Tensor& w, const torch::Tensor& grad,
    const torch::Tensor& w_init_norm, int64_t norm_dim) {
    TORCH_CHECK(norm_dim == 1, "frob_norm_update: only norm_dim==1 supported");
    auto wc = w.contiguous();
    auto gc = grad.contiguous();
    auto wn = w_init_norm.contiguous();
    int64_t B = wc.size(0), A = wc.size(1), C = wc.size(2);

    auto out = torch::empty_like(wc);
    const int threads = 256;
    const int64_t blocks = B * C;
    const size_t shmem = threads * sizeof(float);
    const c10::cuda::CUDAGuard guard(wc.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, wc.scalar_type(), "frob_norm_update", [&] {
            frob_norm_update_kernel<scalar_t><<<blocks, threads, shmem, stream>>>(
                wc.data_ptr<scalar_t>(), gc.data_ptr<scalar_t>(),
                wn.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), B, A, C);
        });
    return out;
}

// ===================== BACKWARD kernels (Plan A) =====================

// silu'(x) = s*(1 + x*(1-s)),  s=sigmoid(x)
__device__ __forceinline__ float silu_p_f(float x) {
    float s = sigmoid_f(x);
    return s * (1.0f + x * (1.0f - s));
}
// silu''(x) = s(1-s)*(2 + x*(1-2s))
__device__ __forceinline__ float silu_pp_f(float x) {
    float s = sigmoid_f(x);
    float s1 = s * (1.0f - s);
    return s1 * (2.0f + x * (1.0f - 2.0f * s));
}

// elementwise: out_p = silu'(x), out_pp = silu''(x)
template <typename scalar_t>
__global__ void silu_derivs_kernel(
    const scalar_t* __restrict__ x, scalar_t* __restrict__ out_p,
    scalar_t* __restrict__ out_pp, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xv = to_f(x[i]);
    out_p[i] = static_cast<scalar_t>(silu_p_f(xv));
    out_pp[i] = static_cast<scalar_t>(silu_pp_f(xv));
}

// Frobenius-normalize vjp: gx = r*(gy - (r/n)*<gy,x>*x), r=1/(n+eps),
// n=||x||_F per batch matrix [b,A,C]. One block per batch b, reduce over A*C.
template <typename scalar_t>
__global__ void frobnorm_bwd_kernel(
    const scalar_t* __restrict__ gy, const scalar_t* __restrict__ x,
    scalar_t* __restrict__ gx, int64_t B, int64_t AC, float eps) {
    int64_t b = blockIdx.x;
    if (b >= B) return;
    const scalar_t* gyb = gy + b * AC;
    const scalar_t* xb = x + b * AC;
    scalar_t* gxb = gx + b * AC;

    extern __shared__ float sh[];  // [blockDim] for two reductions sequentially
    int tid = threadIdx.x;

    // pass 1: n^2 = <x,x> and dot = <gy,x>
    float p_xx = 0.0f, p_gx = 0.0f;
    for (int64_t i = tid; i < AC; i += blockDim.x) {
        float xv = to_f(xb[i]); float gv = to_f(gyb[i]);
        p_xx += xv * xv; p_gx += gv * xv;
    }
    // reduce p_xx
    sh[tid] = p_xx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float xx = sh[0]; __syncthreads();
    sh[tid] = p_gx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float dot = sh[0]; __syncthreads();

    float n = sqrtf(xx);
    float r = 1.0f / (n + eps);
    float coef = (n > 0.0f) ? (r / n) * dot : 0.0f;  // (r/n)*dot
    for (int64_t i = tid; i < AC; i += blockDim.x) {
        float xv = to_f(xb[i]); float gv = to_f(gyb[i]);
        gxb[i] = static_cast<scalar_t>(r * (gv - coef * xv));
    }
}

// weight-norm vjp (per-column over dim=1):
//   gx = wn*( rc*gy - (rc^2/nc)*<gy,w_pre>_col * w_pre ),  rc=1/(nc+eps)
// w_pre,gy: [B,A,C]; nc,wn: per (b,c) reduced over A. One block per (b,c).
template <typename scalar_t>
__global__ void weightnorm_bwd_kernel(
    const scalar_t* __restrict__ gy, const scalar_t* __restrict__ w_pre,
    const scalar_t* __restrict__ wn_target,  // [B,1,C]
    scalar_t* __restrict__ gx, int64_t B, int64_t A, int64_t C, float eps) {
    int64_t bc = blockIdx.x;
    int64_t b = bc / C, c = bc % C;
    if (b >= B) return;
    extern __shared__ float sh[];
    int tid = threadIdx.x;

    // pass 1: nc^2 = sum_a w_pre^2 ; dot = sum_a gy*w_pre  (over column c)
    float p_ww = 0.0f, p_gw = 0.0f;
    for (int64_t a = tid; a < A; a += blockDim.x) {
        int64_t idx = (b * A + a) * C + c;
        float wv = to_f(w_pre[idx]); float gv = to_f(gy[idx]);
        p_ww += wv * wv; p_gw += gv * wv;
    }
    sh[tid] = p_ww; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float ww = sh[0]; __syncthreads();
    sh[tid] = p_gw; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float dot = sh[0]; __syncthreads();

    float nc = sqrtf(ww);
    float rc = 1.0f / (nc + eps);
    float wn = to_f(wn_target[b * C + c]);
    float coef = (nc > 0.0f) ? wn * (rc * rc / nc) * dot : 0.0f;
    float rcwn = wn * rc;
    for (int64_t a = tid; a < A; a += blockDim.x) {
        int64_t idx = (b * A + a) * C + c;
        float wv = to_f(w_pre[idx]); float gv = to_f(gy[idx]);
        gx[idx] = static_cast<scalar_t>(rcwn * gv - coef * wv);
    }
}

std::vector<torch::Tensor> silu_derivs(const torch::Tensor& x) {
    auto xc = x.contiguous();
    auto op = torch::empty_like(xc);
    auto opp = torch::empty_like(xc);
    int64_t n = xc.numel();
    const int threads = 256; const int64_t blocks = (n + threads - 1) / threads;
    const c10::cuda::CUDAGuard guard(xc.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, xc.scalar_type(), "silu_derivs", [&] {
            silu_derivs_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                xc.data_ptr<scalar_t>(), op.data_ptr<scalar_t>(), opp.data_ptr<scalar_t>(), n);
        });
    return {op, opp};
}

torch::Tensor frobnorm_bwd(const torch::Tensor& gy, const torch::Tensor& x, double eps) {
    auto gyc = gy.contiguous(); auto xc = x.contiguous();
    int64_t B = xc.size(0), AC = xc.numel() / B;
    auto gx = torch::empty_like(xc);
    const int threads = 256; const size_t shmem = threads * sizeof(float);
    const c10::cuda::CUDAGuard guard(xc.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, xc.scalar_type(), "frobnorm_bwd", [&] {
            frobnorm_bwd_kernel<scalar_t><<<B, threads, shmem, stream>>>(
                gyc.data_ptr<scalar_t>(), xc.data_ptr<scalar_t>(),
                gx.data_ptr<scalar_t>(), B, AC, (float)eps);
        });
    return gx;
}

torch::Tensor weightnorm_bwd(
    const torch::Tensor& gy, const torch::Tensor& w_pre,
    const torch::Tensor& wn_target, double eps) {
    auto gyc = gy.contiguous(); auto wc = w_pre.contiguous(); auto wn = wn_target.contiguous();
    int64_t B = wc.size(0), A = wc.size(1), C = wc.size(2);
    auto gx = torch::empty_like(wc);
    const int threads = 256; const size_t shmem = threads * sizeof(float);
    const c10::cuda::CUDAGuard guard(wc.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, wc.scalar_type(), "weightnorm_bwd", [&] {
            weightnorm_bwd_kernel<scalar_t><<<B * C, threads, shmem, stream>>>(
                gyc.data_ptr<scalar_t>(), wc.data_ptr<scalar_t>(), wn.data_ptr<scalar_t>(),
                gx.data_ptr<scalar_t>(), B, A, C, (float)eps);
        });
    return gx;
}

// ===================== INFERENCE single-step fused kernel =====================
//
// Fuses the whole infer_step core for ONE token into a SINGLE kernel:
//   gate = q @ w0 ; up = q @ w2 ; h = silu(gate)*up ; o = h @ w1 ;
//   o = RMSNorm(o) * o_norm_weight
// q: [N, D] (N = B*heads rows, one token each), w0/w2: [N, D, H], w1: [N, H, D].
// (here D = head_dim = H = hidden per head, square.) One block per row n;
// blockDim.x = H threads. q and h live in shared memory; no global intermediates.
//
// Replaces ~19 tiny kernel launches/step with 1. fp32 math internally.
template <typename scalar_t>
__global__ void infer_step_kernel(
    const scalar_t* __restrict__ q,    // [N, D]
    const scalar_t* __restrict__ w0,   // [N, D, H]
    const scalar_t* __restrict__ w2,   // [N, D, H]
    const scalar_t* __restrict__ w1,   // [N, H, D]
    const scalar_t* __restrict__ onw,  // [D] o_norm weight (RMSNorm, per head_dim)
    scalar_t* __restrict__ out,        // [N, D]
    int64_t N, int D, int H, float eps) {
    int n = blockIdx.x;
    if (n >= N) return;
    int t = threadIdx.x;  // 0..H-1 (and reused for 0..D-1; D==H here)

    extern __shared__ float sh[];
    float* sq = sh;          // [D] the query row
    float* shd = sh + D;     // [H] hidden = silu(gate)*up

    // load q row into shared
    for (int i = t; i < D; i += blockDim.x) sq[i] = to_f(q[n * D + i]);
    __syncthreads();

    // each thread j computes gate_j and up_j = sum_i q_i * w[i,j]
    if (t < H) {
        float gate = 0.f, up = 0.f;
        const scalar_t* w0n = w0 + (int64_t)n * D * H;
        const scalar_t* w2n = w2 + (int64_t)n * D * H;
        for (int i = 0; i < D; ++i) {
            float qi = sq[i];
            gate += qi * to_f(w0n[i * H + t]);
            up   += qi * to_f(w2n[i * H + t]);
        }
        float s = gate / (1.f + __expf(-gate));   // silu(gate)
        shd[t] = s * up;
    }
    __syncthreads();

    // each thread d computes o_d = sum_j h_j * w1[j,d]
    float o = 0.f;
    if (t < D) {
        const scalar_t* w1n = w1 + (int64_t)n * H * D;
        for (int j = 0; j < H; ++j) o += shd[j] * to_f(w1n[j * D + t]);
    }
    // RMSNorm over the D outputs of this row: need sum of squares across threads
    __shared__ float red[1024];
    red[t] = (t < D) ? o * o : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(red[0] / D + eps);
    if (t < D) {
        out[n * D + t] = static_cast<scalar_t>(o * inv_rms * to_f(onw[t]));
    }
}

torch::Tensor infer_step(
    const torch::Tensor& q,    // [N, D]
    const torch::Tensor& w0,   // [N, D, H]
    const torch::Tensor& w2,   // [N, D, H]
    const torch::Tensor& w1,   // [N, H, D]
    const torch::Tensor& o_norm_weight,  // [D]
    double eps) {
    auto qc = q.contiguous(), w0c = w0.contiguous(), w2c = w2.contiguous(),
         w1c = w1.contiguous(), onw = o_norm_weight.contiguous();
    int64_t N = qc.size(0);
    int D = qc.size(1);
    int H = w0c.size(2);
    auto out = torch::empty_like(qc);
    int threads = (D > H ? D : H);
    // round up to power of 2 for the reduction
    int tp = 1; while (tp < threads) tp <<= 1;
    size_t shmem = (D + H) * sizeof(float);
    const c10::cuda::CUDAGuard guard(qc.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(), "infer_step", [&] {
            infer_step_kernel<scalar_t><<<N, tp, shmem, stream>>>(
                qc.data_ptr<scalar_t>(), w0c.data_ptr<scalar_t>(), w2c.data_ptr<scalar_t>(),
                w1c.data_ptr<scalar_t>(), onw.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), N, D, H, (float)eps);
        });
    return out;
}

// ---- Stage-1 mega: fuse q-norm + apply + RMSNorm into ONE kernel ----
// Same as infer_step_kernel but takes the UN-normalized per-head q and does the
// L2 q-normalization (q / (||q||+1e-5)) inside, eliminating the separate q-norm
// launch (the single biggest infer_step segment, ~37%).
template <typename scalar_t>
__global__ void infer_step_mid_kernel(
    const scalar_t* __restrict__ q,    // [N, D] UN-normalized
    const scalar_t* __restrict__ w0,   // [N, D, H]
    const scalar_t* __restrict__ w2,   // [N, D, H]
    const scalar_t* __restrict__ w1,   // [N, H, D]
    const scalar_t* __restrict__ onw,  // [D]
    scalar_t* __restrict__ out,        // [N, D]
    int64_t N, int D, int H, float eps, float qeps) {
    int n = blockIdx.x;
    if (n >= N) return;
    int t = threadIdx.x;

    extern __shared__ float sh[];
    float* sq = sh;          // [D] query row (post q-norm)
    float* shd = sh + D;     // [H] hidden

    // load q + accumulate sum of squares for the L2 norm
    __shared__ float qred[1024];
    float qv = (t < D) ? to_f(q[n * D + t]) : 0.f;
    for (int i = t; i < D; i += blockDim.x) sq[i] = to_f(q[n * D + i]);
    qred[t] = (t < D) ? qv * qv : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) qred[t] += qred[t + s];
        __syncthreads();
    }
    float qinv = 1.0f / (sqrtf(qred[0]) + qeps);   // 1/(||q||+1e-5)
    // normalize sq in place
    for (int i = t; i < D; i += blockDim.x) sq[i] *= qinv;
    __syncthreads();

    if (t < H) {
        float gate = 0.f, up = 0.f;
        const scalar_t* w0n = w0 + (int64_t)n * D * H;
        const scalar_t* w2n = w2 + (int64_t)n * D * H;
        for (int i = 0; i < D; ++i) {
            float qi = sq[i];
            gate += qi * to_f(w0n[i * H + t]);
            up   += qi * to_f(w2n[i * H + t]);
        }
        float s = gate / (1.f + __expf(-gate));
        shd[t] = s * up;
    }
    __syncthreads();

    float o = 0.f;
    if (t < D) {
        const scalar_t* w1n = w1 + (int64_t)n * H * D;
        for (int j = 0; j < H; ++j) o += shd[j] * to_f(w1n[j * D + t]);
    }
    __shared__ float red[1024];
    red[t] = (t < D) ? o * o : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(red[0] / D + eps);
    if (t < D) out[n * D + t] = static_cast<scalar_t>(o * inv_rms * to_f(onw[t]));
}

torch::Tensor infer_step_mid(
    const torch::Tensor& q,    // [N, D] UN-normalized per-head q
    const torch::Tensor& w0, const torch::Tensor& w2, const torch::Tensor& w1,
    const torch::Tensor& o_norm_weight, double eps, double qeps) {
    auto qc = q.contiguous(), w0c = w0.contiguous(), w2c = w2.contiguous(),
         w1c = w1.contiguous(), onw = o_norm_weight.contiguous();
    int64_t N = qc.size(0);
    int D = qc.size(1), H = w0c.size(2);
    auto out = torch::empty_like(qc);
    int threads = (D > H ? D : H);
    int tp = 1; while (tp < threads) tp <<= 1;
    size_t shmem = (D + H) * sizeof(float);
    const c10::cuda::CUDAGuard guard(qc.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(), "infer_step_mid", [&] {
            infer_step_mid_kernel<scalar_t><<<N, tp, shmem, stream>>>(
                qc.data_ptr<scalar_t>(), w0c.data_ptr<scalar_t>(), w2c.data_ptr<scalar_t>(),
                w1c.data_ptr<scalar_t>(), onw.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), N, D, H, (float)eps, (float)qeps);
        });
    return out;
}

}  // namespace ttt_cuda
