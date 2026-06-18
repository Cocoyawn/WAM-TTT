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

}  // namespace ttt_cuda
