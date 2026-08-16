// FlashAttention-2 forward kernel (educational). Tests run flash_attn.py.
//
// Tiling (FA-2):
//   Br = 64  — Q rows (and output rows) owned by this CTA
//   Bc = 64  — K/V columns consumed per inner iteration
//
// SRAM (per CTA, d = head dim):
//   Qi  [Br x d]   — resident for the whole K/V loop
//   Kj  [Bc x d]   — streamed
//   Vj  [Bc x d]   — streamed
//   S   [Br x Bc]  — scores for this tile only; NEVER an (N, N) buffer
//   Oi  [Br x d]   — running weighted outputs
//   mi  [Br]       — running row-max
//   li  [Br]       — running row-sum of exp
//
// Algorithm (Dao et al. FA-2, synchronous):
//   for each Q-tile i:
//     O=0; l=0; m=-inf
//     for each K/V-tile j:
//       S = Qi @ Kj^T / sqrt(d)     // optionally causal-mask
//       m' = max(m, rowmax(S))
//       P  = exp(S - m')
//       l' = exp(m-m')*l + rowsum(P)
//       O  = exp(m-m')*O + P @ Vj
//       m,l = m',l'
//     O /= l
//
// FA-3 (Shah et al. 2024, Hopper only — documented, not required):
//   Same math, different schedule: TMA async copies, WGMMA, producer/consumer
//   warp specialization, FP8. This file is the FA-2 synchronous algorithm and
//   will compile on any sm that has CUDA; it does not use Hopper async.

#include <cuda_runtime.h>
#include <math.h>

#ifndef BR
#define BR 64
#endif
#ifndef BC
#define BC 64
#endif

extern "C" __global__ void flash_attn_fwd(
    const float* __restrict__ Q,  // (N, d) row-major, one head
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, int causal
) {
    // One CTA handles one Q-tile of Br rows. Grid.x = ceil(N / Br).
    const int i0 = blockIdx.x * BR;
    if (i0 >= N) return;

    extern __shared__ float smem[];
    // Layout: Qi[Br*d] | Kj[Bc*d] | Vj[Bc*d] | S[Br*Bc] | Oi[Br*d] | m[Br] | l[Br]
    float* Qi = smem;
    float* Kj = Qi + BR * d;
    float* Vj = Kj + BC * d;
    float* S  = Vj + BC * d;
    float* Oi = S + BR * BC;
    float* m  = Oi + BR * d;
    float* l  = m + BR;

    const int rows = min(BR, N - i0);
    const float scale = rsqrtf((float)d);

    // Load Qi, zero Oi, init m/l.
    for (int idx = threadIdx.x; idx < rows * d; idx += blockDim.x) {
        Qi[idx] = Q[(i0 + idx / d) * d + (idx % d)];
        Oi[idx] = 0.f;
    }
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        m[r] = -1e30f;
        l[r] = 0.f;
    }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BC) {
        const int cols = min(BC, N - j0);
        for (int idx = threadIdx.x; idx < cols * d; idx += blockDim.x) {
            Kj[idx] = K[(j0 + idx / d) * d + (idx % d)];
            Vj[idx] = V[(j0 + idx / d) * d + (idx % d)];
        }
        __syncthreads();

        // S = Qi @ Kj^T * scale  (rows x cols), then online softmax update.
        for (int r = threadIdx.x; r < rows; r += blockDim.x) {
            float rowmax = -1e30f;
            for (int c = 0; c < cols; ++c) {
                float acc = 0.f;
                for (int kk = 0; kk < d; ++kk) acc += Qi[r * d + kk] * Kj[c * d + kk];
                acc *= scale;
                if (causal && (j0 + c) > (i0 + r)) acc = -1e30f;
                S[r * BC + c] = acc;
                if (acc > rowmax) rowmax = acc;
            }
            const float m_new = fmaxf(m[r], rowmax);
            float psum = 0.f;
            for (int c = 0; c < cols; ++c) {
                const float p = expf(S[r * BC + c] - m_new);
                S[r * BC + c] = p;
                psum += p;
            }
            const float alpha = expf(m[r] - m_new);
            for (int kk = 0; kk < d; ++kk) {
                float acc = alpha * Oi[r * d + kk];
                for (int c = 0; c < cols; ++c) acc += S[r * BC + c] * Vj[c * d + kk];
                Oi[r * d + kk] = acc;
            }
            l[r] = alpha * l[r] + psum;
            m[r] = m_new;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < rows * d; idx += blockDim.x) {
        const int r = idx / d;
        O[(i0 + r) * d + (idx % d)] = Oi[idx] / fmaxf(l[r], 1e-20f);
    }
}
