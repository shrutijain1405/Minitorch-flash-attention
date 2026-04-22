/*
 * Flash Attention 1 — Forward and Backward CUDA kernels
 *
 * Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with
 *         IO-Awareness", Tri Dao et al., NeurIPS 2022.
 *
 * Design
 * ------
 * Inputs  Q, K, V : (batch, T, d)   batch = B * n_heads, must be contiguous
 * Outputs O       : (batch, T, d)   attention output
 *         L       : (batch, T)      per-row log-sum-exp, needed by backward
 *
 * Tile sizes FA_Br = FA_Bc = 32.  Max head-dim supported: FA_D_MAX = 64.
 * For larger d the Python wrapper falls back to vanilla attention.
 *
 * Thread layout: one CUDA thread per query row within the tile.
 *   Grid  = (batch,  ceil(T / FA_Br))
 *   Block = (FA_Br,)
 *
 * Shared memory per block (forward)  : 2 * FA_Bc * d * 4 bytes  (Kj + Vj)
 * Shared memory per block (backward) : 2 * FA_Bc * d * 4 bytes  (Kj + Vj)
 *
 * dK and dV are accumulated across query-tile thread-blocks using
 * global atomicAdd (correct but serialises conflicting updates).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Compile-time constants                                             */
/* ------------------------------------------------------------------ */
#define FA_Br     32   /* query  tile size (rows of Q per block) */
#define FA_Bc     32   /* key/value tile size                    */
#define FA_D_MAX  64   /* maximum supported head dimension       */


/* ================================================================== */
/*  FORWARD KERNEL                                                     */
/* ================================================================== */
/*
 * Each block handles one (batch, query-tile) pair.
 * One thread  →  one query row.
 *
 * Algorithm (Flash Attention 1 forward):
 *   For each KV tile j:
 *     Load Kj, Vj into SRAM
 *     Compute row-wise dot products  Sij = Qi · Kj^T * scale
 *     Update running max mi, running sum li, and output accumulator Oi
 *       using the numerically-stable online softmax trick.
 *   Finalise:  O[i] = Oi / li
 *              L[i] = mi + log(li)      (logsumexp, saved for backward)
 */
__global__ void flashAttnFwdKernel(
    const float* __restrict__ Q,   /* (batch, T, d) */
    const float* __restrict__ K,   /* (batch, T, d) */
    const float* __restrict__ V,   /* (batch, T, d) */
    float*       __restrict__ O,   /* (batch, T, d) */
    float*       __restrict__ L,   /* (batch, T)    */
    int T, int d, float scale, int causal
)
{
    int b        = blockIdx.x;   /* batch index */
    int qb       = blockIdx.y;   /* query-tile index */
    int tid      = threadIdx.x;  /* row within tile, 0..FA_Br-1 */

    int qi_start = qb * FA_Br;
    int qi       = qi_start + tid;   /* actual query row */

    const float* Qb = Q + (size_t)b * T * d;
    const float* Kb = K + (size_t)b * T * d;
    const float* Vb = V + (size_t)b * T * d;
    float*       Ob = O + (size_t)b * T * d;
    float*       Lb = L + (size_t)b * T;

    /* Shared memory: Kj tile then Vj tile  (dynamically sized by host) */
    extern __shared__ float smem[];
    float* Kj_sh = smem;              /* FA_Bc * d floats */
    float* Vj_sh = smem + FA_Bc * d;  /* FA_Bc * d floats */

    /* ---------- per-thread registers ---------- */
    float qi_r[FA_D_MAX];
    float oi_r[FA_D_MAX];
    float mi = -1e30f;
    float li =  0.0f;

    /* Load this thread's query row (zero-pad if out of bounds) */
    for (int e = 0; e < d; e++) {
        qi_r[e] = (qi < T) ? Qb[qi * d + e] : 0.0f;
        oi_r[e] = 0.0f;
    }

    int num_kv = (T + FA_Bc - 1) / FA_Bc;

    for (int kvb = 0; kvb < num_kv; kvb++) {
        int kj_start = kvb * FA_Bc;
        int kj_load  = kj_start + tid;   /* row of K/V this thread loads */

        /* Cooperatively load one Kj / Vj tile (each thread loads 1 row) */
        for (int e = 0; e < d; e++) {
            Kj_sh[tid * d + e] = (kj_load < T) ? Kb[kj_load * d + e] : 0.0f;
            Vj_sh[tid * d + e] = (kj_load < T) ? Vb[kj_load * d + e] : 0.0f;
        }
        __syncthreads();

        if (qi < T) {
            /* ---- compute attention scores Sij for this query row ---- */
            float sij[FA_Bc];
            float mi_new = mi;

            for (int c = 0; c < FA_Bc; c++) {
                int kj_idx = kj_start + c;
                if (kj_idx >= T || (causal && qi < kj_idx)) {
                    sij[c] = -1e30f;  /* masked or padding */
                } else {
                    float dot = 0.0f;
                    for (int e = 0; e < d; e++)
                        dot += qi_r[e] * Kj_sh[c * d + e];
                    sij[c] = dot * scale;
                }
                if (sij[c] > mi_new) mi_new = sij[c];
            }

            /* ---- online softmax update ---- */
            float alpha  = expf(mi - mi_new);   /* rescale factor for old stats */
            float li_new = li * alpha;

            /* rescale previous output accumulator */
            for (int e = 0; e < d; e++)
                oi_r[e] *= alpha;

            /* add contribution from this KV tile */
            for (int c = 0; c < FA_Bc; c++) {
                float p = expf(sij[c] - mi_new);
                li_new += p;
                for (int e = 0; e < d; e++)
                    oi_r[e] += p * Vj_sh[c * d + e];
            }

            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    /* Finalise and write output */
    if (qi < T) {
        float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;
        for (int e = 0; e < d; e++)
            Ob[qi * d + e] = oi_r[e] * inv_l;
        Lb[qi] = (li > 0.0f) ? (mi + logf(li)) : -1e30f;
    }
}


/* ================================================================== */
/*  BACKWARD KERNEL                                                    */
/* ================================================================== */
/*
 * Recompute-based Flash Attention 1 backward (Algorithm 4 in the paper).
 *
 * Given Q, K, V, O, L (from forward) and dO:
 *   Di    = rowsum(Oi * dOi)                         shape (T,)
 *   Pij   = exp(Sij − Li)                            recomputed from Q,K
 *   dSij  = Pij * (dPij − Di)   where dPij = dOi·Vj^T
 *   dQi  += dSij · Kj * scale
 *   dKj  += dSij^T · Qi * scale  (atomicAdd, multi-block writes)
 *   dVj  += Pij^T · dOi          (atomicAdd, multi-block writes)
 *
 * Grid  = (batch, ceil(T / FA_Br))
 * Block = (FA_Br,)
 */
__global__ void flashAttnBwdKernel(
    const float* __restrict__ Q,   /* (batch, T, d) */
    const float* __restrict__ K,   /* (batch, T, d) */
    const float* __restrict__ V,   /* (batch, T, d) */
    const float* __restrict__ O,   /* (batch, T, d) */
    const float* __restrict__ L,   /* (batch, T)    */
    const float* __restrict__ dO,  /* (batch, T, d) */
    float*       __restrict__ dQ,  /* (batch, T, d) output, pre-zeroed  */
    float*       __restrict__ dK,  /* (batch, T, d) output, pre-zeroed  */
    float*       __restrict__ dV,  /* (batch, T, d) output, pre-zeroed  */
    int T, int d, float scale, int causal
)
{
    int b        = blockIdx.x;
    int qb       = blockIdx.y;
    int tid      = threadIdx.x;

    int qi_start = qb * FA_Br;
    int qi       = qi_start + tid;

    const float* Qb  = Q  + (size_t)b * T * d;
    const float* Kb  = K  + (size_t)b * T * d;
    const float* Vb  = V  + (size_t)b * T * d;
    const float* Ob  = O  + (size_t)b * T * d;
    const float* Lb  = L  + (size_t)b * T;
    const float* dOb = dO + (size_t)b * T * d;
    float*       dQb = dQ + (size_t)b * T * d;
    float*       dKb = dK + (size_t)b * T * d;
    float*       dVb = dV + (size_t)b * T * d;

    extern __shared__ float smem[];
    float* Kj_sh = smem;
    float* Vj_sh = smem + FA_Bc * d;

    /* ---------- per-thread registers ---------- */
    float qi_r[FA_D_MAX];
    float oi_r[FA_D_MAX];
    float doi_r[FA_D_MAX];
    float dqi_r[FA_D_MAX];
    float Li_val = -1e30f;
    float Di     =  0.0f;

    if (qi < T) {
        Li_val = Lb[qi];
        for (int e = 0; e < d; e++) {
            qi_r[e]  = Qb [qi * d + e];
            oi_r[e]  = Ob [qi * d + e];
            doi_r[e] = dOb[qi * d + e];
            dqi_r[e] = 0.0f;
            Di += oi_r[e] * doi_r[e];  /* Di = dot(O[i], dO[i]) */
        }
    } else {
        for (int e = 0; e < d; e++) {
            qi_r[e] = doi_r[e] = dqi_r[e] = 0.0f;
        }
    }

    int num_kv = (T + FA_Bc - 1) / FA_Bc;

    for (int kvb = 0; kvb < num_kv; kvb++) {
        int kj_start = kvb * FA_Bc;
        int kj_load  = kj_start + tid;

        for (int e = 0; e < d; e++) {
            Kj_sh[tid * d + e] = (kj_load < T) ? Kb[kj_load * d + e] : 0.0f;
            Vj_sh[tid * d + e] = (kj_load < T) ? Vb[kj_load * d + e] : 0.0f;
        }
        __syncthreads();

        if (qi < T) {
            float sij [FA_Bc];
            float pij [FA_Bc];
            float dpij[FA_Bc];
            float dsij[FA_Bc];

            /* Recompute Sij (same masking as forward) */
            for (int c = 0; c < FA_Bc; c++) {
                int kj_idx = kj_start + c;
                if (kj_idx >= T || (causal && qi < kj_idx)) {
                    sij[c] = -1e30f;
                } else {
                    float dot = 0.0f;
                    for (int e = 0; e < d; e++)
                        dot += qi_r[e] * Kj_sh[c * d + e];
                    sij[c] = dot * scale;
                }
                /* Pij = exp(Sij − Li);  guard against -inf − (-inf) = NaN */
                pij[c] = (sij[c] > -1e29f) ? expf(sij[c] - Li_val) : 0.0f;
            }

            /* dpij[c] = dot(dOi, Vj[c]) */
            for (int c = 0; c < FA_Bc; c++) {
                float dot = 0.0f;
                for (int e = 0; e < d; e++)
                    dot += doi_r[e] * Vj_sh[c * d + e];
                dpij[c] = dot;
                dsij[c] = pij[c] * (dpij[c] - Di) * scale;
            }

            /* dQi += dsij · Kj */
            for (int c = 0; c < FA_Bc; c++) {
                for (int e = 0; e < d; e++)
                    dqi_r[e] += dsij[c] * Kj_sh[c * d + e];
            }

            /* dKj += dsij^T · Qi  and  dVj += pij^T · dOi  (global atomics) */
            for (int c = 0; c < FA_Bc; c++) {
                int kj_idx = kj_start + c;
                if (kj_idx < T) {
                    for (int e = 0; e < d; e++) {
                        atomicAdd(&dKb[kj_idx * d + e], dsij[c] * qi_r[e]);
                        atomicAdd(&dVb[kj_idx * d + e], pij[c]  * doi_r[e]);
                    }
                }
            }
        }
        __syncthreads();
    }

    /* Write dQi */
    if (qi < T) {
        for (int e = 0; e < d; e++)
            dQb[qi * d + e] = dqi_r[e];
    }
}


/* ================================================================== */
/*  HOST WRAPPER — FORWARD                                             */
/* ================================================================== */
/*
 * All pointers are HOST pointers (float*).
 * The function allocates device memory, copies in, runs the kernel,
 * copies out, and frees — matching the pattern of combine.cu.
 *
 * Q, K, V : (batch, T, d)  contiguous float32
 * O       : (batch, T, d)  output  (pre-allocated by caller)
 * L       : (batch, T)     log-sum-exp output (pre-allocated by caller)
 */
extern "C"
void flashAttentionForward(
    float* Q, float* K, float* V,
    float* O, float* L,
    int batch, int T, int d, float scale, int causal
)
{
    size_t qkv_bytes = (size_t)batch * T * d * sizeof(float);
    size_t lse_bytes = (size_t)batch * T     * sizeof(float);

    float *dQ, *dK, *dV, *dO, *dL;
    cudaMalloc(&dQ, qkv_bytes);
    cudaMalloc(&dK, qkv_bytes);
    cudaMalloc(&dV, qkv_bytes);
    cudaMalloc(&dO, qkv_bytes);
    cudaMalloc(&dL, lse_bytes);

    cudaMemcpy(dQ, Q, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, qkv_bytes, cudaMemcpyHostToDevice);

    int q_tiles  = (T + FA_Br - 1) / FA_Br;
    dim3 grid(batch, q_tiles);
    dim3 block(FA_Br);
    size_t shmem = 2 * FA_Bc * d * sizeof(float);

    flashAttnFwdKernel<<<grid, block, shmem>>>(
        dQ, dK, dV, dO, dL,
        T, d, scale, causal
    );

    cudaMemcpy(O, dO, qkv_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(L, dL, lse_bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flashAttentionForward error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dL);
}


/* ================================================================== */
/*  HOST WRAPPER — BACKWARD                                            */
/* ================================================================== */
/*
 * Q, K, V, O, L : saved from forward pass (HOST pointers)
 * dO            : upstream gradient for O (HOST pointer)
 * dQ, dK, dV    : output gradients (HOST pointers, pre-allocated by caller)
 */
extern "C"
void flashAttentionBackward(
    float* Q, float* K, float* V,
    float* O, float* L,
    float* dO,
    float* dQ, float* dK, float* dV,
    int batch, int T, int d, float scale, int causal
)
{
    size_t qkv_bytes = (size_t)batch * T * d * sizeof(float);
    size_t lse_bytes = (size_t)batch * T     * sizeof(float);

    float *dQ_d, *dK_d, *dV_d;
    float *Q_d,  *K_d,  *V_d, *O_d, *L_d, *dO_d;

    cudaMalloc(&Q_d,  qkv_bytes);
    cudaMalloc(&K_d,  qkv_bytes);
    cudaMalloc(&V_d,  qkv_bytes);
    cudaMalloc(&O_d,  qkv_bytes);
    cudaMalloc(&L_d,  lse_bytes);
    cudaMalloc(&dO_d, qkv_bytes);
    cudaMalloc(&dQ_d, qkv_bytes);
    cudaMalloc(&dK_d, qkv_bytes);
    cudaMalloc(&dV_d, qkv_bytes);

    cudaMemcpy(Q_d,  Q,  qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(K_d,  K,  qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d,  V,  qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(O_d,  O,  qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(L_d,  L,  lse_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dO_d, dO, qkv_bytes, cudaMemcpyHostToDevice);

    /* dK and dV are accumulated via atomicAdd — must start at zero */
    cudaMemset(dQ_d, 0, qkv_bytes);
    cudaMemset(dK_d, 0, qkv_bytes);
    cudaMemset(dV_d, 0, qkv_bytes);

    int q_tiles = (T + FA_Br - 1) / FA_Br;
    dim3 grid(batch, q_tiles);
    dim3 block(FA_Br);
    size_t shmem = 2 * FA_Bc * d * sizeof(float);

    flashAttnBwdKernel<<<grid, block, shmem>>>(
        Q_d, K_d, V_d, O_d, L_d, dO_d,
        dQ_d, dK_d, dV_d,
        T, d, scale, causal
    );

    cudaMemcpy(dQ, dQ_d, qkv_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(dK, dK_d, qkv_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(dV, dV_d, qkv_bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flashAttentionBackward error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(Q_d);  cudaFree(K_d);  cudaFree(V_d);
    cudaFree(O_d);  cudaFree(L_d);  cudaFree(dO_d);
    cudaFree(dQ_d); cudaFree(dK_d); cudaFree(dV_d);
}
