#include <cuda_runtime.h>
#include <stdio.h>

#define FLASH_D   64
#define FLASH_BR  64
#define FLASH_BC  64

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ---------------------------------------------------------------------------
// D kernel: D_i = rowsum(dO_i * O_i)  — one scalar per query row
// Shared by FA1 and FA2 backward.
// Grid: (BH, Tr),  Block: (Br,)
// ---------------------------------------------------------------------------
__global__ void computeDKernel(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* D,
    int N
) {
    int bh  = blockIdx.x;
    int row = blockIdx.y * FLASH_BR + threadIdx.x;
    if (row >= N) return;

    const float* dO_bh = dO + bh * N * FLASH_D;
    const float* O_bh  = O  + bh * N * FLASH_D;
    float* D_bh = D + bh * N;

    float d = 0.0f;
    for (int c = 0; c < FLASH_D; c++)
        d += dO_bh[row * FLASH_D + c] * O_bh[row * FLASH_D + c];
    D_bh[row] = d;
}


// ===========================================================================
// FA1 Backward Kernel
//
// Grid: (BH, Tc) — one block per (batch*head, K/V tile)
// Block: (Bc,)   — one thread owns one K/V row in the tile
//
// Loop structure (outer K/V, inner Q):
//   • dK_j, dV_j accumulated in registers — no race, written at end
//   • dQ_i contributions from multiple KV blocks → atomicAdd
// ===========================================================================
__global__ void flashAttentionBackwardKernelFA1(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    float* dQ,
    float* dK,
    float* dV,
    int N,
    float scale
) {
    int bh      = blockIdx.x;
    int kv_tile = blockIdx.y;
    int tid     = threadIdx.x;
    int kk      = kv_tile * FLASH_BC + tid;

    const float* Q_bh  = Q  + bh * N * FLASH_D;
    const float* K_bh  = K  + bh * N * FLASH_D;
    const float* V_bh  = V  + bh * N * FLASH_D;
    const float* dO_bh = dO + bh * N * FLASH_D;
    const float* L_bh  = L  + bh * N;
    const float* D_bh  = D  + bh * N;
    float* dQ_bh = dQ + bh * N * FLASH_D;
    float* dK_bh = dK + bh * N * FLASH_D;
    float* dV_bh = dV + bh * N * FLASH_D;

    __shared__ float Q_tile[FLASH_BR][FLASH_D];
    __shared__ float dO_tile[FLASH_BR][FLASH_D];
    __shared__ float L_tile[FLASH_BR];
    __shared__ float D_tile[FLASH_BR];

    bool valid_kv = (kk < N);

    float k_reg[FLASH_D];
    float v_reg[FLASH_D];
    if (valid_kv) {
        for (int c = 0; c < FLASH_D; c++) {
            k_reg[c] = K_bh[kk * FLASH_D + c];
            v_reg[c] = V_bh[kk * FLASH_D + c];
        }
    } else {
        for (int c = 0; c < FLASH_D; c++) k_reg[c] = v_reg[c] = 0.0f;
    }

    float dK_reg[FLASH_D];
    float dV_reg[FLASH_D];
    for (int c = 0; c < FLASH_D; c++) dK_reg[c] = dV_reg[c] = 0.0f;

    int Tr = (N + FLASH_BR - 1) / FLASH_BR;

    for (int i = 0; i < Tr; i++) {
        int q_base = i * FLASH_BR;

        int q_row_load = q_base + tid;
        if (q_row_load < N) {
            for (int c = 0; c < FLASH_D; c++) {
                Q_tile[tid][c]  = Q_bh [q_row_load * FLASH_D + c];
                dO_tile[tid][c] = dO_bh[q_row_load * FLASH_D + c];
            }
            L_tile[tid] = L_bh[q_row_load];
            D_tile[tid] = D_bh[q_row_load];
        } else {
            for (int c = 0; c < FLASH_D; c++) Q_tile[tid][c] = dO_tile[tid][c] = 0.0f;
            L_tile[tid] = D_tile[tid] = 0.0f;
        }
        __syncthreads();

        if (valid_kv) {
            for (int ii = 0; ii < FLASH_BR; ii++) {
                int q_row = q_base + ii;
                if (q_row >= N) break;

                float S = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    S += Q_tile[ii][c] * k_reg[c];
                S *= scale;

                float P = expf(S - L_tile[ii]);

                float dP = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    dP += dO_tile[ii][c] * v_reg[c];

                float dS = P * (dP - D_tile[ii]);

                for (int c = 0; c < FLASH_D; c++)
                    dV_reg[c] += P * dO_tile[ii][c];

                for (int c = 0; c < FLASH_D; c++)
                    dK_reg[c] += dS * Q_tile[ii][c] * scale;

                for (int c = 0; c < FLASH_D; c++)
                    atomicAdd(&dQ_bh[q_row * FLASH_D + c], dS * k_reg[c] * scale);
            }
        }

        __syncthreads();
    }

    if (valid_kv) {
        for (int c = 0; c < FLASH_D; c++) {
            dK_bh[kk * FLASH_D + c] = dK_reg[c];
            dV_bh[kk * FLASH_D + c] = dV_reg[c];
        }
    }
}


// ===========================================================================
// FA2 Backward Kernel 1: Compute dK and dV
//
// Grid: (BH, Tc) — one block per (batch*head, K/V tile)
// Block: (Bc,)   — one thread owns one K/V row
//
// Outer K/V, inner Q — dK and dV accumulated in registers.
// No atomicAdd: each block has exclusive ownership of its KV rows.
// ===========================================================================
__global__ void flashAttentionBackwardDKVKernelFA2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    float* dK,
    float* dV,
    int N,
    float scale
) {
    int bh      = blockIdx.x;
    int kv_tile = blockIdx.y;
    int tid     = threadIdx.x;
    int kk      = kv_tile * FLASH_BC + tid;

    const float* Q_bh  = Q  + bh * N * FLASH_D;
    const float* K_bh  = K  + bh * N * FLASH_D;
    const float* V_bh  = V  + bh * N * FLASH_D;
    const float* dO_bh = dO + bh * N * FLASH_D;
    const float* L_bh  = L  + bh * N;
    const float* D_bh  = D  + bh * N;
    float* dK_bh = dK + bh * N * FLASH_D;
    float* dV_bh = dV + bh * N * FLASH_D;

    // Q and dO loaded cooperatively into shared mem; K/V stay in registers
    __shared__ float Q_tile[FLASH_BR][FLASH_D];
    __shared__ float dO_tile[FLASH_BR][FLASH_D];
    __shared__ float L_tile[FLASH_BR];
    __shared__ float D_tile[FLASH_BR];

    bool valid_kv = (kk < N);

    float k_reg[FLASH_D];
    float v_reg[FLASH_D];
    if (valid_kv) {
        for (int c = 0; c < FLASH_D; c++) {
            k_reg[c] = K_bh[kk * FLASH_D + c];
            v_reg[c] = V_bh[kk * FLASH_D + c];
        }
    } else {
        for (int c = 0; c < FLASH_D; c++) k_reg[c] = v_reg[c] = 0.0f;
    }

    float dK_reg[FLASH_D];
    float dV_reg[FLASH_D];
    for (int c = 0; c < FLASH_D; c++) dK_reg[c] = dV_reg[c] = 0.0f;

    int Tr = (N + FLASH_BR - 1) / FLASH_BR;

    for (int i = 0; i < Tr; i++) {
        int q_base     = i * FLASH_BR;
        int q_row_load = q_base + tid;

        if (q_row_load < N) {
            for (int c = 0; c < FLASH_D; c++) {
                Q_tile[tid][c]  = Q_bh [q_row_load * FLASH_D + c];
                dO_tile[tid][c] = dO_bh[q_row_load * FLASH_D + c];
            }
            L_tile[tid] = L_bh[q_row_load];
            D_tile[tid] = D_bh[q_row_load];
        } else {
            for (int c = 0; c < FLASH_D; c++) Q_tile[tid][c] = dO_tile[tid][c] = 0.0f;
            L_tile[tid] = D_tile[tid] = 0.0f;
        }
        __syncthreads();

        if (valid_kv) {
            for (int ii = 0; ii < FLASH_BR; ii++) {
                int q_row = q_base + ii;
                if (q_row >= N) break;

                float S = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    S += Q_tile[ii][c] * k_reg[c];
                S *= scale;

                float P  = expf(S - L_tile[ii]);
                float dP = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    dP += dO_tile[ii][c] * v_reg[c];
                float dS = P * (dP - D_tile[ii]);

                for (int c = 0; c < FLASH_D; c++)
                    dV_reg[c] += P * dO_tile[ii][c];
                for (int c = 0; c < FLASH_D; c++)
                    dK_reg[c] += dS * Q_tile[ii][c] * scale;
            }
        }

        __syncthreads();
    }

    if (valid_kv) {
        for (int c = 0; c < FLASH_D; c++) {
            dK_bh[kk * FLASH_D + c] = dK_reg[c];
            dV_bh[kk * FLASH_D + c] = dV_reg[c];
        }
    }
}


// ===========================================================================
// FA2 Backward Kernel 2: Compute dQ
//
// Grid: (BH, Tr) — one block per (batch*head, Q tile)
// Block: (Br,)   — one thread owns one Q row
//
// Outer Q, inner K/V — dQ accumulated in registers.
// No atomicAdd: each block has exclusive ownership of its Q rows.
// ===========================================================================
__global__ void flashAttentionBackwardDQKernelFA2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    float* dQ,
    int N,
    float scale
) {
    int bh     = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid    = threadIdx.x;
    int qq     = q_tile * FLASH_BR + tid;

    const float* Q_bh  = Q  + bh * N * FLASH_D;
    const float* K_bh  = K  + bh * N * FLASH_D;
    const float* V_bh  = V  + bh * N * FLASH_D;
    const float* dO_bh = dO + bh * N * FLASH_D;
    const float* L_bh  = L  + bh * N;
    const float* D_bh  = D  + bh * N;
    float* dQ_bh = dQ + bh * N * FLASH_D;

    // K and V loaded cooperatively into shared mem; Q/dO/L/D stay in registers
    __shared__ float K_tile[FLASH_BC][FLASH_D];
    __shared__ float V_tile[FLASH_BC][FLASH_D];

    bool valid_q = (qq < N);

    // Each thread loads its own Q row, dO row, L, D into registers
    float q_reg[FLASH_D];
    float dO_reg[FLASH_D];
    float L_i = 0.0f, D_i = 0.0f;

    if (valid_q) {
        for (int c = 0; c < FLASH_D; c++) {
            q_reg[c]  = Q_bh [qq * FLASH_D + c];
            dO_reg[c] = dO_bh[qq * FLASH_D + c];
        }
        L_i = L_bh[qq];
        D_i = D_bh[qq];
    }

    float dQ_reg[FLASH_D];
    for (int c = 0; c < FLASH_D; c++) dQ_reg[c] = 0.0f;

    int Tc = (N + FLASH_BC - 1) / FLASH_BC;

    for (int j = 0; j < Tc; j++) {
        int k_base     = j * FLASH_BC;
        int k_row_load = k_base + tid;

        // Load K_j and V_j cooperatively
        if (k_row_load < N) {
            for (int c = 0; c < FLASH_D; c++) {
                K_tile[tid][c] = K_bh[k_row_load * FLASH_D + c];
                V_tile[tid][c] = V_bh[k_row_load * FLASH_D + c];
            }
        } else {
            for (int c = 0; c < FLASH_D; c++) K_tile[tid][c] = V_tile[tid][c] = 0.0f;
        }
        __syncthreads();

        if (valid_q) {
            for (int jj = 0; jj < FLASH_BC; jj++) {
                int k_row = k_base + jj;
                if (k_row >= N) break;

                float S = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    S += q_reg[c] * K_tile[jj][c];
                S *= scale;

                float P  = expf(S - L_i);
                float dP = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    dP += dO_reg[c] * V_tile[jj][c];
                float dS = P * (dP - D_i);

                for (int c = 0; c < FLASH_D; c++)
                    dQ_reg[c] += dS * K_tile[jj][c] * scale;
            }
        }

        __syncthreads();
    }

    if (valid_q) {
        for (int c = 0; c < FLASH_D; c++)
            dQ_bh[qq * FLASH_D + c] = dQ_reg[c];
    }
}


extern "C" {

// ---------------------------------------------------------------------------
// FA1 backward wrapper
// Outer K/V blocks, atomicAdd for dQ.
// ---------------------------------------------------------------------------
void flashAttentionBackwardFA1(
    float* Q, float* K, float* V, float* O, float* dO, float* L,
    float* dQ, float* dK, float* dV,
    int BH, int N, float scale
) {
    CUDA_CHECK(cudaSetDevice(0));

    size_t bytes   = (size_t)BH * N * FLASH_D * sizeof(float);
    size_t l_bytes = (size_t)BH * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_L, *d_D, *d_dQ, *d_dK, *d_dV;

    CUDA_CHECK(cudaMalloc(&d_Q,   bytes));
    CUDA_CHECK(cudaMalloc(&d_K,   bytes));
    CUDA_CHECK(cudaMalloc(&d_V,   bytes));
    CUDA_CHECK(cudaMalloc(&d_O,   bytes));
    CUDA_CHECK(cudaMalloc(&d_dO,  bytes));
    CUDA_CHECK(cudaMalloc(&d_L,   l_bytes));
    CUDA_CHECK(cudaMalloc(&d_D,   l_bytes));
    CUDA_CHECK(cudaMalloc(&d_dQ,  bytes));
    CUDA_CHECK(cudaMalloc(&d_dK,  bytes));
    CUDA_CHECK(cudaMalloc(&d_dV,  bytes));

    CUDA_CHECK(cudaMemcpy(d_Q,  Q,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,  K,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,  V,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O,  O,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO, dO, bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_L,  L,  l_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D,  0,  l_bytes));
    CUDA_CHECK(cudaMemset(d_dQ, 0,  bytes));
    CUDA_CHECK(cudaMemset(d_dK, 0,  bytes));
    CUDA_CHECK(cudaMemset(d_dV, 0,  bytes));

    int Tr = (N + FLASH_BR - 1) / FLASH_BR;
    int Tc = (N + FLASH_BC - 1) / FLASH_BC;
    dim3 grid_tr(BH, Tr);
    dim3 grid_tc(BH, Tc);
    dim3 block(FLASH_BR);

    computeDKernel<<<grid_tr, block>>>(d_dO, d_O, d_D, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    flashAttentionBackwardKernelFA1<<<grid_tc, block>>>(
        d_Q, d_K, d_V, d_dO, d_L, d_D,
        d_dQ, d_dK, d_dV, N, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dQ, d_dQ, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dK, d_dK, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dV, d_dV, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_Q);  cudaFree(d_K);  cudaFree(d_V);
    cudaFree(d_O);  cudaFree(d_dO); cudaFree(d_L);
    cudaFree(d_D);  cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
}

// ---------------------------------------------------------------------------
// FA2 backward wrapper
// Two kernels: DKV (outer K/V) + DQ (outer Q) — no atomicAdd anywhere.
// ---------------------------------------------------------------------------
void flashAttentionBackwardFA2(
    float* Q, float* K, float* V, float* O, float* dO, float* L,
    float* dQ, float* dK, float* dV,
    int BH, int N, float scale
) {
    CUDA_CHECK(cudaSetDevice(0));

    size_t bytes   = (size_t)BH * N * FLASH_D * sizeof(float);
    size_t l_bytes = (size_t)BH * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_L, *d_D, *d_dQ, *d_dK, *d_dV;

    CUDA_CHECK(cudaMalloc(&d_Q,   bytes));
    CUDA_CHECK(cudaMalloc(&d_K,   bytes));
    CUDA_CHECK(cudaMalloc(&d_V,   bytes));
    CUDA_CHECK(cudaMalloc(&d_O,   bytes));
    CUDA_CHECK(cudaMalloc(&d_dO,  bytes));
    CUDA_CHECK(cudaMalloc(&d_L,   l_bytes));
    CUDA_CHECK(cudaMalloc(&d_D,   l_bytes));
    CUDA_CHECK(cudaMalloc(&d_dQ,  bytes));
    CUDA_CHECK(cudaMalloc(&d_dK,  bytes));
    CUDA_CHECK(cudaMalloc(&d_dV,  bytes));

    CUDA_CHECK(cudaMemcpy(d_Q,  Q,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,  K,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,  V,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O,  O,  bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO, dO, bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_L,  L,  l_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D,  0,  l_bytes));
    CUDA_CHECK(cudaMemset(d_dQ, 0,  bytes));
    CUDA_CHECK(cudaMemset(d_dK, 0,  bytes));
    CUDA_CHECK(cudaMemset(d_dV, 0,  bytes));

    int Tr = (N + FLASH_BR - 1) / FLASH_BR;
    int Tc = (N + FLASH_BC - 1) / FLASH_BC;
    dim3 grid_tr(BH, Tr);
    dim3 grid_tc(BH, Tc);
    dim3 block(FLASH_BR);

    // Step 1: D_i = rowsum(dO_i * O_i)
    computeDKernel<<<grid_tr, block>>>(d_dO, d_O, d_D, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2a: dK and dV (outer K/V, no atomicAdd)
    flashAttentionBackwardDKVKernelFA2<<<grid_tc, block>>>(
        d_Q, d_K, d_V, d_dO, d_L, d_D,
        d_dK, d_dV, N, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2b: dQ (outer Q, no atomicAdd)
    flashAttentionBackwardDQKernelFA2<<<grid_tr, block>>>(
        d_Q, d_K, d_V, d_dO, d_L, d_D,
        d_dQ, N, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dQ, d_dQ, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dK, d_dK, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dV, d_dV, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_Q);  cudaFree(d_K);  cudaFree(d_V);
    cudaFree(d_O);  cudaFree(d_dO); cudaFree(d_L);
    cudaFree(d_D);  cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
}

} // extern "C"
