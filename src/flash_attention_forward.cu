#include <cuda_runtime.h>
#include <stdio.h>

#define FLASH_D   64
#define FLASH_BR  64
#define FLASH_BC  64

__global__ void flashAttentionForwardKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* O,
    int N,
    float scale
) {
    // Grid : (BH, Tr) — one block per (batch*head, Q-tile)
    // Block: (Br,)    — one thread per row in the Q-tile
    int bh     = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid    = threadIdx.x;
    int q_row  = q_tile * FLASH_BR + tid;

    const float* Q_bh = Q + bh * N * FLASH_D;
    const float* K_bh = K + bh * N * FLASH_D;
    const float* V_bh = V + bh * N * FLASH_D;
    float*       O_bh = O + bh * N * FLASH_D;

    // K and V tiles live in shared memory (~32 KB total)
    __shared__ float K_tile[FLASH_BC][FLASH_D];
    __shared__ float V_tile[FLASH_BC][FLASH_D];

    // Load this thread's Q row into registers — never reloaded from HBM
    float q_reg[FLASH_D];
    bool valid_q = (q_row < N);
    if (valid_q) {
        for (int c = 0; c < FLASH_D; c++)
            q_reg[c] = Q_bh[q_row * FLASH_D + c];
    }

    // Per-row running accumulators (stay in registers throughout)
    float O_reg[FLASH_D];
    for (int c = 0; c < FLASH_D; c++) O_reg[c] = 0.0f;
    float m_i = -1e9f;   // running row-max
    float l_i = 0.0f;    // running softmax denominator

    int Tc = (N + FLASH_BC - 1) / FLASH_BC;

    for (int j = 0; j < Tc; j++) {
        int k_base = j * FLASH_BC;

        // Cooperatively load K and V tiles: thread tid owns row tid of the tile
        int k_row = k_base + tid;
        if (k_row < N) {
            for (int c = 0; c < FLASH_D; c++) {
                K_tile[tid][c] = K_bh[k_row * FLASH_D + c];
                V_tile[tid][c] = V_bh[k_row * FLASH_D + c];
            }
        } else {
            for (int c = 0; c < FLASH_D; c++) {
                K_tile[tid][c] = 0.0f;
                V_tile[tid][c] = 0.0f;
            }
        }
        __syncthreads();

        if (valid_q) {
            // Pass 1: compute local rowmax over this K tile
            float m_ij = -1e9f;
            for (int jj = 0; jj < FLASH_BC; jj++) {
                if (k_base + jj >= N) break;
                float dot = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    dot += q_reg[c] * K_tile[jj][c];
                m_ij = fmaxf(m_ij, dot * scale);
            }

            // Online softmax: rescale existing O and l with the updated global max
            float m_new = fmaxf(m_i, m_ij);
            float alpha = expf(m_i - m_new);   // correction for previously accumulated state
            float l_new = alpha * l_i;
            for (int c = 0; c < FLASH_D; c++)
                O_reg[c] *= alpha;

            // Pass 2: compute unnormalized softmax weights P and accumulate P @ V
            for (int jj = 0; jj < FLASH_BC; jj++) {
                if (k_base + jj >= N) break;
                float dot = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    dot += q_reg[c] * K_tile[jj][c];
                float p = expf(dot * scale - m_new);
                l_new += p;
                for (int c = 0; c < FLASH_D; c++)
                    O_reg[c] += p * V_tile[jj][c];
            }

            m_i = m_new;
            l_i = l_new;
        }

        __syncthreads();
    }

    // Final normalization and write to HBM — only write once, no N×N matrix ever stored
    if (valid_q) {
        float inv_l = 1.0f / l_i;
        for (int c = 0; c < FLASH_D; c++)
            O_bh[q_row * FLASH_D + c] = O_reg[c] * inv_l;
    }
}


extern "C" {

// Q, K, V, O: host pointers to (BH, N, FLASH_D) row-major float32 arrays
// BH    : batch_size * num_heads (combined)
// N     : sequence length
// scale : 1 / sqrt(d), precomputed on Python side
// Macro to check every CUDA call and print the failing line before aborting
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

void flashAttentionForward(
    float* Q,
    float* K,
    float* V,
    float* O,
    int BH,
    int N,
    float scale
) {
    // Force CUDA runtime initialisation — needed when pycuda (driver API)
    // has already claimed the device before this runtime-API library loads.
    CUDA_CHECK(cudaSetDevice(0));

    float *d_Q, *d_K, *d_V, *d_O;
    size_t bytes = (size_t)BH * N * FLASH_D * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, bytes));

    int Tr = (N + FLASH_BR - 1) / FLASH_BR;
    dim3 grid(BH, Tr);
    dim3 block(FLASH_BR);

    flashAttentionForwardKernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, scale);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O, d_O, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

} // extern "C"
