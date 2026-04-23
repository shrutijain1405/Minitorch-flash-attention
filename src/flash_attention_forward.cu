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
// FA1 Forward Kernel  (Algorithm 1, Dao et al. 2022 — original loop order)
//
// Outer loop: K/V tiles (j = 0..Tc-1)  — K_j, V_j loaded into shared memory
// Inner loop: Q tiles   (i = 0..Tr-1)  — Q_i, O_i reloaded from HBM each time
//
// O_i is NORMALIZED at every inner step and written back to HBM.
// This costs more HBM traffic than FA2 (Q read Tc times, O r/w Tc times)
// but is the algorithm as written in the original paper.
//
// Grid : (BH,)  — one block per batch*head; both loops are serial inside
// Block: (Br,)  — one thread per row in the current tile
// ---------------------------------------------------------------------------
__global__ void flashAttentionForwardKernelFA1(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* O,
    float* L,       // output: (BH, N) logsumexp — saved for backward
    float* M_run,   // scratch: (BH, N) running row-max
    float* L_run,   // scratch: (BH, N) running softmax denominator
    int N,
    float scale
) {
    int bh  = blockIdx.x;
    int tid = threadIdx.x;

    const float* Q_bh  = Q     + bh * N * FLASH_D;
    const float* K_bh  = K     + bh * N * FLASH_D;
    const float* V_bh  = V     + bh * N * FLASH_D;
    float*       O_bh  = O     + bh * N * FLASH_D;
    float*       L_bh  = L     + bh * N;
    float*       M_bh  = M_run + bh * N;
    float*       Ls_bh = L_run + bh * N;

    // K and V tiles stay in shared memory for the entire outer-loop iteration
    __shared__ float K_tile[FLASH_BC][FLASH_D];
    __shared__ float V_tile[FLASH_BC][FLASH_D];

    int Tr = (N + FLASH_BR - 1) / FLASH_BR;
    int Tc = (N + FLASH_BC - 1) / FLASH_BC;

    for (int j = 0; j < Tc; j++) {             // ── outer: K/V tiles ──

        // Load K_j, V_j cooperatively (thread tid owns row tid of the tile)
        int k_row = j * FLASH_BC + tid;
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
        __syncthreads();    // K_tile / V_tile ready before any thread reads them

        for (int i = 0; i < Tr; i++) {          // ── inner: Q tiles ──

            int q_row  = i * FLASH_BR + tid;
            bool valid = (q_row < N);

            // Reload Q_i row, current O_i row, and running m/l from HBM
            float q_reg[FLASH_D], o_reg[FLASH_D];
            float m_i = -1e9f, l_i = 0.0f;

            if (valid) {
                for (int c = 0; c < FLASH_D; c++) {
                    q_reg[c] = Q_bh[q_row * FLASH_D + c];
                    o_reg[c] = O_bh[q_row * FLASH_D + c];
                }
                m_i = M_bh[q_row];
                l_i = Ls_bh[q_row];
            }

            if (valid) {
                // Pass 1: local rowmax m_ij over K/V tile j
                float m_ij = -1e9f;
                for (int jj = 0; jj < FLASH_BC; jj++) {
                    if (j * FLASH_BC + jj >= N) break;
                    float s = 0.0f;
                    for (int c = 0; c < FLASH_D; c++)
                        s += q_reg[c] * K_tile[jj][c];
                    m_ij = fmaxf(m_ij, s * scale);
                }

                // Pass 2: local P_ij @ V_j and row-sum l_ij
                float l_ij = 0.0f;
                float pv[FLASH_D];
                for (int c = 0; c < FLASH_D; c++) pv[c] = 0.0f;

                for (int jj = 0; jj < FLASH_BC; jj++) {
                    if (j * FLASH_BC + jj >= N) break;
                    float s = 0.0f;
                    for (int c = 0; c < FLASH_D; c++)
                        s += q_reg[c] * K_tile[jj][c];
                    float p = expf(s * scale - m_ij);
                    l_ij += p;
                    for (int c = 0; c < FLASH_D; c++)
                        pv[c] += p * V_tile[jj][c];
                }

                // FA1 per-step normalization (Algorithm 1, line 12)
                // O_i ← diag(l_new)^{-1} * (exp(m_i - m_new)*l_i*O_i + exp(m_ij - m_new)*P_ij*V_j)
                float m_new    = fmaxf(m_i, m_ij);
                float corr_old = expf(m_i  - m_new);
                float corr_new = expf(m_ij - m_new);
                float l_new    = corr_old * l_i + corr_new * l_ij;
                float inv_l    = 1.0f / l_new;

                for (int c = 0; c < FLASH_D; c++)
                    o_reg[c] = (corr_old * l_i * o_reg[c] + corr_new * pv[c]) * inv_l;

                // Write normalized O_i, updated m and l back to HBM
                for (int c = 0; c < FLASH_D; c++)
                    O_bh[q_row * FLASH_D + c] = o_reg[c];
                M_bh[q_row]  = m_new;
                Ls_bh[q_row] = l_new;
            }
        }

        __syncthreads();    // all threads done reading K_tile/V_tile before next j
    }

    // Final pass: compute logsumexp L = m + log(l) for backward
    for (int i = 0; i < Tr; i++) {
        int q_row = i * FLASH_BR + tid;
        if (q_row < N)
            L_bh[q_row] = M_bh[q_row] + logf(Ls_bh[q_row]);
    }
}


// ---------------------------------------------------------------------------
// FA2 Forward Kernel  (outer Q, inner K/V, deferred normalization)
//
// Grid : (BH, Tr) — one block per (batch*head, Q-tile)
// Block: (Br,)    — one thread per row in the Q-tile
// O is normalized only once at the very end (one HBM write per Q tile).
// ---------------------------------------------------------------------------
__global__ void flashAttentionForwardKernelFA2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* O,
    float* L,
    int N,
    float scale
) {
    int bh     = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid    = threadIdx.x;
    int q_row  = q_tile * FLASH_BR + tid;

    const float* Q_bh = Q + bh * N * FLASH_D;
    const float* K_bh = K + bh * N * FLASH_D;
    const float* V_bh = V + bh * N * FLASH_D;
    float*       O_bh = O + bh * N * FLASH_D;
    float*       L_bh = L + bh * N;

    __shared__ float K_tile[FLASH_BC][FLASH_D];
    __shared__ float V_tile[FLASH_BC][FLASH_D];

    // Load Q row into registers once — never reloaded from HBM
    float q_reg[FLASH_D];
    bool valid_q = (q_row < N);
    if (valid_q) {
        for (int c = 0; c < FLASH_D; c++)
            q_reg[c] = Q_bh[q_row * FLASH_D + c];
    }

    // Unnormalized O accumulator — normalized only at the end
    float O_reg[FLASH_D];
    for (int c = 0; c < FLASH_D; c++) O_reg[c] = 0.0f;
    float m_i = -1e9f;
    float l_i = 0.0f;

    int Tc = (N + FLASH_BC - 1) / FLASH_BC;

    for (int j = 0; j < Tc; j++) {
        int k_base = j * FLASH_BC;

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
            // Pass 1: local rowmax
            float m_ij = -1e9f;
            for (int jj = 0; jj < FLASH_BC; jj++) {
                if (k_base + jj >= N) break;
                float dot = 0.0f;
                for (int c = 0; c < FLASH_D; c++)
                    dot += q_reg[c] * K_tile[jj][c];
                m_ij = fmaxf(m_ij, dot * scale);
            }

            // FA2: rescale existing accumulator, then accumulate new tile
            float m_new = fmaxf(m_i, m_ij);
            float alpha = expf(m_i - m_new);
            float l_new = alpha * l_i;
            for (int c = 0; c < FLASH_D; c++)
                O_reg[c] *= alpha;

            // Pass 2: accumulate P @ V (O stays unnormalized)
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

    // Single normalization + HBM write at the end
    if (valid_q) {
        float inv_l = 1.0f / l_i;
        for (int c = 0; c < FLASH_D; c++)
            O_bh[q_row * FLASH_D + c] = O_reg[c] * inv_l;
        L_bh[q_row] = m_i + logf(l_i);
    }
}


extern "C" {

// ---------------------------------------------------------------------------
// FA1 host wrapper
// Allocates M_run / L_run scratch, initializes them, launches FA1 kernel.
// ---------------------------------------------------------------------------
void flashAttentionForwardFA1(
    float* Q, float* K, float* V,
    float* O, float* L,
    int BH, int N, float scale
) {
    CUDA_CHECK(cudaSetDevice(0));

    size_t bytes   = (size_t)BH * N * FLASH_D * sizeof(float);
    size_t l_bytes = (size_t)BH * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_M_run, *d_L_run;
    CUDA_CHECK(cudaMalloc(&d_Q,     bytes));
    CUDA_CHECK(cudaMalloc(&d_K,     bytes));
    CUDA_CHECK(cudaMalloc(&d_V,     bytes));
    CUDA_CHECK(cudaMalloc(&d_O,     bytes));
    CUDA_CHECK(cudaMalloc(&d_L,     l_bytes));
    CUDA_CHECK(cudaMalloc(&d_M_run, l_bytes));
    CUDA_CHECK(cudaMalloc(&d_L_run, l_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O,     0, bytes));
    CUDA_CHECK(cudaMemset(d_L,     0, l_bytes));
    CUDA_CHECK(cudaMemset(d_L_run, 0, l_bytes));   // l starts at 0

    // Initialize M_run to -1e9 (can't use memset — not a zero pattern)
    {
        float* h_init = (float*)malloc(l_bytes);
        for (size_t idx = 0; idx < (size_t)BH * N; idx++) h_init[idx] = -1e9f;
        CUDA_CHECK(cudaMemcpy(d_M_run, h_init, l_bytes, cudaMemcpyHostToDevice));
        free(h_init);
    }

    dim3 grid(BH);
    dim3 block(FLASH_BR);

    flashAttentionForwardKernelFA1<<<grid, block>>>(
        d_Q, d_K, d_V, d_O, d_L, d_M_run, d_L_run, N, scale
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O, d_O, bytes,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L, d_L, l_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    cudaFree(d_M_run); cudaFree(d_L_run);
}

// ---------------------------------------------------------------------------
// FA2 host wrapper  (original entry point, unchanged interface)
// ---------------------------------------------------------------------------
void flashAttentionForwardFA2(
    float* Q, float* K, float* V,
    float* O, float* L,
    int BH, int N, float scale
) {
    CUDA_CHECK(cudaSetDevice(0));

    size_t bytes   = (size_t)BH * N * FLASH_D * sizeof(float);
    size_t l_bytes = (size_t)BH * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));
    CUDA_CHECK(cudaMalloc(&d_L, l_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, bytes));
    CUDA_CHECK(cudaMemset(d_L, 0, l_bytes));

    int Tr = (N + FLASH_BR - 1) / FLASH_BR;
    dim3 grid(BH, Tr);
    dim3 block(FLASH_BR);

    flashAttentionForwardKernelFA2<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_L, N, scale);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O, d_O, bytes,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L, d_L, l_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
}

} // extern "C"
