#include "cuda_1_utils.cuh"

// ============================================================================
// 1. Naive GPU SGEMM
// ============================================================================
__global__ void naiveSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

// ============================================================================
// 2. SGEMM V1: Tiling in Shared Memory
// ============================================================================
__global__ void sgemm_V1(float *a, float *b, float *c, const int M, const int N, const int K) {
    const int BM = 32, BN = 32, BK = 32;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c = 0.0;
    int load_a_smem_m = tid / 32;
    int load_a_smem_k = tid % 32;
    int load_b_smem_k = tid / 32;
    int load_b_smem_n = tid % 32;

    int load_a_gmem_m = by * 32 + load_a_smem_m;
    int load_b_gmem_n = bx * 32 + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        s_a[load_a_smem_m][load_a_smem_k] = a[load_a_gmem_addr];

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        s_b[load_b_smem_k][load_b_smem_n] = b[load_b_gmem_addr];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            r_c += s_a[ty][k] * s_b[k][tx];
        }
        __syncthreads();
    }
    c[OFFSET(by * BM + ty, bx * BN + tx, N)] = r_c;
}

// ============================================================================
// 3. SGEMM V2: 1D Block Tiling with Register Privatization
// ============================================================================
__global__ void sgemm_V2(float *a, float *b, float *c, const int M, const int N, const int K) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid % 128;
    int load_a_smem_k = tid / 128;
    int load_b_smem_k = tid / 128;
    int load_b_smem_n = tid % 128;

    int load_a_gmem_m = by * 128 + load_a_smem_m;
    int load_b_gmem_n = bx * 128 + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
        s_b[load_b_smem_k][load_b_smem_n] = r_load_b[0];
        s_b[load_b_smem_k + 1][load_b_smem_n] = r_load_b[1];
        s_b[load_b_smem_k + 2][load_b_smem_n] = r_load_b[2];
        s_b[load_b_smem_k + 3][load_b_smem_n] = r_load_b[3];
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * 128 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * 128 + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * 128 + ty * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = bx * 128 + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

// ============================================================================
// 4. SGEMM V3: Double Buffering (Ping-Pong Buffer)
// ============================================================================
__global__ void sgemm_V3(float *a, float *b, float *c, const int M, const int N, const int K) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4], r_load_b[4];
    float r_comp_a[TM], r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid % 128;
    int load_a_smem_k = tid / 128;
    int load_b_smem_k = tid / 128;
    int load_b_smem_n = tid % 128;

    int load_a_gmem_m = by * 128 + load_a_smem_m;
    int load_b_gmem_n = bx * 128 + load_b_smem_n;

    // Load first tile
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
        s_b[0][load_b_smem_k][load_b_smem_n] = r_load_b[0];
        s_b[0][load_b_smem_k + 1][load_b_smem_n] = r_load_b[1];
        s_b[0][load_b_smem_k + 2][load_b_smem_n] = r_load_b[2];
        s_b[0][load_b_smem_k + 3][load_b_smem_n] = r_load_b[3];
    }
    __syncthreads();

    int smem_sel = 0, smem_sel_next = 1;
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        // Asynchronously load next tile
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // Compute current tile
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        s_b[smem_sel_next][load_b_smem_k][load_b_smem_n] = r_load_b[0];
        s_b[smem_sel_next][load_b_smem_k + 1][load_b_smem_n] = r_load_b[1];
        s_b[smem_sel_next][load_b_smem_k + 2][load_b_smem_n] = r_load_b[2];
        s_b[smem_sel_next][load_b_smem_k + 3][load_b_smem_n] = r_load_b[3];
        __syncthreads();

        smem_sel = smem_sel_next;
        smem_sel_next = !smem_sel_next;
    }

    // Compute last tile
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * 128 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * 128 + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * 128 + ty * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = bx * 128 + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}