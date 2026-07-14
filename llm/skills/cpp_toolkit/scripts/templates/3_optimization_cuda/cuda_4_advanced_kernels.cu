#include <cuda_runtime.h>
#include <math.h>
// Note: cutlass headers would be required for the dump utility to compile in a real project
// #include "cutlass/cutlass.h"
// #include "cutlass/util/debug.h"

// ============================================================================
// 1. Flash Attention (Forward Pass Core)
// ============================================================================
constexpr int d = 128;
constexpr int B_r = 8;
constexpr int B_c = 32;
constexpr int block_dim_x = 128;
constexpr int block_dim_y = 8;
constexpr int d_over_bdx = d / block_dim_x;
constexpr int B_r_over_bdy = B_r / block_dim_y;

__global__ void flash_attention_k(
    float *out, float *out_l, 
    float *Q, float *K, float *V, 
    float scaling, int n, int T_r, int T_c
) {
    int tid_x = threadIdx.x, tid_y = threadIdx.y; 
    int bdy = blockDim.y, bdx = blockDim.x, bix = blockIdx.x; 

    // 利用共享内存加载 Q/K/V 分块
    __shared__ float Q_i[B_r][d];
    __shared__ float K_j[B_c][d];
    __shared__ float V_j[B_c][d];
    __shared__ float S[B_r][B_c];

    // 利用寄存器(Registers)私有化累加器，避免反复写回全局内存
    float l_i[B_r_over_bdy];
    float m_i[B_r_over_bdy];
    float O_i[B_r_over_bdy][d_over_bdx];

    for (int i = bix; i < bix + 1; i++) {
        for (int ii = tid_y; ii < B_r; ii += bdy) {
            for (int dd = tid_x; dd < d; dd += bdx) {
                Q_i[ii][dd] = Q[(ii + i * B_r) * d + dd];
            }
        }
        for (int ii = 0; ii < B_r_over_bdy; ii ++) {
            for (int dd = 0; dd < d_over_bdx; dd ++) O_i[ii][dd] = 0.f;
            l_i[ii] = 0.f;
            m_i[ii] = 1e-30f;
        }

        for (int j = 0; j < T_c; j++){
            for (int jj = tid_y; jj < B_c; jj += bdy) {
                for (int dd = tid_x; dd < d; dd += bdx) {
                    K_j[jj][dd] = K[(jj + j * B_c) * d + dd];
                    V_j[jj][dd] = V[(jj + j * B_c) * d + dd];
                }
            }
            __syncthreads();
            
            for (int ii = tid_x; ii < B_r; ii += bdx) {
                for (int jj = tid_y; jj < B_c; jj += bdy) {
                    float S_ij = 0.0f;
                    for (int dd = 0; dd < d; dd ++) S_ij += Q_i[ii][dd] * K_j[jj][dd];
                    S[ii][jj] = scaling * S_ij;
                }
            }
            __syncthreads();
            
            for (int ii = 0; ii < B_r_over_bdy; ii ++) {
                float m = m_i[ii];
                float last_m = m;
                for (int jj = 0; jj < B_c; jj++) {
                    if (m < S[ii * bdy + tid_y][jj]) m = S[ii * bdy + tid_y][jj];
                }
                m_i[ii] = m;
                float l = expf(last_m - m) * l_i[ii];

                for (int dd = 0; dd < d_over_bdx; dd ++) O_i[ii][dd] *= expf(last_m - m);
                
                for (int jj = 0; jj < B_c; jj++) {
                    float P_ij = expf(S[ii * bdy + tid_y][jj] - m);
                    l += P_ij;
                    for (int dd = 0; dd < d_over_bdx; dd ++) {
                        O_i[ii][dd] +=  P_ij * V_j[jj][dd * bdx + tid_x];
                    }
                }
                l_i[ii] = l;
            }
        }
        
        for (int ii = 0; ii < B_r_over_bdy; ii ++) {
            for (int dd = 0; dd < d_over_bdx; dd ++) {
                out[(ii * bdy + tid_y + i * B_r) * d + dd * bdx + tid_x] = O_i[ii][dd] / l_i[ii];
            }
            out_l[ii * bdy + tid_y + i * B_r] = m_i[ii] + logf(l_i[ii]);
        }
        __syncthreads();
    }
}

// ============================================================================
// 2. Cutlass Debugging Utilities
// ============================================================================
/*
// Example of how to dump CUTLASS fragments and shared memory.
// Commented out to avoid compilation errors without cutlass headers.

template <typename Fragment>
__device__ void dump_cutlass_fragment(Fragment const& frag, const char* name) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("--- Dumping Fragment: %s ---\n", name);
        cutlass::debug::dump_fragment(frag);
    }
    __syncthreads();
}

template <typename SharedStorage>
__device__ void dump_cutlass_shmem(SharedStorage const& shmem, const char* name) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("--- Dumping Shared Memory: %s ---\n", name);
        cutlass::debug::dump_shmem(shmem);
    }
    __syncthreads();
}
*/