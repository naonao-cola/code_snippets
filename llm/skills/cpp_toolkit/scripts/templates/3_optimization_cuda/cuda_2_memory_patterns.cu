#include <cuda_runtime.h>

// ============================================================================
// 1. 访存合并 (Memory Coalescing)
// ============================================================================
// Non-Coalesced
__global__ void copyDataNonCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[(index * 2) % n]; // 跨步访问导致带宽浪费
    }
}

// Coalesced
__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index]; // Warp内的线程访问连续内存
    }
}

// ============================================================================
// 2. 线程粗化 (Thread Coarsening)
// ============================================================================
__global__ void VecAddCoarsened(float* A, float* B, float* C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; 
    if (i < n)
        C[i] = A[i] + B[i];
    if (i + 1 < n) 
        C[i + 1] = A[i + 1] + B[i + 1];
}

// ============================================================================
// 3. 寄存器私有化 (Privatization)
// ============================================================================
__global__ void vectorAddPrivatized(const float *a, const float *b, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        float a_private = a[index]; // Load into private memory
        float b_private = b[index]; // Load into private memory
        result[index] = a_private + b_private;
    }
}

// ============================================================================
// 4. 共享内存归约 (Shared Memory Reduction)
// ============================================================================
#define BLOCK_DIM 1024
__global__ void SharedMemoryReduction(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    
    input_s[t] = input[t] + input[t  + BLOCK_DIM];
    
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

// ============================================================================
// 5. 共享内存分块矩阵乘法 (Tiling)
// ============================================================================
#define TILE_WIDTH 16
__global__ void matrixMulTiled(float* A, float* B, float* C, int width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float value = 0;
    for (int m = 0; m < width/TILE_WIDTH; ++m) {
        As[ty][tx] = A[Row*width + (m*TILE_WIDTH + tx)];
        Bs[ty][tx] = B[(m*TILE_WIDTH + ty)*width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if(Row < width && Col < width) {
        C[Row*width + Col] = value;
    }
}