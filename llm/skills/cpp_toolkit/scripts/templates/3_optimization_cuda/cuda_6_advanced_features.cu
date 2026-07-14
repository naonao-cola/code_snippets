/*
 * CUDA 高级特性与并行模式 (来自 cuda-sample/src 的提炼)
 * 包含：Warp-Level 原语、协作组 (Cooperative Groups)、原子操作防冲突、动态共享内存与 Zero-Copy
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// ============================================================================
// 1. 归约优化演进 (Reduction Evolutions)
// ============================================================================

// 1.1 消除 Warp Divergence 的归约 (Interleaved Reduction)
// 原理：通过改变 stride 方式，使得活跃的线程都在同一个 Warp 内，减少控制流发散
__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    // 步长从 blockDim.x/2 开始递减，保证活跃线程集中在前面
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 1.2 极限优化：循环展开 (Unrolling) + Warp Shuffle 归约
// 结合了取数阶段的 8 展开与最后的 Warp 级硬编码归约，性能拉满
__global__ void reduceCompleteUnrollWarps8(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 第一步：Unrolling 8 (每个线程处理8个数据)
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx], a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x], a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x], b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x], b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    // 第二步：Block 级归约 (只处理到 64，剩下的给 Warp)
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512]; __syncthreads();
    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256]; __syncthreads();
    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128]; __syncthreads();
    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64]; __syncthreads();

    // 第三步：Warp 级展开 (无需 __syncthreads，因为在同一个 Warp 内执行步调是一致的)
    if (tid < 32) {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// ============================================================================
// 2. Cooperative Groups (协作组) 与 Warp Shuffle
// ============================================================================
// 利用 cg::thread_block_tile<32> 进行优雅的 Block 内归约
#if __CUDA_ARCH__ >= 700
template<bool writeSquareRoot>
__device__ void reduceBlockData(cuda::barrier<cuda::thread_scope_block>& barrier, 
                                cg::thread_block_tile<32>& tile32, 
                                double& threadSum, double* result) 
{
    extern __shared__ double tmp[];
    
    // Warp 级别的蝶形归约 (Shuffle Down)
    #pragma unroll
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        threadSum += tile32.shfl_down(threadSum, offset);
    }
    // 每个 Warp 的 leader 写入 Shared Memory
    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = threadSum;
    }

    auto token = barrier.arrive();
    barrier.wait(std::move(token));

    // 让 Warp 0 来完成最后一轮合并
    if (tile32.meta_group_rank() == 0) {
        double beta = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
        #pragma unroll
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            beta += tile32.shfl_down(beta, offset);
        }
        if (tile32.thread_rank() == 0) {
            *result = writeSquareRoot ? sqrt(beta) : beta;
        }
    }
}
#endif

// ============================================================================
// 3. 动态共享内存与 Bank Conflict 防范
// ============================================================================
#define BDIMX 32
#define BDIMY 16
#define IPAD 2 // Padding 用于打破 Bank Conflict

__global__ void setRowReadColDynPad(int* out) {
    // extern 声明动态共享内存（大小在 kernel launch 时指定）
    extern __shared__ int tile[];
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;

    // 写入时加上 IPAD，使得同一列的数据被错开存入不同的 Bank
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    tile[row_idx] = g_idx;
    __syncthreads();
    out[g_idx] = tile[col_idx];
}

// ============================================================================
// 4. 原子操作的 CAS 自旋锁实现 (Atomic Compare-And-Swap)
// ============================================================================
// 适用场景：标准 atomicAdd 不支持的数据类型（如 double 在老架构下），或自定义的归约逻辑
__device__ int myAtomicAdd(int* address, int incr) {
    int guess = *address; // Initial guess
    int oldValue = atomicCAS(address, guess, guess + incr);
    // 自旋直到 CAS 成功
    while (oldValue != guess) {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }
    return oldValue;
}

// ============================================================================
// 5. Zero-Copy 内存 (Mapped Memory)
// ============================================================================
// 直接将主机内存映射到设备空间，GPU 读写时通过 PCIe 实时传输，省去了显式的 Memcpy。
// 适用场景：数据只读/写一次，且数据量不大，或者 APU/Jetson 等物理内存与显存同一块板子的设备。
void zero_copy_example() {
    float *h_A, *d_A;
    size_t nBytes = 1024 * sizeof(float);
    
    // 1. 分配页锁定 (Pinned) 且可映射 (Mapped) 的内存
    cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped);
    
    // 2. 获取该主机内存在设备端的指针
    cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0);
    
    // 3. 直接传入 Kernel 即可，无需 cudaMemcpy
    // kernel<<<grid, block>>>(d_A);
    
    cudaFreeHost(h_A);
}
