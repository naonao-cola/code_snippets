#include "distance_calc.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <iostream>
#include <algorithm>

// 辅助宏用于检查 CUDA 错误
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        } \
    } while (0)


// CUDA 核函数：计算每个点到另一个点云中所有点的最小距离
__global__ void minDistanceKernel(const float* cloudA, int numA, const float* cloudB, int numB, float* min_dists) {
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;

    if (idxA < numA) {
        float ptA_x = cloudA[idxA * 3 + 0];
        float ptA_y = cloudA[idxA * 3 + 1];
        float ptA_z = cloudA[idxA * 3 + 2];

        float local_min_sq = FLT_MAX;

        for (int i = 0; i < numB; ++i) {
            float dx = ptA_x - cloudB[i * 3 + 0];
            float dy = ptA_y - cloudB[i * 3 + 1];
            float dz = ptA_z - cloudB[i * 3 + 2];
            float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < local_min_sq) {
                local_min_sq = dist_sq;
            }
        }

        min_dists[idxA] = local_min_sq;
    }
}

// Host 端封装函数
float calculateMinDistanceCUDA(const float* cloudA, int numA, const float* cloudB, int numB) {
    if (numA == 0 || numB == 0) return FLT_MAX;

    float *d_cloudA = nullptr, *d_cloudB = nullptr, *d_min_dists = nullptr;

    // 1. 分配 GPU 内存
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cloudA, numA * 3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cloudB, numB * 3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_min_dists, numA * sizeof(float)));

    // 2. 将数据从 Host 拷贝到 Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_cloudA, cloudA, numA * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cloudB, cloudB, numB * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // 3. 配置 Kernel 执行参数
    int threadsPerBlock = 512;
    int blocksPerGrid = (numA + threadsPerBlock - 1) / threadsPerBlock;

    // 4. 执行 Kernel
    minDistanceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_cloudA, numA, d_cloudB, numB, d_min_dists);

    // 等待设备完成
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 5. 将结果拷回 Host
    float* h_min_dists = new float[numA];
    CHECK_CUDA_ERROR(cudaMemcpy(h_min_dists, d_min_dists, numA * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. 在 Host 端进行一次简单的归约寻找全局最小值
    float global_min_sq = FLT_MAX;
    for (int i = 0; i < numA; ++i) {
        if (h_min_dists[i] < global_min_sq) {
            global_min_sq = h_min_dists[i];
        }
    }

    // 7. 清理资源
    delete[] h_min_dists;
    cudaFree(d_cloudA);
    cudaFree(d_cloudB);
    cudaFree(d_min_dists);

    return std::sqrt(std::max(0.0f, global_min_sq));
}