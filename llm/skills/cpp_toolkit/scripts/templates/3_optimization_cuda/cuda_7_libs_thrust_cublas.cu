/*
 * CUDA 高阶库使用范式 (cuBLAS, Thrust)
 * 提炼自 cuda-sample/src 
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <numeric>

// ============================================================================
// 1. cuBLAS 矩阵乘法 (SGEMM) 范式
// ============================================================================
// cuBLAS 默认使用列主序 (Column-Major)，而在 C++ 中通常是行主序 (Row-Major)。
// 为了在不转置数据的情况下计算 C = A * B (行主序)：
// 利用转置的性质：C^T = (A * B)^T = B^T * A^T
// 因此，我们将 B 作为第一个参数传入，A 作为第二个参数，且不进行转置 (CUBLAS_OP_N)。
void cublas_sgemm_example() {
    constexpr size_t M = 200;
    constexpr size_t N = 400;
    constexpr size_t K = 300;

    std::vector<float> mat_a(M * K, 1.0f);
    std::vector<float> mat_b(K * N, 2.0f);
    std::vector<float> mat_c(M * N, 0.0f);

    float* device_mat_a = nullptr;
    float* device_mat_b = nullptr;
    float* device_mat_c = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&device_mat_a), M * K * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&device_mat_b), K * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&device_mat_c), M * N * sizeof(float));

    // 1. 创建 cuBLAS 句柄
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);

    // 2. 将数据从 Host 拷贝到 Device (这里假设已经是连续内存)
    // 也可以使用 cublasSetMatrix
    cudaMemcpy(device_mat_a, mat_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat_b, mat_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta  = 0.0f;

    // 3. 执行 SGEMM
    // 注意：因为 C++ 是行主序，我们交换 A 和 B 的顺序，使得计算出的是正确的行主序 C
    // cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                device_mat_b, N, // B 作为第一个参数，LDA=N
                device_mat_a, K, // A 作为第二个参数，LDB=K
                &beta, 
                device_mat_c, N); // LDC=N

    // 4. 拷贝回 Host
    cudaMemcpy(mat_c.data(), device_mat_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. 销毁句柄并释放内存
    cublasDestroy(handle);
    cudaFree(device_mat_a);
    cudaFree(device_mat_b);
    cudaFree(device_mat_c);
}

// ============================================================================
// 2. Thrust 库常用范式 (类似 C++ STL)
// ============================================================================
void thrust_examples() {
    int num_elements = 1024;

    // 1. Host Vector 和 Device Vector 自动内存管理
    thrust::host_vector<float> h_vec(num_elements);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // 隐式拷贝：从 Host 到 Device
    thrust::device_vector<float> d_vec = h_vec;

    // 2. 并行归约 (Reduction)
    // thrust::reduce 默认在设备端执行高效的归约算法
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    // 3. 并行排序 (Sorting)
    // 默认使用 Radix Sort (基数排序) 或 Merge Sort (归并排序)，极其高效
    thrust::sort(d_vec.begin(), d_vec.end());

    // 4. 并行 Transform (例如：将每个元素乘以 2)
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), 
                      [] __device__ (float x) { return x * 2.0f; });

    // 5. 隐式拷贝：从 Device 回 Host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
