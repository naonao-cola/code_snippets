#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

// ============================================================================
// 1. 核心宏工具 (Core Macros)
// ============================================================================
// 向量化访存宏：强制使用 ld.global.v4.f32 128-bit 指令
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// 矩阵寻址偏移宏
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// 2. CPU 参考实现与误差计算 (Error Verification Tools)
// ============================================================================
inline void cpuSgemm(float* a, float* b, float* c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

inline float testError(void (*gpuSgemm)(float*, float*, float*, const int, const int, const int), dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float*)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++) h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++) h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) max_error = -NAN;
        else max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c); free(h_d_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return max_error;
}

// ============================================================================
// 3. 性能压测工具 (Performance Benchmarking Tools)
// ============================================================================
inline float testPerformance(void (*gpuSgemm)(float*, float*, float*, const int, const int, const int), dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return sec;
}