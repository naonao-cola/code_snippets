#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Note: NVTX is a profiling tool from NVIDIA
// #include <nvtx3/nvToolsExt.h>

// ============================================================================
// 1. 耗时统计工具 (Timing Tools & GPU Timer)
// ============================================================================
// CPU 端高精度计时宏
#ifndef TICK
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#endif
#ifndef TOCK
#define TOCK(x) printf("%s: %lfs\n", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());
#endif

// GPU 端基于 cudaEvent 的高精度计时器封装
struct GpuTimer {
    cudaStream_t _stream_id;
    cudaEvent_t  _start;
    cudaEvent_t  _stop;

    GpuTimer() : _stream_id(0) {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }
    void start(cudaStream_t stream_id = 0) {
        _stream_id = stream_id;
        cudaEventRecord(_start, _stream_id);
    }
    void stop() { 
        cudaEventRecord(_stop, _stream_id); 
    }
    float elapsed_millis() {
        float elapsed = 0.0;
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&elapsed, _start, _stop);
        return elapsed;
    }
};

// ============================================================================
// 2. CUDA Graph 捕获与回放 (CUDA Graph Capture & Launch)
// ============================================================================
// 适用场景：大量小 Kernel 频繁 Launch 导致 CPU 端 launch overhead 成为瓶颈
// 解决方案：使用 Stream Capture 将 Kernel 和 Memcpy 录制为图，一次性提交给 GPU
__global__ void graph_kernel1(float* out, const float* in, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) out[idx] = in[idx] + 1.0f;
}

__global__ void graph_kernel2(float* out, const float* in, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) out[idx] = in[idx] * 2.0f;
}

inline void test_cuda_graph(float* d_in, float* h_in, float* d_out, float* h_out, size_t size, int numElements) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 1. 声明图与实例
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // 2. 开始捕获 (Capture)
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
    graph_kernel1<<<1, numElements, 0, stream>>>(d_out, d_in, numElements);
    graph_kernel2<<<1, numElements, 0, stream>>>(d_in, d_out, numElements);
    cudaMemcpyAsync(h_out, d_in, size, cudaMemcpyDeviceToHost, stream);
    
    // 3. 结束捕获
    cudaStreamEndCapture(stream, &graph);

    // 4. 实例化 CUDA 图
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // 5. NVTX Profiling 埋点 (供 Nsight Systems 分析)
    // nvtxRangePushA("cuda_graph_launch_loop");
    for (int i = 0; i < 10; ++i) {
        // 重复 Launch 实例化后的图，消除 Launch 延迟
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
    }
    // nvtxRangePop();

    // 清理资源
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
}

// ============================================================================
// 3. 常用生态库错误检查宏 (cuBLAS, cuRAND, cuFFT, cuSPARSE, CUTLASS)
// ============================================================================
#define CHECK_CUBLAS(call) \
    { cublasStatus_t err; if ((err = (call)) != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); exit(1); } }

#define CHECK_CURAND(call) \
    { curandStatus_t err; if ((err = (call)) != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, __LINE__); exit(1); } }
