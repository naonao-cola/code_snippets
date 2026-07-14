/*
 * CUDA 高阶库使用范式 (cuFFT, cuSPARSE)
 * 提炼自 cuda-sample/src/08
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// 1. cuFFT 快速傅里叶变换
// ============================================================================
void cufft_example() {
    int n = 2048;
    cufftHandle plan = 0;
    cufftComplex *dComplexSamples, *complexSamples, *complexFreq;

    complexSamples = (cufftComplex*)malloc(sizeof(cufftComplex) * n);
    complexFreq = (cufftComplex*)malloc(sizeof(cufftComplex) * n);

    // 假设数据初始化
    for (int i = 0; i < n; i++) {
        complexSamples[i].x = cos(i * M_PI / 4.0);
        complexSamples[i].y = 0.0f;
    }

    // Setup the cuFFT plan
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);

    // Allocate device memory
    cudaMalloc((void**)&dComplexSamples, sizeof(cufftComplex) * n);

    // Transfer inputs into device memory
    cudaMemcpy(dComplexSamples, complexSamples, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);

    // Execute a complex-to-complex 1D FFT
    cufftExecC2C(plan, dComplexSamples, dComplexSamples, CUFFT_FORWARD);

    // Retrieve the results into host memory
    cudaMemcpy(complexFreq, dComplexSamples, sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost);

    // Cleanup
    free(complexSamples);
    free(complexFreq);
    cudaFree(dComplexSamples);
    cufftDestroy(plan);
}

// ============================================================================
// 2. cuSPARSE 稀疏矩阵操作
// ============================================================================
void cusparse_example() {
    int M = 1024, N = 1024;
    float *A, *dA, *dX, *X, *dY, *Y;
    int *dNnzPerRow, *dCsrRowPtrA, *dCsrColIndA;
    float *dCsrValA;
    int totalNnz;
    float alpha = 3.0f, beta = 4.0f;
    
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    // Create the cuSPARSE handle
    cusparseCreate(&handle);

    // Construct a descriptor of the matrix A
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // 假设 A 为稠密矩阵，已被分配并拷贝至 dA
    // 假设 X, Y 已被分配并拷贝至 dX, dY
    // 假设 dNnzPerRow 已被分配 (大小为 M)

    /*
    // 1. Compute the number of non-zero elements in A
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dA, M, dNnzPerRow, &totalNnz);

    // 2. Allocate device memory to store the sparse CSR representation of A
    cudaMalloc((void**)&dCsrValA, sizeof(float) * totalNnz);
    cudaMalloc((void**)&dCsrRowPtrA, sizeof(int) * (M + 1));
    cudaMalloc((void**)&dCsrColIndA, sizeof(int) * totalNnz);

    // 3. Convert A from a dense formatting to a CSR formatting, using the GPU
    cusparseSdense2Csr(handle, M, N, descr, dA, M, dNnzPerRow, dCsrValA, dCsrRowPtrA, dCsrColIndA);

    // 4. Perform matrix-vector multiplication with the CSR-formatted matrix A
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, totalNnz, &alpha, descr, 
                   dCsrValA, dCsrRowPtrA, dCsrColIndA, dX, &beta, dY);
    */

    // Cleanup
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
}
