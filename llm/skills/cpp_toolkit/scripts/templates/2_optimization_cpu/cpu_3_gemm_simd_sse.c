/*
 * CPU GEMM 优化演进: 第三阶段 - SIMD 向量化 (SSE 指令集)
 * 参考 FLAME "How To Optimize GEMM" 教程 (MMult_4x4_10 ~ MMult_4x4_12)
 */

#include <xmmintrin.h>   // SSE
#include <pmmintrin.h>   // SSE2
#include <emmintrin.h>   // SSE3
#include <mmintrin.h>

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

// ============================================================================
// MMult_4x4_10/12: 引入 SSE 向量指令
// ============================================================================
// 定义 union 方便在 __m128d 和 double 数组间进行数据提取
typedef union {
    __m128d v;
    double  d[2];
} v2df_t;

void AddDot4x4_SSE(int k, double* a, int lda, double* b, int ldb, double* c, int ldc) {
    int p;
    
    // 向量寄存器：每个 __m128d 可以装载 2 个 double
    v2df_t c_00_c_10_vreg, c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg;
    v2df_t c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg;
    
    v2df_t a_0p_a_1p_vreg, a_2p_a_3p_vreg;
    v2df_t b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

    double *b_p0_pntr = &B(0, 0);
    double *b_p1_pntr = &B(0, 1);
    double *b_p2_pntr = &B(0, 2);
    double *b_p3_pntr = &B(0, 3);

    // 寄存器清零
    c_00_c_10_vreg.v = _mm_setzero_pd(); c_01_c_11_vreg.v = _mm_setzero_pd();
    c_02_c_12_vreg.v = _mm_setzero_pd(); c_03_c_13_vreg.v = _mm_setzero_pd();
    c_20_c_30_vreg.v = _mm_setzero_pd(); c_21_c_31_vreg.v = _mm_setzero_pd();
    c_22_c_32_vreg.v = _mm_setzero_pd(); c_23_c_33_vreg.v = _mm_setzero_pd();

    for (p = 0; p < k; p++) {
        // _mm_load_pd：一次加载 2 个连续 double
        a_0p_a_1p_vreg.v = _mm_load_pd((double*)&A(0, p));
        a_2p_a_3p_vreg.v = _mm_load_pd((double*)&A(2, p));

        // _mm_loaddup_pd：加载 1 个 double 并复制给向量的低位和高位
        b_p0_vreg.v = _mm_loaddup_pd((double*)b_p0_pntr++); 
        b_p1_vreg.v = _mm_loaddup_pd((double*)b_p1_pntr++); 
        b_p2_vreg.v = _mm_loaddup_pd((double*)b_p2_pntr++); 
        b_p3_vreg.v = _mm_loaddup_pd((double*)b_p3_pntr++); 

        // 乘加运算
        c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
        c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
        c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
        c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

        c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
        c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
        c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
        c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
    }

    // 写回主存
    C(0, 0) += c_00_c_10_vreg.d[0]; C(0, 1) += c_01_c_11_vreg.d[0];
    C(0, 2) += c_02_c_12_vreg.d[0]; C(0, 3) += c_03_c_13_vreg.d[0];

    C(1, 0) += c_00_c_10_vreg.d[1]; C(1, 1) += c_01_c_11_vreg.d[1];
    C(1, 2) += c_02_c_12_vreg.d[1]; C(1, 3) += c_03_c_13_vreg.d[1];

    C(2, 0) += c_20_c_30_vreg.d[0]; C(2, 1) += c_21_c_31_vreg.d[0];
    C(2, 2) += c_22_c_32_vreg.d[0]; C(2, 3) += c_23_c_33_vreg.d[0];

    C(3, 0) += c_20_c_30_vreg.d[1]; C(3, 1) += c_21_c_31_vreg.d[1];
    C(3, 2) += c_22_c_32_vreg.d[1]; C(3, 3) += c_23_c_33_vreg.d[1];
}
