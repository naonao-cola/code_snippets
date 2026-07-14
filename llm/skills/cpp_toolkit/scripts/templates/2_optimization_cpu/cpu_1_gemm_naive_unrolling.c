/*
 * CPU GEMM 优化演进: 第一阶段 - 基础实现与初步循环展开
 * 参考 FLAME "How To Optimize GEMM" 教程
 *
 * 此阶段展示了最原始的矩阵乘法，以及通过提取内积和初步循环展开来减少分支开销。
 */

// 矩阵寻址宏 (列主序 Column-major)
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

// ============================================================================
// 1. MMult0: 最朴素的三层循环 (Naive)
// ============================================================================
// 问题：对 A 的访存是跨步的 (stride = lda)，导致严重的 Cache Miss。
void MY_MMult0( int m, int n, int k, double *a, int lda, 
                                     double *b, int ldb,
                                     double *c, int ldc )
{
  int i, j, p;
  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      for ( p=0; p<k; p++ ){
        C( i,j ) = C( i,j ) +  A( i,p ) * B( p,j );
      }
    }
  }
}

// ============================================================================
// 2. MMult1 & MMult2: 提取 AddDot 并展开外层循环 (Loop Unrolling)
// ============================================================================
// 提取内积为独立函数
#define X(i) x[ (i)*incx ]
void AddDot( int k, double *x, int incx,  double *y, double *gamma ) {
  int p;
  for ( p=0; p<k; p++ ){
    *gamma += X(p) * y[p];
  }
}

// MMult2: 将外层 j 循环展开 4 次
void MY_MMult2( int m, int n, int k, double *a, int lda, 
                                     double *b, int ldb,
                                     double *c, int ldc )
{
  int i, j;
  for ( j=0; j<n; j+=4 ){        // Unrolled by 4
    for ( i=0; i<m; i+=1 ){
      AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );
      AddDot( k, &A( i,0 ), lda, &B( 0,j+1 ), &C( i,j+1 ) );
      AddDot( k, &A( i,0 ), lda, &B( 0,j+2 ), &C( i,j+2 ) );
      AddDot( k, &A( i,0 ), lda, &B( 0,j+3 ), &C( i,j+3 ) );
    }
  }
}
