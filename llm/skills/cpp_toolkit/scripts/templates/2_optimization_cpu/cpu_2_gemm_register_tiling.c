/*
 * CPU GEMM 优化演进: 第二阶段 - 寄存器分块 (Register Tiling) 与指针运算
 * 参考 FLAME "How To Optimize GEMM" 教程 (MMult_4x4_5 ~ MMult_4x4_9)
 */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

// ============================================================================
// 1. MMult_4x4_5: 4x4 微内核的雏形 (合并内积)
// ============================================================================
// 我们不再一次计算一个点，而是一次性计算 C 的 4x4 块 (16个元素)
void AddDot4x4_Basic( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc ) {
  int p;
  for ( p=0; p<k; p++ ){
    // 第一行
    C( 0, 0 ) += A( 0, p ) * B( p, 0 ); C( 0, 1 ) += A( 0, p ) * B( p, 1 );
    C( 0, 2 ) += A( 0, p ) * B( p, 2 ); C( 0, 3 ) += A( 0, p ) * B( p, 3 );
    // 第二行
    C( 1, 0 ) += A( 1, p ) * B( p, 0 ); C( 1, 1 ) += A( 1, p ) * B( p, 1 );
    C( 1, 2 ) += A( 1, p ) * B( p, 2 ); C( 1, 3 ) += A( 1, p ) * B( p, 3 );
    // 第三、四行同理...
    C( 2, 0 ) += A( 2, p ) * B( p, 0 ); C( 2, 1 ) += A( 2, p ) * B( p, 1 );
    C( 2, 2 ) += A( 2, p ) * B( p, 2 ); C( 2, 3 ) += A( 2, p ) * B( p, 3 );
    C( 3, 0 ) += A( 3, p ) * B( p, 0 ); C( 3, 1 ) += A( 3, p ) * B( p, 1 );
    C( 3, 2 ) += A( 3, p ) * B( p, 2 ); C( 3, 3 ) += A( 3, p ) * B( p, 3 );
  }
}

// ============================================================================
// 2. MMult_4x4_9: 纯 C 极致优化 (指针偏移 + 寄存器驻留 + 循环展开)
// ============================================================================
// 通过显式声明 register 和指针自增，消除数组下标计算开销，压榨 CPU 寄存器
void AddDot4x4_PureC( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc ) {
  int p;
  register double 
    // 累加器驻留寄存器 (16个)
    c_00_reg=0.0, c_01_reg=0.0, c_02_reg=0.0, c_03_reg=0.0,  
    c_10_reg=0.0, c_11_reg=0.0, c_12_reg=0.0, c_13_reg=0.0,  
    c_20_reg=0.0, c_21_reg=0.0, c_22_reg=0.0, c_23_reg=0.0,  
    c_30_reg=0.0, c_31_reg=0.0, c_32_reg=0.0, c_33_reg=0.0,
    // 数据缓存寄存器
    a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
    b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;

  double *b_p0_pntr = &B(0, 0);
  double *b_p1_pntr = &B(0, 1);
  double *b_p2_pntr = &B(0, 2);
  double *b_p3_pntr = &B(0, 3);

  for ( p=0; p<k; p++ ){
    a_0p_reg = A( 0, p ); a_1p_reg = A( 1, p );
    a_2p_reg = A( 2, p ); a_3p_reg = A( 3, p );

    b_p0_reg = *b_p0_pntr++; b_p1_reg = *b_p1_pntr++;
    b_p2_reg = *b_p2_pntr++; b_p3_reg = *b_p3_pntr++;

    c_00_reg += a_0p_reg * b_p0_reg; c_10_reg += a_1p_reg * b_p0_reg;
    c_01_reg += a_0p_reg * b_p1_reg; c_11_reg += a_1p_reg * b_p1_reg;
    c_02_reg += a_0p_reg * b_p2_reg; c_12_reg += a_1p_reg * b_p2_reg;
    c_03_reg += a_0p_reg * b_p3_reg; c_13_reg += a_1p_reg * b_p3_reg;

    c_20_reg += a_2p_reg * b_p0_reg; c_30_reg += a_3p_reg * b_p0_reg;
    c_21_reg += a_2p_reg * b_p1_reg; c_31_reg += a_3p_reg * b_p1_reg;
    c_22_reg += a_2p_reg * b_p2_reg; c_32_reg += a_3p_reg * b_p2_reg;
    c_23_reg += a_2p_reg * b_p3_reg; c_33_reg += a_3p_reg * b_p3_reg;
  }

  C(0, 0) += c_00_reg; C(0, 1) += c_01_reg; C(0, 2) += c_02_reg; C(0, 3) += c_03_reg;
  C(1, 0) += c_10_reg; C(1, 1) += c_11_reg; C(1, 2) += c_12_reg; C(1, 3) += c_13_reg;
  C(2, 0) += c_20_reg; C(2, 1) += c_21_reg; C(2, 2) += c_22_reg; C(2, 3) += c_23_reg;
  C(3, 0) += c_30_reg; C(3, 1) += c_31_reg; C(3, 2) += c_32_reg; C(3, 3) += c_33_reg;
}
