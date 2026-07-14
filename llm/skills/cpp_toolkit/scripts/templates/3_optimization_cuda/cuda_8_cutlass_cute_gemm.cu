/*
 * CUTLASS 与 CuTe 核心架构与优化范式
 * 提炼自 cuda-sample/cutlass_mode
 */

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/debug.h>
#include <cutlass/util/device_dump.h>
#include <cute/tensor.hpp>

// ============================================================================
// 1. CUTLASS 经典 Device GEMM (2.x API)
// ============================================================================
// 适用场景：需要快速调用高性能的泛型 GEMM，支持任意精度和布局。
void cutlass_classic_gemm_example(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // 定义 GEMM 算子的模板配置
    using Gemm = cutlass::gemm::device::Gemm<
        float,                           // A 的数据类型
        cutlass::layout::ColumnMajor,    // A 的布局 (列主序)
        float,                           // B 的数据类型
        cutlass::layout::ColumnMajor,    // B 的布局 (列主序)
        float,                           // C 的数据类型
        cutlass::layout::ColumnMajor     // C 的布局 (列主序)
    >;

    Gemm gemm_op;
    // 传入参数：问题规模 (M,N,K)，数据指针及其主维度(Leading Dimension)，以及 alpha/beta
    cutlass::Status status = gemm_op({{M, N, K}, {A, M}, {B, K}, {C, M}, {C, M}, {alpha, beta}});
    if (status != cutlass::Status::kSuccess) {
        // 错误处理
    }
}

// ============================================================================
// 2. CUTLASS HostTensor 与测试工具
// ============================================================================
// 适用场景：编写单元测试、生成随机数据、以及 Host/Device 数据对比。
void cutlass_utilities_example(int M, int N, int K) {
    // 自动管理 Host 和 Device 的内存分配与同步
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});

    // 设备端高斯随机初始化
    cutlass::reference::device::TensorFillRandomGaussian(A.device_view(), 2080 /*seed*/, 0.0_hf /*mean*/, 5.0_hf /*stddev*/, 0);

    // 将设备端数据同步回主机端进行对比
    A.sync_host();
    
    // 使用 TensorEquals 比较 bit 级一致性
    // cutlass::reference::host::TensorEquals(C_reference.host_view(), C_cutlass.host_view());
}

// ============================================================================
// 3. CUTLASS 寄存器与共享内存调试 (Dump Utils)
// ============================================================================
// 适用场景：在内核中直接打印 Fragment (寄存器片段) 和 Shared Memory 数据以排查对齐或数值错误。
template<typename GmemIterator>
__global__ void cutlass_debug_dump_kernel(typename GmemIterator::Params params, typename GmemIterator::TensorRef ref) {
    extern __shared__ cutlass::half_t shared_storage[];
    
    // 初始化迭代器与加载 Fragment
    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    GmemIterator gmem_iterator(params, ref.data(), {64, 32}, tb_thread_id);
    typename GmemIterator::Fragment frag;
    frag.clear();
    gmem_iterator.load(frag);

    // 1. 打印 Fragment
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
    }
    // 参数: (fragment, 打印线程数 N=1, 打印元素数 M=16, 跨度 S=8)
    cutlass::debug::dump_fragment(frag, 1, 16, 8);

    // 2. 打印 Shared Memory
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\nDump shared memory elements:\n");
        // 参数: (shmem_ptr, 元素总数, 跨度 S=8)
        cutlass::debug::dump_shmem(shared_storage, 64 * 32, 8);
    }
}

// ============================================================================
// 4. CuTe 布局与张量操作 (TiledCopy & TiledMMA)
// ============================================================================
// CuTe 是 CUTLASS 3.0 的核心，通过 Layout (Shape, Stride) 代替传统的数组下标计算。
template<class TensorS, class TensorD, class Tiled_Copy>
__global__ void cute_vectorized_copy_kernel(TensorS S, TensorD D, Tiled_Copy tiled_copy) {
    using namespace cute;
    // 切片：获得每个 Block 处理的 Tile
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);

    // 线程级切片：获取当前线程负责拷贝的部分
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    Tensor thr_tile_S = thr_copy.partition_S(tile_S);
    Tensor thr_tile_D = thr_copy.partition_D(tile_D);

    // 构建寄存器片段 (Fragment)
    Tensor fragment = make_fragment_like(thr_tile_D);

    // 自动向量化的拷贝：GMEM -> RMEM -> GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}

// ============================================================================
// 5. SM80 (Ampere) CP.ASYNC 与 LDSM 流水线
// ============================================================================
// 适用场景：极致的 SM80 异步内存加载与 LDSM (Load Matrix) 优化。
void cute_sm80_pipeline_example() {
    using namespace cute;

    // 定义异步拷贝原子 (SM80 cp.async 指令)
    using Copy_Atom_Type = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>;
    
    // 定义 TiledCopy: 线程布局为 16x8, 向量化拷贝值为 1x8
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom_Type{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},   // Thr layout
        Layout<Shape<_1, _8>>{}                     // Val layout
    );

    // 定义 LDSM 拷贝原子 (从 Shared Memory 加载到 Register)
    using S2R_Atom_Type = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;

    // 定义 TiledMMA: 结合 Tensor Core (SM80 16x8x8)
    TiledMMA mmaC = make_tiled_mma(
        SM80_16x8x8_F16F16F16F16_TN{},
        Layout<Shape<_2, _2>>{},   // 2x2x1 MMA Atoms
        Tile<_32, _32, _16>{}      // Tiled MMA for LDSM
    );

    // 核心异步流水线循环范例 (伪代码，详见 sgemm_sm80.cu):
    // 1. cp_async_fence()
    // 2. cp_async_wait<K_PIPE_MAX - 2>()
    // 3. __syncthreads()
    // 4. copy(s2r_atom, smem, rmem)
    // 5. gemm(mma, rmem_A, rmem_B, rmem_C)
}