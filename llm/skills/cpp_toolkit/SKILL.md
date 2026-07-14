---
name: "cpp_toolkit"
description: "C++ development, performance optimization, and TensorRT deployment toolkit. MUST be invoked whenever the user asks to write, debug, refactor, or optimize C++/CUDA code, manage xmake/CMake, implement multi-threading/SIMD/cuBLAS/CUTLASS, or deploy models (YOLO, OCR, UNet, etc.) via TensorRT. Do not attempt C++ tasks without invoking this first."
---

# C++ Toolkit (P10 Architecture Router)

作为一个高级 Agent，在直接跳入特定领域前，**必须先执行环境侦察**，拒绝盲人摸象。本技能采用了模块化路由架构，用于处理 C++ 相关的开发、构建、调试和性能优化任务。

## 必读上下文 (Mandatory Context)

**在接手任何任务前，你必须首先读取以下文档以对齐架构约束与代码准则：**
1. 读取 `references/0_project_memory_and_constraints.md`：这是本项目的全局架构基因、红线约束和避坑指南。
2. 读取 `references/1_basics_and_modern_cpp.md`：C++ 基础架构、宏与内存池。
3. 读取 `references/2_optimization_and_parallelism.md`：多线程、SIMD 与 CUDA 极客级优化。
4. 读取 `references/3_tensorrt_deployment.md`：TensorRT 高并发部署与模型分离架构。

## Agent SOP (Standard Operating Procedure)

1. **环境与上下文收集 (Reconnaissance):**
   - 检查编译器环境 (`gcc --version`, `clang --version`, `msvc` 环境)。
   - 检查构建系统 (`xmake --version`, `cmake --version`)。
   - 确认 C++ 标准 (C++11/14/17/20/23) 及目标硬件平台（如是否涉及 CUDA）。

2. **路由分配 (Routing):**
   基于用户意图，加载对应的领域规则文件与模板代码。**不要把所有文件都读入，导致上下文污染。**
   - **涉及基础语法、设计模式、内存指针或宏工具：** 加载 `references/1_basics_and_modern_cpp.md`。
     - *细粒度触发*：`shared_ptr` 优化、`unique_ptr` 数组、降低编译依存性 (Handle/Interface 类)、`defer` 宏、时间跨平台获取、强制类型转换、`perf`、`FlameGraph`、`uftrace`、火焰图。
     - *关联模板*：`scripts/templates/1_basic/tools.h`, `scripts/templates/1_basic/cpp_tips_snippets.cpp`, `scripts/templates/1_basic/perf_uftrace_demo.cpp`, `scripts/templates/1_basic/perf_uftrace_cmds.sh`
   - **涉及并发、多线程、OpenMP、CPU GEMM 压榨或 CUDA 优化：** 加载 `references/2_optimization_and_parallelism.md`。
     - *细粒度触发*：`atomic_flag` 自旋锁防重排 (`memory_order`)、死锁排查、OpenMP 降级、xsimd 指令级加速。
     - *CPU GEMM 触发*：寄存器分块、`_mm_load_pd` (SSE)、循环展开、Cache Tiling (L1/L2 Packing)。
     - *CUDA 触发*：`FLOAT4` 向量化访存、双缓冲 Ping-Pong、Flash Attention、CUDA Graph (`cudaStreamBeginCapture`)、CUTLASS 调试 (`dump_fragment`)、CuTe (`Layout`, `TiledMMA`)。
     - *高阶库触发*：cuBLAS 行列主序转换 (`cublasSgemm`), Thrust 并行容器 (`thrust::reduce`, `thrust::sort`), CUTLASS (`HostTensor`, `SM80_CP_ASYNC_CACHEALWAYS`)。
     - *关联模板*：`scripts/templates/2_optimization_cpu/thread_pool.h`、`openmp_simd_examples.cpp`、`cpu_1_gemm_...` (1-4阶段)、`3_optimization_cuda/cuda_...` (1-8模块)。
   - **涉及 TensorRT、ONNX 部署与推理：** 加载 `references/3_tensorrt_deployment.md`。
     - *细粒度触发*：`TRT Builder`, `nvonnxparser`, `OptimizationProfile`, `MixMemory`, `MonopolyAllocator`, `InferController`, `std::promise`, `WarpAffine`, `GPU NMS`, `多流推理`, `多上下文`, `YOLO`, `OCR`, `分类`, `分割`, `UNet`, `YOLOv8_OBB`, `anomalib`, `deim`, `TensorRT Plugin`, `IPluginV2DynamicExt`, `IPluginCreator`, `自定义算子`。
     - *行为*：提取 `scripts/templates/4_deployment_tensorrt/` 目录下的高并发推理框架代码。对于具体的模型（如 YOLO, OCR, 分类, 分割, OBB 旋转框, anomalib, deim 等）的前后处理源码，直接查阅 `scripts/templates/4_deployment_tensorrt/models/` 目录下对应的真实业务代码（非占位符）。对于自定义插件，查阅 `trt_6_plugin.*`。

3. **执行与闭环 (Execution & Closed-loop):**
   - 所有的辅助脚本必须通过统一入口 `scripts/cpp_cli.py` 执行，以强制执行物理隔离和解耦。
   - 确保修改遵循 100% 无损合并原则。在进行复杂算法重构前，主动通过 `web-access` 验证最佳实践。
   - 完成后，提供可验证的测试或断言，确保结果闭环，并主动向用户展示测试输出。

## 核心原则 (Core Principles)

- **【最高红线】完全解耦的独立生成 (Standalone Generation)**：技能目录 (`.trae/skills/cpp_toolkit/scripts/templates/`) 下的所有源码和模板**仅仅是知识库和参考资料**。当你为用户生成代码时，**绝对禁止**直接 `#include` 或依赖技能库内部的路径。生成的代码必须是 100% 独立的（例如，直接将 CUDA Kernel 或核心类提炼并内嵌到用户指定的工程目录中），确保用户复制该工程到任何其他机器上都能“开箱即用”。
- **【进化红线】自我进化与复盘沉淀 (Self-Evolution & Post-Mortem)**：每次在实际项目中解决了一个复杂的 C++/CUDA/TensorRT Bug 或完成了高难度的架构设计后，**必须强制进行复盘**。将踩过的坑、报错信息及最终的解决方案提炼为“Lessons Learned”或代码片段，反向补充写入到技能包对应的 `references/*.md` 文档和 `scripts/templates/` 目录中。技能不能是一潭死水，必须随着项目的推进持续变强。
- **【质量红线】Web 搜索二次校验 (Web-Search Validation)**：在生成复杂算法、CUDA Kernel、高阶 API 调用（如 TensorRT 新版本接口）或任何可能随环境变化的代码时，**必须使用联网工具 (WebSearch)** 进行全网交叉比对和二次校验。只有在确保代码符合最新最佳实践、无已知坑点后，才允许将代码写入目标文件。拒绝“凭记忆盲写”。
- **高内聚低耦合**：尽量隐藏实现细节（如使用 Pimpl 惯用法），不向调用端暴露复杂的底层依赖（如 TensorRT、CUDA 句柄）。
- **现代 C++ 实践 (C++ Core Guidelines 强制对齐)**：
  - 必须严格遵循 `cpp-coding-standards` 技能中的规范：RAII 无处不在（禁止裸 `new`/`delete`）；默认使用 `const`/`constexpr`；使用智能指针 (`std::unique_ptr`/`std::shared_ptr`) 表达所有权；使用 `std::future`/`std::promise` 进行安全的异步编程。
  - 对于所有的代码审查和重构请求，必须先调用 `cpp-coding-standards` 技能拉取核心准则。
- **性能榨干**：绝不在高频路径上发生动态内存分配（使用内存池/MixMemory），绝不在主线程发生阻塞计算。

## 工具箱入口 (CLI Router)

本技能的各项功能通过统一的 CLI 脚本暴露，以防止碎片化：
- `python .trae/skills/cpp_toolkit/scripts/cpp_cli.py --help`
