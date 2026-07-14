/*
 * TensorRT 视觉应用：多流与多上下文推理机制 (Multi-Stream & Multi-Context)
 * 提取自 algo_ai/temp/trt_infer.hpp & trt_infer.cpp
 */

#include <NvInfer.h>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <cuda_runtime.h>
// 假设已有 MixMemory, Tensor 等结构定义
// #include "trt-tensor.hpp" 

namespace TRT {

    class InferEngine;

    /*
     * MutiContext: 多上下文容器
     * 使用 std::enable_shared_from_this<MutiContext> 技法的目的：
     * 1. 安全地在类成员函数内部（如 create_context）获取指向自身的 shared_ptr，
     *    并将其注册到 InferEngine 的 context_vec_ 中进行统一生命周期管理。
     * 2. 避免了在外部或内部重复构造 shared_ptr 导致的“双重释放 (double free)”问题。
     */
    class MutiContext : public std::enable_shared_from_this<MutiContext> {
    public:
        // 在 InferEngine (全局唯一模型) 基础上创建独立的 ExecutionContext
        bool create_context(InferEngine& input_engine) {
            // 弱引用保存 Engine 和 Runtime，避免循环引用
            engine_  = input_engine.engine_;
            runtime_ = input_engine.runtime_;
            
            // 为每个 Context 创建独立的 CUDA Stream
            owner_stream_ = true;
            cudaStreamCreate(&stream_);
            
            // 复用 Engine 创建独立的 ExecutionContext，实现多线程并发推理
            context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
                input_engine.engine_->createExecutionContext(),
                [](nvinfer1::IExecutionContext* p){ if(p) p->destroy(); }
            );

            // 【核心技法】：使用 shared_from_this() 将当前实例注册到全局 Engine 的管理队列中
            // 如果不继承 std::enable_shared_from_this，这里无法安全地获取 shared_ptr<MutiContext>
            input_engine.context_vec_.emplace_back(shared_from_this());
            return true;
        }

        // 将当前 Context 绑定的独立 Stream 和 Tensor 数据压入 TensorRT 队列
        void forward(bool sync = true) {
            // ... (动态 batch 处理和维度设置) ...
            
            // 获取所有的 Binding 指针 (包含输入和输出的 GPU 地址)
            // void** bindingsptr = bindingsPtr_.data();

            // 真正的多流并发核心：将任务异步派发到独立的 stream_ 中
            // bool execute_result = context_->enqueueV2(bindingsptr, stream_, nullptr);

            if (sync) {
                cudaStreamSynchronize(stream_);
            }
        }

    public:
        cudaStream_t                                 stream_       = nullptr;
        bool                                         owner_stream_ = false;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;
        
        // 弱引用指向全局唯一的 Engine
        std::weak_ptr<nvinfer1::ICudaEngine>         engine_;
        std::weak_ptr<nvinfer1::IRuntime>            runtime_;
        
        // 每个 Context 拥有独立的输入/输出 Tensor 内存池，避免多线程数据踩踏
        // std::vector<std::shared_ptr<Tensor>> inputs_;
        // std::vector<std::shared_ptr<Tensor>> outputs_;
    };

    /*
     * InferEngine: 引擎的全局单例包装器
     */
    class InferEngine {
    public:
        ~InferEngine() { destroy(); }
        
        void destroy() {
            // 释放所有通过 shared_from_this() 注册过来的 Context
            for (auto& context : context_vec_) {
                context = nullptr;
            }
            context_vec_.clear();
            runtime_.reset();
            engine_.reset();
        }

    public:
        // 管理所有派生出来的 MutiContext 的生命周期
        std::vector<std::shared_ptr<MutiContext>> context_vec_;
        std::shared_ptr<nvinfer1::ICudaEngine>    engine_;
        std::shared_ptr<nvinfer1::IRuntime>       runtime_ = nullptr;
    };

} // namespace TRT
