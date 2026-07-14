/*
 * TensorRT 模型构建器 (TRT Builder)
 * 提取自 algo_ai/src/trt/trt_common/trt_builder.cpp
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>
#include <vector>

namespace TRT {

    enum class Mode : int {
        FP32,
        FP16
    };

    // 智能指针自定义删除器，自动调用 destroy()
    template <typename _T>
    std::shared_ptr<_T> make_nvshared(_T* ptr) {
        return std::shared_ptr<_T>(ptr, [](_T* p) { if(p) p->destroy(); });
    }

    // 自定义日志收集器
    class Logger : public nvinfer1::ILogger {
    public:
        virtual void log(Severity severity, const char* msg) noexcept override {
            // 根据 severity 级别输出日志，忽略 kINFO 或仅在 Debug 时输出
            if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
                // LOG_ERROR
            }
        }
    };
    static Logger gLogger;

    bool compile(Mode mode, unsigned int maxBatchSize, const std::string& source_onnx, const std::string& saveto, const size_t maxWorkspaceSize = 1ul << 30) {
        auto builder = make_nvshared(nvinfer1::createInferBuilder(gLogger));
        if (!builder) return false;

        auto config = make_nvshared(builder->createBuilderConfig());
        if (mode == Mode::FP16) {
            if (builder->platformHasFastFp16()) {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
        }

        // 创建网络时设置 ExplicitBatch 标志
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = make_nvshared(builder->createNetworkV2(explicitBatch));

        // 创建 ONNX Parser
        auto onnxParser = make_nvshared(nvonnxparser::createParser(*network, gLogger));
        if (!onnxParser->parseFromFile(source_onnx.c_str(), 1)) {
            return false;
        }

        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(maxWorkspaceSize);

        // 创建 Optimization Profile 以支持动态 Batch
        auto profile = builder->createOptimizationProfile();
        int net_num_input = network->getNbInputs();
        for (int i = 0; i < net_num_input; ++i) {
            auto input = network->getInput(i);
            auto input_dims = input->getDimensions();
            input_dims.d[0] = 1; // Min batch
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
            input_dims.d[0] = maxBatchSize / 2 > 0 ? maxBatchSize / 2 : 1; // Opt batch
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
            input_dims.d[0] = maxBatchSize; // Max batch
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
        }
        config->addOptimizationProfile(profile);

        // 构建引擎
        auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
        if (!engine) return false;

        // 序列化引擎并保存到文件
        auto seridata = make_nvshared(engine->serialize());
        FILE* f = fopen(saveto.c_str(), "wb");
        if (f) {
            fwrite(seridata->data(), 1, seridata->size(), f);
            fclose(f);
            return true;
        }
        return false;
    }
}
