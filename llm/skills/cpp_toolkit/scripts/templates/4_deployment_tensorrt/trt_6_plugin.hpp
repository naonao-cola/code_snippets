/*
 * TensorRT 自定义插件 (Plugin) 架构范式
 * 提取自 cuda-sample/trt_mode/plugin/custom_scalar_plugin
 */

#ifndef TRT_6_PLUGIN_HPP
#define TRT_6_PLUGIN_HPP

#include "NvInfer.h"
#include <string>
#include <vector>

using namespace nvinfer1;

namespace custom {

    static const char* PLUGIN_NAME {"GridSample"};
    static const char* PLUGIN_VERSION {"1"};

    /*
     * 1. 插件核心类：继承自 IPluginV2DynamicExt
     * 负责：定义输入输出维度、执行推理计算 (enqueue)、序列化与反序列化
     */
    class CustomGridSamplePlugin : public IPluginV2DynamicExt {
    public:
        // 解析、克隆时的构造函数
        CustomGridSamplePlugin(const std::string &name);  
        // 反序列化时的构造函数
        CustomGridSamplePlugin(const std::string &name, const void* buffer, size_t length); 
        ~CustomGridSamplePlugin();

        const char* getPluginType() const noexcept override;
        const char* getPluginVersion() const noexcept override;
        int32_t     getNbOutputs() const noexcept override;
        size_t      getSerializationSize() const noexcept override;
        const char* getPluginNamespace() const noexcept override;

        DataType    getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
        
        // 核心：推导输出 Tensor 的维度
        DimsExprs   getOutputDimensions(int32_t outputIndex, const DimsExprs* input, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
        
        size_t      getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;

        int32_t     initialize() noexcept override;
        void        terminate() noexcept override;
        void        serialize(void *buffer) const noexcept override;
        void        destroy() noexcept override;

        // 核心：执行 CUDA Kernel 的入口
        int32_t     enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
        
        IPluginV2DynamicExt* clone() const noexcept override;

        // 核心：判断输入输出的数据格式和类型组合是否支持
        bool        supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOuts, int32_t nbInputs, int32_t nbOutputs) noexcept override; 
        
        void        configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override; 
        void        setPluginNamespace(const char* pluginNamespace) noexcept override;
        void        attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
        void        detachFromContext() noexcept override;

    private:
        const std::string mName;
        std::string       mNamespace;
    };

    /*
     * 2. 插件工厂类：继承自 IPluginCreator
     * 负责：在 TRT 解析 ONNX 时根据节点名字注册和创建插件实例，并接收来自 ONNX 的权重/参数
     */
    class CustomGridSamplePluginCreator : public IPluginCreator {
    public:
        CustomGridSamplePluginCreator(); 
        ~CustomGridSamplePluginCreator();

        const char*                     getPluginName() const noexcept override;
        const char*                     getPluginVersion() const noexcept override;
        const PluginFieldCollection*    getFieldNames() noexcept override;
        const char*                     getPluginNamespace() const noexcept override;
        
        // 核心：解析 ONNX 时调用，根据 PluginFieldCollection (参数) 创建 Plugin
        IPluginV2*                      createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;  
        
        // 核心：反序列化 Engine 时调用，直接读取序列化数据创建 Plugin
        IPluginV2*                      deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
        
        void                            setPluginNamespace(const char* pluginNamespace) noexcept override;

    private:
        static PluginFieldCollection    mFC;           
        static std::vector<PluginField> mAttrs;        
        std::string                     mNamespace;
    };

    // 在 cpp 文件中必须有：REGISTER_TENSORRT_PLUGIN(CustomGridSamplePluginCreator);

} // namespace custom

#endif // TRT_6_PLUGIN_HPP