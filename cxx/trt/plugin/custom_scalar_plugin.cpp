#include <assert.h>
#include "custom_scalar_plugin.hpp"
#include <map>
#include <cstring>

/* customScalar的核函数接口部分 */
void customScalarImpl(const float *input, const float *grid, float *output,
    int N, int C, int H_in, int W_in, int H_out,
    int W_out, cudaStream_t stream);

using namespace nvinfer1;

namespace custom
{
/******************************************************************/
/********************注册PluginCreator*****************************/
/******************************************************************/
REGISTER_TENSORRT_PLUGIN(CustomGridSamplePluginCreator);

/******************************************************************/
/*********************静态变量的申明*******************************/
/******************************************************************/
PluginFieldCollection   CustomGridSamplePluginCreator::mFC {};
std::vector<PluginField> CustomGridSamplePluginCreator::mAttrs;

/******************************************************************/
/*********************CustomScalarPlugin实现部分***********************/
/******************************************************************/

CustomGridSamplePlugin::CustomGridSamplePlugin(const std::string &name):
    mName(name)
{
}

CustomGridSamplePlugin::CustomGridSamplePlugin(const std::string &name,const void* buffer, size_t length):
    mName(name)
{
}

CustomGridSamplePlugin::~CustomGridSamplePlugin()
{
    /* 这里的析构函数不需要做任何事情，生命周期结束的时候会自动调用terminate和destroy */
    return;
}

const char* CustomGridSamplePlugin::getPluginType() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_NAME;
}

const char* CustomGridSamplePlugin::getPluginVersion() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_VERSION;
}

int32_t CustomGridSamplePlugin::getNbOutputs() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return 1;
}

size_t CustomGridSamplePlugin::getSerializationSize() const noexcept
{
    /* 如果把所有的参数给放在mParams中的话, 一般来说所有插件的实现差不多一致 */
    return 0;
}

const char* CustomGridSamplePlugin::getPluginNamespace() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return mNamespace.c_str();
}

DataType CustomGridSamplePlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    //ASSERT(inputTypes && nbInputs > 0 && index == 0);

    return inputTypes[0];
}

DimsExprs CustomGridSamplePlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    nvinfer1::DimsExprs ret(inputs[0]);
    // ret.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
    // *ret.d[0], *exprBuilder.constant(2));
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = inputs[1].d[1];
    ret.d[3] = inputs[1].d[2];
    return ret;
}

size_t CustomGridSamplePlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CustomGridSamplePlugin::initialize() noexcept
{
    return 0;
}

void CustomGridSamplePlugin::terminate() noexcept
{
    return;
}

void CustomGridSamplePlugin::serialize(void *buffer) const noexcept
{
    return;
}

void CustomGridSamplePlugin::destroy() noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    delete this;
    return;
}

int32_t CustomGridSamplePlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    int inputBatch = inputDesc[0].dims.d[0];
    int inputChannels = inputDesc[0].dims.d[1];
    int inputHeight = inputDesc[0].dims.d[2];
    int inputWidth = inputDesc[0].dims.d[3];

    int outputHeight = inputDesc[1].dims.d[1];
    int outputWidth = inputDesc[1].dims.d[2];

    customScalarImpl(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<float*>(outputs[0]),
        inputBatch,
        inputChannels,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        stream);

    return 0;
}

IPluginV2DynamicExt* CustomGridSamplePlugin::clone() const noexcept
{
    /* 克隆一个Plugin对象，所有的插件的实现都差不多*/
    auto p = new CustomGridSamplePlugin(mName);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

bool CustomGridSamplePlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(0 <= pos && pos < 4);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;

    switch (pos) {
        case 0: return in[0].format == nvinfer1::TensorFormat::kLINEAR;
        case 1: return in[1].type == in[0].type && in[1].format == nvinfer1::TensorFormat::kLINEAR;
        case 2: return out[0].type == in[0].type && out[0].format == nvinfer1::TensorFormat::kLINEAR;
        case 3: return true;
    }
    return false;
}

void CustomGridSamplePlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    assert(nbInputs == 2 && in[0].desc.dims.d[1] != -1);
}
void CustomGridSamplePlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
    return;
}
void CustomGridSamplePlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    return;
}
void CustomGridSamplePlugin::detachFromContext() noexcept
{
    return;
}

/******************************************************************/
/*********************CustomGridSamplePluginCreator********************/
/******************************************************************/

CustomGridSamplePluginCreator::CustomGridSamplePluginCreator()
{
    /*
     * 每个插件的Creator构造函数需要定制，主要就是获取参数以及传递参数
     * 初始化creator中的PluginField以及PluginFieldCollection
     * - PluginField::            负责获取onnx中的参数
     * - PluginFieldCollection：  负责将onnx中的参数传递给Plugin
    */

    mAttrs.emplace_back(PluginField("GridSample", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

CustomGridSamplePluginCreator::~CustomGridSamplePluginCreator()
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
}

const char* CustomGridSamplePluginCreator::getPluginName() const noexcept
{
    /* 所有插件实现都差不多 */
    return PLUGIN_NAME;
}

const char* CustomGridSamplePluginCreator::getPluginVersion() const noexcept
{
    /* 所有插件实现都差不多 */
    return PLUGIN_VERSION;
}

const char* CustomGridSamplePluginCreator::getPluginNamespace() const noexcept
{
    /* 所有插件实现都差不多 */
    return mNamespace.c_str();
}

IPluginV2* CustomGridSamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    /*
     * 通过Creator创建一个Plugin的实现，这个时候会通过mFC中取出需要的参数, 并实例化一个Plugin
     * 这个案例中，参数有scalar和scale两个参数。从fc中取出来对应的数据来初始化这个plugin
    */
    // float scalar = 0;
    // float scale  = 0;
    // std::map<std::string, float*> paramMap = {{"scalar", &scalar}, {"scale", &scale}};

    // for (int i = 0; i < fc->nbFields; i++) {
    //     if (paramMap.find(fc->fields[i].name) != paramMap.end()){
    //         *paramMap[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
    //     }
    // }
    return new CustomGridSamplePlugin(name);
}

IPluginV2* CustomGridSamplePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    /* 反序列化插件其实就是实例化一个插件，所有插件实现都差不多 */
    return new CustomGridSamplePlugin(name, serialData, serialLength);
}

void CustomGridSamplePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    /* 所有插件实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}

const PluginFieldCollection* CustomGridSamplePluginCreator::getFieldNames() noexcept
{
    /* 所有插件实现都差不多 */
    return &mFC;
}

} // namespace custom
