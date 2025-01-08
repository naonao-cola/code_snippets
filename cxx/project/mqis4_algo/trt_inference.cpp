#include "logger.h"
#include "trt_inference.h"

TrtInference::TrtInference(char *ptr, int size, int device_id) {
    cudaSetDevice(device_id);

    m_logger.setReportableSeverity(Severity::kWARNING); // kVERBOSE

    //TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(m_logger)};

    std::shared_ptr<nvinfer1::IRuntime> runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger), [](nvinfer1::IRuntime* s) {s->destroy();});
    // assert(!runtime);

    //m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(ptr, size));

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(ptr, size, nullptr), [](nvinfer1::ICudaEngine* s) {
        s->destroy();});
    // assert(!m_engine);
}

TrtInference::~TrtInference() {
}

json TrtInference::forward(cv::Mat img, const json& in_param) {
    int batch_size = 1;

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    cudaStreamCreate(&m_stream);

    TrtBufferManager buffers(m_engine, batch_size);

    preprocess(img, in_param, buffers);

    buffers.copyInputToDeviceAsync(m_stream);

    m_context->enqueueV2(buffers.getDeviceBindings().data(), m_stream, nullptr);

    buffers.copyOutputToHostAsync(m_stream);

    cudaStreamSynchronize(m_stream);

    cudaStreamDestroy(m_stream);

    m_context->destroy();

    return post_process(img, in_param, buffers);
}
