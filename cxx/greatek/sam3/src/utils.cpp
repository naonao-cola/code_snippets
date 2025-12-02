/*
 * @FilePath     : /sam_test/src/utils.cu
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2025-12-02 13:07:36
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2025-12-02 13:25:47
 * Copyright (c) 2025 by G, All Rights Reserved.
 */
#include "utils.h"
#include "clip_bpe.h"
#include <cmath>


// Functor must have external linkage (file scope) to be used in device kernels.
struct SigmoidOp
{
    __host__ __device__ float operator()(float x) const
    {
        return 1.f / (1.f + expf(-x));
    }
};


std::shared_ptr<float> sam_preprocess(const cv::Mat& img, int target_width, int target_height, const float mean, float std)
{
    cv::Mat canvas, mat_inplace;
    cv::cvtColor(img, canvas, cv::COLOR_BGR2RGB);
    cv::resize(canvas, canvas, cv::Size(target_width, target_height), cv::INTER_LINEAR);
    if (canvas.type() != CV_32FC3)
        canvas.convertTo(mat_inplace, CV_32FC3);
    else
        mat_inplace = canvas;
    mat_inplace = mat_inplace / 255.0;
    mat_inplace = (mat_inplace - mean) / std;
    std::shared_ptr<float> inBlob(new float[3 * target_width * target_height], [](float* s) { delete[] s; });
    for (int i = 0; i < mat_inplace.channels(); ++i) {
        cv::extractChannel(mat_inplace, cv::Mat(mat_inplace.rows, mat_inplace.cols, CV_32FC1, inBlob.get() + i * mat_inplace.rows * mat_inplace.cols), i);
    }
    return inBlob;
}



std::vector<char> load_engine(const std::string& path)
{
    std::ifstream in_file(path, std::ios::in | std::ios::binary);
    if (!in_file.is_open())
        return {};
    in_file.seekg(0, std::ios::end);
    size_t            length = in_file.tellg();
    std::vector<char> data;
    if (length > 0) {
        in_file.seekg(0, std::ios::beg);
        data.resize(length);
        in_file.read(data.data(), length);
    }
    in_file.close();
    return data;
}
static MyLogger gLogger;

void infer(std::string engine_path, std::string img_path, std::string prompt)
{
    bool didInitPlugins = initLibNvInferPlugins(&gLogger, "");
    /* 1. 反序列化 engine */
    auto                                         engine_buf = load_engine(engine_path);
    std::shared_ptr<nvinfer1::IRuntime>          runtime(nvinfer1::createInferRuntime(gLogger), [](nvinfer1::IRuntime* s) { s->destroy(); });
    std::shared_ptr<nvinfer1::ICudaEngine>       engine(runtime->deserializeCudaEngine(engine_buf.data(), engine_buf.size()), [](nvinfer1::ICudaEngine* s) { s->destroy(); });
    std::shared_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext(), [](nvinfer1::IExecutionContext* s) { s->destroy(); });
    int                                          nbBindings = engine->getNbBindings();
    /* 2. 获取维度 */
    nvinfer1::Dims pix_dims    = engine->getBindingDimensions(0);   // 1,3,1008,1008
    nvinfer1::Dims ids_dims    = engine->getBindingDimensions(1);   // 1,32
    nvinfer1::Dims mask_dims   = engine->getBindingDimensions(2);   // 1,32
    nvinfer1::Dims logits_dims = engine->getBindingDimensions(3);   // 1,200,288,288
    nvinfer1::Dims boxes_dims  = engine->getBindingDimensions(4);   // 1,200,4
    nvinfer1::Dims masks_dims  = engine->getBindingDimensions(5);   // 1,200

    // 获取名字
    std::cout << "engine->getBindingName(0)" << engine->getBindingName(0) << std::endl;
    std::cout << "engine->getBindingName(1)" << engine->getBindingName(1) << std::endl;
    std::cout << "engine->getBindingName(2)" << engine->getBindingName(2) << std::endl;
    std::cout << "engine->getBindingName(3)" << engine->getBindingName(3) << std::endl;
    std::cout << "engine->getBindingName(4)" << engine->getBindingName(4) << std::endl;
    std::cout << "engine->getBindingName(5)" << engine->getBindingName(5) << std::endl;
    std::cout << "输入绑定信息:" << std::endl;
    std::cout << "pix_dims: " << pix_dims.d[0] << " " << pix_dims.d[1] << " " << pix_dims.d[2] << " " << pix_dims.d[3] << std::endl;
    std::cout << "ids_dims: " << ids_dims.d[0] << " " << ids_dims.d[1] << std::endl;
    std::cout << "mask_dims: " << mask_dims.d[0] << " " << mask_dims.d[1] << std::endl;
    std::cout << "输出绑定信息:" << std::endl;
    std::cout << "logits_dims: " << logits_dims.d[0] << " " << logits_dims.d[1] << std::endl;
    std::cout << "boxes_dims: " << boxes_dims.d[0] << " " << boxes_dims.d[1] << " " << boxes_dims.d[2] << std::endl;
    std::cout << "masks_dims: " << masks_dims.d[0] << " " << masks_dims.d[1] << " " << masks_dims.d[2] << " " << masks_dims.d[3] << std::endl;
    // pix_dims  3*1080*1008
    const int img_c = pix_dims.d[1];
    const int img_h = pix_dims.d[2];
    const int img_w = pix_dims.d[3];
    // ids_dims  32
    const int seq_len = ids_dims.d[1];
    // masks_dims 200*288*288
    const int num_inst = masks_dims.d[1];
    const int mask_h   = masks_dims.d[2];
    const int mask_w   = masks_dims.d[3];
    // 维度
    ctx->setBindingDimensions(0, pix_dims);
    ctx->setBindingDimensions(1, ids_dims);
    ctx->setBindingDimensions(2, mask_dims);
    ctx->setBindingDimensions(3, logits_dims);
    ctx->setBindingDimensions(4, boxes_dims);
    ctx->setBindingDimensions(5, masks_dims);

    /* 3. CPU 前处理 → 得到 float RGB [0,1] 已归一化 */
    cv::Mat                img          = cv::imread(img_path);
    std::shared_ptr<float> pix_host     = sam_preprocess(img, 1008, 1008, 0.5, 0.5);
    std::string            input_prompt = prompt;
    std::locale::global(std::locale("en_US.UTF-8"));
    Tokenizer tok("E:/test/sam_test/model/vocab.json", "E:/test/sam_test/model/merges.txt", 32);
    auto [token_ids, mask] = tok.encode_with_mask(prompt);

    // 设备端输入
    void* dImage = nullptr;
    cudaMalloc(&dImage, sizeof(float) * img_c * img_h * img_w);
    cudaMemcpy(dImage, pix_host.get(), sizeof(float) * img_c * img_h * img_w, cudaMemcpyHostToDevice);
    void* dIds = nullptr;
    cudaMalloc(&dIds, sizeof(int64_t) * seq_len);
    cudaMemcpy(dIds, token_ids.data(), sizeof(int64_t) * seq_len, cudaMemcpyHostToDevice);
    void* dAttn = nullptr;
    cudaMalloc(&dAttn, sizeof(int64_t) * seq_len);
    cudaMemcpy(dAttn, mask.data(), sizeof(int64_t) * seq_len, cudaMemcpyHostToDevice);

    // 设备端输出
    void* dLogits = nullptr;
    cudaMalloc(&dLogits, sizeof(float) * num_inst);
    void* dBoxes = nullptr;
    cudaMalloc(&dBoxes, sizeof(float) * num_inst * 4);
    void* dMasks = nullptr;
    cudaMalloc(&dMasks, sizeof(float) * num_inst * mask_h * mask_w);

    /* 7. 绑定地址（顺序必须和 engine 一致） */
    std::vector<void*> bindings = {dImage, dIds, dAttn, dLogits, dBoxes, dMasks};
    ctx->enqueueV2(bindings.data(), 0, nullptr);

    /* 8. 获取输出 */
    std::vector<float> h_logits(num_inst);
    std::vector<float> h_boxes(num_inst * 4);
    std::vector<float> h_masks(num_inst * mask_h * mask_w);
    cudaMemcpy(h_logits.data(), dLogits, sizeof(float) * num_inst, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boxes.data(), dBoxes, sizeof(float) * num_inst * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_masks.data(), dMasks, sizeof(float) * num_inst * mask_h * mask_w, cudaMemcpyDeviceToHost);

    // 后处理
    std::transform(h_logits.begin(), h_logits.end(), h_logits.begin(), [&](float x) { return 1.f / (1.f + std::exp(-x)); });
    std::vector<int> h_keep_idx(num_inst);
    std::iota(h_keep_idx.begin(), h_keep_idx.end(), 0);
    std::stable_sort(h_keep_idx.begin(), h_keep_idx.end(), [&](int a, int b) { return h_logits[a] > h_logits[b]; });
    for (std::vector<int>::iterator it = h_keep_idx.begin(); it != h_keep_idx.end();) {
        if (h_logits[*it] < 0.5) {
            it = h_keep_idx.erase(it);
        }
        else {
            ++it;
        }
    }
    cv::Mat dis = img.clone();
    for (size_t i = 0; i < h_keep_idx.size(); ++i) {
        auto    mask_ptr = h_masks.data() + h_keep_idx[i] * mask_h * mask_w;
        cv::Mat c_mask_mat(mask_h, mask_w, CV_32FC1, mask_ptr);
        cv::resize(c_mask_mat, c_mask_mat, cv::Size(img.cols, img.rows), cv::INTER_NEAREST);
        cv::threshold(c_mask_mat, c_mask_mat, 0.5, 1, cv::THRESH_BINARY);
        c_mask_mat.convertTo(c_mask_mat, CV_8UC1, 255.0);
        float x1 = h_boxes[h_keep_idx[i] * 4 + 0];
        float y1 = h_boxes[h_keep_idx[i] * 4 + 1];
        float x2 = h_boxes[h_keep_idx[i] * 4 + 2];
        float y2 = h_boxes[h_keep_idx[i] * 4 + 3];
        x1 *= img.cols;
        x2 *= img.cols;
        y1 *= img.rows;
        y2 *= img.rows;
        x1 = std::max(0.f, std::min(x1, (float)(img_w - 1)));
        x2 = std::max(0.f, std::min(x2, (float)(img_w - 1)));
        y1 = std::max(0.f, std::min(y1, (float)(img_h - 1)));
        y2 = std::max(0.f, std::min(y2, (float)(img_h - 1)));
        cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
        std::cout << "box: " << box << std::endl;
        std::cout << "score: " << h_logits[h_keep_idx[i]] << std::endl;
        cv::rectangle(dis, box, cv::Scalar(0, 0, 255), 2, 8);
    }
    std::string filename = "E:/test/sam_test/res.jpg";
    cv::imwrite(filename, dis);
    cudaFree(dImage);
    cudaFree(dIds);
    cudaFree(dAttn);
    cudaFree(dMasks);
    cudaFree(dBoxes);
    cudaFree(dLogits);
}
