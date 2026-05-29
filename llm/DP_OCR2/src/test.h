#ifndef OCR_INFERENCE_H
#define OCR_INFERENCE_H

#include <string>
#include <functional>
#include <memory>

class OCRInference {
public:
    // 设计目标：
    // - 构造一次：加载 GGUF 文字模型 + 加载 mmproj 视觉投影 + 初始化 mtmd
    // - 多次推理：后续每张图只走 tokenize/encode/decode，不重复初始化，适合长驻服务
    OCRInference(const std::string& model_path,
                 const std::string& mmproj_path,
                 int n_gpu_layers = 99,
                 int n_ctx = 8192);

    ~OCRInference();

    // runOCR：一次性返回完整字符串（适合 bench 和简单调用）
    std::string runOCR(const std::string& image_path,
                       const std::string& prompt = "</td>\n识别并提取图片中的所有文字。");

    // runOCRStream：边生成边回调（适合 UI/服务端流式返回）
    void runOCRStream(const std::string& image_path,
                      const std::string& prompt,
                      std::function<void(const std::string&)> callback);

    std::string getLastMessage() const { return last_message_; }

private:
    // 使用 PImpl 隔离 llama.cpp / mtmd 的头文件依赖，避免把大量 C API 暴露到外部头文件，
    // 同时也避免因为 llama.cpp API 变动导致外部编译单元大量重编。
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_message_;
    int n_ctx_;
    int n_gpu_layers_;
};

#endif // OCR_INFERENCE_H
