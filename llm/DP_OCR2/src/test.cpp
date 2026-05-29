#include "test.h"
#include <iostream>
#include <cstring>

#include <stdexcept>
#include <vector>

// llama.cpp core headers
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

struct OCRInference::Impl {
    // 文字模型（GGUF）和推理上下文（持有 KV cache / CUDA buffers 等）
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;

    // 多模态上下文（持有 mmproj 模型以及图像编码/投影的内部状态）
    mtmd_context * mtmd = nullptr;

    // 词表接口（用于判断 EOG / token->piece）
    const llama_vocab * vocab = nullptr;

    // 采样器（这里用 greedy，等价于 temp=0 的确定性输出）
    llama_sampler * smpl = nullptr;

    // n_past 表示当前序列已写入 KV cache 的 token 位置
    llama_pos n_past = 0;

    // mtmd_helper_eval_chunks 会用这个 n_batch 控制内部的 batch 大小（更大通常更快，但占用更高）
    int32_t n_batch = 512;

    Impl() = default;

    ~Impl() {
        // 释放顺序：sampler -> mtmd -> llama_context -> llama_model -> backend
        // 原因：
        // - mtmd 依赖 text model 的某些信息（metadata/embedding dim 等）
        // - context 依赖 model
        // - backend 最后释放，避免提前清理 CUDA 资源导致析构阶段访问非法资源
        if (smpl) {
            llama_sampler_free(smpl);
            smpl = nullptr;
        }
        if (mtmd) {
            mtmd_free(mtmd);
            mtmd = nullptr;
        }
        if (lctx) {
            llama_free(lctx);
            lctx = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        llama_backend_free();
    }
};

OCRInference::OCRInference(const std::string& model_path,
                           const std::string& mmproj_path,
                           int n_gpu_layers,
                           int n_ctx)
    : impl_(std::make_unique<Impl>())
    , last_message_()
    , n_ctx_(n_ctx)
    , n_gpu_layers_(n_gpu_layers) {
    // 初始化全局 backend（CPU/GPU 资源、CUDA 等）
    llama_backend_init();

    // 1) 加载文字模型（GGUF）
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers_;
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    impl_->model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!impl_->model) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }

    // 2) 创建推理上下文（KV cache 等都在这里）
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx_;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;

    impl_->lctx = llama_init_from_model(impl_->model, cparams);
    if (!impl_->lctx) {
        throw std::runtime_error("Failed to create llama context");
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);

    // 3) 初始化 mtmd：加载 mmproj（视觉编码器/投影器）
    // mtmd_init_from_file() 会根据 mmproj GGUF 的 metadata 决定 projector 类型，
    // 并创建图像编码/投影需要的内部结构。
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = true;
    mp.print_timings = false;
    mp.n_threads = 8;
    mp.warmup = true;
    impl_->mtmd = mtmd_init_from_file(mmproj_path.c_str(), impl_->model, mp);
    if (!impl_->mtmd) {
        throw std::runtime_error("Failed to init mtmd from mmproj: " + mmproj_path);
    }

    // 4) 采样器：greedy（确定性输出）
    impl_->smpl = llama_sampler_init_greedy();
    if (!impl_->smpl) {
        throw std::runtime_error("Failed to init sampler");
    }
}

OCRInference::~OCRInference() = default;

std::string OCRInference::runOCR(const std::string& image_path,
                                 const std::string& prompt)
{
    std::string result;
    runOCRStream(image_path, prompt, [&result](const std::string& token) {
        result += token;
    });
    return result;
}

void OCRInference::runOCRStream(const std::string& image_path,
                                const std::string& prompt,
                                std::function<void(const std::string&)> callback)
{
    if (!impl_ || !impl_->model || !impl_->lctx || !impl_->mtmd || !impl_->vocab) {
        last_message_ = "Not initialized";
        throw std::runtime_error(last_message_);
    }

    // 这里选择“每次推理独立”：
    // - 清空 KV cache，避免上一张图/上一轮对话影响下一次结果
    // - 重置 sampler，避免 sampler 内部状态残留
    // 如果你想做多轮对话（保留上下文），就不要清 KV cache，并用新的 seq_id 管理历史。
    llama_memory_clear(llama_get_memory(impl_->lctx), true);
    impl_->n_past = 0;
    if (impl_->smpl) {
        llama_sampler_reset(impl_->smpl);
    }

    // 1) 读取并预处理图片：mtmd_helper_bitmap_init_from_file 内部用 stb_image 解码等
    mtmd::bitmap_ptr bmp(mtmd_helper_bitmap_init_from_file(impl_->mtmd, image_path.c_str()));
    if (!bmp) {
        last_message_ = "Failed to load image";
        throw std::runtime_error(last_message_);
    }

    // 2) 组装 prompt：
    // mtmd_tokenize 要求 prompt 内包含 marker（默认 "<__media__>"），用于把图片 chunk 插入到文本 token 序列中。
    // 终端工具 llama-mtmd-cli 也是这么做的：如果用户没写 marker，会自动补到 prompt 前面。
    std::string full_prompt = prompt;
    const char * marker = mtmd_default_marker();
    if (full_prompt.find(marker) == std::string::npos) {
        full_prompt = std::string(marker) + full_prompt;
    }

    // 3) 文本输入配置：
    // - add_special=true：让 tokenizer 加 BOS 等必要的 special token
    // - parse_special=true：允许解析 prompt 中显式写出的 special token
    mtmd_input_text text;
    text.text = full_prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    // 4) tokenize：把 “文本 + 图片” 转成 chunks（chunk 里包含文本 token 和 image_tokens）
    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    const mtmd_bitmap * bitmaps[1] = { bmp.get() };
    const int32_t tok_res = mtmd_tokenize(impl_->mtmd, chunks.get(), &text, bitmaps, 1);
    if (tok_res != 0) {
        last_message_ = "mtmd_tokenize failed";
        throw std::runtime_error(last_message_);
    }

    // 5) eval chunks：
    // - 文本 chunk：等价于把 tokens 喂进 llama_decode
    // - 图片 chunk：先 mtmd_encode 得到 embedding，再按模型要求把 embedding “解码进” llama 的上下文
    // 这一步结束后，llama 上下文已经具备生成下一 token 的 logits。
    llama_pos new_n_past = 0;
    const int32_t eval_res = mtmd_helper_eval_chunks(
        impl_->mtmd,
        impl_->lctx,
        chunks.get(),
        impl_->n_past,
        0,
        impl_->n_batch,
        true,
        &new_n_past
    );
    if (eval_res != 0) {
        last_message_ = "mtmd_helper_eval_chunks failed";
        throw std::runtime_error(last_message_);
    }
    impl_->n_past = new_n_past;

    // 6) 逐 token 生成：
    // - llama_sampler_sample 从 logits 里选 token（greedy=取最大）
    // - llama_token_to_piece 把 token 转成字符串片段输出
    // - llama_decode 把 token 写入 KV cache，为下一步生成提供新的 logits
    for (int i = 0; i < 4096; i++) {
        const llama_token id = llama_sampler_sample(impl_->smpl, impl_->lctx, -1);
        llama_sampler_accept(impl_->smpl, id);

        if (llama_vocab_is_eog(impl_->vocab, id)) {
            break;
        }

        char piece[4096];
        const int32_t n = llama_token_to_piece(impl_->vocab, id, piece, (int32_t) sizeof(piece), 0, false);
        if (n > 0) {
            callback(std::string(piece, piece + n));
        }

        llama_token token_1 = id;
        llama_batch batch = llama_batch_get_one(&token_1, 1);
        const int32_t dec_res = llama_decode(impl_->lctx, batch);
        if (dec_res != 0) {
            last_message_ = "llama_decode failed";
            callback("\n[ERROR] llama_decode failed\n");
            break;
        }
    }

    last_message_ = "Success";
}
