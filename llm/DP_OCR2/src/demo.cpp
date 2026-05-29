#include "test.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <limits>

int main(int argc, char** argv) {
    // 说明：
    // 1) 这个 demo 只做一件事：复用同一个 OCRInference 实例，循环多次对同一张图片做推理，
    //    统计每次耗时，并输出平均/最小/最大耗时。
    // 2) 之所以要“复用同一个实例”，是为了把“模型加载/显存初始化/图像投影模型初始化”等一次性成本剥离出去，
    //    得到更接近实际服务场景（长驻进程）的单次推理耗时。

    std::string model_path = "/home/greatek/wangw/py/ocr/models/DeepSeek-OCR-2-Q4_K_M.gguf";
    std::string mmproj_path = "/home/greatek/wangw/py/ocr/models/DeepSeek-OCR-2-mmproj-f16.gguf";
    std::string image_path = "/home/greatek/wangw/py/ocr/test_image.jpg";

    // prompt 里不需要手动写 <__media__>，库代码会在必要时自动补 marker；
    // 你也可以自己在 prompt 前面加 <__media__>，两种方式等价。
    std::string prompt = "<tr>\n识别并提取图片中的所有文字。";

    // bench_runs：正式统计次数（默认 10）
    // warmup_runs：预热次数（默认 1，不纳入统计）
    // 预热的原因：第一次推理通常包含 CUDA graph warmup / 内存 page-in / cache 建立等一次性开销，
    //            不预热会让平均值被“首次开销”拉高。
    int warmup_runs = 1;
    int bench_runs = 10;
    if (argc >= 2) {
        bench_runs = std::max(1, std::atoi(argv[1]));
    }
    if (argc >= 3) {
        warmup_runs = std::max(0, std::atoi(argv[2]));
    }

    try {
        std::cout << "初始化 OCR 引擎..." << std::endl;
        auto init_start = std::chrono::high_resolution_clock::now();

        // 只初始化一次：加载文字模型 + 加载 mmproj + 初始化 mtmd 上下文 + 初始化 sampler。
        // 后续每次推理只传入图片路径和 prompt。
        OCRInference ocr(model_path, mmproj_path, 99, 8192);

        auto init_end = std::chrono::high_resolution_clock::now();
        const long long init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();

        std::cout << "模型: " << model_path << std::endl;
        std::cout << "MMProj: " << mmproj_path << std::endl;
        std::cout << "图片: " << image_path << std::endl;
        std::cout << "warmup_runs=" << warmup_runs << ", bench_runs=" << bench_runs << std::endl;
        std::cout << "init_ms=" << init_ms << std::endl;

        // 预热：不关心输出，只让 runtime 把一次性开销走完
        for (int i = 0; i < warmup_runs; i++) {
            (void) ocr.runOCR(image_path, prompt);
        }

        long long sum_ms = 0;
        long long min_ms = std::numeric_limits<long long>::max();
        long long max_ms = 0;
        size_t last_out_size = 0;

        // 正式测速：循环 N 次，统计耗时
        for (int i = 0; i < bench_runs; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            std::string result = ocr.runOCR(image_path, prompt);
            auto end = std::chrono::high_resolution_clock::now();
            const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            sum_ms += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
            last_out_size = result.size();
            std::cout << "run " << (i + 1) << "/" << bench_runs << ": " << ms << " ms" << std::endl;
        }

        const double avg_ms = bench_runs > 0 ? (double) sum_ms / (double) bench_runs : 0.0;
        std::cout << "avg_ms=" << avg_ms << ", min_ms=" << min_ms << ", max_ms=" << max_ms << ", last_output_bytes=" << last_out_size << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
