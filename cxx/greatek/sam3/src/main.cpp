// ---------- 测试 ----------
#include "clip_bpe.h"
#include "utils.h"
#include <codecvt>
#include <iostream>
#include <locale>
#include <string>

void test_clip_bpe()
{
    std::locale::global(std::locale("en_US.UTF-8"));
    Tokenizer tok("E:/test/sam_test/model/vocab.json", "E:/test/sam_test/model/merges.txt", 32);

    std::string prompt     = "ni shi sha bi ma";
    auto [token_ids, mask] = tok.encode_with_mask(prompt);
    std::cout << "token_ids.size(): " << token_ids.size() << std::endl;
    std::cout << "[";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::cout << token_ids[i] << (i == token_ids.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::cout << "mask.size(): " << mask.size() << std::endl;
    std::cout << "[";
    for (size_t i = 0; i < mask.size(); ++i) {
        std::cout << mask[i] << (i == mask.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
    std::cout << " sum(mask): " << std::accumulate(mask.begin(), mask.end(), 0) << std::endl;
}


void test_preprocess()
{

    cv::Mat                img      = cv::imread("/home/greatek/wangww/demo/cxx/sam_test/data/cat.jpg");
    std::shared_ptr<float> img_data = sam_preprocess(img, 1004, 1008, 0.5, 0.5);
}
int main()
{


    infer("F:/zhang/onnx_detect_modify/sam3.engine", "E:/test/sam_test/data/cat.jpg", "ear");

    return 0;
}