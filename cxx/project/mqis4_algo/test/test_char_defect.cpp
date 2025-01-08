#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include "../char_defect_det_algo.h"
#include "../logger.h"
#include "test_utils.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

bool test_gqft(std::string test_pic_dir, bool show_ng_list=true) {
    LOG_INFO("@@@ test_pic_dir{}", test_pic_dir);
    std::vector<std::string> ng_imgs;
    std::vector<std::string> ok_imgs;
    std::string ref_img_dir = "D:\\work_dir\\gtmc_ocr_algo\\ref_data";
    int bin_thr = 230;
    int error_std = 200;
    int morph_size = 7;
    CharDefectDetAlgo* char_defect = new CharDefectDetAlgo();
    char_defect->config(ref_img_dir, bin_thr, error_std, morph_size,"");
    assert(fs::exists(test_pic_dir));
    fs::directory_iterator pic_dir(test_pic_dir);
    int ng_cnt = 0;
    int cnt = 0;
    for(auto& pic_path : pic_dir) {
        if(pic_path.path().extension() != ".jpg") {
            continue;
        }
        cnt ++;
        LOG_INFO("{}", pic_path.path().filename().string());
        cv::Mat img = cv::imread(pic_path.path().string());
        auto rst = char_defect->forward(img, "GQFT");
        if(rst.size()>0) {
            ng_cnt++;
            ng_imgs.push_back(pic_path.path().string());
            // LOG_INFO("rst: {}", rst.dump());
        } else {
            ok_imgs.push_back(pic_path.path().string());
        }
    }
    if (show_ng_list){
        for(auto img : ng_imgs){
            LOG_INFO("NG img: {}", img);
        }
    }else {
        for(auto img : ok_imgs){
            LOG_INFO("ok img: {}", img);
        }
    }
    // float acc = ng_cnt/cnt;
    LOG_INFO("ng_cnt: {}, cnt: {} ", ng_cnt, cnt);
    return true;
}



json test_infer(CharDefectDetAlgo::m_ptr char_defect,
                std::string test_pic_dir,
                const json& gt_label,
                bool is_gqft = false) {
    LOG_INFO("test_pic_dir: {}", test_pic_dir);
    assert(fs::exists(test_pic_dir));
    fs::directory_iterator pic_dir(test_pic_dir);
    int ng_cnt = 0;
    int cnt = 0;
    for ( auto& pic_path : pic_dir ) {
        if ( pic_path.path().extension() != ".jpg" ) {
            continue;
        }
        // LOG_INFO("cnt:{} {}", cnt, pic_path.path().filename().string());
        cv::Mat img = cv::imread(pic_path.path().string());
        std::string label;
        std::string paper_type;
        std::string pic_name = pic_path.path().filename().string();
        if ( gt_label.contains(pic_name) && is_gqft == false ) {
            label = gt_label[pic_name]["gt_label"];
            paper_type = gt_label[pic_name]["paper_type"];
            char_defect->set_ref_img_type(paper_type);
        } else if ( is_gqft == true ) {
            label = "GQFT";
        } else {
            continue;
        }
        cnt++;
        auto rst = char_defect->forward(img, label);
        if ( rst.size() > 0 ) {
            std::string dst = "D:\\work_dir\\gtmc_ocr_algo\\ng3_test\\test1\\" + pic_name;
            LOG_INFO("save ng image at {}", dst);
            cv::imwrite(dst, img);
            ng_cnt++;
        }
    }
    json rst = {
        {"NG", ng_cnt},
        {"Count", cnt}
    };
    return rst;
}


TEST_CASE("Test character defect detection") {
    json all_result;
    std::string ref_img_root = "D:\\work_dir\\gtmc_ocr_algo\\ref_data\\";
    std::string test_img_root = "D:\\work_dir\\gtmc_ocr_algo\\ng3_test\\";
    // std::string test_img_root = "D:\\work_dir\\gtmc_ocr_algo\\ng3_test\\test";
    json COC_param = {
        {"bin_thr", 200},
        {"morph_size", 7},
    };
    json HGZ_B_param = {
        {"bin_thr", 230},
        {"morph_size", 7},
    };
    json RYZ_param = {
        {"bin_thr", 100},
        {"morph_size", 7},
    };
    json GQFT_param = {
        {"bin_thr", 200},
        {"morph_size", 7},
    };
    json HGZ_A_param = {
        {"bin_thr", 180},
        {"morph_size", 7},
    };
    json HBZ_A_param = {
        {"bin_thr", 180},
        {"morph_size", 7},
    };

    json HBZ_B_param = {
        {"bin_thr", 180},
        {"morph_size", 7},
    };
    REQUIRE(fs::exists(ref_img_root));
    REQUIRE(fs::exists(test_img_root));
    CharDefectDetAlgo::m_ptr char_defect = std::make_shared<CharDefectDetAlgo>();
    REQUIRE(char_defect != nullptr);

    SUBCASE("Test config") {
        char_defect->config(ref_img_root, 100, 200, 20, "HGZ_A");
        char_defect->config(ref_img_root, 100, 200, 20, "HGZ_B");
        char_defect->config(ref_img_root, 100, 200, 20, "COC");
        char_defect->config(ref_img_root, 100, 200, 20, "HBZ_A");
        char_defect->config(ref_img_root, 100, 200, 20, "HBZ_B");
        char_defect->config(ref_img_root, 100, 200, 20, "RYZ");
    }

    SUBCASE("Test NG3") {
        json test_data = {
            {"COC_B_1_fjdh", COC_param},
            {"COC_B_1_vin", COC_param},
            {"COC_B_3_fjdh", COC_param},
            {"COC_B_3_vin", COC_param},
            {"HBZ_A", HBZ_A_param},
            {"HBZ_B", HBZ_A_param},
            {"HGZ_B_fjdh",  HGZ_B_param},
            {"HGZ_B_vin", HGZ_B_param},
            {"RYZ_vin", RYZ_param},
            {"GQFT_OK", GQFT_param},
            {"GQFT_NG", GQFT_param}
        };
        // std::vector<std::string> test_type = {"COC_B_1_vin", "COC_B_1_vin", "COC_B_3_vin", "COC_B_3_fjdh",
        // "HBZ_A", "HBZ_B", "HGZ_B_fjdh", "HGZ_B_vin", "RYZ_vin"};
        std::vector<std::string> test_type = {"GQFT_OK", "GQFT_NG"};
        for ( auto type : test_type ) {
            json param = test_data[type];
            LOG_INFO(param.dump());
            std::string test_dir = test_img_root + type;
            char_defect->config(ref_img_root, param["bin_thr"], 200, param["morph_size"], "");
            MESSAGE(test_dir);
            json gt_label;
            if ( type == "GQFT_OK" || type == "GQFT_NG" ) {
                char_defect->config(ref_img_root, 180, 200, 7, "");
                json rst = test_infer(char_defect, test_dir, gt_label, true);
                all_result[type] = rst;
            } else {
                std::string json_p = "D:\\work_dir\\gtmc_ocr_algo\\ng3_test\\" + type + ".json";
                MESSAGE(json_p);
                bool read_ok = TestUtils::read_all_json_file(json_p, gt_label);
                CHECK(read_ok);
                if ( read_ok ) {
                    MESSAGE("start test "<< type);
                    json rst = test_infer(char_defect, test_dir, gt_label);
                    all_result[type] = rst;
                }
            }
        }

        for ( auto rst : all_result.items() ) {
            std::string type = rst.key();
            json test_rst = rst.value();
            MESSAGE(type << " NG: " << test_rst["NG"] << " ALL: " << test_rst["Count"]);
        }
    }
}
