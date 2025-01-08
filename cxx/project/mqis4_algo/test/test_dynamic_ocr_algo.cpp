#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <filesystem>
#include <map>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../src/dynamic_ocr_algo.h"
#include "test_utils.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

bool infer_one(DynamicOCR::DynamicCharDet::m_ptr dymic_det,
               const cv::Mat& img,
               const std::string& label) {
    std::vector<cv::Mat> line_text_imgs;
    std::vector<cv::RotatedRect> OK_rrect;
    std::vector<cv::RotatedRect> NG_rrect;
    dymic_det->spilt_muti_line_text_img(img, line_text_imgs, OK_rrect, NG_rrect, label);
    return NG_rrect.size() > 0;
}

std::vector<std::string> infer(const std::string& test_pic_dir,
                               const json& pic_info,
                               int step = 500,
                               std::string label = "",
                               std::string paper_type = "") {
    DynamicOCR::DynamicCharDet::m_ptr dymic_det = std::make_shared<DynamicOCR::DynamicCharDet>();
    std::vector<std::string> ng_imgs;
    fs::directory_iterator pic_dir(test_pic_dir);
    bool read_label_from_json{false};
    bool read_paper_type_from_json{false};
    if ( label == "" ) {
        read_label_from_json = true;
    }

    if ( paper_type == "" ) {
        read_paper_type_from_json = true;
    }

    int ok_cnt = 0;
    int cnt = 0;
    for ( auto& pic_path : pic_dir ) {
        if ( pic_path.path().extension() != ".jpg" ) {
            LOG_WARN(pic_path.path().string());
            continue;
        }
        std::string pic_abs_path = pic_path.path().string();
        std::string pic_path_name = pic_path.path().filename().string();
        json info;

        if (read_label_from_json || read_paper_type_from_json) {
            if (pic_info.contains(pic_path_name)) {
                info = pic_info[pic_path_name];
            } else {
                continue;
            }
        }

        if (read_label_from_json) {
            label = info["label"];
        } else {
            static int i = 0;
            label = label + std::to_string(i);
            i++;
        }

        if (read_paper_type_from_json) {
            paper_type = info["paper_type"];
        }

        if (label == "zhgk") {
            continue;
        }

        cnt++;
        if ( cnt > 5000 ) break;
        dymic_det->config(paper_type, true);
        cv::Mat img = cv::imread(pic_abs_path);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        bool is_ng = infer_one(dymic_det, img, label);
        if (is_ng == false) {
            ok_cnt++;
        } else {
            ng_imgs.push_back(pic_abs_path);
            cv::imwrite("D:\\ng_ocr\\" + pic_path_name, img);
        }
        if ( cnt % step == 0 ) {
            LOG_INFO("ok_cnt: {}  cnt : {}", ok_cnt, cnt);
        }
    }
    LOG_INFO("ok_cnt: {}  cnt : {}", ok_cnt, cnt);
    return ng_imgs;
}

TEST_CASE("Test dynamic ocr det") {
    std::vector<std::string> ng_imgs;
    SUBCASE("Test zhizhaoqiye") {
        MESSAGE("Test zhizhaoqiye");
        std::string test_pic_dir = "D:\\blob_test\\zhizhaoqiye";
        REQUIRE(fs::exists(test_pic_dir));
        std::string test_json_file = "D:\\work_dir\\gtmc_ocr_algo\\src\\test\\test_data\\ng2\\ryz_info.json";
        json pic_label_json;
        bool read_rst = TestUtils::read_all_json_file(test_json_file, pic_label_json);
        REQUIRE(read_rst);
        ng_imgs = infer(test_pic_dir, pic_label_json, 100, "AAA", "HGZ_B");
        for ( auto ng_img : ng_imgs ) {
            MESSAGE(ng_img);
        }
        CHECK(ng_imgs.empty());
    }

    SUBCASE("Test RYZ") {
        MESSAGE("Test RYZ");
        std::string test_pic_dir = "D:\\work_dir\\gtmc_ocr_algo\\src\\test\\test_data\\ng2\\RYZ";
        REQUIRE(fs::exists(test_pic_dir));
        std::string test_json_file = "D:\\work_dir\\gtmc_ocr_algo\\src\\test\\test_data\\ng2\\ryz_info.json";
        json pic_label_json;
        bool read_rst = TestUtils::read_all_json_file(test_json_file, pic_label_json);
        REQUIRE(read_rst);
        ng_imgs = infer(test_pic_dir, pic_label_json);
        for ( auto ng_img : ng_imgs ) {
            MESSAGE(ng_img);
        }
        CHECK(ng_imgs.empty());
    }

    SUBCASE("Test RYZ NG") {
        MESSAGE("Test RYZ NG");
        std::string test_pic_dir = "D:\\work_dir\\gtmc_ocr_algo\\src\\test\\test_data\\ng2\\RYZ_NG";
        REQUIRE(fs::exists(test_pic_dir));
        std::string test_json_file = "D:\\work_dir\\gtmc_ocr_algo\\src\\test\\test_data\\ng2\\ryz_info.json";
        json pic_label_json;
        bool read_rst = TestUtils::read_all_json_file(test_json_file, pic_label_json);
        ng_imgs = infer(test_pic_dir, pic_label_json);
        CHECK_FALSE(ng_imgs.empty());
        for ( auto ng_img : ng_imgs ) {
            MESSAGE(ng_img);
        }
    }

    SUBCASE("Test COC") {
        MESSAGE("Test COC");
        std::string test_pic_dir = "D:\\work_dir\\gtmc_ocr_algo\\COC\\COC";
        REQUIRE(fs::exists(test_pic_dir));
        std::string test_json_file = "D:\\work_dir\\gtmc_ocr_algo\\src\\test\\test_data\\ng2\\coc_info.json";
        json pic_label_json;
        bool read_rst = TestUtils::read_all_json_file(test_json_file, pic_label_json);
        ng_imgs = infer(test_pic_dir, pic_label_json, 1000, "", "COC");
        CHECK_LT(ng_imgs.size(), 5);
        for ( auto ng_img : ng_imgs ) {
            MESSAGE(ng_img);
        }
    }
}
