#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../dynamic_binaray_threshold.h"

namespace fs = std::filesystem;

void infer(std::string img_p,
           std::string type,
           int low, int high,
           double scale = 0.1,
           const std::string& save_root_dir = "D:\\") {
    DynamicThreshold dythr;
    cv::Mat thr_img;
    dythr.config(type, scale);
    cv::Mat img = cv::imread(img_p);
    cv::resize(img, img, cv::Size(1000, 1420));
    REQUIRE_FALSE(img_p.empty());
    if (fs::exists(save_root_dir) == false) {
        fs::create_directories(save_root_dir);
    }
    cv::imwrite(save_root_dir + type + ".jpg", img);
    dythr.forwad(img, thr_img, high, low);
    CHECK(img.empty() == false);
    CHECK(thr_img.empty() == false);
    CHECK_EQ(thr_img.channels(), 1);
    CHECK_EQ(thr_img.size, img.size);
    double max_v, min_v;
    cv::Point max_loc, min_loc;
    cv::minMaxLoc(thr_img, &min_v, &max_v, &max_loc, &min_loc);
    CHECK_EQ(max_v, std::min(255, high));
    CHECK_EQ(min_v, std::max(0, low));
    cv::imwrite(save_root_dir + type + "_thr.jpg", thr_img);
    cv::Mat rst_img;
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    cv::addWeighted(gray_img, 0.5, thr_img, 0.5, 0, rst_img);
    cv::imwrite(save_root_dir + type + std::to_string(scale*10) + "_rst.jpg", rst_img);
    MESSAGE("HGZ_A_result save at: " << save_root_dir + type << "_rst.jpg");
}


TEST_CASE("testing dynamic_binaray_threshold") {
    DynamicThreshold dythr;
    std::string test_data_root = "D:\\work_dir\\gtmc_ocr_algo\\src\\test\\test_data\\";
    std::string rst_save_dir = "D:\\dynamic_thr_test\\";
    // MESSAGE(fs::exists(test_data_root));
    REQUIRE(fs::exists(test_data_root));

    SUBCASE("test config") {
        dythr.config("HGZ_A");
        // dythr.config("aaaa");
    }
    SUBCASE("test COC") {
        std::string img_path = test_data_root + "coc.jpg";
        infer(img_path, "COC", 0, 255, 0.2, rst_save_dir);
        // infer(img_path, "COC", 0, 255, 0.5, rst_save_dir);
        // infer(img_path, "COC", 0, 255, 1.0, rst_save_dir);
    }
    SUBCASE("test HGZ_A") {
        std::string img_path = test_data_root + "hgz_a.jpg";
        infer(img_path, "HGZ_A", 0, 255, 0.2, rst_save_dir);
        // infer(img_path, "HGZ_A", 0, 255, 0.5, rst_save_dir);
        // infer(img_path, "HGZ_A", 0, 255, 1.0, rst_save_dir);
    }
    SUBCASE("test HGZ_B") {
        std::string img_path = test_data_root + "hgz_b.jpg";
        infer(img_path, "HGZ_B", 0, 255, 0.2, rst_save_dir);
        // infer(img_path, "HGZ_B", 0, 255, 0.5, rst_save_dir);
        // infer(img_path, "HGZ_B", 0, 255, 1.0, rst_save_dir);
    }
    SUBCASE("test HBZ_A") {
        std::string img_path = test_data_root + "hbz_a.jpg";
        infer(img_path, "HBZ_A", 0, 255, 0.2, rst_save_dir);
        // infer(img_path, "HBZ_A", 0, 255, 0.5, rst_save_dir);
        // infer(img_path, "HBZ_A", 0, 255, 1.0, rst_save_dir);
    }
    SUBCASE("test HBZ_B") {
        std::string img_path = test_data_root + "hbz_b.jpg";
        infer(img_path, "HBZ_B", 0, 255, 0.2, rst_save_dir);
        // infer(img_path, "HBZ_B", 0, 255, 0.5, rst_save_dir);
        // infer(img_path, "HBZ_B", 0, 255, 1.0, rst_save_dir);
    }
    SUBCASE("test RYZ") {
        std::string img_path = test_data_root + "ryz.jpg";
        infer(img_path, "RYZ", 0, 255, 0.2, rst_save_dir);
        // infer(img_path, "RYZ", 0, 255, 0.5, rst_save_dir);
        // infer(img_path, "RYZ", 0, 255, 1.0, rst_save_dir);
    }
}
