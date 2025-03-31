#include "../../modules/tv_algo_base/src/Interface.h"
#include "../../modules/tv_algo_base/src/utils/StringConvert.h"
/**
 * @FilePath     : /connector_ai/src/test/main.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-11-01 10:41:50
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-11-04 14:52:21
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/
#include "fs.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <windows.h>

using json                  = nlohmann::json;
void*               pHandle = nullptr;
std::deque<cv::Mat> g_imgs;
int finish_cnt = 0;

template<typename T>
static T GetProperty(const json& json_obj, const std::string& key, const T& def_val)
{
    if (json_obj.contains(key)) {
        return json_obj[key].get<T>();
    }
    else {
        return def_val;
    }
}

json ReadJsonFile(std::string filepath)
{
    std::ifstream     conf_i(filepath);
    std::stringstream ss_config;
    ss_config << conf_i.rdbuf();
    json jsonObj = json::parse(ss_config.str());
    return std::move(jsonObj);
}

std::string DumpJson(json jsonObj, bool toAnsi = true)
{
    return toAnsi ? StringConvert::Utf8ToAnsi(jsonObj.dump(2)) : jsonObj.dump(2);
}

void RecultCallback(const char* img_info_json, const char* result_json)
{
    std::cout << "### [Result_Callback] --------" << std::endl;
    std::cout << "[IMG_INFO]" << img_info_json << std::endl;
    std::cout << "[Result]" << result_json << std::endl;
    g_imgs.pop_back();
}

void LogCallback(int level, const char* log_msg)
{
    std::cout << "[LOG_CALLBACK] [" << level << "]:" << log_msg << std::endl;
}

/**
 * @brief 两张图的算法
 * @param algo_file
 * @param image_file
 * @param img_1_path
 * @param img_2_path
 */
void test_pin(const std::string& algo_file, const std::string& image_file,
	const std::string& img_1_path, const std::string& img_2_path) {
	//pHandle = tapp_init();
	json image_info = ReadJsonFile(image_file);
	json image_info_2 = ReadJsonFile(image_file);


	cv::Mat img1 = cv::imread(img_1_path);
	cv::Mat img2 = cv::imread(img_2_path);
	g_imgs.push_back(img1);
	g_imgs.push_back(img2);
	/*
	* real img info
	*/
	image_info["img_w"] = img1.cols;
	image_info["img_h"] = img1.rows;
	image_info["img_c"] = img1.channels();
	image_info["img_path"] = img_1_path;
	image_info_2["img_w"] = img2.cols;
	image_info_2["img_h"] = img2.rows;
	image_info_2["img_c"] = img2.channels();
	image_info_2["img_path"] = img_2_path;
	tapp_run2(pHandle, img1.data, DumpJson(image_info).c_str(), img2.data,
		DumpJson(image_info_2).c_str());
	while (finish_cnt < 1) {
		Sleep(50);
	}
}


/**
 * @brief 单张图的测试方法
 * @param image_file image_json file
 * @param img_1_path image path
 */
void test_single(const std::string& image_file, const std::string& img_1_path)
{
    // 查找最后一个斜杠的位置
    size_t lastSlashPos = img_1_path.find_last_of("/\\");
    size_t start        = (lastSlashPos == std::string::npos) ? 0 : lastSlashPos + 1;

    // 查找最后一个点的位置
    size_t dotPos = img_1_path.find_last_of('.');
    std::string imgName = img_1_path.substr(start, dotPos - start);

    json    image_info;
    cv::Mat img1;
    img1                   = cv::imread(img_1_path);
    g_imgs.push_back(img1);
    image_info             = ReadJsonFile(image_file);
    image_info["img_w"]    = img1.cols;
    image_info["img_h"]    = img1.rows;
    image_info["img_c"]    = img1.channels();
    image_info["img_path"] = img_1_path;
    image_info["img_name"] = imgName;
    tapp_run(pHandle, img1.data, DumpJson(image_info).c_str());

}

int main_pin(){
    json common_cfg = ReadJsonFile("D:/work/0_HF/HF/common_cfg.json");
    json algo_cfg   = ReadJsonFile("D:/work/0_HF/new/sbzg.json");
    json image_info = ReadJsonFile("D:/work/0_HF/HF/HF_configs/image_info_RZ_24_8.json");

    std::cout << "CommonConfig main: " << common_cfg.dump() << std::endl;
    std::cout << "algo_cfg main: " << algo_cfg.dump() << std::endl;
    std::cout << "image_info main: " << DumpJson(image_info) << std::endl;

    pHandle = tapp_init();

    try {
        // json  input_imgs = ReadJsonFile("E:/demo/cxx/connector/config/image_info.json");
        tapp_common_config(pHandle, DumpJson(common_cfg).c_str());
        tapp_algo_config(pHandle, DumpJson(algo_cfg).c_str());
        tapp_register_result_callback(pHandle, RecultCallback);


        cv::Mat img1           = cv::imread("D:/work/0_HF/dataset/sbzg/1/FishEyePin_A.PNG");
        cv::Mat img2           = cv::imread("D:/work/0_HF/dataset/sbzg/1/FishEyePin_B.PNG");
        image_info["img_name"] = "8_A";
        g_imgs.push_back(img1);
        g_imgs.push_back(img2);
        tapp_run2(pHandle, img1.data, DumpJson(image_info).c_str(), img2.data, DumpJson(image_info).c_str());
        /*while (finish_cnt < 1) {
            Sleep(50);
        }*/
        tapp_destroy(pHandle);
        //std::cout << "Main app exit. finish cnt:" << finish_cnt << std::endl;
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }


    return 0;
    }


int main(int args, char** argv)
{
    json                     common_cfg;
    json                     algo_cfg;
    std::vector<std::string> img_file;
    int                      count      = 0;
    int                      errCode    = 0;
    std::string              image_file = R"(D:\work\0_HF\AI\connector_ai\config\image_info.json)";
    common_cfg                          = ReadJsonFile(R"(D:\work\0_HF\AI\connector_ai\config\ocr_common_cfg_test.json)");
    algo_cfg                            = ReadJsonFile(R"(D:\work\0_HF\AI\new_config\algo_other.json)");

    pHandle = tapp_init();
    if (pHandle == nullptr) {
        printf("Init inference engine fail. \n");
        return 0;
    }

    tapp_common_config(pHandle, DumpJson(common_cfg).c_str());
    tapp_algo_config(pHandle, DumpJson(algo_cfg).c_str());
    tapp_register_result_callback(pHandle, RecultCallback);
    //nao::fl::getAllFormatFiles("./images/", img_file);
    nao::fl::getAllFormatFiles(R"(D:\work\0_HF\AI\new_config\20241223\20241223\)", img_file);
    for (int i = 0; i < img_file.size(); i++) {
        test_single(image_file, img_file[i]);
        Sleep(3000);
    }
    Sleep(3000);
    tapp_destroy(pHandle);
    printf("==================== End Destroy.");
    return EXIT_SUCCESS;
}

int main_test(int args, char** argv)
{
    json                     common_cfg;
    json                     algo_cfg;
    std::vector<std::string> img_file;
    int                      count      = 0;
    int                      errCode    = 0;
    std::string              image_file = "./configs_test/image_info.json";
    common_cfg                          = ReadJsonFile("./configs_test/ocr_common_cfg_test.json");
    algo_cfg                            = ReadJsonFile("./configs_test/ocr_algo_cfg.json");

    pHandle = tapp_init();
    if (pHandle == nullptr) {
        printf("Init inference engine fail. \n");
        return 0;
    }

    tapp_common_config(pHandle, DumpJson(common_cfg).c_str());
    tapp_algo_config(pHandle, DumpJson(algo_cfg).c_str());
    tapp_register_result_callback(pHandle, RecultCallback);
    nao::fl::getAllFormatFiles("./images_test/", img_file);
    //nao::fl::getAllFormatFiles(R"(D:\work\0_HF\AI\images\3\)", img_file);
    for (int i = 0; i < img_file.size(); i++) {
        test_single(image_file, img_file[i]);
    }
    Sleep(1000);
    tapp_destroy(pHandle);
    printf("==================== End Destroy.");
    return EXIT_SUCCESS;
}