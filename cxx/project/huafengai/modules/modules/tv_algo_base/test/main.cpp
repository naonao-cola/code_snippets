#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "nlohmann/json.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "../src/Interface.h"
#include "../src/utils/StringConvert.h"
#include <windows.h>


namespace fs = std::filesystem;
using namespace std;
using json = nlohmann::json;


static const std::string WORK_DIR = filesystem::current_path().string();
const int task_cnt = 5;
int finish_cnt = 0;
std::vector<cv::Mat> g_img_list;

template<typename T>
static T GetProperty(const json& json_obj, const std::string& key, const T& def_val)
{
    if (json_obj.contains(key)) {
        return json_obj[key].get<T>();
    } else {
        return def_val;
    }
}

json ReadJsonFile(std::string filepath)
{
    std::ifstream conf_i(filepath);
    std::stringstream ss_config;
    ss_config << conf_i.rdbuf();
    json jsonObj = json::parse(ss_config.str());
    return std::move(jsonObj);
}

std::string DumpJson(json jsonObj, bool toAnsi=true)
{
    return toAnsi ? StringConvert::Utf8ToAnsi(jsonObj.dump(2)) : jsonObj.dump(2);
}


void RecultCallback(const char* img_info_json, const char* result_json)
{
    std::cout << "### [Result_Callback] --------" << std::endl;
    std::cout << "[IMG_INFO]" << img_info_json << std::endl;
    std::cout << "[Result]" << result_json << std::endl;
    finish_cnt++;
}

void LogCallback(int level, const char* log_msg)
{
    std::cout << "[LOG_CALLBACK] [" << level << "]:" << log_msg << std::endl;
}

void* pHandle = nullptr;


int main(int args, char** argv) {
    std::filesystem::current_path(std::filesystem::path(WORK_DIR));

    // std::string test_folder = "./test_data/tayin/";
    json common_cfg = ReadJsonFile("./config/ocr_common_cfg.json");
    json algo_cfg = ReadJsonFile("./config/ocr_algo_cfg.json");
    json image_info = ReadJsonFile("./config/image_info.json");
    
    std::cout << "CommonConfig main: " << common_cfg.dump(2) << std::endl;
    std::cout << "algo_cfg main: " << algo_cfg.dump(2) << std::endl;
    std::cout << "image_info main: "<< DumpJson(image_info) << std::endl;

    pHandle = tapp_init();

    if (pHandle == nullptr) {
        cout<<"Init inference engine fail."<<endl;
        return 0;
    }

    int errCode = tapp_common_config(pHandle, DumpJson(common_cfg).c_str());
    if (errCode != 0) {
        cout<<"tapp_common_config fail. errCode:"<<errCode<<endl;
        return 0;
    }

    errCode = tapp_algo_config(pHandle, DumpJson(algo_cfg).c_str());
    if (errCode != 0) {
        cout<<"tapp_algo_config fail. errCode:"<<errCode<<endl;
        return 0;
    }

    tapp_register_result_callback(pHandle, RecultCallback);

    int count = 0;
    if (image_info.is_array() && !image_info.empty()) {
        for (auto item : image_info) {
            std::string img_dir = GetProperty(item, "img_dir", std::string(""));
            std::string img_path = GetProperty(item, "img_path", std::string(""));

            // 遍历文件目录
            if (img_dir.length() > 0 && fs::exists(img_dir)) {
                for (const auto& entry : fs::directory_iterator(img_dir)) {
                    if (entry.is_regular_file()) {
                        std::string filePath = entry.path().string();
                        std::string fileName = entry.path().filename().string();
                        std::string extension = entry.path().extension().string();
                        if (extension == ".jpg" || extension == ".png" || extension == ".JPG" || extension == ".bmp") {
                            std::string imgName = fileName.substr(0, fileName.find_last_of('.'));
                            std::string debugPath = entry.path().filename().string() + "\\" + imgName;
                            auto item = image_info.at(0);
                            
                            cv::Mat img  = cv::imread(filePath, cv::IMREAD_COLOR);
                            item["img_name"] = imgName;
                            item["img_w"] = img.cols;
                            item["img_h"] = img.rows;
                            g_img_list.push_back(img);
                            if (img.empty())
                                continue;
                            int ret = tapp_run(pHandle, img.data, DumpJson(item).c_str());
                            // std::cout<<"@@Result["<< count <<"]:" << ""<<std::string(result)<<std::endl;
                            count++;
                        }
                    }
                    else if (entry.is_directory()) {
                        std::string filePath = entry.path().string();
                        for (auto & entry2 : fs::directory_iterator(filePath)) {
                            if (entry2.is_regular_file()) {
                                std::string filePath = entry2.path().string();
                                std::string fileName = entry2.path().filename().string();
                                std::cout << "fileName1: " << fileName << std::endl;
                                std::cout << "filePath1: " << filePath << std::endl;
                                std::string imgName = fileName.substr(0, fileName.find_last_of('.'));
                                std::string debugPath = entry.path().filename().string() + "\\" + imgName;
                                auto item = image_info.at(0);

                                 cv::Mat img  = cv::imread(filePath);
                                item["img_name"] = debugPath;
                                item["img_w"] = img.cols;
                                item["img_h"] = img.rows;
                               
                                g_img_list.push_back(img);
                                if (img.empty())
                                    continue;
                                int ret = tapp_run(pHandle, img.data, DumpJson(item).c_str());
                                // std::cout<<"@@Result["<< count <<"]:" << ""<<std::string(result)<<std::endl;
                                count++;
                            }
                        }
                    }
                }
            } // end if (img_dir.length() > 0)
            else if (img_path.length() > 0 && fs::exists(img_path)) {
                std::string fileName = fs::path(img_path).filename().string();
                
                cv::Mat img  = cv::imread(img_path);
                item["img_name"] = fileName;
                item["img_w"] = img.cols;
                item["img_h"] = img.rows;
                g_img_list.push_back(img);
                int ret = tapp_run(pHandle, img.data, DumpJson(item).c_str());
                count++;
            }
        }
    }

    while (count > 0 && finish_cnt < count) {
        Sleep(20);
    }
    Sleep(100);
    std::cout << "Finished count:" << finish_cnt << std::endl;
    std::cout << "-------------------- Begin Destroy." << std::endl;
    tapp_destroy(pHandle);
    std::cout << "==================== End Destroy." << std::endl;
    system("pause");
    return 0;
}