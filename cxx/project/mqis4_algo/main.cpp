#include <iostream>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <argparse/argparse.hpp>
#include "interface.h"
#include "logger.h"
#include "utils.h"
#include <thread>
#include <chrono>
#include <locale>
#include <string>
#include "TivalCore.h"
#include <NvInfer.h>
// Test

namespace fs = std::filesystem;

using namespace std;
using json = nlohmann::json;

const static std::string WORK_DIR = "E:/demo/cxx/mqis4_algo";
// const static std::string WORK_DIR = "D:/gtmc_model_convert"; // RTX5000工控机运行目录

void run() {
    LOG_INFO("open handle");
    int *handle = tapp_model_open((WORK_DIR + "/models").c_str(), 0);

    std::vector<std::string> data_dir_list;

    for (const auto & entry : fs::directory_iterator(WORK_DIR + "/configs/")) {
        if (fs::is_directory(entry.status())) {
            std::string data_pth = entry.path().string();
            LOG_INFO("@@ data_pth:{}", data_pth);
            if (data_pth.find("ignore") == -1) {
                data_dir_list.push_back(data_pth);
            }
        }
    }

    for (auto data_path: data_dir_list) {
        LOG_INFO("begin process: {}", data_path);
        LOG_INFO("load config: {}", data_path + "/config.json");
        std::ifstream conf_i(data_path + "/config.json");
        std::stringstream ss_config;
        ss_config << conf_i.rdbuf();
        std::string config = Utf8ToAnsi(ss_config.str());

        LOG_INFO("config");
        tapp_model_config(handle, config.c_str());


        LOG_INFO("load in_param: {}", data_path + "/in_param.json");
        std::ifstream in_param_i(data_path + "/in_param.json");
        std::stringstream ss_in_param;
        ss_in_param << in_param_i.rdbuf();
        std::string in_param = Utf8ToAnsi(ss_in_param.str());

        // json in_param;
        // in_param_i >> in_param;

        LOG_INFO("start run");
        const char *result = tapp_model_run(handle, in_param.c_str());
        LOG_INFO("tapp_model_run done");

        std::string out_path = data_path.replace(data_path.begin(), data_path.begin() + 6, WORK_DIR+"/out");
        LOG_INFO("write result: {}", out_path + "/result.json");
        std::filesystem::create_directories(out_path);
        std::ofstream result_o(out_path + "/result.json");
        result_o << std::setw(4) << AnsiToUtf8(std::string(result)) << std::endl;
        LOG_INFO("end process: {}", data_path);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    LOG_INFO("end process all");
    tapp_model_close(handle);
    LOG_INFO("close handle");
}

int main(int argc, char** argv)
{

    std::cout << "TensorRT version: " << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;
    LOG_INFO("---------- main ------------");
    argparse::ArgumentParser program("gtmc_ocr_algo_cmd");

    program.add_argument("--thread")
        .help("multi thread test")
        .default_value(false)
        .implicit_value(true);
    
    program.add_argument("--work_dir")
        .help("work directory")
        .default_value(std::string(""));

    program.add_argument("--package_model")
        .help("package tapp")
        .default_value(std::string(""));

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string package_model = program.get<std::string>("--package_model");
    std::string work_dir = program.get<std::string>("--work_dir");
    if (work_dir == "") {
        work_dir = WORK_DIR;
    }
    std::filesystem::current_path(std::filesystem::path(work_dir));
    LOG_INFO("Set work dir: {}", work_dir);

    if (package_model != "") {
        std::string tapp_path = "./models/";
        LOG_INFO("Package model: {}  => {}", package_model, tapp_path);
        tapp_model_package(tapp_path.c_str(), "./models/origin", const_cast<char *>(package_model.c_str())); 
        return 0;
    }
    
    if (program["--thread"] == true) {
        LOG_INFO("[Run thread mode]");
        while (1000) {
            std::thread t1 (run);
            /*std::thread t2 (run);
            std::thread t3 (run);
            std::thread t4 (run);
            std::thread t5 (run);
            std::thread t6 (run);*/
            t1.join();
            /*t2.join();
            t3.join();
            t4.join();
            t5.join();
            t6.join();*/
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    } else {
        LOG_INFO("[Run in main thread]");
        
        // while (1000) {
            run();
            // std::this_thread::sleep_for(std::chrono::seconds(3));
        // }
    }
    return 0;
}
