// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include <iostream>
#include "utils/easylogging++.h"
#include <filesystem>

INITIALIZE_EASYLOGGINGPP

namespace fs = std::filesystem;

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    {
        std::cout << "----------- tv_algorithm DLL_PROCESS_ATTACH -----------------" << std::endl;
        el::Loggers::removeFlag(el::LoggingFlag::AllowVerboseIfModuleNotSpecified);
        std::string path1 = "./algo_log.conf";
        std::string path2 = "./config/algo_log.conf";
        std::string log_conf_path = fs::exists(path1) ? path1 : (fs::exists(path2) ? path2 : "");
        if (fs::exists(log_conf_path)) {
            el::Configurations conf(log_conf_path);
            el::Loggers::reconfigureLogger("tv_algo", conf);
            el::Loggers::reconfigureAllLoggers(conf);
        }
        else {
            el::Configurations conf;
            conf.setToDefault();
            conf.setGlobally(el::ConfigurationType::Format, "%datetime %levshort %thread %msg");
            conf.setGlobally(el::ConfigurationType::Filename, "Algorithm.log");
            conf.setGlobally(el::ConfigurationType::ToFile, "false");
            conf.setGlobally(el::ConfigurationType::ToStandardOutput, "true");
        }
        break;
    }
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

