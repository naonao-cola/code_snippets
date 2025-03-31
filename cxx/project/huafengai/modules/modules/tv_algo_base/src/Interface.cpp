#include <iostream>
#include "Interface.h"
#include "utils/StringConvert.h"
#include "nlohmann/json.hpp"
#include "framework/InferenceEngine.h"
#include "utils/Utils.h"
#include "framework/ErrorDefine.h"
#include "utils/Logger.h"

#include "tv_license.h"
using json = nlohmann::json;


#define LICENSE_SOLUTION_ID 2
#define MODULE_IDS          "100"


#if USE_LICENSE
//extern int verify_license(int m, int b, int *c);
//extern std::string get_hwid();

const char* get_hardware_id()
{
    return TVLicense::Instance()->GetHWID();
}

int tapp_license_verify(void* handle) {
    InferenceEngine * pEngine = static_cast<InferenceEngine*>(handle);
   return (int)pEngine->LicenseVerify();
}
#endif

int tapp_package(const char* out_model_path, const char* origin_model_dir, const char* model_name, const char* model_type,
						  int input_w, int input_h, int input_c, unsigned int major_version, unsigned int minor_version)
{
    return 0;
}


void* tapp_init()
{
    InferenceEngine* pEngine = InferenceEngine::get_instance();
    ErrorCode ec = pEngine->Init();
   if (ec != ErrorCode::OK) {
        LOGE("tapp_init fail!");
        pEngine->Destroy();
        return nullptr;
    }
    return (int*)pEngine;
}

void tapp_common_config(void* handle, const char* common_config_json)
{
    std::string utf8_str = StringConvert::AnsiToUtf8(std::string(common_config_json));
    json config = json::parse(utf8_str);
    InferenceEngine* pEngine = static_cast<InferenceEngine*>(handle);
    pEngine->ConfigSystemParams(config);
}

void tapp_algo_config(void* handle, const char* algo_config_json)
{
    std::string utf8_str = StringConvert::AnsiToUtf8(std::string(algo_config_json));
    json config = json::parse(utf8_str);
    InferenceEngine* pEngine = static_cast<InferenceEngine*>(handle);
    pEngine->ConfigAlgoParams(config);
}

 void tapp_register_result_callback(void* handle, ResultCallbackFunc callback)
 {
    InferenceEngine* pEngine = static_cast<InferenceEngine*>(handle);
    pEngine->RegisterResultCallback(callback);
 }

 void tapp_register_log_callback(void* handle, LogCallbackFunc callback)
 {
    InferenceEngine* pEngine = static_cast<InferenceEngine*>(handle);
    pEngine->RegisterLogCallback(callback);
 }

 int tapp_run(void* handle, unsigned char* img_data, const char* img_info)
 {
     LOGI(" algo tapp run start !");

     int ret = 0;
     InferenceEngine* pEngine = static_cast<InferenceEngine*>(handle);

//#if USE_LICENSE
//     ret = (int)pEngine->LicenseVerify();
//     if (ret != 0)
//     {
//         LOGE("tapp_run fail!! ErrorCode:{}", ret);
//         return ret;
//     }
//#endif
    InferTaskPtr task = std::make_shared<stInferTask>();
    task->image_info = Utils::ParseJsonText(img_info);
    task->img_data = img_data;
    task->image = Utils::GenCvImage(img_data, task->image_info);

    LOGI("image width:{}", task->image.cols);
    if (task->image.empty()) {
        LOGE("Gen image obj fail! Wrong image parameter!!");
        return int(ErrorCode::WRONG_PARAM);
    }
    //cv::imwrite("D:/test_save.jpg", task->image);
    pEngine->CommitInferTask(task);
    return ret;
 }

 int tapp_run2(void* handle, unsigned char* img_data, const char* img_info, unsigned char* img_data2, const char* img_info2)
 {
     LOGI("tapp_run2() ++");
     int ret = 0;
     InferenceEngine* pEngine = static_cast<InferenceEngine*>(handle);

//#if USE_LICENSE
//     ret = (int)pEngine->LicenseVerify();
//     if (ret != 0)
//     {
//         LOGE("tapp_run fail!! ErrorCode:{}", ret);
//         return ret;
//     }
//#endif
    InferTaskPtr task = std::make_shared<stInferTask>();
    task->image_info = Utils::ParseJsonText(img_info);
    task->img_data = img_data;
    task->image = Utils::GenCvImage(img_data, task->image_info);

    if (task->image.empty()) {
        LOGE("Gen image obj fail! Wrong image parameter!!");
        return int(ErrorCode::WRONG_PARAM);
    }
    task->image_info2 = Utils::ParseJsonText(img_info2);
    task->img_data2 = img_data;
    task->image2 = Utils::GenCvImage(img_data2, task->image_info);
    if (task->image.empty()) {
        LOGE("Gen image obj fail! Wrong image parameter!!");
        return int(ErrorCode::WRONG_PARAM);
    }
    pEngine->CommitInferTask(task);
    return ret;
 }

void tapp_destroy(void* handle)
{
    InferenceEngine* pEngine = static_cast<InferenceEngine*>(handle);
    pEngine->Destroy();
    TVLicense::Instance()->Destroy();
}