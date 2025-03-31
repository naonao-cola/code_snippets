#include "Utils.h"
#include "Logger.h"
/**
 * @FilePath     : /connector_ai/modules/tv_algo_base/src/utils/Utils.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-11-04 13:24:43
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-11-04 13:57:00
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/

json Utils::ReadJsonFile(std::string filepath)
{
    std::ifstream conf_i(filepath);
    std::stringstream ss_config;
    ss_config << conf_i.rdbuf();
    json jsonObj = json::parse(ss_config.str());
    return std::move(jsonObj);
}


json Utils::ParseJsonText(const char* json_text, bool is_ansi)
{
    std::string utf8_text = is_ansi ? StringConvert::AnsiToUtf8(std::string(json_text)) : std::string(json_text);
    json jsonObj = json::parse(utf8_text);
    return std::move(jsonObj);
}

std::string Utils::DumpJson(json jsonObj, bool toAnsi)
{
    return toAnsi ? StringConvert::Utf8ToAnsi(jsonObj.dump()) : jsonObj.dump();
}

cv::Mat Utils::GenCvImage(unsigned char* img_data, const json& img_info)
{
    int img_w = Utils::GetProperty(img_info, "img_w", 0);
    int img_h = Utils::GetProperty(img_info, "img_h", 0);
    int img_c = Utils::GetProperty(img_info, "img_c", 0);
    if (img_w <= 0 || img_h <=0 || img_c < 1 || img_c > 4) {
        LOGE("Wrong image info!!  img_w:{}, img_h:{} img_c:{}", img_w, img_h, img_c);
        return cv::Mat();
    } else {
        return cv::Mat(img_h, img_w, Utils::GetCvType(img_c), img_data);
    }
}

int Utils::GetCvType(int img_c)
{
    int cv_type = CV_8UC1;
    switch (img_c)
    {
    case 1:
        cv_type = CV_8UC1;
        break;
    case 2:
        cv_type = CV_8UC2;
        break;
    case 3:
        cv_type = CV_8UC3;
        break;
    case 4:
        cv_type = CV_8UC4;
        break;
    default:
        break;
    }
    return cv_type;
}

std::string Utils::GetStatusCode(RunStatus status)
{
    switch (status)
    {
    case RunStatus::OK:
        return "OK";
    case RunStatus::ABNORMAL_IMAGE:
        return "ABNORMAL_IMAGE";
    case RunStatus::ABNORMAL_ANGLE:
        return "ABNORMAL_ANGLE";
    case RunStatus::NOT_FOUND_TARGET:
        return "NOT_FOUND_TARGET";
    case RunStatus::OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";
    case RunStatus::WRONG_PARAM:
        return "WRONG_PARAM";
    case RunStatus::UNKNOWN_ERROR:
        return "UNKOWN_ERROR";
    default:
        return "UNKOWN_ERROR";
    }
}

std::string Utils::GetErrorCode(ErrorCode errCode)
{
    switch (errCode)
    {
    case ErrorCode::OK:
        return "OK";
    case ErrorCode::NOT_READY:
        return "NOT_READY";
    case ErrorCode::TIME_OUT:
        return "TIME_OUT";
    case ErrorCode::WRONG_PARAM:
        return "WRONG_PARAM";
    case ErrorCode::DUPLICATE_ALGO_NAME:
        return "DUPLICATE_ALGO_NAME";
    case ErrorCode::INVALID_IMG_DATA:
        return "INVALID_IMG_DATA";
    case ErrorCode::QUEUE_OVERFLOW:
        return "QUEUE_OVERFLOW";
    case ErrorCode::WRONG_STATE:
        return "WRONG_STATE";
    case ErrorCode::UNKNOWN_ERROR:
        return "UNKNOWN_ERROR";
    default:
        return "UNKNOWN_ERROR";
    }
}