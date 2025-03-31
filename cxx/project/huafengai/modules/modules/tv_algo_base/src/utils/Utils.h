#pragma once

#include "../framework/Defines.h"
#include "../framework/ErrorDefine.h"
#include "StringConvert.h"
#include "nlohmann/json.hpp"
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using json = nlohmann::json;

class Utils {
public:
    static json ReadJsonFile(std::string filepath);
    static json ParseJsonText(const char* json_text, bool is_ansi = true);
    static std::string DumpJson(json jsonObj, bool toAnsi = true);
    static cv::Mat GenCvImage(unsigned char* img_data, const json& img_info);
    static int GetCvType(int img_c);

    static std::string GetStatusCode(RunStatus status);
    static std::string GetErrorCode(ErrorCode errCode);

    template <typename T>
    static T GetProperty(const json& json_obj, const std::string& key, const T& def_val)
    {
        if (json_obj.contains(key)) {
            return json_obj[key].get<T>();
        } else {
            return def_val;
        }
    }
};
