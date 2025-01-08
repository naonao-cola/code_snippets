#pragma once
#include <filesystem>
#include <iostream>
#include<fstream>
#include  "../src/logger.h"

namespace TestUtils {

bool read_all_json_file(const std::string &strFileName, json &ref_char_info) {
    std::ifstream in(strFileName, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        LOG_INFO("Could not open json file: {}", strFileName);
        return false;
    }
    ref_char_info = json::parse(in);
    LOG_INFO("size: {}", ref_char_info.size());
    std::string json_str = ref_char_info.dump();
    return true;
}
}  //  namespace TestUtils
