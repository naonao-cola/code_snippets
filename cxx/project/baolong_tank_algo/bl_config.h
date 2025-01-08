#pragma once
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
// #include <stdio.h>
#define struct_size 10
using json = nlohmann::json;


// namespace BL_CONFIG {
        //模型检测参数
struct Paras
{
    /* data */
    std::string detect_classes;
    std::string model_path;
    std::string label_path;
};

struct Shapes
{
    /* data */
    std::string label;
    std::vector<std::vector<int>> point;
    std::string shape_type; 
};

struct Param_1
{
    /* data */
    float threshold;
    float radius_min_distance;
    int centerX;
    int centerY;
    Shapes box_param;
};

struct Param
{
    /* data */
    std::string detect_Item;
    float score;
    // Param_1 param;
};



//模板参数
struct Templates
{
    /* data */
    std::string img_path;
    Shapes shapes[struct_size];
};

//json 转结构体
struct bl_config
        {
            /* data */
            std::string type_id;
            std::string device_id;
            Paras params;
            Templates templates[struct_size];
        };

struct in_param
        {
            std::string type_id;
            std::string img_name;
            std::string img_path;
            int img_w;
            int img_h;
            int channel;
        };

struct img_check{
    std::string check_value = "OK";
    int gv_value[3] = {0, 0, 0};
};

struct new_bl_config
{
    std::string type_id;
    std::string device_id;
    std::string model_path;
    std::vector<Param> params;
    // Templates templates[struct_size];
};

struct algo_sequence
{
    std::string type_id;
    std::vector<std::string> detect_class;
};

class BlParameter {
 public:
    BlParameter() {};
    ~BlParameter() {};  
    void from_bl_json(const json &json_file, bl_config&      bl_json);
    void from_bl_json(const json &json_file, in_param&       bl_json);
    void from_new_bl_json (const json &json_file, new_bl_config& bl_json);
    bool const_char_to_json(const char* json_str, json&                 m_config);
    bool result_to_json(json& result_info, bl_config model_config, std::vector<std::string> result);
    bool result_to_json2(json& result_info, in_param model_config, std::vector<std::string>& result);
    bool result_to_json3(json& result_info, in_param model_config, std::vector<std::string>& result, img_check imgcheck);
public:
    float acc = 1.0;
    // bool result_to_json_arry(json& result_arry, json result_info);
};
 
// } // BL_CONFIG;




