#pragma once
#include <nlohmann/json.hpp>
#include "bl_config.h"
// using namespace BL_CONFIG;
using json = nlohmann::json;
struct BL_CV_PARM
{
    int cv_threshold;  //二值化
    int circle_radius; // 半径
    int color_thr_value; // 颜色阈值


};


class Cv_Pram : public BlParameter{

    public:
        Cv_Pram();
        ~Cv_Pram();
    
    public:
        void get_cv_pram(const char* value, json& cv_pram_json);
        void json2cvres(BL_CV_PARM& pram, json cv_pram_json);

    public:
        json cv_pram_json;
        BL_CV_PARM cv_algorithm_pram;

};


