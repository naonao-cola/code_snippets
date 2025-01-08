#pragma once

#include "cv_detect_procesing.h"
#include "resmodel_procesing.h"
#include "bl_config.h"
#include "tapp.h"
#include "defines.h"
// using namespace BL_CONFIG;

// enum ClassType
// {
//     O_RING = 0, C_RING, CO, PLUG
// };

class BlAlgoDetect : public BlParameter, public Tapp {

    public:
        
        BlAlgoDetect();

        ~BlAlgoDetect();
        virtual const char* run(const char* in_param_str, unsigned char* data);
        virtual const char* run2(const char* in_param_str, unsigned char** data);
        virtual bool load_model_config();
        virtual void tools_func();
        virtual void model_config(const char *config_json_str);
        virtual void model_config2(const char *config_json_str);
        virtual void package_model(std::string model_dir_path, std::string model_key);
        virtual void encode_onnx_model_to_binn(const char* model_file_path, const char* binn_file_path);
        virtual bool load(std::string model_dir_path, std::string model_name);
        virtual void load_model(std::string model_dir_path, std::string model_key);

        // void BlAlgoDetect::load_model(std::string model_dir_path);
        virtual std::string get_path(std::string tapp_dir_path);

    public:
        cv_detect* cv_process;
        ONNXClassifier* resnet_process_o;
        ONNXClassifier* resnet_process_c;
        cv::Mat* src;
        bl_config resnet_config;
        bl_config cv_config;
        bool cv_detect_flag;
        bool resnet_detect_flag_O;
        bool resnet_detect_flag_c;
        in_param in_json_param;
        img_check imgcheck;
        bool cv_model_switch;
        // ClassType flag;
        new_bl_config algo_pram;
        BlClassType Type;

        cv_detect* C_process;
        cv_detect* O_process;
        cv_detect* PLUG_process;
        bool C_switch;
        bool O_switch;
        bool PLUG_switch;
        bool C_RING_switch;
        bool O_RING_switch;

        std::vector<algo_sequence> bl_algo_sequence;

        
        

};