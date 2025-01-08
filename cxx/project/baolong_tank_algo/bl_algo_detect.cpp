#pragma once
#include<iostream>
// #include  <boost/cast.hpp>
#include "defines.h"
#include <time.h>
#include "utils.h"
using namespace std;
#include "bl_algo_detect.h"
BlAlgoDetect::BlAlgoDetect() : Tapp("D:/Ronnie/pakges/bldataset/station2_model/O"){
    resnet_process_o = new ONNXClassifier();
    cv_process = new cv_detect();
    C_process = new cv_detect();
    O_process = new cv_detect();
    PLUG_process = new cv_detect();
    resnet_process_c = new ONNXClassifier();
    cv_detect_flag = false;
    resnet_detect_flag_O = false;//模型开关
    resnet_detect_flag_c = false;
    cv_model_switch = false;//CO算法和PLUG算法开关

    C_switch = false;
    O_switch = false;
    PLUG_switch = false;
    
    
}

BlAlgoDetect::~BlAlgoDetect(){
    if(!resnet_process_o){
        delete resnet_process_o;
    }
    if(!cv_process){
        delete cv_process;
    }
    if(!resnet_process_c){
        delete resnet_process_c;
    }
}

const char* BlAlgoDetect::run(const char* in_param_str, unsigned char* data){
    // LOG_INFO("model running......");
    // cv::Mat get_in_image;
    // std::string result = "";
    // json config;
    // std::string img_single_dir;
    // json result_info;
    // if (!in_param_str) {
    //     LOG_WARN("in_param_str is null!!! result=[ {null} ]");
    //     return result.c_str();
    // }
    // // if (!data){
    // //     LOG_WARN("image ptr is null!!!  result=[ {null} ]");
    // //     return result.c_str();
    // // }
    // try
    // {
    //     const_char_to_json(in_param_str, config);
    //     from_bl_json(config, in_json_param);
    //     img_single_dir = in_json_param.img_path;
    //     LOG_INFO("config: {}", config.dump());
    // }
    // catch(const std::exception& e)
    // {
    //     LOG_ERROR("error const char* to config====={}", e.what());
    //     return result.c_str();
    // }   

    // if(img_single_dir == "") {
    //     if (!data){
    //         LOG_WARN("image ptr is null!!!  result=[ {null} ]");
    //         return result.c_str();
    //     }
    //     int channel = config["channel"].get<int>();
    //     if (channel == 3) {
    //         LOG_INFO("channel == 3 Convert char* to CV_8UC3");
    //         get_in_image = cv::Mat(in_json_param.img_h, in_json_param.img_w, CV_8UC3, data, 0);
    //     } else if (channel == 1) {
    //         LOG_INFO("channel == 1 Convert char* to CV_8UC1");
    //         get_in_image = cv::Mat(in_json_param.img_h, in_json_param.img_w, CV_8UC1, data, 0);
    //     } else {
    //         LOG_ERROR("Channel = {}, channel must be 1 or 3", channel);
    //     }
    //     LOG_INFO("Get img ptr data.......");
    // }
    // else {
    //     get_in_image = cv::imread(img_single_dir);
    //     LOG_INFO("===Detect single img test========img dir:{}", img_single_dir);//D:\\Ronnie\\proj\\baolong\\tank\\dataset\\B\\dataset\\AT002\\0817\\ok_ng_10\\1.bmp
    // }
    // if (resnet_detect_flag_c){
    //     LOG_INFO("ResNet_Detect_Flag = [ {} ]\n deep learn model start... detect classes=[ {} ]", resnet_detect_flag_c, resnet_config.params.detect_classes);   
    //     vector<std::string> resnet_result;
    //     float start = clock();
    //     float acc = resnet_process_c->Classify(get_in_image, result, FIND_C_RING);
    //     float end = clock();
    //     std::cout<< "img:"<< img_single_dir << "::::::::::::::>" << "result: "<<result << "    time:   " << ((double)end - start) / CLOCKS_PER_SEC << "sec." << std::endl;
    //     json result_info;
    //     // result_to_json(result_info, result, resnet_config.params.detect_classes, resnet_config.type_id);
    //     if(result == "NG"){
    //         resnet_result.push_back(result);
    //     }
    //     result_to_json(result_info, resnet_config, resnet_result);
    //     LOG_INFO("result info json:{}", result_info.dump());
    //     const char* res = Utf8ToAnsi(result_info.dump().c_str());
    //     return res;
    // }
    // if (resnet_detect_flag_O){
    //     LOG_INFO("ResNet_Detect_Flag = [ {} ]\n deep learn model start... detect classes=[ {} ]", resnet_detect_flag_O, resnet_config.params.detect_classes);   
    //     vector<std::string> resnet_result;
    //     float start = clock();
    //     resnet_process_o->Classify(get_in_image, result, FIND_O_RING);
    //     float end = clock();
    //     std::cout<< "img:"<< img_single_dir << "::::::::::::::>" << "result: "<<result << "    time:   " << ((double)end - start) / CLOCKS_PER_SEC << "sec." << std::endl;
    //     json result_info;
    //     // result_to_json(result_info, result, resnet_config.params.detect_classes, resnet_config.type_id);
    //     if(result == "NG"){
    //         resnet_result.push_back(result);
    //     }
    //     result_to_json(result_info, resnet_config, resnet_result);
    //     LOG_INFO("result info json:{}", result_info.dump());
    //     const char* res = Utf8ToAnsi(result_info.dump().c_str());
    //     return res;
    // }
    
    // if(cv_detect_flag)
    // {
    //     cv_process->run_config = in_json_param;
        
    //     if(cv_model_switch){
    //     LOG_INFO("CV_Detect_Flag = [ {} ] \n Detect Classes=[ {} ]",cv_detect_flag, resnet_config.params.detect_classes);
    //     bool find = cv_process->find_circle_center_easy(get_in_image);
    //     if(!find){cv_process->find_circle_center_hard(get_in_image);}
    //     vector<std::string> cv_result;
    //     std::string result_c;
    //     std::string result_o;
    //     float start = clock();

    //     result_c = cv_process->check_c_circle(get_in_image);
    //     if (result_c == "C_NG"){
            
    //         cv_result.push_back(result_c);
    //     }
    //     result_o = cv_process->check_o_circle(get_in_image);
    //     if (result_o == "O_NG"){
    //         cv_result.push_back(result_o);
    //     }

    //     float end = clock();
    //     LOG_INFO("{} detect cons:{} msc.", cv_config.params.detect_classes, ((double)end - start) / CLOCKS_PER_SEC);
    //     result_to_json(result_info, cv_config, cv_result);
    //     LOG_INFO("result info json:{}", result_info.dump());
    //     const char* res = Utf8ToAnsi(result_info.dump().c_str());
    //     return res;
    //     }
    //     else{
    //         //胶塞检测
    //         vector<std::string> result_plug;
    //         std::string res_plug;
    //         float start = clock();
    //         res_plug = cv_process->check_plug(get_in_image);
    //         if (res_plug == "PLUG_NG"){
    //             result_plug.push_back(res_plug);
    //         }
    //         result_to_json(result_info, cv_config, result_plug);
    //         result_plug.clear();
    //         float end = clock();
    //         LOG_INFO("result info json:{}  \t cons:{} msc.", result_info.dump(), ((double)end - start) / CLOCKS_PER_SEC);
    //         const char* res = Utf8ToAnsi(result_info.dump().c_str());
    //         return res;
    //     }
    // }
    // else
    // {
    //     LOG_WARN("error detect xxxxxxxxx check model config switch [ cv_detect_flag, resnet_detect_flag] and cheac in prams!!! result={null}");
    //     return result.c_str();
    // }
    return "";
/*
批量推理

    // cv::String WORK_DIR = "D:\\Ronnie\\proj\\baolong\\tank\\dataset\\B\\dataset\\AT002\\0817\\ok_ng_10";
    // vector<cv::String> img_nane;
    // cv::glob(WORK_DIR, img_nane);
    // for (int i = 0; i < img_nane.size(); i++){
    //     cv::Mat test_image = cv::imread(img_nane[i]);
        // }
*/
}

const char* BlAlgoDetect::run2(const char* in_param_str, unsigned char** data){
    LOG_INFO("model running(new run2)......");
    vector<std::string> cv_result;
    cv::Mat get_in_image;
    std::string result = "";
    json config;
    std::string img_single_dir;
    
    json result_info;       //单张图像的Json 结果
    json summary_result;    //汇总结果
    if (!in_param_str){
        LOG_WARN("in_param_str is null!!! result=[ {null} ]");
        return result.c_str();
    }

    std::string utf8_config_json_str = AnsiToUtf8(std::string(in_param_str));
	json m_config = json::parse(utf8_config_json_str);
	// std::cout << AnsiToUtf8(m_config.dump()) << std::endl;
    int img_num = 0;
	for (auto t = m_config.begin(); t != m_config.end(); t++){
		
		// std::cout << (*t).dump() << std::endl;
		json t2(*t);
		std::cout << "========================" << t2.dump() << "=========================" << std::endl;
        try
        {
            from_bl_json(t2, in_json_param);
        }
        catch(const std::exception& e)
        {
            LOG_ERROR("error json to run config====={}", e.what());
            return result.c_str();
            assert(false);
        }
        if (data[img_num]){
            
            int channel = in_json_param.channel;
            LOG_INFO("channel: {}", channel);
            if (channel == 3) {
                LOG_INFO("channel == 3 Convert char* to CV_8UC3");
                get_in_image = cv::Mat(in_json_param.img_h, in_json_param.img_w, CV_8UC3, data[img_num], 0);

            } else if (channel == 1) {
                LOG_INFO("channel == 1 Convert char* to CV_8UC1");
                get_in_image = cv::Mat(in_json_param.img_h, in_json_param.img_w, CV_8UC1, data[img_num], 0);
                // cv::cvtColor(get_in_image, get_in_image, cv::, cv::COLOR_GRAY2RGB);
            } else {
                LOG_ERROR("Channel = {}, channel must be 1 or 3", channel);
            }
            // get_in_image = cv::Mat(in_json_param.img_h, in_json_param.img_w, CV_8UC3, data[img_num], 0);
            LOG_INFO("Get img ptr data. image cols: {}, image rows: {}, image channels: {}", get_in_image.cols, get_in_image.rows, get_in_image.channels());
        }
        else {
            break;
        }
        

        for (auto sequence: bl_algo_sequence) {
            if (sequence.type_id.compare(in_json_param.type_id) == STATUS_OK) {
                for (auto turn_detect_class_switch: sequence.detect_class) {
                    if (turn_detect_class_switch.compare("C") == STATUS_OK) {
                        C_switch = true;
                    }
                    if (turn_detect_class_switch.compare("O") == STATUS_OK) {
                        O_switch = true;
                    }
                    if (turn_detect_class_switch.compare("PLUG") == STATUS_OK) {
                        PLUG_switch = true;
                    }
                    if (turn_detect_class_switch == "C_RING") {
                        C_RING_switch = true;
                    }
                    if (turn_detect_class_switch == "O_RING") {
                        O_RING_switch = true;
                    }
                }
            }
            else 
            {

                LOG_INFO("Without this algo configuration : type id = {}", sequence.type_id);
            }
        }
                LOG_INFO("typee_id={}, C_switch={}, O_switch={}, PLUG_switch={}, C_RING_switch={}, O_RING_switch={}", in_json_param.type_id, C_switch, O_switch, PLUG_switch, C_RING_switch, O_RING_switch);
                //开始算法推理返回单张图像jason结果
                if (C_RING_switch && in_json_param.type_id == "BLCV003"){
                    LOG_INFO("ResNet_Detect_Flag = [ {} ]\n deep learn model start... detect classes=[ {} ]", C_RING_switch, resnet_process_c->bl_dl_pram.detect_Item);   
                    vector<std::string> resnet_result;
                    float start = clock();
                    LOG_INFO("Start Classify");
                    if (resnet_process_c->gv_abnormal(get_in_image, 100, 235)) {
                        float confidence = resnet_process_c->Classify(get_in_image, result, FIND_C_RING);
                        this->acc = confidence;
                        LOG_INFO("Classify done!");
                        float end = clock();
                        LOG_INFO("img:::::::::::::::>result: {}  Tact: {}sec.", result, ((double)end - start) / CLOCKS_PER_SEC );
                        std::cout<< "img:"<< img_single_dir << "::::::::::::::>" << "result: "<<result << "    time:   " << ((double)end - start) / CLOCKS_PER_SEC << "sec." << std::endl;
                        json result_info;
                        // result_to_json(result_info, result, resnet_config.params.detect_classes, resnet_config.type_id);
                        if(result == "NG"){
                            cv_result.push_back(result);
                        }
                    }
                    else {
                        result = "NG";
                        cv_result.push_back(result);
                        LOG_INFO("{}  GV Abnormal!!!", resnet_process_c->bl_dl_pram.detect_Item);

                    }
                    resnet_process_c->updateImgCheck(imgcheck);
                    // result_to_json(result_info, resnet_config, resnet_result);
                    // LOG_INFO("result info json:{}", result_info.dump());
                    // const char* res = Utf8ToAnsi(result_info.dump().c_str());
                    // return res;
                }           
    
                if (O_RING_switch  && in_json_param.type_id == "BLCV004"){
                    LOG_INFO("ResNet_Detect_Flag = [ {} ]\n deep learn model start... detect classes=[ {} ]", O_RING_switch, resnet_process_o->bl_dl_pram.detect_Item);   
                    vector<std::string> resnet_result;
                    float start = clock();
                    LOG_INFO("Start Classify");
                    if (resnet_process_o->gv_abnormal(get_in_image, 100, 235)) {
                        float confidence = resnet_process_o->Classify(get_in_image, result, FIND_O_RING);
                        this->acc = confidence;
                        LOG_INFO("Classify done!");
                        float end = clock();
                        LOG_INFO("img:::::::::::::::>result: {}  Tact: {}sec.", result, ((double)end - start) / CLOCKS_PER_SEC );
                        std::cout<< "img:"<< img_single_dir << "::::::::::::::>" << "result: "<<result << "    time:   " << ((double)end - start) / CLOCKS_PER_SEC << "sec." << std::endl;
                        json result_info;
                        // result_to_json(result_info, result, resnet_config.params.detect_classes, resnet_config.type_id);
                        if(result == "NG"){
                            std::string O_reslut = "O_RING_NG";
                            cv_result.push_back(O_reslut);
                        }
                    }
                    else {
                            std::string O_reslut = "O_RING_NG";
                            cv_result.push_back(O_reslut);
                            LOG_INFO("{}  GV Abnormal!!!", resnet_process_o->bl_dl_pram.detect_Item);
                    }
                    resnet_process_o->updateImgCheck(imgcheck);
                    // result_to_json(result_info, resnet_config, resnet_result);
                    // LOG_INFO("result info json:{}", result_info.dump());
                    // const char* res = Utf8ToAnsi(result_info.dump().c_str());
                    // return res;
                }
    
                if (C_switch && in_json_param.type_id == "BLCV001") {
                    LOG_INFO("CV_Detect_Flag = [ {} ] \n Detect Classes=[ {} ]",C_switch, C_process->bl_cv_pram.detect_Item);
                    // cv::imwrite("hjf.jpg", get_in_image);
                    std::string result_C;
                    float start = clock();
                    if(C_process->gv_abnormal(get_in_image, 10, 240)) {
                        C_process->run_config = in_json_param;
                        bool find = C_process->find_circle_center_easy(get_in_image);
                        // bool find  = false;
                        if(!find){C_process->find_circle_center_hard(get_in_image);}
                        
                        


                        result_C = C_process->check_c_circle(get_in_image);
                        if (result_C == "C_NG"){
                            
                            cv_result.push_back(result_C);
                            // cv::imwrite("D:\\Ronnie\\pakges\\bldataset\\test_algo\\ngggg.jpg", get_in_image);
                        }
                    }
                    else {
                        result_C = "C_NG";
                        cv_result.push_back(result_C);
                        LOG_INFO("{}  GV Abnormal!!!", C_process->bl_cv_pram.detect_Item);
                    }
                    C_process->updateImgCheck(imgcheck);
                    float end = clock();
                    LOG_INFO("{} detect cons:{} msc.", C_process->bl_cv_pram.detect_Item, ((double)end - start) / CLOCKS_PER_SEC);
                    C_switch = false;
                }
    
                if (O_switch && in_json_param.type_id == "BLCV001") {
                    LOG_INFO("CV_Detect_Flag = [ {} ] \n Detect Classes=[ {} ]",O_switch, O_process->bl_cv_pram.detect_Item);
                    O_process->run_config = in_json_param;
                    std::string result_O;
                    float start = clock();
                    if(O_process->gv_abnormal(get_in_image, 80, 200)) {
                        bool find = O_process->find_circle_center_easy(get_in_image);
                        // bool find = false;
                        if(!find){O_process->find_circle_center_hard(get_in_image);}

                        
                        // std::string result_o;
                        

                        result_O = O_process->check_o_circle(get_in_image);
                        if (result_O == "O_NG"){
                            
                            cv_result.push_back(result_O);
                        }
                    }
                    else {
                        result_O == "O_NG";
                        cv_result.push_back(result_O);
                        LOG_INFO("{}  GV Abnormal!!!", O_process->bl_cv_pram.detect_Item);
                    }
                    O_process->updateImgCheck(imgcheck);
                    float end = clock();
                    LOG_INFO("{} detect cons:{} msc.", O_process->bl_cv_pram.detect_Item, ((double)end - start) / CLOCKS_PER_SEC);
                    O_switch = false;
                }
    
                if (PLUG_switch && in_json_param.type_id == "BLCV002") {
                    LOG_INFO("CV_Detect_Flag = [ {} ] \n Detect Classes=[ {} ]",PLUG_switch, PLUG_process->bl_cv_pram.detect_Item);
                    PLUG_process->run_config = in_json_param;
                    // bool find = PLUG_process->find_circle_center_easy(get_in_image);
                    // if(!find){PLUG_process->find_circle_center_hard(get_in_image);}
                    std::string result_PLUG;
                    float start = clock();
                    if (PLUG_process->gv_abnormal(get_in_image, 10, 240)) {
                        
                        // std::string result_o;
                        
                        cv_result.clear();
                        result_PLUG = PLUG_process->check_plug(get_in_image);
                        if (result_PLUG == "PLUG_NG"){
                            
                            cv_result.push_back(result_PLUG);
                        }
                    }
                    else {
                        cv_result.clear();
                        result_PLUG = "PLUG_NG";
                        cv_result.push_back(result_PLUG);
                        LOG_INFO("{}  GV Abnormal!!!", PLUG_process->bl_cv_pram.detect_Item);
                    }
                    PLUG_process->updateImgCheck(imgcheck);
                    float end = clock();
                    LOG_INFO("{} detect cons:{} msc.", PLUG_process->bl_cv_pram.detect_Item, ((double)end - start) / CLOCKS_PER_SEC);
                    PLUG_switch = false;
                }

                //结果汇总到shapes
                // result_to_json2(result_info, , cv_result);
                // result_to_json2(result_info, in_json_param, cv_result);
                result_to_json3(result_info, in_json_param, cv_result, imgcheck);
                LOG_INFO("result info json:{}", result_info.dump());
                //                 for(auto m:cv_result){
                //     LOG_INFO("cv_result = {}", m);
                // }
                cv_result.clear();

                for( auto k : cv_result) {

                    cv_result.pop_back();
                }


            //关闭单个检测开关，等下一张图重置
            C_switch = false;
            O_switch = false;
            PLUG_switch = false;
            O_RING_switch = false;//模型开关
            C_RING_switch = false;
        
        //获取到单张图像的result_info
        //放入json数组 summary_result
        /* code */
        // result_to_json_arry(summary_result, result_info);
        summary_result[img_num++] = result_info;

    }
    //每张图结果已完成算法检测，返回汇总后的json数组转const char*
    const char* res = Utf8ToAnsi(summary_result.dump().c_str());
    return res;
}

bool BlAlgoDetect::load_model_config(){
    
    return true;
}

void BlAlgoDetect::tools_func(){

    return;
}

void BlAlgoDetect::encode_onnx_model_to_binn(const char* model_file_path, const char* binn_file_path) {
    // Load ONNX model from file
    std::ifstream model_file(model_file_path, std::ios::binary);
    std::vector<char> model_data((std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>());

    // Encode ONNX model to Binn format
    binn* obj = binn_object();
    binn_object_set_blob(obj, "onnx_model", reinterpret_cast<void*>(model_data.data()), static_cast<int>(model_data.size()));
    void* binn_data;
    int binn_size1 = binn_size(obj);
    // binn_data = malloc(binn_size1);
    // binn_store(obj, binn_data);
    binn_data = binn_ptr(obj);
    // size_t binn_size1 = binn_size(obj);
    
    LOG_INFO("weight size:{}", binn_size1);
    // Write Binn data to file
    std::ofstream binn_file(binn_file_path, std::ios::binary);
    binn_file.write(reinterpret_cast<const char*>(binn_data), binn_size1);

    // Clean up memory
    // free(binn_data);
    binn_free(obj);
}


void BlAlgoDetect::package_model(std::string model_dir_path, std::string model_key)
{
    m_tapp_path = model_dir_path;
    int file_len = 0;
    // char *trt_file;

    std::string model_path = model_dir_path + "/" + model_key + ".onnx";
    std::string save_path = model_dir_path + "/" + model_key + ".tapp";
    LOG_INFO("++++ Load Onnx model: {}", model_path);
    // file_len = Tapp::read_file(model_path, &trt_file);
    LOG_INFO("package model:{}", model_key);
    // set_blob(model_key, trt_file, file_len);
    // delete trt_file;
    encode_onnx_model_to_binn(model_path.c_str(), save_path.c_str());
}

bool BlAlgoDetect::load(std::string model_dir_path, std::string model_name) {
    LOG_INFO("load tapp");
    // Tapp::load();

    load_model(model_dir_path, model_name);

    // load_model(model_dir_path, "C_RING");
    // load_model(model_dir_path, "O_RING");

    return true;
}


std::string BlAlgoDetect::get_path(std::string tapp_dir_path){
    std::string tapp_name;
	for (int i= tapp_dir_path.size()-1;i>0;i--) {
		if (tapp_dir_path[i] == '\\' || tapp_dir_path[i] == '/') {
			tapp_name = tapp_dir_path.substr(i+1);
            break;
		}
	}    
    return std::strtok(tapp_name.data(), ".");
    // for (auto i : tapp_name){

    //     std::cout << i << std::endl;
    // }
    // return tapp_name;
}

void BlAlgoDetect::load_model(std::string model_dir_path, std::string model_key)
{
    // char *buffer;
    // int length = Tapp::read_file(model_dir_path, &buffer);
    // m_obj = binn_open(buffer);
    // char *blob_ptr;
    // int size;
    // std::string model_key = get_path(model_dir_path);
    // get_blob(model_key, &blob_ptr, &size);

     // Load Binn data from file
    std::ifstream binn_file(model_dir_path, std::ios::binary);
    std::vector<char> binn_data((std::istreambuf_iterator<char>(binn_file)), std::istreambuf_iterator<char>());

    // Decode Binn data to ONNX model binary format
    binn* obj = binn_open(reinterpret_cast<void*>(binn_data.data()));
    void* model_data;
    int model_size;
    binn_object_get_blob(obj, "onnx_model", &model_data, &model_size);

    LOG_INFO("load model: {}, blob size:{}", model_key, model_size);
    ONNXClassifier* netInference = nullptr;
    size_t sizeBuffer = model_size;
    // resnet_process_c->init_model((char*)model_data, sizeBuffer, cv::Size(1440, 1080));
    // resnet_process_o->init_model((char*)model_data, sizeBuffer, cv::Size(1440, 1080));
    if (model_key == "C_RING") {
        resnet_process_c->init_model((char*)model_data, sizeBuffer, cv::Size(1440, 1080));
    } else if (model_key == "O_RING") {
        resnet_process_o->init_model((char*)model_data, sizeBuffer, cv::Size(1440, 1080));
    }

    free_buffer(m_obj);
    return;
}

void BlAlgoDetect::model_config(const char *config_json_str){
    
    json config;
    try
    {
        const_char_to_json(config_json_str, config);
        // LOG_INFO("config: {}", config.dump());
        from_bl_json(config, cv_config);
        from_bl_json(config, resnet_config);
        // resnet_process = new ONNXClassifier("d:/", "dasasd", cv::Size(640, 640));
    }
    catch(const std::exception& e)
    {
        // std::cerr << e.what() << '\n';
        LOG_INFO("const char* to json error!!! {}", e.what());
    }
    

    if(resnet_config.params.detect_classes == "O_RING"){
        LOG_INFO("\n Detect class <====>{} \n \n Deep Learn model init....\n",resnet_config.params.detect_classes);
        // if(resnet_process_o->init_model(resnet_config.params.model_path, cv::Size(1440, 1080))){
            resnet_detect_flag_O = true;
        // }
    }
    else if( resnet_config.params.detect_classes == "C_RING" ) {
        LOG_INFO("\n Detect class <====>{} \n \n Deep Learn model init....\n",resnet_config.params.detect_classes);
        // if(resnet_process_c->init_model(resnet_config.params.model_path, cv::Size(1440, 1080))){
            resnet_detect_flag_c = true;
        // }
    }
    else if (resnet_config.params.detect_classes == "CO" || resnet_config.params.detect_classes == "PLUG"){
        LOG_INFO("\n CV-Detect class <====>{} \n cv model config done!!!\n",resnet_config.params.detect_classes);
        cv_process->bl_json = cv_config;
        cv_detect_flag = true;
        if (resnet_config.params.detect_classes == "CO"){
            //打开工位1 CO形环检测
            this->cv_model_switch = true;
        }
    }
    else{
        LOG_ERROR("\n\n\n\nModel Load Fail.\n\nThe configuration file is invalid!!!!\n");
        assert(false);
    }
    return;
}

void BlAlgoDetect::model_config2(const char *config_json_str) {
    std::string utf8_config_json_str = AnsiToUtf8(std::string(config_json_str));
	json m_config = json::parse(utf8_config_json_str);
	// std::cout << AnsiToUtf8(m_config.dump()) << std::endl;
    algo_sequence single_algo_sequence;
	for (auto t = m_config.begin(); t != m_config.end(); t++){
		new_bl_config algo_pram;
		// std::cout << (*t).dump() << std::endl;
		json t2(*t);
        LOG_INFO("algo params = {}",t2.dump());
		std::cout << t2.dump() << std::endl;
        
        try
        {
            /* code */
            from_new_bl_json(t2, algo_pram);
        }
        catch(const std::exception& e)
        {
            // std::cerr << e.what() << '\n';
            LOG_INFO("const char* to json error!!! {}", e.what());
            assert(false);
        }
        
        single_algo_sequence.type_id = algo_pram.type_id;
        std::cout << "model path=============================" << algo_pram.model_path << std::endl;
        LOG_INFO("model path============================={}", algo_pram.model_path);
        if (algo_pram.model_path.empty()) {
            for (auto detect_class : algo_pram.params) {
                std::cout << "tyid:" << algo_pram.type_id << "class:" << detect_class.detect_Item << std::endl;
                if(detect_class.detect_Item == "C") {
                    //配置C算法参数
                    // LOG_INFO("\n CV-Detect class <====>{} \n cv model config done!!!\n",detect_class.detect_Item);
                    C_process->bl_cv_pram = detect_class;
                    
                    single_algo_sequence.detect_class.push_back(detect_class.detect_Item);
                    // C_switch = true;
                }
                else if (detect_class.detect_Item == "O") {
                    //配置O算法参数
                    // LOG_INFO("\n CV-Detect class <====>{} \n cv model config done!!!\n",detect_class.detect_Item);
                    O_process->bl_cv_pram = detect_class;
                    
                    single_algo_sequence.detect_class.push_back(detect_class.detect_Item);
                    // O_switch = true;
                }
                else if (detect_class.detect_Item == "PLUG") {
                    //配置PLUG算法参数
                    // LOG_INFO("\n CV-Detect class <====>{} \n cv model config done!!!\n",detect_class.detect_Item);
                    PLUG_process->bl_cv_pram = detect_class;

                    single_algo_sequence.detect_class.push_back(detect_class.detect_Item);
                    // PLUG_switch = true;
                }
            }
        }
        else {
            for (auto detect_class : algo_pram.params) {
                if(detect_class.detect_Item == "C_RING") {
                    //配置C_RING算法参数
                    LOG_INFO("\n Detect class <====>{} \n \n Deep Learn model init....\n",detect_class.detect_Item);
                    resnet_process_c->bl_dl_pram = detect_class;
                    load(algo_pram.model_path, detect_class.detect_Item);


                    single_algo_sequence.detect_class.push_back(detect_class.detect_Item);
                    // resnet_detect_flag_c = true;
                }
                else if (detect_class.detect_Item == "O_RING") {
                    //配置O_RING算法参数
                    LOG_INFO("\n Detect class <====>{} \n \n Deep Learn model init....\n",detect_class.detect_Item);
                    resnet_process_o->bl_dl_pram = detect_class;
                    load(algo_pram.model_path,  detect_class.detect_Item);

                    single_algo_sequence.detect_class.push_back(detect_class.detect_Item);
                    // resnet_detect_flag_O = true;
                }
            }
        
        
		// const json &json_file = t2;
		// BL_CONFIG::new_bl_config& bl_json = check_param;
		// t1.from_new_bl_json(t2, check_param);

        }
    //bl_algo_sequence.push_back(single_algo_sequence);
    bl_algo_sequence.push_back(single_algo_sequence);
    for (auto i : single_algo_sequence.detect_class) {

        // std::cout << "+++++++++++++" << i << std::endl;
        single_algo_sequence.detect_class.pop_back();
    }
    }
}  