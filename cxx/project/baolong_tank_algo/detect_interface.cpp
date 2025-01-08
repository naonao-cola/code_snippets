
#include "detect_interFace.h"
#include <string>
#include <nlohmann/json.hpp>
#include "utils.h"
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include "Windows.h"
// #include "bl_config.h"
// #include "cv_detect_procesing.h"
#include "bl_algo_detect.h"
#include "logger.h"
using namespace std;
using json = nlohmann::json;
using namespace cv;


extern int verify_license(int m, int b, int *c);
extern std::string get_hwid();
/*
*/
const char* get_hardware_id()
{
    return get_hwid().c_str();
}

int tapp_license_verify() {
    long int bbb = static_cast<long int> (std::time(NULL)) && 0xffffff;
    int ccc;
    int err = verify_license(0, bbb, &ccc);
    return err;
}


int* tapp_model_init()
{
	// int* handle = nullptr;
	// cv_detect* handle = new cv_detect();
	BlAlgoDetect* handle = new BlAlgoDetect();
	if (handle != nullptr){
		LOG_INFO("===========Creat BlAlgoDetect Model done.");
		return (int*)handle;
	}	
	LOG_ERROR("==============Create BlAlgoDetect Model fail.");
	return nullptr;

}
void tapp_model_open(int* handle, const char *model_path, int device_id) {
	BlAlgoDetect* handle1 = (BlAlgoDetect*) handle;
    LOG_INFO("handle: {}", (void*)handle1);
	return;
}

void tapp_model_config(int* handle, const char* config_json_str) {	
	// cv_detect* handle1 = (cv_detect*)handle;
	
	BlAlgoDetect* handle1 = (BlAlgoDetect*) handle;
	if (handle1 != nullptr){
		handle1->model_config2(config_json_str);
		LOG_INFO("json data to model init...");
	}else{
		LOG_ERROR("model init fail!!!!!!!!!!!!!!");
	}
	
	// json config;
	// handle1->const_char_to_json(config_json_str, config);
	// // LOG_INFO("config: {}", config.dump());
	// handle1->from_bl_json(config, handle1->bl_json);
	// config.clear();

	return;
}

int tapp_model_package(int* handle, const char *model_path, char *origin_model_dir, char *model_key){
	BlAlgoDetect* handel_t = (BlAlgoDetect*) handle;
	handel_t->package_model(origin_model_dir, model_key);
    // handel_t->save(model_key);
	return true;
}

const char* tapp_model_run(int* handle, unsigned char** data, const char* in_param_json_str) {//
	BlAlgoDetect* handel_t = (BlAlgoDetect*) handle;
	
	// return "q34213";
	return handel_t->run2(in_param_json_str, data);
	// cv_detect* handle1 = (cv_detect*)handle;
	// std::cout << handle1->bl_json.device_id << std::endl;
	// handle1->find_circle_center();
	// LOG_INFO("handle1->center_pos:({},{})", (handle1->center_pos).x, (handle1->center_pos).y);
	// handle1->stat_gray_ratio();
	
}

void tapp_model_destroy(int* handle) {
	delete handle;
	LOG_INFO("model destroy...");
	return;
}

