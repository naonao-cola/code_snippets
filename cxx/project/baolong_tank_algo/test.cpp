#pragma once
#include "detect_interface.h"
#include <iostream>
#include <string.h>
#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "bl_config.h"
#include <typeinfo>
#include "logger.h"
#include "utils.h"
#include <filesystem>
#define N 300
using namespace cv;
using namespace std;
namespace fs = std::filesystem;


using json = nlohmann::json;
// using namespace BL_CONFIG;

// std::string Json_Path = "D:/Ronnie/pakges/baolong_tank_algo/src/test.json";
// std::string Json_Path1 = "D:/Ronnie/pakges/baolong_tank_algo/src/parameter.json";
/// @brief BLCV001 BLCV002  BLCV003 	 BLCV004 
/// @brief C\O算法 PLUG算法 内侧C形环算法  内侧O形圈
std::string Json_new_run_Path = "E:/project/BL/baolong_tank_algo/config/run_single.json";//run.json  runBLCV002.json
std::string Json_new_Path = "E:/project/BL/baolong_tank_algo/config/config_new.json";
// std::string model_c_path = "D:/Ronnie/pakges/baolong_tank_algo/models/Inter_ring/C_RINGdadasfa.tapp";
// std::string model_o_path = "D:/Ronnie/pakges/baolong_tank_algo/models/Inter_ring/O_RING1.tapp";
std::string model_C = "C_RING";
std::string model_O = "O_RING";
std::string model_pathO = "E:/project/BL/dataset/station2_model/O";
std::string model_pathC = "E:/project/BL/dataset/station2_model/C";
/// @brief config 配置json文件修改 2023.1.7
/// @brief config 一次配置多个类别和算法参数
/// @brief 取消tapp_model_open  根据config配置是否打开模型
/// tapp_model_run 取消一次调用推理单张图片，改为一次调用推理多张图片（默认4张）
/// 返回结果：加入多张结果到result json->"shapes"中
void run(){

	LOG_INFO("Load Json Params!!!");
	std::ifstream in(Json_new_Path.c_str(), std::ios::binary);
	if(!in.is_open()){
		std::cout << "open json file error ..." << std::endl;
		LOG_ERROR("Open json File Error...");
		return;
	}
	LOG_INFO("Open Json File Successd...");
	std::cout << "open json file successd ..." << std::endl;

	std::stringstream s_in;
	s_in << in.rdbuf();
	std::string string_json = Utf8ToAnsi(s_in.str());
	// std::cout << string_json << std::endl;
	//handle 初始化
	LOG_INFO("Start init model....");
	int *handle = tapp_model_init();
	LOG_INFO("finished init model....");


	//模型保存
	tapp_model_package(handle, model_pathO.c_str(), model_pathO.data(), model_O.data());
	tapp_model_package(handle, model_pathC.c_str(), model_pathC.data(), model_C.data());
	// tapp 模型加载
	// tapp_model_open(handle, model_c_path.c_str(), 0);
	// tapp_model_open(handle, model_o_path.c_str(), 0);
	//模型配置

	tapp_model_config(handle, string_json.c_str());


	// cv::Mat im = cv::imread("D:\\Ronnie\\pakges\\bldataset\\num1\\1229\\CO\\20221229112222537.bmp");
	// unsigned char* test = im.data;
	//模型检测

	std::ifstream in_p(Json_new_run_Path.c_str(), std::ios::binary);
	if(!in_p.is_open()){
		std::cout << "open json file error ..." << std::endl;
		LOG_ERROR("Open json File Error...");
		return;
	}
	std::stringstream p_in;
	p_in << in_p.rdbuf();
	std::string string_json1 = Utf8ToAnsi(p_in.str());

	// const char* result = tapp_model_run(handle, test, string_json1.c_str());
	// 批量推理测试
	// cv::String WORK_DIR = "D:\\Ronnie\\pakges\\bldataset\\num1\\1229\\PLUG";
    // std::vector<cv::String> img_name;
    // cv::glob(WORK_DIR, img_name);
    // for (int i = 0; i < img_name.size(); i++){
    //     cv::Mat test_image = cv::imread(img_name[i]);
		
	// 	unsigned char* test = test_image.data;
		
	// 	std::cout << "-----" << img_name[i] << std::endl;
	// 	const char* result = tapp_model_run(handle, test, string_json1.c_str());
	// 	std::cout << "result = " << result << std::endl;
    //     }

	//多张图片传入run测试
	unsigned char* img_arr[N];
	cv::Mat test_image1[N];
	cv::String WORK_DIR = "E:\\project\\BL\\dataset\\test_algo";
    std::vector<cv::String> img_name;
    cv::glob(WORK_DIR, img_name);
	BlParameter test_res_json;
	std::vector<cv::String> ng_path;
	json res;
    for (int i = 0; i < img_name.size(); i++){
        test_image1[i] = cv::imread(img_name[i]);
		std::cout << img_name[i] << std::endl;
		// cv::Mat src = cv::imread(img_name[i]);
		// cv::imshow('1', src);
		// cv::waitKey(0);
		unsigned char* test = test_image1[i].data;
		// img_arr[i] = test;
		img_arr[0] = test;
		unsigned char** img_ptr = img_arr;
		const char* result = tapp_model_run(handle, img_ptr, string_json1.c_str());
		test_res_json.const_char_to_json(result, res);
		json label_json = res[0].at("label_set");
		if(label_json[0].at("shapes") != json::array()) {
			ng_path.push_back(img_name[i]);
			cv::String save_path = img_name[i].replace(img_name[i].find("test_algo"), 9, "algo_NG");
			// int last_slash = save_path.find_last_of("\\/");
			// cv::String new_str = save_path.substr(0, last_slash+1);
			cv::imwrite(save_path, test_image1[i]);
			// std::cout << "1111111111111111111" << std::endl;
		}
		std::cout << res[0] << std::endl;
		std::cout << result << std::endl;
		std::cout << "---------------------------------------------------" << std::endl;
	}
	int n = 1;
	for (auto p : ng_path) {
		
		std::cout << n << "   NG_ING_PATH:" << p << std::endl;
		n++;
	}
	// unsigned char** img_ptr = img_arr;
	// const char* result = tapp_model_run(handle, img_ptr, string_json1.c_str());
	// std::cout << result << std::endl;
	//销毁算法模型
	tapp_model_destroy(handle);

	return;
}
//test tapp 
void test_tapp(){
	int *handle_onnx = tapp_model_init();
	// std::cout << string_json << std::endl;
	tapp_model_package(handle_onnx, model_pathC.c_str(), model_pathC.data(), model_C.data());

	// 读 ONNX

	// c 读 C_RING.onnx
	return;
}

void test_new_json(){

	std::ifstream in_p(Json_new_Path.c_str(), std::ios::binary);
	if(!in_p.is_open()){
		std::cout << "open json file error ..." << std::endl;
		LOG_ERROR("Open json File Error...");
		return;
	}
	std::stringstream p_in;
	p_in << in_p.rdbuf();
	std::string string_json1 = Utf8ToAnsi(p_in.str());
	std::cout << string_json1 << std::endl;
	const char* const_string = string_json1.c_str();
	BlParameter t1;
	json t_config;
	new_bl_config check_param;
	std::string utf8_config_json_str = AnsiToUtf8(std::string(const_string));
	json m_config = json::parse(utf8_config_json_str);
	// std::cout << AnsiToUtf8(m_config.dump()) << std::endl;

	for (auto t = m_config.begin(); t != m_config.end(); t++){
		
		// std::cout << (*t).dump() << std::endl;
		json t2(*t);
		std::cout << t2.dump() << std::endl;
		// const json &json_file = t2;
		// BL_CONFIG::new_bl_config& bl_json = check_param;
		t1.from_new_bl_json(t2, check_param);
		
		// if ( json_file.contains("type_id")) {
        //     json_file.at("type_id").get_to(bl_json.type_id);
        // }
        // json_file.at("device_id").get_to(bl_json.device_id);
		// Param cv_param;
		// for (auto i = json_file["params"].begin(); i != json_file["params"].end(); i++){

		// 	i->at("detect_Item").get_to(cv_param.detect_Item);
		// 	i->at("param").at("threshold").get_to(cv_param.param.threshold);
        //     i->at("param").at("radius_min_distance").get_to(cv_param.param.radius_min_distance);
        //     i->at("param").at("centerX").get_to(cv_param.param.centerX);
        //     i->at("param").at("centerY").get_to(cv_param.param.centerY);
		// 	if (i->contains("box_param")){
		// 		std::cout << "sdasdasdasdasdas" << std::endl;
		// 	}
		// 	bl_json.params.push_back(cv_param);
		// }
		// // for(auto item : json_file["params"]) {
			
		// // }

        // json_file.at("template").at("img_path").get_to(bl_json.templates->img_path);
        // for (int i = 0; i < 2; i++) {
        //     json_file.at("template").at("shapes")[i].at("points").get_to<std::vector<std::vector<int>>>(bl_json.templates->shapes[i].point);
        //     json_file.at("template").at("shapes")[i].at("label").get_to(bl_json.templates->shapes->label);
        //     json_file.at("template").at("shapes")[i].at("shape_type").get_to(bl_json.templates->shapes->shape_type);
        // }
	}
	// for (auto Param: check_param.params){
	// 	std::cout << Param.detect_Item << std::endl;
	// 	std::cout << Param.param.centerX << std::endl;
	// 	std::cout << Param.param.centerY << std::endl;
	// 	std::cout << Param.param.radius_min_distance << std::endl;
	// 	std::cout << Param.param.box_param.label << std::endl;
	// 	std::cout << Param.param.threshold << std::endl;
	// }
	return;
}



void test_img() {
	unsigned char* img_arr[2];
	cv::Mat test_image1[2];
	cv::String WORK_DIR = "D:\\Ronnie\\pakges\\bldataset\\num1\\1229\\PLUG";
    std::vector<cv::String> img_name;
    cv::glob(WORK_DIR, img_name);
    for (int i = 0; i < img_name.size(); i++){
        test_image1[i] = cv::imread(img_name[i]);

		unsigned char* test = test_image1[i].data;
		img_arr[i] = test;
	}
	unsigned char** img_ptr = img_arr;
	cv::Mat get_in_image = cv::Mat(1080, 1440, CV_8UC1, img_ptr[0], 0);
	cv::imwrite("ttttt0.jpg", get_in_image);
	
	// cv::Mat get_in_image = cv::Mat(1080, 1440, CV_8UC3, img_arr[0], 0);
	// cv::imwrite("ttttt0.jpg", get_in_image);
	// cv::Mat get_in_image1 = cv::Mat(1080, 1440, CV_8UC3, img_arr[1], 0);
	// cv::imwrite("ttttt1.jpg", get_in_image1);
	// for (auto img_p: img_ptr){
	// 	cv::Mat get_in_image = cv::Mat(1080, 1440, CV_8UC3, img_p, 0);
	// 	cv::imwrite("ttttt.jpg", get_in_image);
	// }
	// for(int i=0;i<2;i++){
    //     for(int j=0;j<1440*1080*3;j++){
    //         std::cout << img_ptr[i][j] << std::endl;
    //     }
	// }

}

void test_hist() {

	// cv::Mat img = cv::imread("D:\\Ronnie\\pakges\\bldataset\\test_algo\\20230527-105658.jpg", cv::IMREAD_GRAYSCALE);
    // if (img.empty())
    // {
    //     std::cerr << "Failed to load image" << std::endl;
    //     return;
    // }

    // cv::Mat eq_img;
    // cv::equalizeHist(img, eq_img);

    // cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Equalized Image", cv::WINDOW_AUTOSIZE);

    // cv::imshow("Original Image", img);
    // cv::imshow("Equalized Image", eq_img);
	// cv::imwrite("D:\\Ronnie\\pakges\\bldataset\\test_algo\\2.jpg", eq_img);

    // cv::waitKey(0);
	cv::Mat img1 = cv::imread("D:\\Ronnie\\pakges\\bldataset\\test_algo\\1.jpg");
	cv::Mat img2 = cv::imread("D:\\Ronnie\\pakges\\bldataset\\test_algo\\2.jpg");

	cv::threshold(img1,img1, 150, 255, cv::THRESH_BINARY);
	cv::threshold(img2,img2, 150, 255, cv::THRESH_BINARY);

    cv::imshow("Original Image", img1);
    cv::imshow("Equalized Image", img2);
	cv::waitKey(0);
}

double calculate_curvature(const cv::Vec4i& line)
{
    double x1 = double(line[0]);
    double y1 = double(line[1]);
    double x2 = double(line[2]);
    double y2 = double(line[3]);

    double dx = x2 - x1;
    double dy = y2 - y1;
    
    // 计算线段的长度和方向
    double length = std::sqrt(dx * dx + dy * dy);
    dx /= length;
    dy /= length;

    // 计算该点切线的方向，平行于线段的法向量
    double nx = -dy;
    double ny = dx;

    // 计算该点的弯曲方向
    double dotprod = (dx * nx) + (dy * ny);
    double crossprod = (dx * ny) - (dy * nx);
    double angle = std::atan2(crossprod, dotprod);
    
    // 计算弯曲半径
    double radius = length / (2.0 * std::sin(angle));
    
    // 计算曲率
    double curvature = 1.0 / radius;

    return curvature;
}

void new_C() {

	// 读入输入图像
    cv::Mat input_image = cv::imread("D:\\Ronnie\\pakges\\bldataset\\test_algo\\2.jpg", cv::IMREAD_GRAYSCALE);
    if (input_image.empty())
    {
        std::cout << "Open image file failed..." << std::endl;
        return;
    }
    
    // 转换为二值图像
    cv::Mat binary_image;
    cv::threshold(input_image, binary_image, 128, 255, cv::THRESH_BINARY);

    // 边缘检测
    cv::Mat edges;
    cv::Canny(binary_image, edges, 50, 150);
	cv::imshow("edge", edges);
    // 形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // 修改kernel的尺寸以获得更好的效果
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);
	cv::imshow("edge", edges);
    // 霍夫曼分析算法找出直线段
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 30, 10);

    // 筛选C形环
    std::vector<cv::Vec4i> c_rings;
    for (size_t i = 0; i < lines.size(); i++)
    {
        // 计算直线段的中心点
        cv::Point center((lines[i][0] + lines[i][2]) / 2, (lines[i][1] + lines[i][3]) / 2);

        // 计算直线段的长度和曲率
        double length = cv::norm(cv::Point(lines[i][0], lines[i][1]) - cv::Point(lines[i][2], lines[i][3]));
        double curvature = calculate_curvature(lines[i]);

        // 判断是否为C形环
        if (length > 5 && length < 2000 && std::abs(curvature) < 0.8 && std::abs(curvature) > 0.1)
        {
            c_rings.push_back(lines[i]);
        }
    }

    // 输出标记后的结果图像
    cv::Mat output_image;
    cv::cvtColor(input_image, output_image, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < c_rings.size(); i++)
    {
        cv::Point pt1(c_rings[i][0], c_rings[i][1]);
        cv::Point pt2(c_rings[i][2], c_rings[i][3]);
        cv::line(output_image, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("C-rings", output_image);
    cv::waitKey(0);
}

void test_algo() {
	Mat img = imread("D:\\Ronnie\\pakges\\bldataset\\test_algo\\_20230608204301955.BMP");
    Mat gray, thresh;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 127, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		cv::Mat mask(img.size(), CV_8UC1, Scalar(0));
		if (area  < 500 && area > 100) {
			drawContours(mask, contours, i, Scalar(255), cv::FILLED);
			cv::imshow("mask",mask);
			cv::imshow("thresh", thresh);
			imwrite("contour1.jpg", mask);
			cv::waitKey(0);

		}
		

	}


}

void test_ini() {

	char buffer[1024];
    GetPrivateProfileString("Algorithm Params", "C Threshold Value", "", buffer, 1024, "D:\\Ronnie\\pakges\\baolong_tank_algo\\src\\Algorithm.ini");
    std::cout << "Value of Section1/Key1: " << buffer << std::endl;
    
	int k = GetPrivateProfileInt("Algorithm Params", "O Threshold Value", 0, "D:\\Ronnie\\pakges\\baolong_tank_algo\\src\\Algorithm.ini");
    std::cout << "Value of Section1/Key1: " << k << std::endl;


	return;
}


int main()
{

	// test_tapp();
	run();
	// test_ini();
	// test_algo();
	// new_C();
	// test_hist();
	// system("pause");
	// test_new_json();
	// test_img();
	system("pause");
    return 0;
}