#pragma once
#include<iostream>
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
// #include <windows.h>
// #include <tchar.h>
#include "logger.h"
// #pragma comment(lib, "kernel32.lib")

#define DEBUG_SHOW_C
#define DEBUG_SHOW_O
#define DEBUG_SHOW_PLUG
#define DEBUG_CIRCLE_EA
#define DEBUG_CIRCLE_HD
#define DEBUG_SHOW_C_RING
#define DEBUG_SHOW_O_RING

// #define MAX_CUR_PATH 100
// TCHAR buffer[MAX_CUR_PATH];
// GetCurrentDirectory(MAX_CUR_PATH, buffer);
// std::wcout << "Current directory is: " << buffer << std::endl;

#define Algorithm_path './Algorithm.ini'
struct params_C {

    
};
struct params_O {
    std::vector<int> white_L;
    std::vector<int> white_H;
    int min_area;
    int max_area;
    float rate;
    int maxline;
    float judge_cross_ratio;
    float judge_area_plug_ratio;
    float max_nm;

    // 构造函数
    params_O() : 
        white_L({0, 0, 242}), 
        white_H({180, 20, 255}), 
        min_area(6000), 
        max_area(40000), 
        rate(0.7f), 
        maxline(368), 
        judge_cross_ratio(0.7f), 
        judge_area_plug_ratio(0.35f), 
        max_nm(1.7f) {}

    // 带参数的构造函数
    params_O(std::vector<int> wL, std::vector<int> wH, int minA, int maxA, float rt, int mxl, float jcr, float jap, float mxn) :
        white_L(wL), 
        white_H(wH), 
        min_area(minA), 
        max_area(maxA), 
        rate(rt), 
        maxline(mxl), 
        judge_cross_ratio(jcr), 
        judge_area_plug_ratio(jap), 
        max_nm(mxn) {}
};
struct params_plug {

    
}; 
enum DebugType
{
    CIRCLE_EA = 0, CIRCLE_HD, FIND_C, FIND_O, FIND_PLUG, FIND_C_RING, FIND_O_RING
};

enum BlClassType
{
    C = 0, O, PLUG, C_RING, O_RING
};
//OK
#define STATUS_OK 0

void write_debug_img(std::string fpath, cv::Mat img, DebugType debug_type);
void write_rgb_img(std::string fpath, cv::Mat img);

// void init_params(params_O& params)
// {
//     TCHAR ini_file[] = _T("config.ini");
    
//     // Read [WHITE] section
//     TCHAR white_L[256], white_H[256];
//     GetPrivateProfileString(_T("WHITE"), _T("L"), _T(""), white_L, 256, ini_file);
//     GetPrivateProfileString(_T("WHITE"), _T("H"), _T(""), white_H, 256, ini_file);

//     std::vector<int> L_values, H_values;
//     int value;
//     TCHAR* next_token;

//     // Parse comma-separated integers in white_L and white_H
//     for (TCHAR* token = _tcstok_s(white_L, _T(","), &next_token); token != nullptr; token = _tcstok_s(nullptr, _T(","), &next_token))
//     {
//         if (_stscanf_s(token, _T("%d"), &value) == 1)
//         {
//             L_values.push_back(value);
//         }
//     }

//     for (TCHAR* token = _tcstok_s(white_H, _T(","), &next_token); token != nullptr; token = _tcstok_s(nullptr, _T(","), &next_token))
//     {
//         if (_stscanf_s(token, _T("%d"), &value) == 1)
//         {
//             H_values.push_back(value);
//         }
//     }

//     params.white_L = L_values;
//     params.white_H = H_values;

//     // Read [PARAMS] section
//     params.min_area = GetPrivateProfileInt(_T("PARAMS"), _T("min_area"), 0, ini_file);
//     params.max_area = GetPrivateProfileInt(_T("PARAMS"), _T("max_area"), 0, ini_file);
//     params.rate = (float)GetPrivateProfileInt(_T("PARAMS"), _T("rate"), 0, ini_file) / 100.0f;
//     params.maxline = GetPrivateProfileInt(_T("PARAMS"), _T("maxline"), 0, ini_file);
//     params.judge_cross_ratio = (float)GetPrivateProfileInt(_T("PARAMS"), _T("judge_cross_ratio"), 0, ini_file) / 100.0f;
//     params.judge_area_plug_ratio = (float)GetPrivateProfileInt(_T("PARAMS"), _T("judge_area_plug_ratio"), 0, ini_file) / 100.0f;
    
//     TCHAR max_nm_str[256];
//     GetPrivateProfileString(_T("PARAMS"), _T("max_nm"), _T(""), max_nm_str, 256, ini_file);
//     _stscanf_s(max_nm_str, _T("%f"), &params.max_nm);
// }

