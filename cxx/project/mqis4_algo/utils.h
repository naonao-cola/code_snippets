#pragma once
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "defines.h"

using json = nlohmann::json;


wchar_t* AnsiToUnicode(const char* lpszStr);
char* UnicodeToAnsi(const wchar_t* lpszStr);
char* AnsiToUtf8(const char* lpszStr);
char* Utf8ToAnsi(const char* lpszStr);
char* UnicodeToUtf8(const wchar_t* lpszStr);
wchar_t* Utf8ToUnicode(const char* lpszStr);

bool AnsiToUnicode(const char* lpszAnsi, wchar_t* lpszUnicode, int nLen);
bool UnicodeToAnsi(const wchar_t* lpszUnicode, char* lpszAnsi, int nLen);
bool AnsiToUtf8(const char* lpszAnsi, char* lpszUtf8, int nLen);
bool Utf8ToAnsi(const char* lpszUtf8, char* lpszAnsi, int nLen);
bool UnicodeToUtf8(const wchar_t* lpszUnicode, char* lpszUtf8, int nLen);
bool Utf8ToUnicode(const char* lpszUtf8, wchar_t* lpszUnicode, int nLen);

std::wstring AnsiToUnicode(const std::string& strAnsi);
std::string UnicodeToAnsi(const std::wstring& strUnicode);
std::string AnsiToUtf8(const std::string& strAnsi);
std::string Utf8ToAnsi(const std::string& strUtf8);
std::string UnicodeToUtf8(const std::wstring& strUnicode);
std::wstring Utf8ToUnicode(const std::string& strUtf8);

void draw_polygon(cv::Mat& image, const json& pts, const cv::Scalar& color);
std::vector<cv::Point2f> get_rotrect_coords(const json& xywhr, bool lt_first = true);
// std::vector<cv::Point> get_minrect_points(cv::RotatedRect rot_rect, bool lt_first = true);
cv::Mat d6_to_cvMat(double d0, double d1, double d2, double d3, double d4, double d5);
cv::Mat cvMat6_to_cvMat9(const cv::Mat &mtx6);
std::vector<cv::Point> json_to_cv_pts(const json& pts);
cv::Mat vector_angle_to_M(const LocateInfo& v1, const LocateInfo& v2);
cv::Mat vector_angle_to_M(double x1, double y1, double d1, double x2, double y2, double d2);
json affine_points(const cv::Mat& M, const json& pts);
double p2p_distance(double x1, double y1, double x2, double y2);
cv::Mat points2d_to_mat(json points);
json bbox2polygon(double x1, double y1, double x2, double y2);
json bbox2polygon(json bbox);
json polygon2bbox(json polygon);
bool is_intersect(json bbox1, json bbox2);
json bbox_intersect(json bbox1, json bbox2);
cv::RotatedRect cv_rotrect_to_halcon_rotrect(cv::RotatedRect rotrect);

cv::Mat gray_scale_image(cv::Mat img, int r1, int r2, int s1=0, int s2=255);
PaperType get_paper_type(json img_param);
std::string get_paper_type_str(PaperType ptype);
void write_rgb_img(std::string fpath, cv::Mat img, bool cvtRGB=false);
void write_debug_img(std::string fpath, cv::Mat img, bool cvtRGB=false, DebugType dbg_type = DebugType::NORMAL);

template<typename T>
T get_param(const json& param, const std::string& key, const T& def_val)
{
    if (param.contains(key)) {
        return param[key].get<T>();
    } else {
        return def_val;
    }
}

template<typename T>
void get_rotrect_size(std::vector<T> pts, int& width, int& height)
{
    assert(pts.size() == 4);
    double ltx = pts[0].x;
    double lty = pts[0].y;
    double rtx = pts[1].x;
    double rty = pts[1].y;
    double rbx = pts[2].x;
    double rby = pts[2].y;
    width = std::round(std::sqrt((ltx-rtx)*(ltx-rtx) + (lty-rty)*(lty-rty)));
    height = std::round(std::sqrt((rtx-rbx)*(rtx-rbx) + (rty-rby)*(rty-rby)));
}

template<typename T>
void sort_rotrect_pts(std::vector<T>& pts)
{
    assert(pts.size() == 4, "Wrong points size!");
    int lt_idx = 0;
    int min_val = 999999;
    for (int i = 0; i < 4; i++)
    {
        if (pts[i].x + pts[i].y < min_val)
        {
            min_val = pts[i].x + pts[i].y;
            lt_idx = i;
        }
    }

    // 排序左上角第一个
    std::vector<T> sort_pts = std::vector<T>(pts.size());
    for(int i = 0; i < pts.size(); i++) {
        sort_pts[i] = pts[(lt_idx+i) % pts.size()];
    }
    pts = sort_pts;
}
