#pragma once
#include <fstream>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc/types_c.h>
#include <nlohmann/json.hpp>
#include "logger.h"
#include "utils.h"
#include "defines.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct CharRect {
    CharRect(char c, float score, cv::Rect2i& rect): m_char(c), m_rect(rect), m_score(score){}
    inline static bool cmp_x(CharRect& a, CharRect& b) {
        if(a.m_rect.x < b.m_rect.x) {
            return true;
        }
        return false;
    }

    inline static float iou(CharRect a, CharRect b) {
        return (float)(a.m_rect & b.m_rect).area() / (float)(a.m_rect | b.m_rect).area();
    }

    inline static float min_iou(CharRect a, CharRect b) {
        cv::Rect min_rect = a.m_rect.area() < b.m_rect.area() ? a.m_rect : b.m_rect;
        return (float)(a.m_rect & b.m_rect).area() / (float)min_rect.area();
    }

    inline static void swap(CharRect& a, CharRect& b) {
        CharRect c = a;
        a = b;
        b = c;
    }

    char m_char;
    cv::Rect2i m_rect;
    float m_score;
};

class CharDefectDetAlgo {
 public:
    CharDefectDetAlgo();
    typedef std::shared_ptr<CharDefectDetAlgo> m_ptr;
    void config(std::string ref_img_dir, int bin_thr, int error_std, int morph_size, const std::string& papar_type, int ng3_min_area=30);
    std::vector<cv::Rect2i> forward(cv::Mat img, std::string label);
    bool set_ref_img_type(const std::string& ref_img_dir);

 private:
    inline static bool cmp_poly_by_x(const std::vector<cv::Point>& a, const std::vector<cv::Point>& b);
    bool defect_det_by_pixarea(cv::Mat img, cv::Mat ref_img, int error_std);
    bool defect_det_by_binary(cv::Mat img, cv::Mat ref_img, int bin_thr, int min_area);
    cv::Mat calbrate_img_by_template(cv::Mat img, cv::Mat ref_img);
    cv::Mat calbrate_img_by_ecc(cv::Mat img, cv::Mat ref_img);
    bool det_GQFT(cv::Mat img);
    void merge_result(cv::Rect box, json &all_out);
    bool char_defect_by_ecc(cv::Mat ref_img, cv::Mat img, int bin_thr, std::string label);
    void blob_rect(cv::Mat bin_img, int min_area, std::vector<std::vector<cv::Point> > &blob_region);
    void blob_rect(cv::Mat bin_img, int min_area, std::vector<cv::Rect> &blob_region);
    void detect_color_region(cv::Mat img,
                         cv::Scalar low_val,
                         cv::Scalar high_val,
                         std::vector<cv::Rect>& out_rects);

    void split_char_img(cv::Mat input,
                        std::vector<cv::Mat> &out_imgs,
                        std::vector<cv::Rect> &blob_region);

    inline static bool cmp_rect_by_x(cv::Rect a, cv::Rect b);
    void gen_label_bin_img(cv::Mat& img, const std::string& label, cv::Mat &dst_img);
    bool defect_by_blob_nums(cv::Mat img, const std::string& label);
    bool char_defect_detection(const cv::Mat &char_img_o, std::string label);
    int get_char_pix_area(cv::Mat char_img_o, int bin_thr);
    int get_pix_area(cv::Mat input_img, int pix_value);
    void get_img_gradian(const cv::Mat& img, cv::Mat& gradian_img);

 private:
    std::string m_ref_img_dir;
    std::string m_ref_img_root;
    std::string m_paper_type;
    std::string m_name;
    int m_bin_thr{200};
    int m_error_std;
    int m_morph_size;
    int m_defect_min_area;
    float m_gray_scale{2};
    int m_ng3_min_area{30};
};