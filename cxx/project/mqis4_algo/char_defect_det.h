#pragma once
#include <fstream>
#include <filesystem>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc/types_c.h>
#include <nlohmann/json.hpp>
#include "logger.h"
#include "utils.h"
#include "defines.h"
#include "ref_img_tool.h"
#include "char_defect_det_algo.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// /**
//  * @brief NG3检测参数
//  * 
//  */
// struct NG3DetParam {
//    typedef std::shared_ptr<NG3DetParam> m_ptr;
//    int m_morph_size{7};                             // 膨胀尺寸
//    int m_defect_area_thr{30};                       // NG3 缺陷最小面积阈值
//    int m_bin_thr{-1};                               // NG3 不同版面的二值化阈值
// }

/**
 * 法规字符严格检测（法律规定必须打印清晰、不能有错误的内容），主要针对发动机号、车架号等内容检查脏污和残缺
 * 算法主要通过与标准模板配准后相减来实现
*/
class CharDefectDet {
 public:
    CharDefectDet();
    void config(json config);
    void config(json config, RefImgTool *ref);
    json forward(cv::Mat img, json in_param);

 private:
    /**
     * @brief  运行时读取配置 
     * 
     * @param in_param : 运行时参数
     * @return true
     * @return false 
     */
    void config_runtime(const json& in_param);

    bool read_all_json_file(const std::string& strFileName, json& ref_char_info);

    void split_char_img(cv::Mat input,
                        std::vector<cv::Mat> &out_imgs,
                        std::vector<cv::Rect> &char_region);

    int get_pix_area(cv::Mat input_img, int pix_value);

    void merge_result(cv::Rect box, json tfm_pts, std::string task_name, json &all_out);

    bool update_paramter();
    bool update_paramter(const json& runtime_config);

 private:
    json m_config;
    json m_char_defect_param;
    json m_char_pixarea_info;
    json m_ref_char_info;
    std::string m_ref_dir;
    int m_bin_thr{200};
    bool m_use_ecc{true};
    std::string m_paper_name;
    int m_morph_size{9};
    float m_gray_scale{1.5};
    int m_char_pix_std{200};
    int m_ng3_min_area{30};

    RefImgTool *m_ref{nullptr};
    std::shared_ptr<CharDefectDetAlgo> m_char_defect_algo;
};