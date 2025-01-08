#include "stamp_det.h"
#include "logger.h"
#include "utils.h"

#include <filesystem>
#include "defines.h"

namespace fs = std::filesystem;

StampDet::StampDet(json info):
    m_info(info)
{
}

void StampDet::config(json config, RefImgTool *ref)
{
    m_ref = ref;
    m_config = config;
}

json StampDet::forward(cv::Mat img, const json& in_param)
{
    json all_out = json::array();
    if (!m_ref->has_stamp()) return all_out;

    PaperType ptype = get_paper_type(in_param);
    std::string ptype_str = get_paper_type_str(ptype);

    if (ptype_str == "UNKNOWN") return all_out;

    get_param(m_config, std::string("stamp_template_dir"),  std::string(""));
    std::string stamp_template_dir = get_param(m_config, std::string("stamp_template_dir"),  std::string(""));
    fs::path stamp_ref_path(stamp_template_dir);
    stamp_ref_path.append(ptype_str+"_stamp.jpg");
    if (stamp_template_dir.size() == 0 || !fs::exists(stamp_ref_path.string()))
    {
        LOG_WARN("Stamp template path wrong: {}", stamp_ref_path.string());
        return all_out;
    }

    // 读取印章模板图，并二值化
    cv::Mat stamp_temp = cv::imread(stamp_ref_path.string(), cv::IMREAD_GRAYSCALE);
    cv::Mat stamp_temp_bin;
    cv::threshold(stamp_temp, stamp_temp_bin, 127, 255, cv::THRESH_BINARY);
    write_debug_img("./gtmc_debug/stamp_temp_bin.jpg", stamp_temp_bin);
    
    cv::Mat stamp_infer_bin;
    int center_xoff, center_yoff; // 印章中心点偏移值
    // 获取推理图印章，并矫正，对齐模板
    cv::Mat stamp_infer = m_ref->get_stamp_img_keep_black(img, stamp_temp, in_param, center_xoff, center_yoff);
    if (stamp_infer.rows == 0) { // 没有找到印章
            json rst_points = m_ref->transform_result(bbox2polygon(m_ref->get_stamp_bbox()));
            json out = {
                {"label", "NG4"},
                {"shapeType", "polygon"},
                {"points", rst_points},
                {"result", {{"confidence", 1.0f}, {"area", 99999}}},
            };
            all_out.push_back(out);
        return all_out;
    }
    // 推理印章二值化
    cv::cvtColor(stamp_infer, stamp_infer, cv::COLOR_RGB2GRAY);
    cv::threshold(stamp_infer, stamp_infer_bin, 210, 255, cv::THRESH_BINARY_INV);
    cv::morphologyEx(stamp_infer_bin, stamp_infer_bin, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));
    write_debug_img("./gtmc_debug/stamp_infer_bin.jpg", stamp_infer_bin);

    // 二值图相减
    cv::Mat stamp_diff(stamp_infer_bin.rows, stamp_infer_bin.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat mask = (stamp_temp_bin>0) & (stamp_infer_bin==0);
    stamp_diff.setTo(255, mask);
    write_debug_img("./gtmc_debug/stamp_diff_img0.jpg", stamp_diff);

    cv::morphologyEx(stamp_diff, stamp_diff, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,4)));
    cv::morphologyEx(stamp_diff, stamp_diff, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50,50)));
    write_debug_img("./gtmc_debug/stamp_diff_img.jpg", stamp_diff);

    // 剩余面积过滤
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(stamp_diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    json stamp_bbox = m_ref->get_stamp_bbox();
    int size_xoff = (1080 - (stamp_bbox[1][0] - stamp_bbox[0][0])) / 2;
    int size_yoff = (1080 - (stamp_bbox[1][1] - stamp_bbox[0][1])) / 2;

    int area_threshold = get_param<int>(in_param, "stamp_area_threshold", 50);
    for (auto cont : contours)
    {
        double area = cv::contourArea(cont);
        LOG_INFO("Stamp diff area: {}  threshold:{}", area, area_threshold);
        if (area > area_threshold) {
            cv::Rect box = cv::boundingRect(cont);
            box.x -= (center_xoff + 30 + size_xoff);
            box.y -= (center_yoff + 30 + size_yoff);
            box.width += 60;
            box.height += 60;
            json points = {
                box.x, box.y, (box.x + box.width), box.y,
                (box.x + box.width), (box.y + box.height), box.x, (box.y + box.height)
            };
            json rst_points = m_ref->transform_stamp_result(points);
            json out = {
                {"label", "NG4"},
                {"shapeType", "polygon"},
                {"points", rst_points},
                {"result", {{"confidence", 1.0f}, {"area", area}}},
            };
            all_out.push_back(out);
        }
    }
    if (all_out.size() > 0) {
        LOG_INFO("Stamp NG detect!");
    } else {
        LOG_INFO("Stamp OK!");
    }

    return all_out;
}
