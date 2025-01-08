#include "ref_img_tool.h"
#include "logger.h"
// #include <tvcore.h>
#include <iomanip>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include "utils.h"
#include "TAPI.h"
#include "algo_tool.h"

RefImgTool::RefImgTool()
{
}

void RefImgTool::config(const json& config)
{
    std::string img_path = config["ref_img"];
    m_mask_shapes = config["mask_shapes"];

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 裁剪纸张区域
    std::vector<cv::Point2f> coords;
    get_paper_loc(img, coords);
    // LOG_INFO("config paper_roi coords: {}", paper_roi.dump());
    int out_w, out_h;
    m_ref_paper_M = get_perspective_M(coords, 0, 0, out_w, out_h);
    cv::warpPerspective(img, m_ref_img, m_ref_paper_M, cv::Size(out_w, out_h), cv::INTER_CUBIC);
    cv::cvtColor(m_ref_img, m_ref_gray_img, cv::COLOR_RGB2GRAY);

    write_debug_img("./gtmc_debug/ref_crop.jpg", m_ref_img);


    // 根据两个mark点定位文档位置和方向，推理时用于矫正roi区域
    for (auto shape : m_mask_shapes)
    {
        if (shape["label"] == "mark_a") {
            m_mark_a = shape["points"];
            // LOG_INFO("@@@@ markA: {}", m_mark_a.dump());
        } else if (shape["label"] == "mark_b") {
            m_mark_b = shape["points"];
            // LOG_INFO("@@@ markB: {}", m_mark_b.dump());
        }
    }
    if (m_mark_a.empty() || m_mark_a.empty()) {
        LOG_ERROR("Mark point not found!");
        return;
    }

    json mark_a_ply = transform_roi(bbox2polygon(m_mark_a), true);
    json mark_b_ply = transform_roi(bbox2polygon(m_mark_b), true);

    get_roi_img(m_ref_gray_img, m_mark_a_temp, mark_a_ply, 0, 0, TFM_NONE);
    get_roi_img(m_ref_gray_img, m_mark_b_temp, mark_b_ply, 0, 0, TFM_NONE);

    // get_pad_roi_img(m_ref_gray_img, m_mark_a_temp, bbox2polygon(m_mark_a), 0, TFM_REF);
    // get_pad_roi_img(m_ref_gray_img, m_mark_b_temp, bbox2polygon(m_mark_b), 0, TFM_REF);
    double a_tx = 0, a_ty = 0, b_tx = 0, b_ty = 0;
    for (int i = 0; i < mark_a_ply.size() / 2; i++) {
        a_tx += mark_a_ply[i*2];
        b_tx += mark_b_ply[i*2];
        a_ty += mark_a_ply[i*2 + 1];
        b_ty += mark_b_ply[i*2 + 1];
    }
    double a_cx = a_tx / 4.0;
    double a_cy = a_ty / 4.0;
    double b_cx = b_tx / 4.0;
    double b_cy = b_ty / 4.0;

    m_ref_loc.angle = Tival::TAPI::AngleLX(a_cx, a_cy, b_cx, b_cy);

  
    m_ref_loc.x1 = a_cx;
    m_ref_loc.y1 = a_cy;
    m_ref_loc.x2 = b_cx;
    m_ref_loc.y2 = b_cy;
    m_ref_loc.x = (a_cx + b_cx) / 2.0;
    m_ref_loc.y = (a_cy + b_cy) / 2.0;

    // Sign image extract
    m_stamp_bbox = json::array();
    for (auto shape : m_mask_shapes)
    {
        if (shape["label"] == "stamp") {
            m_stamp_bbox = shape["points"];
            break;
        }
    }
}

cv::Mat RefImgTool::get_ref_img(bool gray)
{
    return gray ? m_ref_gray_img : m_ref_img;
}

cv::Mat RefImgTool::set_test_img(cv::Mat &img, const json& in_param, bool& locate_ok)
{
    PaperType ptype = get_paper_type(in_param);
    m_ptype_str = get_paper_type_str(ptype);
    // std:: array no_scale_types = {HBZ_B_CD, HBZ_B_HD1, HBZ_B_HD2, HBZ_B_RY1, HBZ_B_RY2, RYZ_HD, RYZ_RY};
    // bool is_no_scale = std::find(no_scale_types.begin(), no_scale_types.end(), ptype) != no_scale_types.end();

    // 对比度拉伸，去除水印
    // cv::Mat org_img = img;
    // if (!is_no_scale) {
    //     gray_scale_image(img, 0, 210).convertTo(org_img, CV_8U);
    // }

    // 裁剪纸张区域
    std::vector<cv::Point2f> coords;
    get_paper_loc(img, coords);
    int out_w, out_h;
    m_paper_M = get_perspective_M(coords, 0, 0, out_w, out_h);

    cv::Mat crop_img, crop_gray;
    cv::warpPerspective(img, crop_img, m_paper_M, cv::Size(out_w, out_h), cv::INTER_CUBIC);
    cv::resize(crop_img, crop_img, cv::Size(m_ref_img.cols, m_ref_img.rows), cv::INTER_CUBIC);
    cv::cvtColor(crop_img, crop_gray, cv::COLOR_RGB2GRAY);

    // 根据mark点定位
    cv::Rect pad_a_rect = get_pad_roi_img(crop_gray, m_mark_a_pad, bbox2polygon(m_mark_a), 200, TFM_REF);
    cv::Rect pad_b_rect = get_pad_roi_img(crop_gray, m_mark_b_pad, bbox2polygon(m_mark_b), 200, TFM_REF);

    cv::Mat result_a, result_b;
    cv::matchTemplate(m_mark_a_pad, m_mark_a_temp, result_a, cv::TM_CCOEFF_NORMED);
    cv::matchTemplate(m_mark_b_pad, m_mark_b_temp, result_b, cv::TM_CCOEFF_NORMED);

    double min_val_a, max_val_a, min_val_b, max_val_b;
    cv::Point min_loc_a, max_loc_a, min_loc_b, max_loc_b;
    cv::minMaxLoc(result_a, &min_val_a, &max_val_a, &min_loc_a, &max_loc_a);
    if (max_val_a < 0.5) {
        LOG_WARN("Mark A find fail, max val: {}", max_val_a);
        locate_ok = false;
        return img;
    }

    cv::minMaxLoc(result_b, &min_val_b, &max_val_b, &min_loc_b, &max_loc_b);
    if (max_val_b < 0.5) {
        LOG_WARN("Mark A find fail, max val: {}", max_val_a);
        locate_ok = false;
        return img;
    }

    // 坐标计算到paper
    double a_cx = pad_a_rect.x + max_loc_a.x + m_mark_a_temp.cols / 2.0;
    double a_cy = pad_a_rect.y + max_loc_a.y + m_mark_a_temp.rows / 2.0;
    double b_cx = pad_b_rect.x + max_loc_b.x + m_mark_b_temp.cols / 2.0;
    double b_cy = pad_b_rect.y + max_loc_b.y + m_mark_b_temp.rows / 2.0;

    locate_ok = true;
    m_img_loc.angle = Tival::TAPI::AngleLX(a_cx, a_cy, b_cx, b_cy);

   
    m_img_loc.x1 = a_cx;
    m_img_loc.y1 = a_cy;
    m_img_loc.x2 = b_cx;
    m_img_loc.y2 = b_cy;
    m_img_loc.x = (a_cx + b_cx) / 2.0;
    m_img_loc.y = (a_cy + b_cy) / 2.0;

    m_ref2test_M = vector_angle_to_M(m_ref_loc, m_img_loc);
    return crop_img;
}

void RefImgTool::get_paper_loc(const cv::Mat &img, std::vector<cv::Point2f>& roi)
{
    cv::Mat gray_image;
    if (img.channels() > 1) {
        cv::cvtColor(img, gray_image, cv::COLOR_RGB2GRAY);
    } else {
        gray_image = img;
    }
    // cv::resize(gray_image, gray_image, cv::Size(img.cols/4, img.rows/4));

    cv::Mat bin_img;
    cv::threshold(gray_image, bin_img, 80, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contorus;
    cv::findContours(bin_img, contorus, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto cont : contorus) {
        double area = cv::contourArea(cont);
        if (area > 1000000) {
            cv::RotatedRect rotRect = cv::minAreaRect(cont);
            rotRect.size.width -= 30;
            rotRect.size.height -= 30;
            cv::Point2f pts[4];
            rotRect.points(pts);
            for (int i = 0; i < 4; i++) {
                roi.push_back(pts[i]);
            }
            break;
        }
    }

    sort_rotrect_pts(roi);
}

// pts: 输入矩形四个顶点，顺序左上，右上，右下，左下
template<typename T>
cv::Mat RefImgTool::get_perspective_M(std::vector<T> pts, int width, int height, int& out_w, int& out_h)
{
    get_rotrect_size(pts, out_w, out_h);

    if (width != 0 && height != 0) {
        out_w = int((1.0 * out_w / out_h) * height);

        if (out_w > width) {
            out_h = int((1.0 * out_h / out_w) * width);
        } else {
            out_h = height;
        }
    } else {
        width = out_w;
        height = out_h;
    }

    cv::Point2f src_pts[] = {
        cv::Point2f(pts[0]), cv::Point2f(pts[1]),
        cv::Point2f(pts[2]), cv::Point2f(pts[3]),
    };

    cv::Point2f dst_pts[] = {
        cv::Point2f(0, 0), cv::Point2f(out_w, 0),
        cv::Point2f(out_w, out_h), cv::Point2f(0, out_h),
    };
    
    return cv::getPerspectiveTransform(src_pts, dst_pts);
}

cv::Mat RefImgTool::perspective(cv::Mat &img, std::vector<cv::Point> pts, int width, int height)
{
    int out_w, out_h;
    cv::Mat tfm_img;
    cv::Mat M = get_perspective_M(pts, width, height, out_w, out_h);
    cv::warpPerspective(img, tfm_img, M, cv::Size(out_w, out_h), cv::INTER_CUBIC);

    if ((width != 0 && height != 0) && (out_w != width || out_h != height)) {
        cv::Mat pad_img;
        int bottom = height - out_h;
        int right = width - out_w;
        cv::copyMakeBorder(tfm_img, pad_img, 0, bottom, 0, right, cv::BORDER_CONSTANT, cv::Scalar(0));
        tfm_img = pad_img;
    }

    return tfm_img;
}

json RefImgTool::get_roi_img(cv::Mat img, cv::Mat &roi_img, json roi, int widht, int height, TFM_MODE tfmmode)
{
    if (roi.size() == 4) {
        roi = bbox2polygon(roi[0], roi[1], roi[2], roi[3]);
    }
    json roi_tfm;
    if (tfmmode == TFM_NONE) {
        roi_tfm = roi;
    } else {
        roi_tfm = transform_roi(roi, tfmmode == TFM_REF);
    }

    std::vector<cv::Point> pts;
    for (int i=0; i < roi_tfm.size()/2; ++i) {
        pts.push_back(cv::Point(roi_tfm[i*2], roi_tfm[i*2+1]));
    }
    
    roi_img = perspective(img, pts, widht, height);

    json points = json::array();
    for (auto pt: pts) {
        points.push_back(pt.x);
        points.push_back(pt.y);
    }
    return points;
}

cv::Rect RefImgTool::get_pad_roi_img(cv::Mat img, cv::Mat &roi_img, json roi, double pad_px, TFM_MODE tfmmode) {
    json roi_tfm;
    if (tfmmode == TFM_NONE) {
        roi_tfm = roi;
    } else {
        roi_tfm = transform_roi(roi, tfmmode == TFM_REF);
    }

    std::vector<cv::Point2f> pts;
    for (int i=0; i < roi_tfm.size()/2; ++i) {
        pts.push_back(cv::Point2f(roi_tfm[i*2], roi_tfm[i*2+1]));
    }
    cv::Rect rect = cv::boundingRect(pts);
    //LOG_INFO("rect x:{}, y:{}, w:{}, h:{}", rect.x, rect.y, rect.width, rect.height);

    cv::Rect pad_rect;
    pad_rect.x = rect.x - pad_px;
    pad_rect.y = rect.y - pad_px;
    pad_rect.x = pad_rect.x < 0 ? 0 : pad_rect.x;
    pad_rect.y = pad_rect.y < 0 ? 0 : pad_rect.y;

    //LOG_INFO("@ pad_rect x:{}, y:{}, w:{}, h:{}", pad_rect.x, pad_rect.y, pad_rect.width, pad_rect.height);

    int xe = rect.x + rect.width + pad_px;
    int ye = rect.y + rect.height + pad_px;
    xe = xe >= img.cols ? (img.cols-1) : xe;
    ye = ye >= img.rows ? (img.rows-1) : ye;

    pad_rect.width = xe - pad_rect.x;
    pad_rect.height = ye - pad_rect.y;

    //LOG_INFO("@@ pad_rect x:{}, y:{}, w:{}, h:{}", pad_rect.x, pad_rect.y, pad_rect.width, pad_rect.height);

    // int out_w = std::ceil(rect.width + 2 * pad_px);
    // int out_h = std::ceil(rect.height + 2 * pad_px);

    //LOG_INFO("out_w: {}, out_h: {}", out_w, out_h);

    roi_img = cv::Mat::zeros(pad_rect.height, pad_rect.width, img.type());
    // cv::Rect roi_rect;
    // roi_rect.x = (out_w - pad_rect.width)/2;
    // roi_rect.y = (out_h - pad_rect.height)/2;
    // roi_rect.width = pad_rect.width;
    // roi_rect.height = pad_rect.height;
    //LOG_INFO("roi_rect x:{}, y:{}, w:{}, h:{}", roi_rect.x, roi_rect.y, roi_rect.width, roi_rect.height);
    img(pad_rect).copyTo(roi_img);
    return pad_rect;
}

json RefImgTool::transform_roi(json roi, bool is_ref)
{
    json out_coords = json::array();
    cv::Mat pt_mat = points2d_to_mat(roi);
    cv::Mat paper_trans_M = is_ref ? m_ref_paper_M : m_ref_paper_M * m_ref2test_M;
    cv::Mat pt_tfm_mat = paper_trans_M * pt_mat.t();
    
    for (int i = 0; i < roi.size()/2; i++)
    {
        double x = pt_tfm_mat.at<double>(0, i) / pt_tfm_mat.at<double>(2, i);
        double y = pt_tfm_mat.at<double>(1, i) / pt_tfm_mat.at<double>(2, i);
        out_coords.push_back(x);
        out_coords.push_back(y);
    }
    // LOG_INFO("##  roi input: {}", roi.dump());
    // LOG_INFO("##  roi tfm: {}", out_coords.dump());
    return out_coords;
}

json RefImgTool::transform_stamp_result(json polygon, bool need_round)
{
    json stamp_polygon = bbox2polygon(m_stamp_bbox);
    json tfm_stamp_pts = transform_roi(stamp_polygon, true);
    json tfm_stamp_bbox = polygon2bbox(tfm_stamp_pts);

    for (int i = 0; i < polygon.size()/2; i++)
    {
        polygon[i*2] = polygon[i*2] + tfm_stamp_bbox[0][0];
        polygon[i*2+1] = polygon[i*2+1] + tfm_stamp_bbox[0][1];
    }

    // json out_coords = json::array();
    // cv::Mat pt_mat = points2d_to_mat(polygon);
    // cv::Mat pt_tfm_mat = m_paper_M.inv() *  pt_mat.t();
    // std::vector<double> v_pts =  (std::vector<double>)(pt_tfm_mat.reshape(1, 1));
    
    // for (int i = 0; i < polygon.size()/2; i++)
    // {
    //     double x = pt_tfm_mat.at<double>(0, i) / pt_tfm_mat.at<double>(2, i);
    //     double y = pt_tfm_mat.at<double>(1, i) / pt_tfm_mat.at<double>(2, i);
    //     out_coords.push_back(need_round ? std::round(x) : x);
    //     out_coords.push_back(need_round ? std::round(y) : y);
    // }
    return transform_result(polygon, need_round);
}

json RefImgTool::transform_result(json polygon, bool need_round)
{
    json out_coords = json::array();
    cv::Mat pt_mat = points2d_to_mat(polygon);
    cv::Mat pt_tfm_mat = m_paper_M.inv() *  pt_mat.t();
    std::vector<double> v_pts =  (std::vector<double>)(pt_tfm_mat.reshape(1, 1));
    
    for (int i = 0; i < polygon.size()/2; i++)
    {
        double x = pt_tfm_mat.at<double>(0, i) / pt_tfm_mat.at<double>(2, i);
        double y = pt_tfm_mat.at<double>(1, i) / pt_tfm_mat.at<double>(2, i);
        out_coords.push_back(need_round ? std::round(x) : x);
        out_coords.push_back(need_round ? std::round(y) : y);
    }
    return out_coords;
}

bool RefImgTool::is_intersect_with_stamp(json region)
{
    if (!has_stamp()) return false;
    json stamp_polygon = bbox2polygon(m_stamp_bbox);
    json tfm_stamp_pts = transform_roi(stamp_polygon, true);
    return is_intersect(polygon2bbox(tfm_stamp_pts), polygon2bbox(region));
}

cv::Mat RefImgTool::get_stamp_img(cv::Mat img, const json& in_param)
{
    cv::Mat stamp_img;
    if (!has_stamp()) {
        return stamp_img;
    }

    // cv::Mat paper_img = img.clone();
    json tfm_stamp_pts;
    cv::Rect pad_yz_rect;
    json stamp_polygon = bbox2polygon(m_stamp_bbox);

    cv::Mat stamp_org;
    pad_yz_rect = get_pad_roi_img(img, stamp_org, stamp_polygon, 100, TFM_REF);
    // get_roi_img(img, stamp_org, stamp_polygon, 0, 0, TFM_REF);
    double red_ratio = get_param(in_param, "stamp_red_ratio", m_ptype_str == "HBZ" ? 1.8 : 2.1);
    stamp_img = stamp_extract(stamp_org, red_ratio, true);
    write_debug_img("./gtmc_debug/msae_stamp.jpg", stamp_img, true);
    return stamp_img;
}

cv::Mat RefImgTool::get_stamp_img_keep_black(cv::Mat img, cv::Mat stamp_temp, const json& in_param, int& xoff, int& yoff)
{
    cv::Mat stamp_img;
    if (!has_stamp()) {
        return stamp_img;
    }

    // cv::Mat paper_img = img.clone();
    json tfm_stamp_pts;
    cv::Rect pad_yz_rect;
    json stamp_polygon = bbox2polygon(m_stamp_bbox);

    cv::Mat stamp_org;
    pad_yz_rect = get_pad_roi_img(img, stamp_org, stamp_polygon, 100, TFM_REF);
    write_debug_img("./gtmc_debug/stamp_org.jpg", stamp_org);

    cv::Mat stamp_mask;
    // cv::inRange(stamp_org, cv::Scalar(120, 20, 20), cv::Scalar(255, 120, 100), stamp_mask);
    cv::inRange(stamp_org, cv::Scalar(120, 20, 20), cv::Scalar(255, 100, 100), stamp_mask);
    write_debug_img("./gtmc_debug/stamp_mask.jpg", stamp_mask);

    // 封闭缺口，去除毛刺，提取红色印章区域
    cv::morphologyEx(stamp_mask, stamp_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    //   cv::medianBlur(stamp_mask, stamp_mask, 11);
    cv::morphologyEx(stamp_mask, stamp_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(150,150)));
    cv::morphologyEx(stamp_mask, stamp_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10)));
    write_debug_img("./gtmc_debug/stamp_mask2.jpg", stamp_mask);

    // 找边缘轮廓
    cv::Mat mask_bin;
    std::vector<std::vector<cv::Point> > contours;
    cv::threshold(stamp_mask, mask_bin, 50, 255, cv::THRESH_BINARY);
    cv::findContours(mask_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<cv::Point> stamp_cont;
    for (auto cont : contours) {
        double area = cv::contourArea(cont);
        if (area > 300000) // 740949
        {
            stamp_cont = cont;
            break;
        }
    }

    if (stamp_cont.size() < 3) {
        LOG_WARN("Find stamp contour fail!");
        return stamp_img;
    }

    // 轮廓拟合椭圆，投影矫正为标准圆，直径970
    int stamp_img_w = 1080;
    int stamp_img_h = 1080;
    int stamp_radius = 970;
    cv::RotatedRect minrect = cv::fitEllipse(stamp_cont);
    // cv::RotatedRect minrect2 = cv::minAreaRect(stamp_cont);

    int img_cx = mask_bin.cols/2;
    int img_cy = mask_bin.rows/2;
    float x_off = img_cx - minrect.center.x;  // 印章中心和图片中心的偏差
    float y_off = img_cy - minrect.center.y;  // 印章中心和图片中心的偏差
    xoff = int(x_off);
    yoff = int(y_off);
    cv::RotatedRect minrect_correct(minrect);
    minrect_correct.size.width = stamp_radius;
    minrect_correct.size.height = stamp_radius;

    cv::Point2f points1[4], points2[4];
    minrect.points(points1);
    minrect_correct.points(points2);
    for (int i = 0; i < 4; i++) {
        // 移动到图片中心
        points2[i].x += x_off;
        points2[i].y += y_off;
    }
    cv::Mat M = cv::getPerspectiveTransform(points1, points2);
    cv::Mat correct_img;
    cv::warpPerspective(stamp_org, correct_img, M, cv::Size(stamp_org.cols, stamp_org.rows), cv::INTER_CUBIC);

    // 圆度矫正后，通过模板匹配配准
    cv::Mat correct_gray, correct_bin;
    cv::cvtColor(correct_img, correct_gray, cv::COLOR_RGB2GRAY);
    cv::threshold(correct_gray, correct_bin, 180, 255, cv::THRESH_BINARY_INV);
    cv::Mat match_r;
    cv::matchTemplate(correct_bin, stamp_temp, match_r, cv::TM_CCOEFF_NORMED);

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(match_r, &min_val, &max_val, &min_loc, &max_loc);

    write_debug_img("./gtmc_debug/stamp_correct_bin.jpg", correct_bin);
    write_debug_img("./gtmc_debug/stamp_temp.jpg", stamp_temp);

    LOG_INFO("Stamp match score: {}", max_val);
    cv::Point start_pt(img_cx - stamp_img_w/2, img_cy - stamp_img_h/2);
    if (max_val > 0.35) {
        start_pt = max_loc;
    }

    cv::Mat correct_crop_img = correct_img(cv::Rect(start_pt.x, start_pt.y, stamp_img_w, stamp_img_h));
    return correct_crop_img;
}

json RefImgTool::get_shape_pts_by_name(std::string name)
{
    for (auto shape : m_mask_shapes)
    {
        if (shape["label"] == name) {
            return shape["points"];
        }
    }
    return json::array();
}

cv::Mat RefImgTool::get_masked_img(cv::Mat img, const json& in_param)
{
    PaperType ptype = get_paper_type(in_param);
    std::string ptype_str = get_paper_type_str(ptype);

    cv::Mat paper_img = img.clone();
    
    cv::Mat stamp_img;
    json tfm_stamp_pts;
    cv::Rect pad_yz_rect;
    if (has_stamp() && ptype_str != "HBZ_B") {
        json stamp_polygon = bbox2polygon(m_stamp_bbox);
        tfm_stamp_pts = transform_roi(stamp_polygon, true);

        cv::Mat stamp_pad_img;
        pad_yz_rect = get_pad_roi_img(paper_img, stamp_pad_img, stamp_polygon, 0, TFM_REF);
        double red_ratio = get_param<double>(in_param, std::string("stamp_red_ratio"), m_ptype_str == "HBZ" ? 1.8 : 1.9);
        stamp_img = stamp_extract(stamp_pad_img, red_ratio, false);
#ifdef DEBUG_ON
        cv::Mat bgr_yz;
        cv::cvtColor(stamp_img, bgr_yz, cv::COLOR_RGB2BGR);
#endif
    }

    // 对比度拉伸，去除水印、浅色背景
    // if (ptype_str == "HGZ_B") {
    //     gray_scale_image(paper_img, 0, 210).convertTo(paper_img, CV_8U);
    // }

    for (auto shape : m_mask_shapes)
    {
        std::string label_name = shape["label"];
        if (label_name == "mark_a" || label_name == "mark_b" || label_name == "stamp" || label_name == "blank") continue;
        if (ptype_str == "HBZ_B" && label_name != "qrcode") continue;
        if (ptype_str == "HGZ_A" && label_name != "hgz_barcode") continue;
        if (ptype_str == "HGZ_B" && label_name == "qrcode") continue;

        json roi_bbox = shape["points"];

        // HGZ条码去除下方文字
        if (label_name == "hgz_barcode") {
            cv::Mat barcode_img;
            json barcode_pts = bbox2polygon(roi_bbox);
            cv::Rect barcode_pad_rc = get_pad_roi_img(paper_img, barcode_img, barcode_pts, 0, TFM_INFER);
            std::vector<cv::Point> zxm_char_coords;
            cv::Mat barcode_out = barcode_extract(barcode_img, zxm_char_coords);
            barcode_out.copyTo(paper_img(barcode_pad_rc));
            continue;
        }

        json roi_pts = json::array();
        if (shape["shape_type"] == "rectangle") {
            roi_pts = bbox2polygon(roi_bbox);
        } else {
            LOG_WARN("Unsupport shape_type {}", shape["shape_type"]); // TODO: 多边形支持
        }
        json tfm_pts = transform_roi(roi_pts, false);

        // 如果roi和印章不重叠，仅填充颜色
        if (m_stamp_bbox.size() == 0 || !is_intersect(roi_bbox, m_stamp_bbox)) {
            std::vector<cv::Point> vpts;
            for (int i=0; i < tfm_pts.size()/2; ++i) {
                double x = tfm_pts[i*2];
                double y = tfm_pts[i*2+1];
                vpts.push_back(cv::Point(std::round(x), std::round(y)));
            }

            //cv::Scalar fill_color = (ptype_str == "RYZ" || ptype_str == "HBZ_A") ? cv::Scalar(127, 127, 127) : cv::Scalar(255, 255, 255);
            cv::Scalar fill_color = (ptype_str == "RYZ" || ptype_str == "HBZ_A") ? cv::Scalar(127, 127, 127) : cv::Scalar(255, 255, 255);
            //if (ptype_str == "RYZ") {
            //    fill_color = cv::Scalar(255, 255, 255);
            //}
            cv::fillConvexPoly(paper_img, vpts, fill_color);
            continue;
        }

        // roi和印章重叠，提取红色印章，文字部分填充为白色
        json tfm_roi_bbox = polygon2bbox(tfm_pts);
        double tfm_roi_w = tfm_roi_bbox[1][0] - tfm_roi_bbox[0][0];
        double tfm_roi_h = tfm_roi_bbox[1][1] - tfm_roi_bbox[0][1];
        cv::Mat roi_pad_img;
        cv::Rect pad_roi_rect = get_pad_roi_img(paper_img, roi_pad_img, tfm_pts, 0, TFM_NONE);

        // 填充roi区域为白色
        cv::Mat mask(roi_pad_img.rows, roi_pad_img.cols, CV_8U, cv::Scalar(0,0,0));
        std::vector<cv::Point> mask_pts;
        for (int i=0; i < tfm_pts.size()/2; ++i) {
            double x = tfm_pts[i*2];
            double y = tfm_pts[i*2+1];
            x = std::round(x - pad_roi_rect.x);
            y = std::round(y - pad_roi_rect.y);
            mask_pts.push_back(cv::Point(x, y));
        }
        cv::fillConvexPoly(mask, mask_pts, cv::Scalar(1,1,1));
        roi_pad_img.setTo(255, mask > 0);

        // 印章重叠区域，只提取印章部分回填
        json tfm_stamp_bbox = polygon2bbox(tfm_stamp_pts);
        json stamp_inter_bbox = bbox_intersect(tfm_roi_bbox, tfm_stamp_bbox);
        int inter_w = stamp_inter_bbox[1][0] - stamp_inter_bbox[0][0];
        int inter_h = stamp_inter_bbox[1][1] - stamp_inter_bbox[0][1];

        cv::Rect2i stamp_inter_rect(
            stamp_inter_bbox[0][0] - pad_yz_rect.x,
            stamp_inter_bbox[0][1] - pad_yz_rect.y,
            inter_w,
            inter_h);
        cv::Rect2i roi_inter_rect(
            stamp_inter_bbox[0][0] - pad_roi_rect.x,
            stamp_inter_bbox[0][1] - pad_roi_rect.y,
            inter_w,
            inter_h);

        stamp_img(stamp_inter_rect).copyTo(roi_pad_img(roi_inter_rect));
        roi_pad_img.copyTo(paper_img(pad_roi_rect));
    }
    // 去除杂点
    // if (ptype_str != "HBZ_A" || ptype_str != "HBZ_B")
    // {
    //     cv::Mat gray_img, bin_img;
    //     cv::cvtColor(paper_img, gray_img, cv::COLOR_RGB2GRAY);
    //     cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    //     cv::morphologyEx(gray_img, gray_img, cv::MORPH_ERODE, element);
    //     cv::morphologyEx(gray_img, gray_img, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)));

    //     if (ptype_str == "RYZ") {
    //         cv::threshold(gray_img, bin_img, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
    //     } else {
    //         cv::threshold(gray_img, bin_img, 200, 255, cv::THRESH_BINARY_INV);
    //     }
    //     // cv::imwrite("D:/dot_bin.jpg", bin_img);
    //     std::vector<std::vector<cv::Point>> contours;
    //     std::vector<cv::Vec4i> hierarchy(contours.size());
    //     cv::findContours(bin_img, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point());
        
    //     int size = (int)(contours.size());
    //     for (int i = 0; i < size; i++)
    //     {
    //         cv::Vec4i hier = hierarchy[i];
    //         double area = cv::contourArea(contours[i]);
    //         if (area < 100 && area > 2) {
    //             cv::Rect rc = cv::boundingRect(contours[i]);
    //             int pad = 10;
    //             rc.x -= pad;
    //             rc.y -= pad;
    //             rc.width += (2 * pad);
    //             rc.height += (2 * pad);
    //             if (rc.x < 0) rc.x = 0;
    //             if (rc.y < 0) rc.y = 0;
    //             if (rc.x + rc.width > paper_img.cols) rc.width = paper_img.cols - rc.x - 1;
    //             if (rc.y + rc.height > paper_img.rows) rc.height = paper_img.rows - rc.y - 1;
    //             // LOG_INFO("@@ get_masked_img boundingRect:{},{},{},{}", rc.x, rc.y, rc.width, rc.height);
    //             // LOG_INFO("@@ get_masked_img paper_img:{},{}", paper_img.cols, paper_img.rows);
    //             cv::Mat area_crop = paper_img(rc);
    //             cv::Mat mask = cv::Mat::ones(rc.height, rc.width, CV_8UC1);
    //             std::vector<cv::Point> cnt_local_pts;
    //             for (cv::Point pt: contours[i]) {
    //                 cnt_local_pts.push_back(cv::Point(pt.x-rc.x, pt.y-rc.y));
    //             }
    //             cv::fillPoly(mask, cnt_local_pts, cv::Scalar(0));
    //             cv::Scalar mean_outer = cv::mean(area_crop, mask==1);
    //             // cv::Scalar mean_inner = cv::mean(area_crop, mask==0);
    //             // double outer_gray = (mean_outer[0] + mean_outer[1] + mean_outer[2])/3;
    //             // double inner_gray = (mean_inner[0] + mean_inner[1] + mean_inner[2])/3;
    //             // LOG_INFO("#### area: {}  fill:{},{},{}", area, mean_outer[0], mean_outer[1], mean_outer[2]);
    //             // mean_outer = cv::Scalar(0,255,0);
    //             cv::fillPoly(paper_img, contours[i], mean_outer);
    //         }
    //     }
    // }

    return paper_img;
}

cv::Mat RefImgTool::stamp_extract(cv::Mat img, float red_ratio, bool locate)
{
    assert(img.channels() == 3);
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat b, g, r;
    channels.at(0).convertTo(r, CV_32F, 1.0);
    channels.at(1).convertTo(g, CV_32F, 1.0);
    channels.at(2).convertTo(b, CV_32F, 1.0);

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    cv::Mat r_div_b = r / b;

    cv::Rect2i crop_rect(100, 100, img.cols-200, img.rows-200);
    cv::RotatedRect stamp_rotrect;
    if (locate) {
        cv::Mat mask_vis_red = (r_div_b < 2.5) | (hsv_channels[2]<100);
        cv::Mat r0 = channels.at(0).clone();
        cv::Mat g0 = channels.at(1).clone();
        cv::Mat b0 = channels.at(2).clone();
        b0.setTo(255, mask_vis_red);
        g0.setTo(255, mask_vis_red);
        r0.setTo(255, mask_vis_red);
        std::vector<cv::Mat> red_rgb_channels(3);
        red_rgb_channels[0] = r0;
        red_rgb_channels[1] = g0;
        red_rgb_channels[2] = b0;
        cv::Mat red_rgb, red_gray, red_bin;
        cv::merge(red_rgb_channels, red_rgb);
        cv::cvtColor(red_rgb, red_gray, cv::COLOR_RGB2GRAY);
        cv::morphologyEx(red_gray, red_gray, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30)));
        cv::threshold(red_gray, red_bin, 200, 255, cv::THRESH_BINARY_INV);

        std::vector<std::vector<cv::Point>> contours0;
        std::vector<cv::Vec4i> hierarchy0(contours0.size());
        cv::findContours(red_bin, contours0, hierarchy0, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point());
        
        for (size_t i = 0; i < contours0.size(); i++)
        {
            double area = cv::contourArea(contours0[i]);
            if (area > 1e5) {
                stamp_rotrect = cv::minAreaRect(contours0[i]);
                crop_rect.width = img.cols - 200;
                crop_rect.height = img.rows - 200;
                crop_rect.x = int(stamp_rotrect.center.x - crop_rect.width / 2);
                crop_rect.y = int(stamp_rotrect.center.y - crop_rect.height / 2);
                // LOG_INFO("------ stamp_lt:{},{}  area:{}", crop_rect.x, crop_rect.y, area);
            }
        }
    }

    cv::Mat mask = (r_div_b < red_ratio) & (hsv_channels[2]<180);
    cv::Mat r1 = channels.at(0).clone();
    cv::Mat g1 = channels.at(1).clone();
    cv::Mat b1 = channels.at(2).clone();
    b1.setTo(255, mask);
    g1.setTo(255, mask);
    r1.setTo(255, mask);

    // 计算印章颜色均值
    cv::Mat mask_red = r_div_b > 2;
    cv::Scalar mean_r = int(cv::mean(r1, mask_red)[0]);
    cv::Scalar mean_g = int(cv::mean(g1, mask_red)[0]);
    cv::Scalar mean_b = int(cv::mean(b1, mask_red)[0]);

    // 字符重叠到印章区域填充为印章颜色（均值）
    std::vector<cv::Mat> tmp_channels(3);
    std::vector<cv::Mat> tmp_hsv_channels(3);
    tmp_channels[0] = r1;
    tmp_channels[1] = g1;
    tmp_channels[2] = b1;
    cv::Mat tmp_rgb;
    cv::Mat tmp_hsv;
    cv::merge(tmp_channels, tmp_rgb);
    cv::cvtColor(tmp_rgb, tmp_hsv, cv::COLOR_RGB2HSV);
    cv::split(tmp_hsv, tmp_hsv_channels);

    cv::Mat dark_red_mask = tmp_hsv_channels[2] < 180;
    b1.setTo(mean_b, dark_red_mask);
    g1.setTo(mean_g, dark_red_mask);
    r1.setTo(mean_r, dark_red_mask);

    std::vector<cv::Mat> new_channels(3);
    new_channels[0] = r1;
    new_channels[1] = g1;
    new_channels[2] = b1;
    cv::Mat yz_img;
    cv::merge(new_channels, yz_img);
    cv::medianBlur(yz_img, yz_img, 5);
    gray_scale_image(yz_img, 0, 200).convertTo(yz_img, CV_8U);

    if (stamp_rotrect.size.width > 0) {
        cv::Mat stam_outer_mask(yz_img.rows, yz_img.cols, CV_8UC1, cv::Scalar(255));
        int radius =  (stamp_rotrect.size.width + stamp_rotrect.size.height) / 4;
        cv::circle(stam_outer_mask, stamp_rotrect.center, radius, cv::Scalar(0), -1);
        yz_img.setTo(cv::Scalar(255,255,255), stam_outer_mask);
    }

    cv::Mat bin_img;
    cv::Mat gray_img;
    cv::cvtColor(yz_img, gray_img, cv::COLOR_RGB2GRAY);
    cv::threshold(gray_img, bin_img, 250, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy(contours.size());
    cv::findContours(bin_img, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cv::Point());
    int size = (int)(contours.size());
    for (int i = 0; i < size; i++)
    {
        double area = cv::contourArea(contours[i]);
        cv::RotatedRect rot_rc = cv::minAreaRect(contours[i]);
        if (area < 30) {
            cv::fillPoly(yz_img, contours[i], cv::Scalar(255,255,255));
        } else if (std::min(rot_rc.size.width, rot_rc.size.height) < 8) {
            if (std::abs(rot_rc.angle - 90) < 5 || rot_rc.angle < 5) {
                cv::fillPoly(yz_img, contours[i], cv::Scalar(255,255,255));
            }
            // LOG_INFO("AAA: x:{}, y:{}, w:{}  h:{}  angle:{} area:{}", rot_rc.center.x, rot_rc.center.y, rot_rc.size.width, rot_rc.size.height, rot_rc.angle, area);
        }
    }
    
    return locate ? yz_img(crop_rect) : yz_img;
}

// HGZ_A条码提取，讲条码下方文字填充为白色
cv::Mat RefImgTool::barcode_extract(cv::Mat barcode_img, std::vector<cv::Point>& char_coords)
{
    const int char_w = 830;
    const int char_h = 80;
    cv::Mat barcode_gray, bin_img;
    cv::Mat barcode_out = barcode_img.clone();
    cv::cvtColor(barcode_img, barcode_gray, cv::COLOR_RGB2GRAY);
    barcode_gray = gray_scale_image(barcode_gray, 0, 150);
    cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(19, 7));
    cv::morphologyEx(barcode_gray, barcode_gray, cv::MORPH_ERODE, element1);
    cv::morphologyEx(barcode_gray, barcode_gray, cv::MORPH_OPEN, element2);
    cv::threshold(barcode_gray, bin_img, 220, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(bin_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for(auto cnt : contours){
        double area = cv::contourArea(cnt);
        if (area > 30000 && area < 60000){
            cv::RotatedRect rect = cv::minAreaRect(cnt);
            bool hori = rect.size.width > rect.size.height;
            rect.size.width = hori ? char_w : char_h;
            rect.size.height = hori ? char_h : char_w;
            cv::Mat box_pts;
            cv::boxPoints(rect, box_pts);
            box_pts.convertTo(box_pts, CV_32S);
            cv::fillPoly(barcode_out, box_pts, cv::Scalar(255,255,255));

            for (int i = 0; i < 4; i++) {
                int x = box_pts.at<int>(i, 0);
                int y = box_pts.at<int>(i, 1);
                char_coords.push_back(cv::Point(x, y));
            }
            sort_rotrect_pts(char_coords);
            break;
        }
    }
    return barcode_out;
}