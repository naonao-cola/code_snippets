#include "dynamic_ocr_algo.h"

namespace DynamicOCR {

json DynamicCharDet::get_ocr_defect_std(std::string ptype_str) {
    json dynamic_defect_std;
    /*HGZ_B*/
    json HGZ_B_others = {
        {"height_min", 45},
        {"height_max", 100},
        {"width_min", 200},
        {"max_space_dist", 130}
    };

    dynamic_defect_std["HGZ_B"] = {
        {"others", HGZ_B_others},
        {"bin_thresh", 220}
    };

    /*HGZ_A*/
    json HGZ_A_others;
    HGZ_A_others["height_min"] = 60;
    HGZ_A_others["height_max"] = 120;
    HGZ_A_others["width_min"] = 200;
    HGZ_A_others["max_space_dist"] = 130;
    json hgz_barcode = {
        {"height_min", 50},
        {"height_max", 90},
        {"width_min", 800},
        {"stamp_height_min", 180},
        {"stamp_height_max", 240},
        {"stamp_width_min", 1000},
        {"stamp_width_max", 2000},
        {"max_space_dist", 100}
    };
    dynamic_defect_std["HGZ_A"] = {
        {"others", HGZ_A_others},
        {"hgz_barcode", hgz_barcode},
        {"bin_thresh", 200}
    };

    /*COC*/
    json COC_A_others;
    COC_A_others["height_min"] = 40;
    COC_A_others["height_max"] = 100;
    COC_A_others["width_min"] = 200;
    COC_A_others["max_space_dist"] = 130;
    dynamic_defect_std["COC"] = {
        {"others", COC_A_others},
        {"bin_thresh", 220}
    };

    /*RYZ*/
    json RYZ_others;
    RYZ_others["height_min"] = 50;
    RYZ_others["height_max"] = 120;
    RYZ_others["width_min"] = 200;
    RYZ_others["max_space_dist"] = 130;

    json rlxhl = {
        {"height_min", 180},
        {"height_max", 260},
        {"width_min", 400},
        {"max_space_dist", 130}};

    json sjgk = {
        {"height_min", 60},
        {"height_max", 140},
        {"width_min", 200},
        {"max_space_dist", 130}
    };
    json sqgk = {
        {"height_min", 180},
        {"height_max", 260},
        {"width_min", 400},
        {"max_space_dist", 130}
    };
    json zhgk = {
        {"height_min", 60},
        {"height_max", 120},
        {"width_min", 200},
        {"max_space_dist", 130}
    };
    dynamic_defect_std["RYZ"] = {
        {"others", RYZ_others},
        {"sjgk", sjgk},
        {"sqgk", sqgk},
        {"rlxhl", rlxhl},
        {"zhgk", zhgk},
        {"bin_thresh", 100}
    };

    // /*HBZ_A*/
    json HBZ_A_others;
    HBZ_A_others["height_min"] = 50;
    HBZ_A_others["height_max"] = 90;
    HBZ_A_others["width_min"] = 800;
    HBZ_A_others["max_space_dist"] = 130;
    dynamic_defect_std["HBZ_A"] = {
        {"others", HBZ_A_others},
        {"bin_thresh", 200}
    };

    /*HBZ_B*/
    json HBZ_B_others;
    HBZ_B_others["height_min"] = 60;
    HBZ_B_others["height_max"] = 100;
    HBZ_B_others["width_min"] = 200;
    HBZ_B_others["max_space_dist"] = 130;
    dynamic_defect_std["HBZ_B"] = {
        {"others", HBZ_B_others},
        {"bin_thresh", 200}
    };
    if (dynamic_defect_std.contains(ptype_str)) {
        return dynamic_defect_std[ptype_str];
    }
    LOG_ERROR("Wrong paper type str: {}", ptype_str);
    return dynamic_defect_std["HGZ_B"];
}

void DynamicCharDet::config(const std::string& paper_type, bool has_stamp) {
    m_paper_type = paper_type;
    m_dynamic_defect_std = get_ocr_defect_std(paper_type);
    m_has_stamp = has_stamp;
    // LOG_INFO("m_param: {}", m_param.dump());
}

void DynamicCharDet::spilt_muti_line_text_img(const cv::Mat input_img,
                                              std::vector<cv::Mat> &line_text_imgs,
                                              std::vector<cv::RotatedRect> &OK_rrect,
                                              std::vector<cv::RotatedRect> &NG_rrect,
                                              std::string label) {  // NOLINT
    if (input_img.empty()) {
        LOG_WARN("The input image is empty!");
        return;
    }

    if (m_dynamic_defect_std.contains(label)) {
        m_param = m_dynamic_defect_std[label];
    } else {
        m_param = m_dynamic_defect_std["others"];
    }
    m_img_height = input_img.rows;
    m_img_width = input_img.cols;
    std::vector<cv::RotatedRect> det_rrect;
    // 根据文本特征拆分文本区域
    spilt_line_by_blob(input_img, det_rrect, label);
    std::sort(det_rrect.begin(), det_rrect.end(), cmp_rot_rect);
    // 根据行高和横向字符间距离判断是否异常，
    // 高度异常：如果有墨汁脏污等，拆分出来的行高会超过正常高度
    // 横向间距异常：如果文本框之间距离超过阈值，认为是字符缺失
    det_defect(det_rrect, NG_rrect, OK_rrect, label);

    // 将OK文本拆分到line_text_imgs中
    std::sort(OK_rrect.begin(), OK_rrect.end(), cmp_rot_rect);
    crop_single_line_text_img(input_img, OK_rrect, line_text_imgs);
}

/*
* 将单行/多行文本区域按行进行拆分，单行中如果前后空隙较大会拆分成多个区域
* @input 动态文本区域图片
* @det_rst 文本框
*/
void DynamicCharDet::spilt_line_by_blob(cv::Mat input,
                                        std::vector<cv::RotatedRect> &det_rst,
                                        std::string label) {
    cv::Mat bin_img;
    cv::Mat gray_img;
    get_text_mask(input, bin_img);
    remove_stamp(input, input);
    cv::cvtColor(input, gray_img, cv::COLOR_BGR2GRAY);
    int bin_threshold = m_dynamic_defect_std["bin_thresh"];
    // cv::threshold(gray_img, bin_img, 0, 255, cv::THRESH_BINARY_INV);
    cv::blur(gray_img, gray_img, cv::Size(5, 5), cv::Point(-1, -1), cv::BORDER_DEFAULT);
    cv::Canny(gray_img, bin_img, 70, 140, 3, false);
    bin_img.copyTo(m_bin_img);
    int element_width;
    if (label == "sqgk" || label == "rlxhl") {
        element_width = 180;
    } else {
        int max_height = m_param["max_space_dist"];
        element_width = max_height < 120 ? max_height : 120;
    }
    // LOG_INFO("@@@@ element width : {}", element_width);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(element_width, 5));
    cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(element_width, 5));
    // cv::Mat erode_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    // cv::morphologyEx(bin_img, bin_img, cv::MORPH_OPEN, erode_element);
    // cv::medianBlur(bin_img, bin_img, 3);
    cv::dilate(bin_img, bin_img, element);
    cv::erode(bin_img, bin_img, element1);

#ifdef DEBUG_ON
    static int bin_idx = 0;
    cv::imwrite("D:\\OcrTest\\" + label + "_bin.jpg", bin_img);
    cv::imwrite("D:\\OcrTest\\"+ label+"_ori.jpg", input);
    bin_idx++;
#endif

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy(contours.size());
    // std::vector<cv::RotatedRect> det_rst;
    // cv::imwrite("D:/ng_ocr/bin_img.jpg", bin_img);
    cv::findContours(bin_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point()); //寻找轮廓
    int size = (int)(contours.size());                                                                   //轮廓的数量
    for (int i = 0; i < size; i++)
    {
        if (cv::contourArea(contours[i]) < 1600)
        {
            continue;
        }
        cv::Rect box = cv::boundingRect(contours[i]);

        int thickness = box.width < box.height ? box.width : box.height;
        if (thickness < 20)
        {
            if (box.width < box.height)
            {
                // 短竖线，如果出现在框的前后两端则不算（定位框不准可能会框到表格）
                if (box.x < 20 || box.x > (box.x + input.cols - 20))
                {
                    continue;
                }
            }
            else
            {
                // 短横线，如果出现在框的上下两端则不算（定位框不准可能会框到表格）
                if (box.y < 20 || box.y > (input.rows - 20))
                {
                    continue;
                }
            }
        }
        cv::RotatedRect rot_rect = cv::minAreaRect(contours[i]);
        det_rst.push_back(rot_rect);
    }
}

void DynamicCharDet::get_text_mask(const cv::Mat &input_img, cv::Mat &txt_mask) {
    if ( input_img.empty() && input_img.channels() == 1 ) {
        LOG_WARN("The input image is empty or the image channels == 1 !");
        return;
    }
    // cv::inRange(input_img, cv::Scalar(0, 0, 0), cv::Scalar(140, 140, 140), txt_mask);
    // 红色 没有字： (200, 100, 100) 白色无字 (220, 220, 220) 红色黑字 ()
    cv::Mat hsv_img;
    cv::Mat hsv_mask;
    cv::Mat rgb_mask;
    cv::cvtColor(input_img, hsv_img, cv::COLOR_RGB2HSV);
    cv::inRange(input_img, cv::Scalar(0, 0, 0), cv::Scalar(120, 160, 160), rgb_mask);
    cv::inRange(hsv_img, cv::Scalar(40, 0, 0), cv::Scalar(260, 100, 50), hsv_mask);
    cv::bitwise_or(rgb_mask, hsv_mask, txt_mask);
}

void DynamicCharDet::remove_stamp(cv::Mat &input_img, cv::Mat &remove_stamp_img) {
    if (m_has_stamp == false) {
        return;
    }
    if (input_img.empty() && input_img.channels() == 1) {
        LOG_WARN("The input image is empty or the image channels == 1 !");
        return;
    }

    //  找出红色印章的区域
    static int idx = 1;
    idx++;
    cv::Rect2i stamp_bbox;
    // cv::Mat input_img1 = gray_scale_image(input_img, 60, 255);
    get_stamp_region(input_img, stamp_bbox);
    if (stamp_bbox.area() < 1e3) {
        return;
    }

    //  截取红色印章区域
    cv::Mat stamp_sub_img = input_img(stamp_bbox);
    cv::Mat mask;
    text_extract_algo(stamp_sub_img);
    idx++;
}

void DynamicCharDet::det_defect(std::vector<cv::RotatedRect> &det_rst,
                              std::vector<cv::RotatedRect> &NG_rrst,
                              std::vector<cv::RotatedRect> &OK_rrst,
                              std::string label) {
    std::vector<int> centers_y;
    std::vector<int> centers_x;
    std::vector<int> heights;
    std::vector<int> widths;
    std::list<cv::RotatedRect> det_list(det_rst.begin(), det_rst.end());
    std::set<int> defect_idx;
    int data_cnt = heights.size();
    int height_min = m_param["height_min"].get<int>();
    int height_max = m_param["height_max"].get<int>();
    int width_min = m_param["width_min"].get<int>();
    int max_space_dist = m_param["max_space_dist"].get<int>();

    //  判断行高
    std::list<cv::RotatedRect>::iterator det = det_list.begin();
    for (; det != det_list.end(); ++det) {
        int height = get_rotatedRect_shortside(*det);
        int width = get_rotatedRect_longside(*det);
        if (label == "hgz_barcode" && width < width_min) {
            NG_rrst.push_back(*det);
            det_list.erase(det);
            continue;
        }

        if (label == "barcode") {
            int stamp_height_min = m_param["stamp_height_min"].get<int>();
            int stamp_height_max = m_param["stamp_height_max"].get<int>();
            int stamp_width_min = m_param["stamp_width_min"].get<int>();
            int stamp_width_max = m_param["stamp_width_max"].get<int>();
            if (height > stamp_height_max &&
                height < stamp_height_min &&
                width > stamp_width_max &&
                width < stamp_width_min) {
                NG_rrst.push_back(*det);
                det_list.erase(det);
                continue;
            }
        }

        if (height < height_min || height > height_max) {
            NG_rrst.push_back(*det);
            det_list.erase(det);
        }
    }

    //  判断横向距离
    det_list.sort(cmp_by_xy);
    cv::RotatedRect first_rct = det_list.front();
    first_rct.center.x = 0;
    first_rct.size.width = 1;
    first_rct.size.height = 1;
    det_list.push_front(first_rct);

    std::list<cv::RotatedRect>::iterator lst_iter = det_list.begin();
    auto fst_iter = lst_iter;
    lst_iter++;
    while (lst_iter != det_list.end()) {
        if (fst_iter->center.y + 50 < lst_iter->center.y) {
            // fst_iter = lst_iter;
            fst_iter = det_list.begin();
            fst_iter->center.x = 0;
            fst_iter->center.y = lst_iter->center.y;
            first_rct.size.width = 1;
            first_rct.size.height = 1;
        } else {
            int w1 = get_rotatedRect_longside(*fst_iter) / 2;
            int w2 = get_rotatedRect_longside(*lst_iter) / 2;
            int dist = lst_iter->center.x - fst_iter->center.x;
            if (dist - w1 - w2 > max_space_dist) {
                NG_rrst.push_back(*lst_iter);
                det_list.erase(lst_iter);
            } else {
                fst_iter = lst_iter;
            }
            ++lst_iter;
        }
    }
    det_list.erase(det_list.begin());

    for (auto det : det_list) {
        OK_rrst.push_back(det);
    }
}

void DynamicCharDet::crop_single_line_text_img(const cv::Mat input_img,
                                               const std::vector<cv::RotatedRect> &OK_rrect,
                                               std::vector<cv::Mat> &line_text_imgs) {
    if (input_img.empty()) {
        LOG_WARN("The input image is empty !");
        return;
    }

    std::vector<std::vector<cv::Point>> inter_polygen;
    cv::Rect2i inter_rret;
    cv::Mat line_text_img;
    int width = input_img.cols;
    int height = input_img.rows;
    cv::Rect2i img_bbox(0, 0, width, height);

    for (auto text_rrect : OK_rrect) {
        //  提取行
        cv::Rect2i line_rect = text_rrect.boundingRect();
        line_rect.height += 30;
        line_rect.y -= 15;

        line_rect.width += 70;
        line_rect.x -= 15;
        line_rect = line_rect & img_bbox;
        line_text_img = input_img(line_rect);
        remove_stamp(line_text_img, line_text_img);
        line_text_imgs.push_back(line_text_img);
    }
}

void DynamicCharDet::get_stamp_region(const cv::Mat &input_img, cv::Rect2i &stamp_bbox) {
    if (input_img.empty() && input_img.channels() == 1) {
        LOG_WARN("The input image is empty or the channel == 1 !");
        return;
    }

    //  获取印章区域mask
    cv::Mat mask;
    cv::inRange(input_img, cv::Scalar(160, 0, 0), cv::Scalar(255, 160, 160), mask);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, element, cv::Point(-1, -1), 3);
    //  找到印章的位置
    std::vector<std::vector<cv::Point>> contous;
    std::vector<cv::Vec4i> hierarchy(contous.size());
    std::vector<cv::Rect2i> det_rst;
    int max_area = -1;
    int minx = 1e7;
    int maxx = -1;

    cv::findContours(mask, contous, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point());
    int size = static_cast<int>(contous.size());
    for (int i = 0; i < size; ++i) {
        if (cv::contourArea(contous[i]) < 1e2) {
            continue;
        }
        stamp_bbox = cv::boundingRect(contous[i]);
        minx = std::min(minx, stamp_bbox.x);
        maxx = std::max(maxx, stamp_bbox.x + stamp_bbox.width);
    }
    if (maxx > minx) {
        int height = input_img.rows;
        stamp_bbox = cv::Rect2i(cv::Point(minx, 0), cv::Point(maxx, height));
    }
}

void DynamicCharDet::text_extract_algo(cv::Mat &input_img) {
    cv::Mat mask;
    get_text_mask(input_img, mask);
    std::vector<cv::Mat> out_arr;
    for (int i = 0; i < 3; i++) {
        cv::Mat s_img = 255 - mask;
        out_arr.push_back(s_img);
    }
    cv::merge(out_arr, input_img);
}

}  // namespace DynamicOCR
