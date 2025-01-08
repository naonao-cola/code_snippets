#include "char_defect_det_algo.h"

CharDefectDetAlgo::CharDefectDetAlgo() {}
void CharDefectDetAlgo::config(std::string ref_img_dir,
                               int bin_thr,
                               int error_std,
                               int morph_size,
                               const std::string& papar_type,
                               int ng3_min_area)
{
    if (fs::exists(ref_img_dir) == false) {
        LOG_ERROR("ref_img_dir {} not exists", ref_img_dir);
        assert(fs::exists(ref_img_dir));
    }
    m_ref_img_root = ref_img_dir;
    m_bin_thr = std::max(0, bin_thr);
    m_error_std = std::max(0, error_std);
    m_morph_size = std::max(0, morph_size);
    m_ng3_min_area = std::max(0, ng3_min_area);
    set_ref_img_type(papar_type);
}

bool CharDefectDetAlgo::set_ref_img_type(const std::string& ref_img_type)
{
    fs::path p(m_ref_img_root);
    p.append(ref_img_type);
    m_ref_img_dir = p.string();
    LOG_INFO("ref_img_dir: {}", m_ref_img_dir);
    assert(fs::exists(m_ref_img_dir));
    m_paper_type = ref_img_type;
    return true;
}

std::vector<cv::Rect2i> CharDefectDetAlgo::forward(cv::Mat img, std::string label)
{
    std::vector<cv::Rect2i> all_out;
    if (img.empty()) {
        LOG_WARN("The input image is empty!");
        return all_out;
    }
    if (label == "GQFT") {
        bool is_defect = det_GQFT(img);
        if (is_defect) {
            cv::Rect rect(0, 0, img.cols, img.rows);
            all_out.push_back(rect);
        }
        return all_out;
    }
    bool is_defect = false;

    // 提取字符
    std::vector<cv::Mat> char_imgs;
    std::vector<cv::Rect> char_Rects;
    LOG_INFO("<<NG3>> split bin_thr:{}, ng3_min_area: {}", m_bin_thr, m_ng3_min_area);
    split_char_img(img, char_imgs, char_Rects);
    if (static_cast<int>(char_Rects.size()) - static_cast<int>(label.size()) >= 3) {
        LOG_INFO("<<NG3>> char_Rects size: {} - label.size {} >= 3", char_Rects.size(), label.size());
#ifdef DEBUG_ON
        for (int i = 0; i < char_imgs.size(); ++i) {
            cv::imwrite("D:\\ng_test\\" + std::to_string(i) + ".jpg", char_imgs[i]);
        }
#endif
        is_defect = true;
    } else if (static_cast<int>(label.size()) - static_cast<int>(char_Rects.size()) > 3) {
        LOG_INFO("<<NG3>> blob size: {} < label.size {} ", char_Rects.size(), label.size());
#ifdef DEBUG_ON
        for (int i = 0; i < char_imgs.size(); ++i) {
            cv::imwrite("D:\\ng_test\\" + std::to_string(i) + ".jpg", char_imgs[i]);
        }
#endif
        is_defect = true;
    } else if (label.size() == char_Rects.size()) {
#ifdef DEBUG_ON
        for (int i = 0; i < char_imgs.size(); ++i) {
            cv::imwrite("D:\\ng_test\\" + std::to_string(i) + ".jpg", char_imgs[i]);
        }
#endif
        for (int idx = 0; idx < label.size(); ++idx) {
            std::string char_label = "";
            char_label += label[idx];
            bool match_rst = char_defect_detection(char_imgs[idx], char_label);
            if (match_rst) {
                is_defect = true;
                break;
            }
        }
        LOG_INFO("<<NG3>>varificate every char, result: {}", is_defect);
    } else {
#ifdef DEBUG_ON
        for (int i = 0; i < char_imgs.size(); ++i) {
            cv::imwrite("D:\\ng_test\\" + std::to_string(i) + ".jpg", char_imgs[i]);
        }
#endif
        LOG_INFO("<<NG3>> blob size: {} , label.size {} ", char_Rects.size(), label.size());
        cv::Mat ref_img;
        gen_label_bin_img(img, label, ref_img);
        is_defect = char_defect_by_ecc(ref_img, img.clone(), m_bin_thr, label);
        LOG_INFO("<<NG3>> varificate by generate image, result: {}", is_defect);
    }

    if (is_defect) {
        cv::Rect rect(0, 0, img.cols, img.rows);
        all_out.push_back(rect);
    }
    return all_out;
}

void CharDefectDetAlgo::detect_color_region(cv::Mat img,
                                            cv::Scalar low_val,
                                            cv::Scalar high_val,
                                            std::vector<cv::Rect>& out_rects)
{
    cv::Mat mask;
    cv::inRange(img, low_val, high_val, mask);
    blob_rect(mask, 40, out_rects);
}

bool CharDefectDetAlgo::det_GQFT(cv::Mat img)
{
    //  去除右边黄色
    static int mask_idx = 0;
    std::vector<cv::Rect> yellow_region;
    detect_color_region(img, cv::Scalar(30, 125, 161), cv::Scalar(180, 240, 255), yellow_region);
    int yellow_x = 99999;
    for (auto rect : yellow_region) {
        if (std::abs(img.cols - rect.x) < 60) {
            yellow_x = std::min(yellow_x, rect.x);
        }
    }
    yellow_x = yellow_x == 99999 ? img.cols - 3 : yellow_x;
    cv::Mat img_ = img.clone();
    cv::Rect2i img_r(0, 0, img.cols, img.rows);
    cv::Rect2i roi = cv::Rect(0, 0, yellow_x, img.rows) & img_r;
    img = img(roi);
    mask_idx++;

    std::string ref_img_path = m_ref_img_dir + "\\GQFT\\0.jpg";
    cv::Mat ref_img;
    ref_img = cv::imread(ref_img_path);
    assert(ref_img.empty() == false);
    bool is_defect = char_defect_by_ecc(ref_img, img.clone(), m_bin_thr, "GQFT");
    return is_defect;
}

bool CharDefectDetAlgo::defect_by_blob_nums(cv::Mat img, const std::string& label)
{
    std::vector<cv::Mat> char_imgs;
    std::vector<cv::Rect> char_Rects;
    split_char_img(img, char_imgs, char_Rects);
    if (char_Rects.size() > label.size()) {
        return true;
    }

    if (label.size() - char_Rects.size() > 3) {
        return true;
    }

    return false;
}

bool CharDefectDetAlgo::char_defect_by_ecc(cv::Mat ref_img, cv::Mat img, int bin_thr, std::string label = "")
{
    if (ref_img.empty() || img.empty()) {
        LOG_WARN("image is empty!");
        return false;
    }
    cv::Mat ref_gray;
    cv::Mat img_gray;
    cv::Mat img_bin;
    cv::Mat ref_bin;
    bool is_defect = false;
    if (ref_img.channels() == 3)
        cv::cvtColor(ref_img, ref_gray, cv::COLOR_RGB2GRAY);
    else {
        ref_gray = ref_img;
    }
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);
    } else {
        img_gray = img;
    }
    // get_img_gradian(img_gray, img_bin);
    // get_img_gradian(ref_gray, ref_bin);
    cv::threshold(img_gray, img_bin, bin_thr, 255, cv::THRESH_BINARY_INV);
    cv::threshold(ref_gray, ref_bin, bin_thr, 255, cv::THRESH_BINARY_INV);
    // static int img_idx = 0;
    // cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "ref_bin1.jpg", ref_bin);

    // 模板匹配
    int pad_w = img_bin.cols + 2 * ref_bin.cols;
    int pad_h = img_bin.rows + 2 * ref_bin.rows;
    cv::Mat pad_img(pad_h, pad_w, CV_8UC1, cv::Scalar(0));
    cv::Mat pad_ref(pad_h, pad_w, CV_8UC1, cv::Scalar(0));
    img_bin.copyTo(pad_img(cv::Rect2i(ref_bin.cols, ref_bin.rows, img_bin.cols, img_bin.rows)));

    cv::Mat match_r;
    cv::matchTemplate(pad_img, ref_bin, match_r, cv::TM_CCOEFF_NORMED);

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(match_r, &min_val, &max_val, &min_loc, &max_loc);
    ref_bin.copyTo(pad_ref(cv::Rect2i(max_loc.x, max_loc.y, ref_img.cols, ref_img.rows)));

    // 判断污渍
    cv::Mat expand_ref;
    cv::Mat expand_img;
    cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(m_morph_size, m_morph_size + 1));
    cv::morphologyEx(pad_img, expand_img, cv::MORPH_DILATE, elem);
    cv::morphologyEx(pad_ref, expand_ref, cv::MORPH_DILATE, elem);
    // 定位缺陷
    std::vector<cv::Rect> img_sub_ref;
    std::vector<cv::Rect> ref_sub_img;

    cv::Rect2i roi(ref_bin.cols, ref_bin.rows, img_bin.cols, img_bin.rows);
    cv::Mat sub_1 = pad_ref - expand_img;
    cv::Mat sub_2 = pad_img - expand_ref;
    sub_1 = sub_1(roi);
    sub_2 = sub_2(roi);
    blob_rect(sub_1, m_ng3_min_area, img_sub_ref);
    blob_rect(sub_2, m_ng3_min_area, ref_sub_img);
    if (label == "GQFT") {
        is_defect = false;
        for (auto box : img_sub_ref) {
            float thickness = std::max(box.width, box.height) / std::min(box.width, box.height);
            if (thickness > 8) {
                continue;
            } else {
                is_defect = true;
                break;
            }
        }
        for (auto box : ref_sub_img) {
            float thickness = std::max(box.width, box.height) / std::min(box.width, box.height);
            if (thickness > 8) {
                continue;
            } else {
                is_defect = true;
                break;
            }
        }
        return is_defect;
    }

    if (img_sub_ref.empty() && ref_sub_img.empty()) {
        is_defect = false;
#ifdef DEBUG_ON_SHOW_OK
        static int img_idx = 0;
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "ref.jpg", ref_img);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "crop_img.jpg", img);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "ref_bin.jpg", ref_bin);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "img_bin.jpg", img_bin);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "img_sub_ref.jpg", sub_1);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "ref_sub_img.jpg", sub_2);
        img_idx++;
#endif
    } else {
#ifdef DEBUG_ON_SHOW_NG
        static int img_idx = 0;
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "ref.jpg", ref_gray);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "crop_img.jpg", img_gray);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "ref_bin.jpg", ref_bin);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "img_bin.jpg", img_bin);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "img_sub_ref.jpg", sub_1);
        cv::imwrite("D:\\ng_test\\" + std::to_string(img_idx) + label + "ref_sub_img.jpg", sub_2);
        img_idx++;
#endif
        is_defect = true;
    }
    return is_defect;
}

void CharDefectDetAlgo::blob_rect(cv::Mat bin_img, int min_area, std::vector<cv::Rect>& blob_region)
{
    // blob提取
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy(contours.size());
    cv::findContours(bin_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point()); // 寻找轮廓
    int size = (int)(contours.size()); // 轮廓的数量
    for (int i = 0; i < size; i++) {
        cv::Rect box = cv::boundingRect(contours[i]);
        int area = cv::contourArea(contours[i]);
        int arc_len = cv::arcLength(contours[i], false);
        if (area < min_area && arc_len < min_area * 2) {
            if (area > 10) {
                LOG_INFO("<<NG3>>Remove too small blob! area threshold: {}, area: {}, arc_len: {}box.x: {}",
                         min_area,
                         area,
                         arc_len,
                         box.x);
            }
            continue;
        }

        // int thickness = box.width < box.height ? box.width : box.height;
        float thickness = std::max(box.width, box.height) / std::min(box.width, box.height);
        float delta = 10;
        int min_len = std::min(box.width, box.height);
        if (thickness > 5 || min_len < 10) {
            if (box.width < box.height) {
                // 短竖线，如果出现在框的前后两端则不算
                if (box.x < delta || box.x > (bin_img.cols - delta)) {
                    LOG_INFO("remove border line. thickness: {}, min_len: {}, box.x: {}", thickness, min_len, box.x);
                    continue;
                }
            } else {
                // 短横线，如果出现在框的上下两端则不算
                if (box.y < delta || box.y > (bin_img.rows - delta)) {
                    LOG_INFO("remove border line. thickness: {}, min_len: {}, box.x: {}", thickness, min_len, box.x);
                    continue;
                }
            }
        }
        LOG_INFO("<<NG3>>NG3 Blob! area threshold: {}, area: {}, arc_len: {} box.x: {}",
                 min_area,
                 area,
                 arc_len,
                 box.x);
        blob_region.push_back(box);
    }
}

void CharDefectDetAlgo::blob_rect(cv::Mat bin_img, int min_area, std::vector<std::vector<cv::Point>>& blob_region)
{
    // blob提取
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy(contours.size());
    cv::findContours(bin_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point());
    int size = (int)(contours.size());
    for (int i = 0; i < size; i++) {
        cv::Rect box = cv::boundingRect(contours[i]);
        int area = cv::contourArea(contours[i]);
        int arc_len = cv::arcLength(contours[i], false);
        if (area < min_area && arc_len < 2 * min_area) {
            if (area > min_area / 2) {
                LOG_INFO("Remove too small blob! area threshold: {}, area: {}, arc_len: {}box.x: {}",
                         min_area,
                         area,
                         arc_len,
                         box.x);
            }
            continue;
        }
        // int thickness = box.width < box.height ? box.width : box.height;
        float thickness = std::max(box.width, box.height) / std::min(box.width, box.height);
        float delta = 10;
        int min_len = std::min(box.width, box.height);
        if (thickness > 5 || min_len < 10) {
            if (box.width < box.height) {
                // 短竖线，如果出现在框的前后两端则不算
                if (box.x < delta || box.x > (box.x + bin_img.cols - delta)) {
                    LOG_INFO("remove border line. thickness: {}, min_len: {}, box.x: {}", thickness, min_len, box.x);
                    continue;
                }
            } else {
                // 短横线，如果出现在框的上下两端则不算
                if (box.y < delta || box.y > (box.y + bin_img.rows - delta)) {
                    LOG_INFO("remove border line. thickness: {}, min_len: {}, box.x: {}", thickness, min_len, box.x);
                    continue;
                }
            }
        }
        blob_region.push_back(contours[i]);
    }
}

void CharDefectDetAlgo::split_char_img(cv::Mat input,
                                       std::vector<cv::Mat>& out_imgs,
                                       std::vector<cv::Rect>& char_region)
{
    LOG_INFO("Start spilt character image");
    cv::Mat bin_img;
    cv::Mat gray_img;
    LOG_INFO("bin_img: {}", m_bin_thr);

    // gray_scale_image(input, 30, 255);
    cv::cvtColor(input, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, bin_img, m_bin_thr, 255, cv::THRESH_BINARY_INV);
    // cv::blur(gray_img, gray_img, cv::Size(5, 5), cv::Point(-1, -1), cv::BORDER_DEFAULT);
    // cv::Canny(gray_img, bin_img, 70, 140, 3, false);

    // cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1));
    // cv::morphologyEx(bin_img, bin_img, cv::MORPH_OPEN, element1);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, m_morph_size));
    cv::morphologyEx(bin_img, bin_img, cv::MORPH_CLOSE, element);
#ifdef DEBUG_ON
    cv::imwrite("D:\\ng_test\\bin_img.jpg", bin_img);
#endif
    // blob提取
    std::vector<std::vector<cv::Point>> char_region_polygen;
    blob_rect(bin_img, 150, char_region_polygen);
    //  排序
    std::sort(char_region_polygen.begin(), char_region_polygen.end(), cmp_poly_by_x);
    // std::sort(char_region.begin(), char_region.end(), CharDefectDetAlgo::cmp_rect_by_x);
    cv::Scalar mask_val(255);
    // if (input.channels() == 3) {
    //     mask_val = cv::Scalar(255, 255, 255);
    // }
    for (auto polygen : char_region_polygen) {
        cv::Rect rect = cv::boundingRect(polygen);
        char_region.push_back(rect);
        cv::Mat char_img = gray_img(rect);
        int dx = rect.x;
        int dy = rect.y;
        // for ( auto& point : polygen ) {
        //     point.x -= dx;
        //     point.x = std::max(point.x, 0);
        //     point.y -= dy;
        //     point.y = std::max(point.y, 0);
        // }
        // 创建mask
        cv::Mat mask = cv::Mat::zeros(char_img.size(), char_img.type());
        std::vector<std::vector<cv::Point>> cont = {polygen};
        cv::drawContours(mask, cont, -1, mask_val, -1, cv::LINE_AA, cv::noArray(), 21478, cv::Point(-dx, -dy));
        // cv::bitwise_and(char_img, mask, char_img);
        cv::Mat out = cv::Mat::ones(char_img.size(), char_img.type()) * 255;
        char_img.copyTo(out, mask);
        // char_img.copyTo(out);
        // cv::imwrite("D:\\char_img.jpg", out);
        out_imgs.push_back(out);
    }
    LOG_INFO("Finish spilt char image");
}

bool CharDefectDetAlgo::cmp_rect_by_x(cv::Rect a, cv::Rect b)
{
    if (a.x < b.x) {
        return true;
    }
    return false;
}

bool CharDefectDetAlgo::cmp_poly_by_x(const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
{
    cv::Rect ra = cv::boundingRect(a);
    cv::Rect rb = cv::boundingRect(b);
    return cmp_rect_by_x(ra, rb);
}

float get_iou(cv::Rect a, cv::Rect b)
{
    cv::Rect union_rect = a | b;
    cv::Rect inter_rect = a & b;
    float iou = static_cast<float>(inter_rect.area()) / static_cast<float>(union_rect.area());
    return iou;
}

void nms(std::map<float, CharRect, std::greater<float>>& top_info,
         float nms_thr,
         int topk,
         std::vector<CharRect>& rst)
{
    std::map<float, CharRect, std::greater<float>>::iterator map_iter = top_info.begin();
    while (map_iter != top_info.end()) {
        std::map<float, CharRect, std::greater<float>>::iterator iter_next = map_iter;
        iter_next++;
        cv::Rect2i front_rect = map_iter->second.m_rect;
        while (iter_next != top_info.end()) {
            cv::Rect2i rear_rect = iter_next->second.m_rect;
            float iou = get_iou(front_rect, rear_rect);
            if (iou > nms_thr) {
                top_info.erase(iter_next++);
            } else {
                iter_next++;
            }
        }
        rst.push_back(map_iter->second);
        map_iter++;
    }
}

void find_top_k(cv::Mat match, char c, int topk, float val_thr, cv::Size ref_size, std::vector<CharRect>& rst)
{
    int w = match.cols;
    int h = match.rows;
    std::map<float, CharRect, std::greater<float>> top_info;

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(match, &min_val, &max_val, &min_loc, &max_loc);
    cv::Rect2i max_v_rect(max_loc.x, max_loc.y, ref_size.width, ref_size.height);
    top_info.insert({max_val, CharRect(c, max_val, max_v_rect)});

    for (int i = 0; i < h; ++i) {
        for (int s = 0; s < w; ++s) {
            float v = match.at<float>(i, s);
            if (v > val_thr) {
                cv::Rect2i rect(s, i, ref_size.width, ref_size.height);
                CharRect c_rect(c, v, rect);
                top_info.insert({v, c_rect});
            }
        }
    }

    // NMS
    nms(top_info, 0.1, topk, rst);
}

void match_label(const std::string& label,
                 std::vector<CharRect>& match_result_v,
                 std::vector<CharRect>& result)
{
    std::sort(match_result_v.begin(), match_result_v.end(), CharRect::cmp_x);
    std::list<CharRect> match_result = {match_result_v.begin(), match_result_v.end()};
    for (auto c : label) {
        CharRect* char_rect = nullptr;
        bool is_found = false;
        for (auto c_rect = match_result.begin(); c_rect != match_result.end();) {
            if (c_rect->m_char == c) {
                char_rect = new CharRect(c_rect->m_char, c_rect->m_score, c_rect->m_rect);
                match_result.erase(c_rect++);
                // result.push_back(*char_rect);
                // LOG_INFO("find {}", c);
                break;
            } else {
                match_result.erase(c_rect++);
            }
        }

        if (char_rect == nullptr) {
            LOG_INFO("<<NG3>> generate image not match index:{}  char: {}", result.size() + 2, c);
            continue;
        }

        for (auto mtc_rst = match_result.begin(); mtc_rst != match_result.end();) {
            // float iou = CharRect::iou(*mtc_rst, *char_rect);
            float iou = CharRect::min_iou(*mtc_rst, *char_rect);
            if (iou > 0.3) {
                if (mtc_rst->m_score > char_rect->m_score &&
                    mtc_rst->m_char == char_rect->m_char) {
                    delete char_rect;
                    char_rect = new CharRect(mtc_rst->m_char, mtc_rst->m_score, mtc_rst->m_rect);
                    // mtc_rst ++;
                    match_result.erase(mtc_rst++);
                } else {
                    match_result.erase(mtc_rst++);
                }
            } else {
                if (mtc_rst->m_rect.x > char_rect->m_rect.x)
                    break;
            }
        }
        if (char_rect != nullptr) {
            result.push_back(*char_rect);
        } else {
            LOG_INFO("not find {}", c);
        }
    }
}

void image_binary(cv::Mat input_img, int bin_thr, bool is_inv, cv::Mat& bin_img)
{
    if (input_img.channels() == 3) {
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2GRAY);
    }
    if (is_inv) {
        cv::threshold(input_img, bin_img, bin_thr, 255, cv::THRESH_BINARY_INV);
    } else {
        cv::threshold(input_img, bin_img, bin_thr, 255, cv::THRESH_BINARY);
    }
}

void CharDefectDetAlgo::gen_label_bin_img(cv::Mat& img, const std::string& label1, cv::Mat& dst_img)
{
    cv::Mat ref_bin;
    cv::Mat img_bin;
    std::string label = label1;
    std::map<char, int> char_cnt;
    std::map<char, cv::Mat> char_img_map;
    std::vector<CharRect> match_rst;
    image_binary(img, m_bin_thr, true, img_bin);
    if (m_paper_type == "HBZ_A") {
        // TODO 为适配测试，修改为substr(4, label.size() 测试后修改回substr(3, label.size()
        label = "@" + label.substr(3, label.size());
    }
    for (auto c : label) {
        if (char_cnt.count(c) == 0) {
            char_cnt[c] = 1;
        } else {
            char_cnt[c]++;
        }
    }
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    dst_img = cv::Mat::ones(img.rows, img.cols, img.type()) * 255;

    for (auto c : char_cnt) {
        std::string ref_img_p = m_ref_img_dir + "\\" + c.first + ".jpg";
        cv::Mat ref_img = cv::imread(ref_img_p, 0);
        char_img_map.insert({c.first, ref_img});
        if (ref_img.empty()) {
            LOG_WARN("Not found ref_img: {}", c.first);
            continue;
        };

        // 模板匹配
        cv::Mat match_r;
        cv::Mat idx_list;
        image_binary(ref_img, m_bin_thr, true, ref_bin);
        cv::matchTemplate(img_bin, ref_bin, match_r, cv::TM_CCOEFF_NORMED);
        find_top_k(match_r, c.first, c.second, 0.5, ref_img.size(), match_rst);
        if (m_paper_type == "HBZ_A" && (c.first == '6')) {
            ref_img_p = m_ref_img_dir + "\\" + c.first + "_1.jpg";
            ref_img = cv::imread(ref_img_p, 0);
            // char_img_map[c.first] = ref_img;
            cv::matchTemplate(img, ref_img, match_r, cv::TM_CCOEFF_NORMED);
            find_top_k(match_r, c.first, c.second, 0.5, ref_img.size(), match_rst);
        }
    }

    std::vector<CharRect> reuslts;
    match_label(label, match_rst, reuslts);
    for (auto c_rect : reuslts) {
        // cv::Rect2i rect(lt_p.x, lt_p.y, ref_img.cols, ref_img.rows);
        cv::Mat ref_img = char_img_map[c_rect.m_char];
        // cv::threshold(ref_img, ref_img, 200, 255, cv::THRESH_BINARY);
        if ((c_rect.m_char == '6') && c_rect.m_rect.size() != ref_img.size()) {
            std::string ref_img_p = m_ref_img_dir + "\\" + c_rect.m_char + "_1.jpg";
            ref_img = cv::imread(ref_img_p, 0);
        }
        assert(ref_img.channels() == dst_img.channels());
        ref_img.copyTo(dst_img(c_rect.m_rect));
    }
    // cv::imwrite("D:\\ng_test\\gen_" + label + ".jpg", dst_img);
}

int CharDefectDetAlgo::get_pix_area(cv::Mat input_img, int pix_value)
{
    int area = 0;
    cv::Mat_<uchar>::iterator it = input_img.begin<uchar>();
    cv::Mat_<uchar>::iterator end = input_img.end<uchar>();
    for (; it != end; ++it) {
        if ((*it) == pix_value) {
            area++;
        }
    }
    return area;
}

int CharDefectDetAlgo::get_char_pix_area(cv::Mat char_img_o, int bin_thr)
{
    cv::Mat char_img;
    cv::Mat bin_img;
    if (char_img_o.channels() == 3)
        cv::cvtColor(char_img_o, char_img, cv::COLOR_BGR2GRAY);
    else {
        char_img = char_img_o;
    }
    cv::threshold(char_img, bin_img, bin_thr, 255, cv::THRESH_BINARY_INV);
    int char_pix_area = get_pix_area(bin_img, 255);
    return char_pix_area;
}

bool CharDefectDetAlgo::char_defect_detection(const cv::Mat& char_img_o, std::string label)
{
    // 像素判断
    cv::Mat bin_img;
    cv::Mat ref_img;
    cv::Mat char_img;
    int bin_thr;
    int pix_avg_area, char_std = m_error_std;
    bool is_defect = false;
    // bool is_found_info = get_char_info(label, bin_thr, char_std, bin_thr, ref_img);
    std::string ref_img_p = m_ref_img_dir + "\\" + label + ".jpg";
    ref_img = cv::imread(ref_img_p, 0);
    if (ref_img.empty()) {
        LOG_WARN("Not found {} template image!", label);

        is_defect = false;
        return is_defect;
    }

    is_defect = char_defect_by_ecc(ref_img, char_img_o.clone(), m_bin_thr, label);
    if (is_defect) {
        LOG_INFO("<<NG3>>char: {} match result: {}", label, is_defect);
    }
    return is_defect;
}

void CharDefectDetAlgo::get_img_gradian(const cv::Mat& img, cv::Mat& gradian_img)
{
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(img, grad_x, -1, 2, 0, 3);
    cv::Sobel(img, grad_y, -1, 0, 2, 3);
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradian_img);
    static int grad_idx = 0;
    // cv::imwrite("D:\\ng_test\\"+std::to_string(grad_idx)+"_grad.jpg", gradian_img);
    grad_idx++;
}