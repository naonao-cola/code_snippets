/**
 * @FilePath     : /yibiao/src/gauge_mask.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-03-30 15:13:51
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-04-03 15:36:02
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#include "gauge_mask.hpp"
#include "../HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo/yolo.hpp"
#include "../HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_seg/yolo_seg.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


// 三个模型,一个检测仪表位置，一个分割刻度盘，一个分割指针
static std::string det_modelPath         = "./GAUGE_DET.engine";
static std::string pointer_seg_modelPath = "./POINTER_SEG.engine";
static std::string tick_seg_modelPath    = "./TICK_SEG.engine";

// 模型指针
static std::shared_ptr<Yolo::Infer>    det_infer;
static std::shared_ptr<YoloSeg::Infer> pointer_seg_infer;
static std::shared_ptr<YoloSeg::Infer> tick_seg_infer;

// 加载指针
static void ensure_infers()
{
    if (!det_infer) {
        det_infer = Yolo::create_infer(det_modelPath, Yolo::Type::V8, 0, 0.25f, 0.5f);
    }
    if (!pointer_seg_infer) {
        pointer_seg_infer = YoloSeg::create_infer(pointer_seg_modelPath, 0, 0.25f, 0.5f, YoloSeg::NMSMethod::FastGPU, 1024, false);
    }
    if (!tick_seg_infer) {
        tick_seg_infer = YoloSeg::create_infer(tick_seg_modelPath, 0, 0.25f, 0.5f, YoloSeg::NMSMethod::FastGPU, 1024, false);
    }
}

bool GaugeMasksFromImage(const cv::Mat& src,
                         cv::Mat&       scale_mask_full,
                         cv::Mat&       pointer_mask_full,
                         int            det_class,
                         int            scale_class,
                         int            pointer_class,
                         float          det_thresh,
                         float          seg_thresh)
{
    ensure_infers();
    if (!det_infer || !pointer_seg_infer || !tick_seg_infer)
        return false;
    auto det_boxes = det_infer->commit(src).get();
    if (det_boxes.empty())
        return false;
    ObjectDetector::Box best_box;
    float               best_conf = 0.f;
    for (auto& b : det_boxes) {
        if (b.class_label == det_class && b.confidence > det_thresh && b.confidence > best_conf) {
            best_conf = b.confidence;
            best_box  = b;
        }
    }
    if (best_conf <= 0.3f)
        return false;
    int x1 = std::max(0, (int)std::floor(best_box.left));
    int y1 = std::max(0, (int)std::floor(best_box.top));
    int x2 = std::min(src.cols, (int)std::ceil(best_box.right));
    int y2 = std::min(src.rows, (int)std::ceil(best_box.bottom));
    if (x2 <= x1 || y2 <= y1)
        return false;

    // 检测长宽比,长宽比过小返回fasle
    int w = x2 - x1;
    int h = y2 - y1;
    if (w >= h && (float)w / h > 2.0f)
        return false;
    if (h >= w && (float)h / w > 2.0f)
        return false;

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat  crop              = src(roi).clone();
    auto     pointer_seg_boxes = pointer_seg_infer->commit(crop).get();
    auto     tick_seg_boxes    = tick_seg_infer->commit(crop).get();
    // 分割结果转掩码
    cv::Mat scale_mask   = cv::Mat::zeros(crop.size(), CV_8UC1);
    cv::Mat pointer_mask = cv::Mat::zeros(crop.size(), CV_8UC1);

    for (auto& sb : pointer_seg_boxes) {
        if (sb.confidence < seg_thresh)
            continue;
        int bx1 = std::max(0, (int)std::floor(sb.left));
        int by1 = std::max(0, (int)std::floor(sb.top));
        int bx2 = std::min(crop.cols, (int)std::ceil(sb.right));
        int by2 = std::min(crop.rows, (int)std::ceil(sb.bottom));
        int bw  = std::max(0, bx2 - bx1);
        int bh  = std::max(0, by2 - by1);
        if (bw <= 0 || bh <= 0)
            continue;
        if (!sb.seg || sb.seg->data == nullptr || sb.seg->width <= 0 || sb.seg->height <= 0)
            continue;
        cv::Mat mask_small(sb.seg->height, sb.seg->width, CV_8UC1, sb.seg->data);
        cv::Mat mask_resized;
        cv::resize(mask_small, mask_resized, cv::Size(bw, bh), 0, 0, cv::INTER_NEAREST);
        if (sb.class_label == pointer_class) {
            cv::Mat dst = pointer_mask(cv::Rect(bx1, by1, bw, bh));
            mask_resized.copyTo(dst, mask_resized);
        }
    }
    for (auto& tb : tick_seg_boxes) {
        if (tb.confidence < seg_thresh)
            continue;
        int tx1 = std::max(0, (int)std::floor(tb.left));
        int ty1 = std::max(0, (int)std::floor(tb.top));
        int tx2 = std::min(crop.cols, (int)std::ceil(tb.right));
        int ty2 = std::min(crop.rows, (int)std::ceil(tb.bottom));
        int tw  = std::max(0, tx2 - tx1);
        int th  = std::max(0, ty2 - ty1);
        if (tw <= 0 || th <= 0)
            continue;
        if (!tb.seg || tb.seg->data == nullptr || tb.seg->width <= 0 || tb.seg->height <= 0)
            continue;
        cv::Mat mask_small(tb.seg->height, tb.seg->width, CV_8UC1, tb.seg->data);
        cv::Mat mask_resized;
        cv::resize(mask_small, mask_resized, cv::Size(tw, th), 0, 0, cv::INTER_NEAREST);
        if (tb.class_label == scale_class) {
            cv::Mat dst = scale_mask(cv::Rect(tx1, ty1, tw, th));
            mask_resized.copyTo(dst, mask_resized);
        }
    }

    scale_mask_full   = scale_mask;
    pointer_mask_full = pointer_mask;
    /* cv::imwrite("./scale_mask.png", scale_mask);
     cv::imwrite("./pointer_mask.png", pointer_mask);*/
    return true;
}

bool GaugeMasksFromImage(
    const cv::Mat& src, cv::Mat& scale_mask_full, cv::Mat& pointer_mask_full, int scale_class, int pointer_class, float det_thresh, float seg_thresh)
{
    return GaugeMasksFromImage(src, scale_mask_full, pointer_mask_full, 0, scale_class, pointer_class, det_thresh, seg_thresh);
}

static cv::Mat letterbox_resize_binary(const cv::Mat& src, int dst_size)
{
    int w = src.cols;
    int h = src.rows;
    if (w <= 0 || h <= 0)
        return cv::Mat();
    float   s  = std::min(dst_size / (float)w, dst_size / (float)h);
    int     nw = std::max(1, (int)std::round(w * s));
    int     nh = std::max(1, (int)std::round(h * s));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_NEAREST);
    cv::Mat canvas = cv::Mat::zeros(dst_size, dst_size, CV_8UC1);
    int     x      = (dst_size - nw) / 2;
    int     y      = (dst_size - nh) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, nw, nh)));
    return canvas;
}

bool NormalizeMasks512(const cv::Mat& scale_mask_full, const cv::Mat& pointer_mask_full, cv::Mat& scale_norm, cv::Mat& pointer_norm)
{
    if (scale_mask_full.empty() || pointer_mask_full.empty())
        return false;
    scale_norm   = letterbox_resize_binary(scale_mask_full, 512);
    pointer_norm = letterbox_resize_binary(pointer_mask_full, 512);
    return !scale_norm.empty() && !pointer_norm.empty();
}



static void build_line_image_internal(const cv::Mat& mask_norm, unsigned char label_value, cv::Mat& line_image)
{
    const int SEG_IMAGE_SIZE = 512;
    const int LINE_HEIGH     = 120;
    const int LINE_WIDTH     = 1570;
    const int CIRCLE_RADIUS  = 250;
    const int cx             = 256;
    const int cy             = 256;
    line_image               = cv::Mat(LINE_HEIGH, LINE_WIDTH, CV_8UC1, cv::Scalar(0));
    const float pi           = 3.1415926536f;
    for (int row = 0; row < LINE_HEIGH; row++) {
        for (int col = 0; col < LINE_WIDTH; col++) {
            float theta = pi * 2.0f / (float)LINE_WIDTH * (col + 1);
            int   rho   = CIRCLE_RADIUS - row - 1;
            int   ix    = (int)std::round(cx + rho * std::cos(theta));
            int   iy    = (int)std::round(cy - rho * std::sin(theta));
            if (ix < 0 || iy < 0 || ix >= SEG_IMAGE_SIZE || iy >= SEG_IMAGE_SIZE)
                continue;
            unsigned char mv = mask_norm.at<unsigned char>(iy, ix);
            if (mv > 0) {
                line_image.at<unsigned char>(row, col) = label_value;
            }
        }
    }
}




void BuildLineImageFromMask(const cv::Mat& mask_norm, unsigned char label_value, std::vector<unsigned char>& output, cv::Mat& debug_mat)
{
    build_line_image_internal(mask_norm, label_value, debug_mat);
    const int LINE_HEIGH = debug_mat.rows;
    const int LINE_WIDTH = debug_mat.cols;
    output.assign(LINE_HEIGH * LINE_WIDTH, 0);
    for (int row = 0; row < LINE_HEIGH; row++) {
        const unsigned char* p = debug_mat.ptr<unsigned char>(row);
        for (int col = 0; col < LINE_WIDTH; col++) {
            output[row * LINE_WIDTH + col] = p[col];
        }
    }
}

void ConvertLineTo1D(const std::vector<unsigned char>& line_output, unsigned char label_value, std::vector<unsigned int>& counts)
{
    const int LINE_HEIGH = 120;
    const int LINE_WIDTH = 1570;
    if ((int)line_output.size() < LINE_HEIGH * LINE_WIDTH) {
        counts.clear();
        return;
    }
    counts.clear();
    counts.reserve(LINE_WIDTH);
    for (int col = 0; col < LINE_WIDTH; col++) {
        unsigned int c = 0;
        for (int row = 0; row < LINE_HEIGH; row++) {
            unsigned char v = line_output[row * LINE_WIDTH + col];
            if (v == label_value)
                c++;
        }
        counts.push_back(c);
    }
}

void MeanFiltration1D(const std::vector<unsigned int>& input, std::vector<unsigned int>& output)
{
    if (input.empty()) {
        output.clear();
        return;
    }
    unsigned long long sum = 0;
    for (auto v : input) {
        sum += v;
    }
    unsigned int mean = (unsigned int)(sum / input.size());
    output.clear();
    output.reserve(input.size());
    for (auto v : input) {
        if (v >= mean)
            output.push_back(v);
        else
            output.push_back(0);
    }
}

bool ReadMeterFrom1D(const std::vector<unsigned int>& scale_mean, const std::vector<unsigned int>& pointer, GaugeReadResult& result)
{
    result      = GaugeReadResult();
    const int n = (int)std::min(scale_mean.size(), pointer.size());
    if (n < 3)
        return false;

    std::vector<int> scale_location;
    int              one_scale_start = -1;
    for (int i = 0; i < n - 1; i++) {
        if (scale_mean[i] > 0 && scale_mean[i + 1] > 0) {
            if (one_scale_start < 0)
                one_scale_start = i;
        }
        if (one_scale_start >= 0) {
            if (scale_mean[i] == 0 && scale_mean[i + 1] == 0) {
                int one_scale_end      = i - 1;
                int one_scale_location = (one_scale_start + one_scale_end) / 2;
                scale_location.push_back(one_scale_location);
                one_scale_start = -1;
            }
        }
    }

    int one_pointer_start = -1;
    int pointer_location  = -1;
    for (int i = 0; i < n - 1; i++) {
        if (pointer[i] > 0 && pointer[i + 1] > 0) {
            if (one_pointer_start < 0)
                one_pointer_start = i;
        }
        if (one_pointer_start >= 0) {
            if (pointer[i] == 0 && pointer[i + 1] == 0) {
                int one_pointer_end = i - 1;
                pointer_location    = (one_pointer_start + one_pointer_end) / 2;
                one_pointer_start   = -1;
            }
        }
    }

    result.pointer_location = pointer_location;             // 指针的像素值
    result.scale_num        = (int)scale_location.size();   // 刻度的个数
    if (result.scale_num < 2 || pointer_location < 0)
        return false;

    for (int i = 0; i < result.scale_num - 1; i++) {
        if (scale_location[i] <= pointer_location && pointer_location < scale_location[i + 1]) {
            result.scales = ((float)(i + 1)) + ((float)(pointer_location - scale_location[i])) / ((float)(scale_location[i + 1] - scale_location[i]));
            break;
        }
    }

    int left  = scale_location.front();
    int right = scale_location.back();
    if (right > left) {
        result.ratio = (float)(pointer_location - left) / (float)(right - left);
    }
    return true;
}

static void mat_to_vector_u8(const cv::Mat& mat, std::vector<unsigned char>& output)
{
    output.clear();
    if (mat.empty() || mat.type() != CV_8UC1)
        return;
    output.resize((size_t)mat.rows * (size_t)mat.cols);
    const int width = mat.cols;
    for (int row = 0; row < mat.rows; row++) {
        const unsigned char* p = mat.ptr<unsigned char>(row);
        std::copy(p, p + width, output.data() + (size_t)row * (size_t)width);
    }
}

static int find_best_shift_by_gap(const cv::Mat& line_mat)
{
    if (line_mat.empty() || line_mat.cols <= 0)
        return 0;
    cv::Mat col_max;
    cv::reduce(line_mat, col_max, 0, cv::REDUCE_MAX, CV_8U);
    const int w = col_max.cols;
    if (w <= 0)
        return 0;
    std::vector<unsigned char> has(w, 0);
    for (int i = 0; i < w; i++) {
        has[i] = col_max.at<unsigned char>(0, i) > 0 ? 1 : 0;
    }
    int best_len   = 0;
    int best_start = 0;
    int cur_len    = 0;
    int cur_start  = 0;
    for (int i = 0; i < 2 * w; i++) {
        int idx = i % w;
        if (has[idx] == 0) {
            if (cur_len == 0)
                cur_start = i;
            cur_len++;
            if (cur_len > best_len && cur_len <= w) {
                best_len   = cur_len;
                best_start = cur_start;
            }
        }
        else {
            cur_len = 0;
        }
    }
    if (best_len <= 0 || best_len >= w)
        return 0;
    if (best_len < std::max(10, w / 20))
        return 0;
    int start = best_start % w;
    int shift = (start + best_len) % w;
    if (shift == w)
        shift = 0;
    return shift;
}

static void apply_col_shift(cv::Mat& mat, int shift)
{
    const int w = mat.cols;
    if (mat.empty() || w <= 0)
        return;
    int s = shift % w;
    if (s < 0)
        s += w;
    if (s == 0)
        return;
    cv::Mat shifted;
    cv::hconcat(mat.colRange(s, w), mat.colRange(0, s), shifted);
    mat = shifted;
}

static void reverse_cols(cv::Mat& mat)
{
    if (mat.empty())
        return;
    cv::flip(mat, mat, 1);
}

bool ReadGaugePipeline(const cv::Mat&      src,
                       int                 det_class,
                       int                 scale_class,
                       int                 pointer_class,
                       float               det_thresh,
                       float               seg_thresh,
                       GaugeReadResult&    result,
                       GaugePipelineDebug* debug)
{
    return ReadGaugePipeline(src, det_class, scale_class, pointer_class, det_thresh, seg_thresh, true, result, debug);
}

bool ReadGaugePipeline(const cv::Mat&      src,
                       int                 det_class,
                       int                 scale_class,
                       int                 pointer_class,
                       float               det_thresh,
                       float               seg_thresh,
                       bool                right_to_left,
                       GaugeReadResult&    result,
                       GaugePipelineDebug* debug)
{
    cv::Mat scale_mask_full;
    cv::Mat pointer_mask_full;
    if (!GaugeMasksFromImage(src, scale_mask_full, pointer_mask_full, det_class, scale_class, pointer_class, det_thresh, seg_thresh))
        return false;
    // 归一化到512
    cv::Mat scale_norm;
    cv::Mat pointer_norm;
    if (!NormalizeMasks512(scale_mask_full, pointer_mask_full, scale_norm, pointer_norm))
        return false;


    /* cv::imwrite("./scale_norm.png", scale_norm);
     cv::imwrite("./pointer_norm.png", pointer_norm);*/

    std::vector<unsigned char> scale_line;
    std::vector<unsigned char> pointer_line;
    cv::Mat                    scale_line_mat;
    cv::Mat                    pointer_line_mat;
    BuildLineImageFromMask(scale_norm, 200, scale_line, scale_line_mat);
    BuildLineImageFromMask(pointer_norm, 100, pointer_line, pointer_line_mat);

    // 极坐标是逆时针
    if (right_to_left) {
        reverse_cols(scale_line_mat);
        reverse_cols(pointer_line_mat);
    }

    int line_shift = find_best_shift_by_gap(scale_line_mat);
    if (line_shift != 0) {
        apply_col_shift(scale_line_mat, line_shift);
        apply_col_shift(pointer_line_mat, line_shift);
    }
    mat_to_vector_u8(scale_line_mat, scale_line);
    mat_to_vector_u8(pointer_line_mat, pointer_line);

    /*cv::imwrite("./scale_line_mat.png", scale_line_mat);
    cv::imwrite("./pointer_line_mat.png", pointer_line_mat);*/

    // 转为一维向量
    std::vector<unsigned int> scale_1d;
    std::vector<unsigned int> pointer_1d;
    ConvertLineTo1D(scale_line, 200, scale_1d);
    ConvertLineTo1D(pointer_line, 100, pointer_1d);

    // 平滑处理
    std::vector<unsigned int> scale_mean_1d;
    MeanFiltration1D(scale_1d, scale_mean_1d);
    std::vector<unsigned int> pointer_mean_1d;
    MeanFiltration1D(pointer_1d, pointer_mean_1d);


    if (!ReadMeterFrom1D(scale_mean_1d, pointer_mean_1d, result))
        return false;

    /* if (debug) {
         debug->scale_mask_full   = scale_mask_full;
         debug->pointer_mask_full = pointer_mask_full;
         debug->scale_norm        = scale_norm;
         debug->pointer_norm      = pointer_norm;
         debug->scale_line_mat    = scale_line_mat;
         debug->pointer_line_mat  = pointer_line_mat;
         debug->line_shift        = line_shift;
         debug->scale_line        = std::move(scale_line);
         debug->pointer_line      = std::move(pointer_line);
         debug->scale_1d          = std::move(scale_1d);
         debug->pointer_1d        = std::move(pointer_1d);
         debug->scale_mean_1d     = std::move(scale_mean_1d);
     }*/
    return true;
}
