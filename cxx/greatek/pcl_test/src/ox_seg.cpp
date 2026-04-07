/**
 * @FilePath     : /pcl_test/src/ox_seg.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-01-16 09:20:11
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-01-16 09:20:11
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#include "ox_seg.h"
#include <opencv2/imgproc.hpp>

YOLOv8SegDetector::YOLOv8SegDetector(const std::string& model_path, bool use_gpu, float conf_threshold, float iou_threshold, float mask_threshold)
    : YoloOnnxBase(model_path, use_gpu, conf_threshold, iou_threshold)
    , mask_threshold_(mask_threshold)
{
}

cv::Mat YOLOv8SegDetector::Preprocess(const cv::Mat& image, std::vector<int64_t>& input_tensor_shape)
{
    // 与检测模型相同的前处理 
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    int input_h = is_dynamic_input_shape_ ? 640 : input_shapes_[0][2];
    int input_w = is_dynamic_input_shape_ ? 640 : input_shapes_[0][3];

    // Letterbox缩放
    float scale = std::min(static_cast<float>(input_h) / rgb_image.rows, static_cast<float>(input_w) / rgb_image.cols);
    int   new_h = static_cast<int>(rgb_image.rows * scale);
    int   new_w = static_cast<int>(rgb_image.cols * scale);

    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(new_w, new_h));

    cv::Mat letterbox_img(input_h, input_w, CV_32FC3, cv::Scalar(114.0f / 255, 114.0f / 255, 114.0f / 255));
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    int      top  = (input_h - new_h) / 2;
    int      left = (input_w - new_w) / 2;
    cv::Rect roi(left, top, new_w, new_h);
    resized.copyTo(letterbox_img(roi));

    // 修正：正确设置letterbox_info_
    letterbox_info_.scale     = scale;
    letterbox_info_.dw        = left;
    letterbox_info_.dh        = top;
    letterbox_info_.new_shape = cv::Size(new_w, new_h);

    // NHWC -> NCHW
    int     H = letterbox_img.rows;
    int     W = letterbox_img.cols;
    cv::Mat input_blob(1, 3 * H * W, CV_32FC1);
    float*  blob_ptr = input_blob.ptr<float>();

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < H; ++h) {
            const float* row_ptr = letterbox_img.ptr<float>(h);
            for (int w = 0; w < W; ++w) {
                blob_ptr[c * H * W + h * W + w] = row_ptr[w * 3 + c];
            }
        }
    }

    input_tensor_shape = {1, 3, input_h, input_w};
    return input_blob;
}

DetectionResult YOLOv8SegDetector::Postprocess(const std::vector<Ort::Value>& output_tensors, const cv::Size& orig_size, const cv::Size& input_size)
{

    DetectionResult result;
    if (output_tensors.size() < 2)
        return result;   // 需要检测和分割两个输出

   // ===== 1. 解析检测输出 (1, 116, 8400) =====
    auto&  detect_tensor = output_tensors[0];
    float* detect_data   = const_cast<Ort::Value&>(detect_tensor).GetTensorMutableData<float>();
    auto   detect_shape  = detect_tensor.GetTensorTypeAndShapeInfo().GetShape();

    auto& mask_proto_tensor = output_tensors[1];
    auto  mask_proto_shape  = mask_proto_tensor.GetTensorTypeAndShapeInfo().GetShape();

    int num_boxes      = detect_shape[2];
    int num_attributes = detect_shape[1];
    int num_protos     = static_cast<int>(mask_proto_shape[1]);
    int num_classes    = static_cast<int>(num_attributes - 4 - num_protos);

    std::vector<cv::Rect>           boxes;
    std::vector<float>              scores;
    std::vector<int>                class_ids;
    std::vector<std::vector<float>> mask_coeffs;

    cv::Mat output0 = cv::Mat(cv::Size(num_boxes, num_attributes), CV_32F, detect_data).t();
    float*  pdata   = (float*)output0.data;

    for (int i = 0; i < num_boxes; ++i) {
        float* detection = pdata + i * num_attributes;

        // 类别分数
        float*    classes_scores = detection + 4;
        cv::Mat   scores_mat(1, num_classes, CV_32F, classes_scores);
        cv::Point class_id_point;
        double    max_score;
        cv::minMaxLoc(scores_mat, nullptr, &max_score, nullptr, &class_id_point);

        int   class_id   = class_id_point.x;
        float confidence = static_cast<float>(max_score);

        if (confidence < conf_threshold_)
            continue;

        // 边界框 (xc, yc, w, h) - 归一化坐标
        float xc = detection[0];
        float yc = detection[1];
        float w  = detection[2];
        float h  = detection[3];

         // 提取mask系数
        std::vector<float> coeffs(num_protos);
        float*             coeff_start = detection + 4 + num_classes;
        std::copy(coeff_start, coeff_start + num_protos, coeffs.begin());
        mask_coeffs.push_back(coeffs);

        // 转换到原始图像坐标
        float x1_ltrb = ((xc - w / 2.f) - letterbox_info_.dw) / letterbox_info_.scale;
        float y1_ltrb = ((yc - h / 2.f) - letterbox_info_.dh) / letterbox_info_.scale;
        float x2_ltrb = ((xc + w / 2.f) - letterbox_info_.dw) / letterbox_info_.scale;
        float y2_ltrb = ((yc + h / 2.f) - letterbox_info_.dh) / letterbox_info_.scale;

        // 修正：确保坐标在图像范围内
        x1_ltrb = std::clamp(x1_ltrb, 0.f, static_cast<float>(orig_size.width - 1));
        y1_ltrb = std::clamp(y1_ltrb, 0.f, static_cast<float>(orig_size.height - 1));
        x2_ltrb = std::clamp(x2_ltrb, 0.f, static_cast<float>(orig_size.width - 1));
        y2_ltrb = std::clamp(y2_ltrb, 0.f, static_cast<float>(orig_size.height - 1));

        boxes.emplace_back(static_cast<int>(x1_ltrb), static_cast<int>(y1_ltrb), static_cast<int>(x2_ltrb - x1_ltrb), static_cast<int>(y2_ltrb - y1_ltrb));
        scores.push_back(confidence);
        class_ids.push_back(class_id);
    }

    // ===== 2. NMS处理 =====
    std::vector<int> indices;
    // 修正：传入mask_coeffs，NMS会返回筛选后的索引
    NMSWithMasks(boxes, scores, class_ids, mask_coeffs, indices);

    // ===== 3. 解析Mask原型 =====
    if (!indices.empty() && output_tensors.size() > 1) {
        float* mask_proto_data = const_cast<Ort::Value&>(mask_proto_tensor).GetTensorMutableData<float>();

        // 筛选通过NMS的检测
        std::vector<cv::Rect>           final_boxes;
        std::vector<std::vector<float>> final_mask_coeffs;
        for (int idx : indices) {
            final_boxes.push_back(boxes[idx]);
            final_mask_coeffs.push_back(mask_coeffs[idx]);
        }

        // 生成最终掩码
        ProcessMaskProtos(mask_proto_data, mask_proto_shape, final_mask_coeffs, final_boxes, result.masks, orig_size);
    }

    // 填充最终结果
    for (int idx : indices) {
        result.boxes.push_back(boxes[idx]);
        result.scores.push_back(scores[idx]);
        result.class_ids.push_back(class_ids[idx]);
    }

    return result;
}

void YOLOv8SegDetector::ProcessMaskProtos(const float* mask_proto_data, const std::vector<int64_t>& mask_proto_shape, const std::vector<std::vector<float>>& mask_coeffs, const std::vector<cv::Rect>& boxes, std::vector<cv::Mat>& output_masks, const cv::Size& target_size)
{

    int num_protos = static_cast<int>(mask_proto_shape[1]);
    int proto_h    = static_cast<int>(mask_proto_shape[2]);
    int proto_w    = static_cast<int>(mask_proto_shape[3]);

    // 重塑protos为(num_protos, proto_h*proto_w)
    cv::Mat protos(num_protos, proto_h * proto_w, CV_32F);
    memcpy(protos.data, mask_proto_data, num_protos * proto_h * proto_w * sizeof(float));

    int input_h = static_cast<int>(input_shapes_[0][2]);
    int input_w = static_cast<int>(input_shapes_[0][3]);

    for (size_t i = 0; i < mask_coeffs.size(); ++i) {
        // 系数矩阵 (1, num_protos) × protos (num_protos, proto_h*proto_w)
        cv::Mat coeff_mat(1, num_protos, CV_32F, const_cast<float*>(mask_coeffs[i].data()));
        cv::Mat mask_raw = coeff_mat * protos;

        // 重塑为原型图大小
        cv::Mat mask_proto = mask_raw.reshape(1, proto_h);

        // Sigmoid激活
        cv::exp(-mask_proto, mask_proto);
        mask_proto = 1.0f / (1.0f + mask_proto);

        // 缩放到letterbox后的大小
        cv::Mat mask_input;
        cv::resize(mask_proto, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

        // 裁剪掉padding部分
        int x = letterbox_info_.dw;
        int y = letterbox_info_.dh;
        int w = letterbox_info_.new_shape.width;
        int h = letterbox_info_.new_shape.height;

        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (x + w > input_w)
            w = input_w - x;
        if (y + h > input_h)
            h = input_h - y;

        cv::Rect roi(x, y, w, h);
        cv::Mat  mask_no_pad = mask_input(roi);

        // 缩放到原始图像大小
        cv::Mat mask_original;
        cv::resize(mask_no_pad, mask_original, target_size, 0, 0, cv::INTER_LINEAR);

        // 二值化
        cv::Mat mask_binary;
        cv::threshold(mask_original, mask_binary, mask_threshold_, 1, cv::THRESH_BINARY);

        output_masks.push_back(mask_binary);
    }
}

// 修正：添加mask_coeffs参数，用于在NMS后筛选系数
void YOLOv8SegDetector::NMSWithMasks(std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& class_ids, std::vector<std::vector<float>>& mask_coeffs, std::vector<int>& indices)
{
    if (boxes.empty())
        return;

    std::vector<int> sorted_indices(scores.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int i1, int i2) { return scores[i1] > scores[i2]; });

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        int idx = sorted_indices[i];
        if (suppressed[idx])
            continue;

        indices.push_back(idx);

        for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
            int other_idx = sorted_indices[j];
            if (suppressed[other_idx])
                continue;

            if (class_ids[idx] == class_ids[other_idx]) {
                float iou = ComputeIOU(boxes[idx], boxes[other_idx]);
                if (iou >= iou_threshold_) {
                    suppressed[other_idx] = true;
                }
            }
        }
    }
}

float YOLOv8SegDetector::ComputeIOU(const cv::Rect& box1, const cv::Rect& box2)
{
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int inter_w      = std::max(0, x2 - x1);
    int inter_h      = std::max(0, y2 - y1);
    int intersection = inter_w * inter_h;
    int union_area   = box1.area() + box2.area() - intersection;

    if (union_area <= 0)
        return 0.0f;
    return static_cast<float>(intersection) / union_area;
}
