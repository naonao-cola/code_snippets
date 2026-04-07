#include "ox_d.h"


// YOLOv8实现
YOLOv8Detector::YOLOv8Detector(const std::string& model_path, bool use_gpu, float conf_threshold, float iou_threshold)
    : YoloOnnxBase(model_path, use_gpu, conf_threshold, iou_threshold)
{
}

inline cv::Mat YOLOv8Detector::Preprocess(const cv::Mat& image, std::vector<int64_t>& input_tensor_shape)
{
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    int input_h = is_dynamic_input_shape_ ? 640 : input_shapes_[0][2];
    int input_w = is_dynamic_input_shape_ ? 640 : input_shapes_[0][3];

    // ====== Letterbox缩放 ======
    float scale = std::min(static_cast<float>(input_h) / rgb_image.rows, static_cast<float>(input_w) / rgb_image.cols);
    int   new_h = static_cast<int>(rgb_image.rows * scale);
    int   new_w = static_cast<int>(rgb_image.cols * scale);

    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(new_w, new_h));

    // 创建letterbox图像（灰色填充）
    cv::Mat letterbox_img(input_h, input_w, CV_32FC3, cv::Scalar(114.0f / 255, 114.0f / 255, 114.0f / 255));
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
    // 计算padding并放置图像
    int      top  = (input_h - new_h) / 2;
    int      left = (input_w - new_w) / 2;
    cv::Rect roi(left, top, new_w, new_h);
    resized.copyTo(letterbox_img(roi));
    // 保存letterbox信息供后处理使用
    letterbox_info_ = {scale, left, top, cv::Size(new_w, new_h)};
    // ====== NHWC -> NCHW ======
    int     H = letterbox_img.rows;
    int     W = letterbox_img.cols;
    cv::Mat input_blob(1, 3 * H * W, CV_32FC1);
    float*  blob_ptr = input_blob.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < H; ++h) {
            const float* row_ptr = letterbox_img.ptr<float>(h);
            for (int w = 0; w < W; ++w) {
                (blob_ptr[c * H * W + h * W + w]) = (row_ptr[w * 3 + c]);
            }
        }
    }
    input_tensor_shape = {1, 3, input_h, input_w};
    return input_blob;
}

inline DetectionResult YOLOv8Detector::Postprocess(const std::vector<Ort::Value>& output_tensors, const cv::Size& orig_size, const cv::Size& input_size)
{
    DetectionResult result;
    if (output_tensors.empty())
        return result;

    // 获取输出数据
    // 移除 const 限定符以允许调用非 const 方法
    auto&  non_const_output_tensor = const_cast<Ort::Value&>(output_tensors[0]);
    float* output_data             = non_const_output_tensor.GetTensorMutableData<float>();
    auto   output_shape            = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // YOLOv8输出格式: (1, 84, 8400) -> (batch, 4+num_classes, num_boxes)
    int num_boxes      = output_shape[2];
    int num_attributes = output_shape[1];   // 4 + num_classes

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      class_ids;

    cv::Mat output0 = cv::Mat(cv::Size(num_boxes, num_attributes), CV_32F, output_data).t();
    float*  pdata   = (float*)output0.data;

    // 解析每个检测框
    for (int i = 0; i < num_boxes; ++i) {
        float* detection = pdata + i * num_attributes;
        // 获取置信度最高的类别
        float* classes_scores = detection + 4;
        int    class_id       = std::max_element(classes_scores, classes_scores + num_attributes - 4) - classes_scores;
        float  confidence     = classes_scores[class_id];

        if (confidence < conf_threshold_)
            continue;
        // 获取边界框 (xc, yc, w, h)，这些是相对于letterbox图像的归一化坐标
        float xc = detection[0];
        float yc = detection[1];
        float w  = detection[2];
        float h  = detection[3];

        // ====== 转换到原始图像坐标 ======
        // 1. 从归一化坐标转换到letterbox图像坐标
        // 模型输出的归一化坐标是相对于网络输入尺寸 (input_size)
        // 应使用 input_size 而不是 letterbox 中的 new_shape
        float x1_ltrb = ((xc - w / 2.f) - letterbox_info_.dw) / letterbox_info_.scale;
        float y1_ltrb = ((yc - h / 2.f) - letterbox_info_.dh) / letterbox_info_.scale;
        float x2_ltrb = ((xc + w / 2.f) - letterbox_info_.dw) / letterbox_info_.scale;
        float y2_ltrb = ((yc + h / 2.f) - letterbox_info_.dh) / letterbox_info_.scale;
        float x1, y1, x2, y2;
        // 限制在原始图像范围内
        x1 = std::max(0.f, x1_ltrb);
        y1 = std::max(0.f, y1_ltrb);
        x2 = std::min(x2_ltrb, orig_size.width - 1.f);
        y2 = std::min(y2_ltrb, orig_size.height - 1.f);
        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(confidence);
        class_ids.push_back(class_id);
    }
    // NMS处理
    std::vector<int> indices;
    NMS(boxes, scores, class_ids, indices);
    result.boxes.reserve(indices.size());
    result.scores.reserve(indices.size());
    result.class_ids.reserve(indices.size());

    for (int idx : indices) {
        result.boxes.push_back(boxes[idx]);
        result.scores.push_back(scores[idx]);
        result.class_ids.push_back(class_ids[idx]);
    }

    return result;
}

inline float YOLOv8Detector::ComputeIOU(const cv::Rect& box1, const cv::Rect& box2)
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

    return static_cast<float>(intersection) / static_cast<float>(union_area);
}

inline void YOLOv8Detector::NMS(std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& class_ids, std::vector<int>& indices)
{
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
