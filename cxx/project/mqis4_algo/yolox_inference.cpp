#include <opencv2/dnn.hpp>
#include "logger.h"
#include "yolox_inference.h"

YoloxInference::YoloxInference(char *ptr, int size, int device_id, json info):
    TrtInference(ptr, size, device_id),
    m_info(info)
{

}

void YoloxInference::preprocess(cv::Mat img, TrtBufferManager &buffers) {
    int index = m_engine->getBindingIndex("inputs");
    auto in_dims = m_engine->getBindingDimensions(index);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("inputs"));
    int h = in_dims.d[2];
    int w = in_dims.d[3];
    LOG_INFO("input size:{}x{}", w, h);
    cv::Mat norm_r(h, w, CV_32F, hostDataBuffer, 0);
    cv::Mat norm_g(h, w, CV_32F, hostDataBuffer + w*h, 0);
    cv::Mat norm_b(h, w, CV_32F, hostDataBuffer + 2*w*h, 0);

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(w, h));

    std::vector<cv::Mat> channels;
    cv::split(resize_img, channels);

    channels[0].convertTo(norm_r, CV_32F, 1.0 / 255);
    channels[1].convertTo(norm_g, CV_32F, 1.0 / 255);
    channels[2].convertTo(norm_b, CV_32F, 1.0 / 255);
}

json YoloxInference::post_process(cv::Mat img, const json& in_param, TrtBufferManager &buffers) {
    LOG_INFO("before post_process:{}", m_info.dump());
    const float conf_threshold = m_info["conf_threshold"];
    const float nms_threshold = m_info["nms_threshold"];
    const int top_k = m_info["top_k"];

    LOG_INFO("get dims");
    int index = m_engine->getBindingIndex("inputs");
    auto in_dims = m_engine->getBindingDimensions(index);
    index = m_engine->getBindingIndex("outputs");
    auto out_dims = m_engine->getBindingDimensions(index);
    const int anum = out_dims.d[1];
    const int num_item = out_dims.d[2];
    const int num_clsses = num_item - 5;

    int ori_h = img.rows;
    int ori_w = img.cols;

    int h = in_dims.d[2];
    int w = in_dims.d[3];

    float scale_h = ori_h / h;
    float scale_w = ori_w / w;

    float* out_ptr = static_cast<float*>(buffers.getHostBuffer("outputs"));

    std::vector<int> labels;
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;



    LOG_INFO("get model output");
    for (int i = 0; i < anum; i++){
        float box_objectness = out_ptr[4];
        for (int cidx=0; cidx < num_clsses; ++cidx) {
            float score = out_ptr[5+cidx] * box_objectness;
            if (score > conf_threshold) {
                float cx = out_ptr[0];
                float cy = out_ptr[1];
                float bw = out_ptr[2];
                float bh = out_ptr[3];

                float x1 = cx - bw / 2;
                float x2 = cx + bw / 2;
                float y1 = cy - bh / 2;
                float y2 = cy + bh / 2;
                if (x1 < 0 || x2 > w  || y1 < 0 || y2 > h){
                    continue;
                }
                cx , cy , bw , bh = cx * scale_w, cy * scale_h, bw * scale_w, bh * scale_h;

                cv::Rect box(cx, cy, bw, bh);
                bboxes.push_back(box);
                scores.push_back(score);
                labels.push_back(cidx);
            }
        }
        out_ptr += num_item;
    }

    LOG_INFO("before nms");
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, conf_threshold, nms_threshold, indices, 1, top_k);

    LOG_INFO("before convert output");
    json all_out = json::array();
    for (auto i: indices) {
        std::string label = m_info["labelset"][labels[i]];
        cv::Rect box = bboxes[i];
        json points = {box.x, box.y,
                       box.x+box.width, box.y,
                       box.x+box.width, box.y+box.height,
                       box.x, box.y+box.height};
        float conf = scores[i];

        json out = {
            {"label", label},
            {"shapeType", "polygon"},
            {"points", points},
            {"result", {{"confidence", conf}}}
        };
        all_out.push_back(out);
    }
    return all_out;
}
