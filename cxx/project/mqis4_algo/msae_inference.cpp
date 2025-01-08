#include <opencv2/dnn.hpp>
#include <algorithm>
#include "msae_inference.h"
#include "logger.h"
#include "utils.h"


Bbox::Bbox(int ix1, int iy1, int ix2, int iy2, float iscore):
    x1(ix1), y1(iy1), x2(ix2), y2(iy2), score(iscore)
{
    w = ix2 - ix1;
    h = iy2 - iy1;
}


Bbox::Bbox(json pts, std::string type, float iscore)
{
    score = iscore;
    std::vector<int> points = pts;
    if("polygon" == type){   
        std::vector<int> x_vec, y_vec;
        for(int i = 0; i < points.size(); i += 2){
            x_vec.push_back(points[i]);
            y_vec.push_back(points[1 + 1]);
        }

        x1 = *std::min_element(x_vec.begin(), x_vec.end());
        y1 = *std::min_element(y_vec.begin(), y_vec.end());
        x2 = *std::max_element(x_vec.begin(), x_vec.end());
        y2 = *std::max_element(y_vec.begin(), y_vec.end());
    } else{
        x1 = pts[0];
        y1 = pts[1];
        x2 = pts[2];
        x2 = pts[3];
    }
    w = x2 - x1;
    h = y2 - y1;
    
}

float Bbox::iou(Bbox obox){
    int mx1 = std::max(x1, obox.x1);
    int my1 = std::max(y1, obox.y1);
    int mx2 = std::min(x2, obox.x2);
    int my2 = std::min(y2, obox.y2);

    int iw = std::max(0, mx2 - mx1);
    int ih = std::max(0, my2 - my1);
    float over_area = iw * ih;
    float cur_arae = w * h;
    float o_area = obox.w * obox.h;
    if (cur_arae == 0 || o_area == 0){
        return 0;
    }
    return over_area / (cur_arae + o_area - over_area);
}


MsaeInference::MsaeInference(char *ptr, int size, int device_id, json info, bool is_stamp):
    TrtInference(ptr, size, device_id),
    m_info(info), m_is_stamp(is_stamp)
{

}

void MsaeInference::config(json config, RefImgTool *ref)
{
    m_config = config;
    m_ref = ref;
}

cv::Mat MsaeInference::preprocess_img(cv::Mat img, const json& in_param)
{
    return m_ref->get_masked_img(img, in_param);
}

void MsaeInference::preprocess(cv::Mat img, const json& in_param, TrtBufferManager &buffers)
{
    if (!m_is_stamp) {
        img = preprocess_img(img, in_param);
        write_debug_img("./gtmc_debug/msae_input.jpg", img, true);
    }

    int index = m_engine->getBindingIndex("inputs");
    auto in_dims = m_engine->getBindingDimensions(index);

    float *hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("inputs"));
    int h = in_dims.d[2];
    int w = in_dims.d[3];
    // LOG_INFO("MSAE input size: {} x {}", w, h);
    cv::Mat norm_r(h, w, CV_32F, hostDataBuffer, 0);
    cv::Mat norm_g(h, w, CV_32F, hostDataBuffer + w * h, 0);
    cv::Mat norm_b(h, w, CV_32F, hostDataBuffer + 2 * w * h, 0);

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(w, h));

    std::vector<cv::Mat> channels;
    cv::split(resize_img, channels);

    channels[0].convertTo(norm_r, CV_32F, 1.0/255);
    channels[1].convertTo(norm_g, CV_32F, 1.0/255);
    channels[2].convertTo(norm_b, CV_32F, 1.0/255);

    m_processed_img = resize_img;
}

json MsaeInference::post_process(cv::Mat img, const json& in_param, TrtBufferManager &buffers)
{
    // LOG_INFO("before post_process:{}", m_info.dump());
    const float area_threshold = 1000.0; // m_info["area_threshold"]; 
    const float border_coef = 1.618;   //m_info["border_coef"];
    const float border_ratio = 0.05; //m_info["border_ratio"];

    PaperType ptype = get_paper_type(in_param);
    std::string ptype_str = get_paper_type_str(ptype);

    float bin_threshold = 70;
    std::string threshold_key = m_is_stamp ? "stamp_threshold" : "msae_threshold";
    if (in_param.contains(threshold_key)) {
        bin_threshold = in_param[threshold_key];
    }
    LOG_INFO("amap threshold: {}", bin_threshold);

    float msae_area_threshold = 8000;
    if (in_param.contains("msae_area_threshold")) {
        msae_area_threshold = in_param["msae_area_threshold"];
    }

    int index = m_engine->getBindingIndex("inputs");
    auto in_dims = m_engine->getBindingDimensions(index);
    index = m_engine->getBindingIndex("outputs");
    auto out_dims = m_engine->getBindingDimensions(index);

    int out_h = out_dims.d[1];
    int out_w = out_dims.d[2];

    int ori_h = img.rows;
    int ori_w = img.cols;

    int h = in_dims.d[2];
    int w = in_dims.d[3];

    float scale_h = ori_h / out_h;
    float scale_w = ori_w / out_w;

    float* out_ptr = static_cast<float*>(buffers.getHostBuffer("outputs"));

    // 边缘弱化
    int border_w = round(out_w * border_ratio);
    int border_h = round(out_h * border_ratio);
    float coef_factor = std::sqrt(border_coef);
    for (int row = 0; row < out_h; row++)
	{
		for (int col = 0; col < out_w; col++)
		{
            float val = out_ptr[row*out_w+col];
            if (row < border_h || row > out_h-border_h-1 || col < border_w || col > out_w-border_w-1) {
                val /= coef_factor;
                // val = 5;
            }
            if (row < 2*border_h || row > (out_h-2*border_h-1) || col < 2*border_w || col > (out_w-2*border_w-1)) {
                val /= coef_factor;
                // val = 10;
            }
            out_ptr[row*out_w+col] = val;
        }
	}

    cv::Mat outmap (out_h, out_w, CV_32F, out_ptr, 0);

    // 根据vMin, vMax缩放
    float v_min = 0, v_max = m_is_stamp ? 8 : 10;
    outmap = (outmap - v_min) / (v_max - v_min);
    outmap.setTo(1, outmap > 1);
    outmap.convertTo(outmap, CV_8UC1, 255);
        
    cv::Mat amap;
    cv::resize(outmap, amap, cv::Size(ori_w, ori_h));

    if (m_is_stamp) {
        write_debug_img("./gtmc_debug/stamp_amap.jpg", amap);
    } else {
        write_debug_img("./gtmc_debug/amap.jpg", amap);
    }

    // 空白区域衰减
    if (!m_is_stamp) {
        cv::Mat blank_mask;
        m_dt_tool.config(ptype_str, m_processed_img.cols * 1.0 / img.cols);
        m_dt_tool.forwad(m_processed_img, blank_mask);
        cv::resize(blank_mask, blank_mask, cv::Size(ori_w, ori_h));
        write_debug_img("./gtmc_debug/blank_mask.jpg", blank_mask);

        double blank_decay_thresh = get_param<double>(in_param, "blank_decay", 0.5);
        cv::Mat decay_amap = amap * blank_decay_thresh;
        decay_amap.copyTo(amap, blank_mask);
        write_debug_img("./gtmc_debug/amap_decay.jpg", amap);
    }

    cv::GaussianBlur(amap, amap, cv::Size(5,5), 1.0);

    // 二值化
    cv::Mat bin_map;
    cv::threshold(amap, bin_map, bin_threshold, 255, cv::THRESH_BINARY);
    cv::medianBlur(bin_map, bin_map, 7);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin_map, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

    json all_out = json::array();
    for(auto cnt:contours){
        double area = cv::contourArea(cnt);
        if (area > area_threshold){
            cv::Rect box = cv::boundingRect(cnt);
            cv::Mat1b mask(amap.rows, amap.cols, uchar(0));
            cv::fillPoly(mask, cnt, cv::Scalar(1));
            
            cv::Scalar avg = cv::mean(amap, mask);
            float area_conf = area / (30000);
            area_conf = (area_conf < 1.0) ? area_conf : 1.0;
            float conf = (avg[0] / (bin_threshold * 1.8)) * area_conf;
            conf = (conf < 1.0) ? conf : 1.0;

            json points = {
                box.x, box.y, (box.x + box.width), box.y,
                (box.x + box.width), (box.y + box.height), box.x, (box.y + box.height)
            };

            json defect_bbox = polygon2bbox(points);
            if (!is_intersect_important_area(defect_bbox) && area < msae_area_threshold)
            {
                // 过滤掉面积小，且位置不在印章区域的NG
                continue;
            }

            json rst_points = m_is_stamp ? m_ref->transform_stamp_result(points) : m_ref->transform_result(points);
            json out = {
                {"label", "NG"},
                {"shapeType", "polygon"},
                {"points", rst_points},
                {"result", {{"confidence", conf}, {"area", area}, {"mean", avg[0]}}},
            };
            all_out.push_back(out);
        }
    }
    
    // LOG_INFO("msae post_process done: {}", all_out.dump());
    return all_out;
}

bool MsaeInference::is_intersect_important_area(const json& bbox)
{
    json shapes = m_config["mask_shapes"];
    for (auto shape: shapes) {
        std::string label = shape["label"];
        if (label == "stamp" || label == "hgz_lsb" || label == "hgz_fwbs")
        {
            json shape_pts = bbox2polygon(shape["points"]);
            json tfm_shape_pts = m_ref->transform_roi(shape_pts, true);
            if (is_intersect(polygon2bbox(tfm_shape_pts), bbox)) {
                return true;
            }
        }
    }
    return false;
}

json MsaeInference::forward(cv::Mat img, const json& in_param){
    // LOG_INFO("MSAE forward in_param: {}", in_param.dump());
    // const float iou_threshold = m_info["iou_threshold"];
    json all_out = json::array();
    json result = TrtInference::forward(img, in_param);

    // for(auto res : result){
    //     for(auto task:m_config["task"]){
    //         LOG_INFO("before iou");
    //         Bbox rbox(res["points"], "polygon");
    //         Bbox tbox(task["iou"], "polygon");
    //         float o_area = rbox.iou(tbox);
    //         LOG_INFO("after iou");

    //         if (o_area > iou_threshold){
    //             res["label"] = task["name"];
    //             all_out.push_back(res);
    //         }
    //     }
    // }
    LOG_INFO("[Result]: {}", result.dump());
    return result;
}
