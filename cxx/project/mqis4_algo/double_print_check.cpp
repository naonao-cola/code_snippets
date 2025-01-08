#include "double_print_check.h"
#include "logger.h"
#include "utils.h"

DoublePrintCheck::DoublePrintCheck(json info):
    m_info(info)
{
    m_info["threshold"] = 0.6;
}

void DoublePrintCheck::config(json config, RefImgTool *ref)
{
    m_config = config;
    m_ref = ref;
    m_roi_list = json::array();
    for (auto shape : m_config["mask_shapes"])
    {
        if (shape["label"] == "mark_a" || shape["name"] == "mark_b")
        {
            m_roi_list.push_back(bbox2polygon(shape["points"]));
        }
    }
    // json first_task = json::object();
    // json last_task = json::object();
    // for (auto task: config["task"]) {
    //     if (task["type"] == "static_text") {
    //         if (!first_task.contains("type")) {
    //             first_task = task;
    //             continue;
    //         }
    //         last_task = task;
    //     }
    // }
    // if (first_task.contains("type")) {
    //     m_task_list.push_back(first_task);
    // }
    // if (last_task.contains("type")) {
    //     m_task_list.push_back(last_task);
    // }
}

json DoublePrintCheck::forward(cv::Mat gray_img)
{
    json all_out = json::array();
    for (auto roi: m_roi_list) {
        // std::string task_type = task["type"];
        // if (task_type != "static_text" && task_type != "dynamic_text")
        // {
        //     continue;
        // }
        cv::Mat ref_gray_img = m_ref->get_ref_img(true);
        cv::Mat temp_roi_img;
        m_ref->get_pad_roi_img(ref_gray_img, temp_roi_img, roi, 0, TFM_REF);
        
        cv::Mat find_roi_img;
        // m_ref->get_pad_roi_img(gray_img, find_roi_img, task["roi"], m_info["pad_ratio"]);
        m_ref->get_pad_roi_img(gray_img, find_roi_img, roi, 200);

        cv::Mat result;
        cv::matchTemplate(find_roi_img, temp_roi_img, result, cv::TM_CCOEFF_NORMED);
        result = result * 255;
        result.convertTo(result, CV_8U);

        cv::Mat mask;
        LOG_INFO("############# dp thresh: {}", m_info["threshold"].dump());
        int threshold = 255 * m_info["threshold"];
        cv::threshold(result, mask, threshold, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point> > contoursQuery;
        cv::findContours(mask, contoursQuery, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
        if (contoursQuery.size() > 1 && contoursQuery.size() < 4) {
            std::string name = m_info["label"];
            json points = {0,0,gray_img.cols,0, gray_img.cols,gray_img.rows, 0,gray_img.rows};
            json out = {
                {"label", name},
                {"shapeType", "polygon"},
                {"points", points},
                {"result", {{"confidence", 1.0}}}
            };
            all_out.push_back(out);
        }
    }
    LOG_INFO("[Result]: {}", all_out.dump());
    return all_out;
}
