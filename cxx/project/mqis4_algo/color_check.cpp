#include "color_check.h"
#include "logger.h"

ColorCheck::ColorCheck(json info):
    m_info(info)
{
}

void ColorCheck::config(RefImgTool *ref)
{
    m_ref = ref;
}

json ColorCheck::forward(cv::Mat img)
{
    json all_out = json::array();
    cv::Mat ref_img = m_ref->get_ref_img();

    cv::Scalar ref_mean = cv::mean(ref_img);
    cv::Scalar cur_mean = cv::mean(img);

    double max_gap = 0;
    for (int i=0; i < 3; ++i) {
        double gap = std::abs(ref_mean[i] - cur_mean[i]);
        if (gap > max_gap) {
            max_gap = gap;
        }
    }

    LOG_INFO("[Result]: max_gap:{:.2f}", max_gap);
    DBG_ASSERT(m_info.contains("threshold"));

    if (max_gap > m_info["threshold"]) {
        DBG_ASSERT(m_info.contains("label"));
        std::string name = m_info["label"];
        json points = {0,0,img.cols,0, img.cols,img.rows, 0,img.rows};
        json out = {
            {"label", name},
            {"shapeType", "polygon"},
            {"points", points},
            {"result", {{"confidence", 1.0}}}
        };
        all_out.push_back(out);
    }
    return all_out;
}
