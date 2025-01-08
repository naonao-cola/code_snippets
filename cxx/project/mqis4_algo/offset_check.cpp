#include "offset_check.h"
#include "logger.h"
#include "utils.h"

OffsetCheck::OffsetCheck(json info):
    m_info(info)
{
}

OffsetCheck::~OffsetCheck() {
    
}

void OffsetCheck::config(json config, RefImgTool *ref)
{
    m_config = config;
    m_ref = ref;
}

json OffsetCheck::forward(cv::Mat gray_img, const json& in_param)
{
    DBG_ASSERT(m_info.contains("offset_threshold"));
    DBG_ASSERT(m_info.contains("angle_threshold"));

    json all_out = json::array();

    LocateInfo ref_loc = m_ref->get_ref_loc();
    LocateInfo img_loc = m_ref->get_img_loc();
    double delta_angle = RadToDeg(img_loc.angle - ref_loc.angle);
    if (std::abs(delta_angle) > 180) {
        delta_angle = delta_angle > 0 ? delta_angle - 360 : delta_angle + 360;
    }

    double angle_threshold = get_param<double>(in_param, "offset_angle_threashold", -1);
    double offset_threshold = get_param<double>(in_param, "offset_translate_threashold", -1);

    PaperType ptype = get_paper_type(in_param);
    std::string ptype_str = get_paper_type_str(ptype);

    if (angle_threshold < 0) {
        angle_threshold = 2;
    }
    if (offset_threshold < 0) {
        offset_threshold = 60;
        if (ptype_str == "HBZ_A" || ptype_str == "HGZ_A") {
            offset_threshold = 150;
        } else if (ptype_str == "HGZ_B") {
            offset_threshold = 70;
        } else if (ptype_str == "HBZ_B" || ptype_str == "RYZ") {
            offset_threshold = 60;
        }
    }

    double dist_a = p2p_distance(ref_loc.x1, ref_loc.y1, img_loc.x1, img_loc.y1);
    double dist_b = p2p_distance(ref_loc.x2, ref_loc.y2, img_loc.x2, img_loc.y2);
    double xoff_a = std::abs(ref_loc.x1 - img_loc.x1);
    double xoff_b = std::abs(ref_loc.x2 - img_loc.x2);
    bool is_ng = std::abs(delta_angle) > angle_threshold || xoff_a > offset_threshold || xoff_b > offset_threshold;

    LOG_INFO("[Result]: theta: {:.4f}  mark_a: {:.2f}  mark_b: {:.2f}  Result: {}", delta_angle, dist_a, dist_b, is_ng?"NG":"OK");
    LOG_INFO("[Result]: theta: {:.4f}  xoff_a: {:.2f}  xoff_b: {:.2f}  Result: {}", delta_angle, xoff_a, xoff_b, is_ng?"NG":"OK");

    if (is_ng) {
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

    return all_out;
}
