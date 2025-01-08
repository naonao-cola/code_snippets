#include "ocr_det_algo.h"

DynamicalOCR::DynamicalOCR(CrnnInference* crnn)
{
    m_crnn = crnn;
#ifdef COLLECT_OCR_DATA
    std::string s = "D:\\OcrRecLabel";
    m_label_save.init_dir(s);
#endif
    m_ptr_dynamic_ocr_det = std::make_shared<DynamicOCR::DynamicCharDet>();
}

DynamicalOCR::~DynamicalOCR()
{
}

void DynamicalOCR::config(json config, RefImgTool *ref)
{
    m_ref = ref;
    m_config = config;
}

json DynamicalOCR::forward( cv::Mat img, const json &in_param )
{
    json all_out = json::array();
    int i = 0;
    PaperType ptype = get_paper_type(in_param);
    std::string ptype_str = get_paper_type_str(ptype);
    if (ptype_str == "HBZ_B" || ptype_str == "HBZ_A" || ptype_str == "HGZ_A") {
        return all_out;
    }
    m_ptr_dynamic_ocr_det->config(ptype_str, m_ref->has_stamp());
    json crnn_info = m_crnn->get_info();
    for (auto task : m_config["mask_shapes"]) {
        std::string label_name = task["label"];
        std::vector<cv::RotatedRect> OK_rrst_list;
        std::vector<cv::RotatedRect> NG_rrst_list;
        std::vector<cv::Mat> line_text_imgs_list;

        if ( label_name == "mark_a" ||
            label_name == "mark_b"  ||
            label_name == "qrcode"  ||
            label_name == "barcode" ||
            label_name == "stamp"   ||
            label_name == "hbz_barcode" ||
            label_name == "zhgk_yb" ||
            label_name == "blank" ) {
            continue;
        }
        cv::Mat crop_img;
        json tfm_pts = m_ref->get_roi_img(img, crop_img, bbox2polygon(task["points"]), 0, 0);
        //  切割为单行
        m_ptr_dynamic_ocr_det->spilt_muti_line_text_img(crop_img, line_text_imgs_list,
                                                        OK_rrst_list, NG_rrst_list, task["label"]);
        std::string rec_rst = "";
#ifdef COLLECT_OCR_DATA
        if (line_text_imgs_list.size() == 1) {
            m_label_save.save_ocr_data(line_text_imgs_list[0], in_param, task);
        }
#endif
        int ii = 0;
        // 识别文本，拼接成一个长字符串。用于比对字符内容是否有缺失
        for (auto cp_img : line_text_imgs_list) {
            cv::Mat item_img;
            json item_bbox = json::array();
            item_bbox.push_back({0, 0});
            item_bbox.push_back({cp_img.cols, cp_img.rows});
            if (cp_img.cols * 1.0 / cp_img.rows > (1.0 * crnn_info["width"] / crnn_info["height"])) {
                LOG_INFO("ocr image too long:  {} x {} ", cp_img.cols, cp_img.rows);
                continue;
            }
            // 按CRNN模型输入尺寸要求裁剪图片
            m_ref->get_roi_img(cp_img, item_img, bbox2polygon(item_bbox),
                               crnn_info["width"], crnn_info["height"], TFM_NONE);
            // item_img = gray_scale_image(item_img, 0, 210);
            cv::cvtColor(item_img, item_img, cv::COLOR_RGB2GRAY);
            cv::cvtColor(item_img, item_img, cv::COLOR_GRAY2RGB);

            if (ptype_str == "COC") {
                // 对比度拉伸
                item_img = gray_scale_image(item_img, 60, 220, 30, 245);
            }

#ifdef DEBUG_ON
            cv::imwrite("D:\\ocr_det\\" + label_name + "_item_"+std::to_string(ii)+".jpg", item_img);
            cv::imwrite("D:\\ocr_det\\" + label_name + "_crop.jpg", crop_img);
#endif
            // OCR识别+拼接
            std::string rst_text = m_crnn->forward_pure(item_img, in_param);
            rec_rst += rst_text;
            ii++;
        }
        // LOG_INFO("@@@ Dynamic OCR: {}", Utf8ToAnsi(rec_rst));

        json ocr_result = {{"text", rec_rst}, {"type", 2}};
        json out = {
            {"label", task["label"]},
            {"shapeType", "polygon"},
            {"points", m_ref->transform_result(tfm_pts)},
            {"result", ocr_result}};
        all_out.push_back(out);
        if (!NG_rrst_list.empty()) {
            json tfm_roi_bbox = polygon2bbox(tfm_pts);
            int det_x = static_cast<int>(tfm_roi_bbox[0][0]);
            int det_y = static_cast<int>(tfm_roi_bbox[0][1]);
            for (auto ng_rrx : NG_rrst_list) {
                cv::Rect box = ng_rrx.boundingRect();
                json points = {
                    box.x + det_x, box.y + det_y,
                    (box.x + det_x + box.width), box.y + det_y,
                    (box.x + det_x + box.width), (box.y + box.height + det_y),
                    box.x + det_x, (box.y + box.height + det_y)};
                json out = {
                    {"label", "NG2"},
                    {"shapeType", "polygon"},
                    {"points", m_ref->transform_result(points)},
                    {"result", {{"confidence", 1.0}, {"area", box.area()}}},
                };
                all_out.push_back(out);
            }
        }
        // LOG_INFO("[DynamicalNG Result]: {}", Utf8ToAnsi(all_out.dump()));
    }
    return all_out;
}
