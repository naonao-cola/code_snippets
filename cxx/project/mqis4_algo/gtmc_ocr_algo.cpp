#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gtmc_ocr_algo.h"
#include "logger.h"
#include "utils.h"
#include "algo_tool.h"

GtmcOcrAlgo::GtmcOcrAlgo(std::string tapp_path, int device_id):
    Tapp(tapp_path),
    m_device_id(device_id)
{
}

bool GtmcOcrAlgo::load() {
    LOG_INFO("load tapp");
    // Tapp::load();

    std::ifstream info_i(m_tapp_path + "/info.json", std::ios::in);
    info_i >> m_info;


    /*
    * Lazy mode
    */ 
    if (!m_crnn) {
        m_crnn = dynamic_cast<CrnnInference*>(load_model("crnn"));
    }
    if (!m_msae_hgz_a) {
        m_msae_hgz_a = dynamic_cast<MsaeInference*>(load_model("msae_hgz_a"));
    }
    if (!m_msae_hgz_b) {
        m_msae_hgz_b = dynamic_cast<MsaeInference*>(load_model("msae_hgz_b"));
    }
    if (!m_msae_hbz_a) {
        m_msae_hbz_a = dynamic_cast<MsaeInference*>(load_model("msae_hbz_a"));
    }
    if (!m_msae_hbz_b) {
        m_msae_hbz_b = dynamic_cast<MsaeInference*>(load_model("msae_hbz_b"));
    }
    if (!m_msae_ryz) {
        m_msae_ryz = dynamic_cast<MsaeInference*>(load_model("msae_ryz"));
    }
    if (!m_msae_coc) {
        m_msae_coc = dynamic_cast<MsaeInference*>(load_model("msae_coc"));
    }
    // m_msae_stamp = dynamic_cast<MsaeInference*>(load_model("msae_stamp"));

    m_decoder = new BarcodeDecoder(m_info["barcode"]);
    m_colorcheck = new ColorCheck(m_info["color_check"]);
    m_offsetcheck = new OffsetCheck(m_info["offset_check"]);
    m_double_print_check = new DoublePrintCheck(m_info["double_print_check"]);
    m_ocr_det = new DynamicalOCR(m_crnn);
    m_char_defect_det = new CharDefectDet();
    m_stamp_det = new StampDet(m_info);
    return true;
}

void _delete_bin_buf(void* buf) {
    delete buf;
}

TrtInference* GtmcOcrAlgo::load_model(std::string model_key)
{
    std::string tapp_path = m_tapp_path + "/" + model_key + ".trtmodel";
    auto data = nao::utils::load_file(tapp_path);
    LOG_INFO("load model: {}, blob size:{}", model_key, data.size());
    TrtInference* trtInference = nullptr;
    if (model_key == "crnn") {
        trtInference = new CrnnInference(data.data(), data.size(), m_device_id, m_info[model_key]);
    } else {
        bool is_stamp = (model_key == "msae_stamp");
        trtInference = new MsaeInference(data.data(), data.size(), m_device_id, m_info[model_key], is_stamp);
    }
    return trtInference;
}

GtmcOcrAlgo::~GtmcOcrAlgo() {
    delete m_crnn;
    delete m_ocr_det;
    delete m_msae_hgz_a;
    delete m_msae_hgz_b;
    delete m_msae_hbz_a;
    delete m_msae_hbz_b;
    delete m_msae_ryz;
    delete m_msae_coc;
    // delete m_msae_stamp;
    delete m_decoder;
    delete m_colorcheck;
    delete m_offsetcheck;
    delete m_double_print_check;
    delete m_char_defect_det;
    delete m_stamp_det;
    LOG_INFO("GtmcOcrAlgo delete done");
}

void GtmcOcrAlgo::config(const char *config_json_str) {
    LOG_INFO("Algo::config ++");
    std::string utf8_config_json_str = AnsiToUtf8(std::string(config_json_str));
    m_config = json::parse(utf8_config_json_str);
    LOG_INFO("Json tasks: {}", Utf8ToAnsi(m_config["task"].dump()));
    if (m_config.contains("ref_img")) {
        m_ref_img_tool.config(m_config);
        m_crnn->config(m_config, &m_ref_img_tool);
        m_ocr_det->config(m_config, &m_ref_img_tool);
        m_char_defect_det->config(m_config, &m_ref_img_tool);
        m_decoder->config(m_config, &m_ref_img_tool);
        m_colorcheck->config(&m_ref_img_tool);
        m_offsetcheck->config(m_config, &m_ref_img_tool);
        m_double_print_check->config(m_config, &m_ref_img_tool);
        m_msae_hgz_a->config(m_config, &m_ref_img_tool);
        m_msae_hgz_b->config(m_config, &m_ref_img_tool);
        m_msae_hbz_a->config(m_config, &m_ref_img_tool);
        m_msae_hbz_b->config(m_config, &m_ref_img_tool);
        m_msae_ryz->config(m_config, &m_ref_img_tool);
        m_msae_coc->config(m_config, &m_ref_img_tool);
        // m_msae_stamp->config(m_config, &m_ref_img_tool);
        m_stamp_det->config(m_config, &m_ref_img_tool);
    }
    LOG_INFO("Algo::config --");
}

const char* GtmcOcrAlgo::run(const char *in_param_json_str) {
    LOG_INFO("Algo::run ++ ");
    std::string utf8_json_str = AnsiToUtf8(std::string(in_param_json_str));
    json in_param = json::parse(utf8_json_str);
    LOG_INFO("Json in_param: {}", Utf8ToAnsi(in_param.dump()));
    //加载图片并旋转180
    std::vector<cv::Mat> imgs = load_imgs(in_param);
    LOG_INFO("load img done");

    json labelset =  json::array();

    for (int i = 0; i < imgs.size(); i++) {
        json out_shapes = json::array();

        auto img = imgs[i];
        auto img_p = in_param[i];
        PaperType ptype = get_paper_type(img_p);
        std::string ptype_str = get_paper_type_str(ptype);
        //临时代码
        // std:: array hbz_a_types = {HBZ_A};
        // bool is_hbz_a = std::find(hbz_a_types.begin(), hbz_a_types.end(), ptype) != hbz_a_types.end();

        //设置测试图片，计算和参考图之间的偏差矩阵
        bool locate_ok;
        cv::Mat img_crop = m_ref_img_tool.set_test_img(img, img_p, locate_ok);

        if (!locate_ok) {
            // 如果没找到mark点，认为是打印偏移
            std::string name = m_info["offset_check"]["label"];
            json points = {0, 0, img.cols, 0, img.cols, img.rows, 0, img.rows};
            json offset_err = {
                {"label", name},
                {"shapeType", "polygon"},
                {"points", points},
                {"result", {{"confidence", 1.0}}}
            };
            out_shapes.push_back(offset_err);
            LOG_WARN("Mark point locate fail, ignore other test!");
            labelset.push_back(shapes_to_labelset(out_shapes));
            continue;
        }

#ifdef DEBUG_ON
        cv::Mat draw_img;
        cv::cvtColor(img, draw_img, cv::COLOR_RGB2BGR);
#endif

        cv::Mat gray_img;
        cv::cvtColor(img_crop, gray_img, cv::COLOR_RGB2GRAY);

        // BarCode 优先读条码（证芯码）
        const json& decoder_out = m_decoder->forward(gray_img);
        for (auto _out: decoder_out) {
            if (ptype_str == "HGZ_A") {
                std::string zxm_text = _out["result"]["text"];
                if (zxm_text.length() > 0) {
                    zxm_text = zxm_text.substr(1, zxm_text.length()-1);
                    img_p.push_back({"zxm_char", zxm_text});
                }
            }
            out_shapes.push_back(_out);
        }

        // ####   OCR   ##########
        const json& crnn_out = m_crnn->forward(img_crop, img_p);
        for (auto _out: crnn_out) {
            out_shapes.push_back(_out);
        }

        const json& char_defect_out = m_char_defect_det->forward(img_crop, img_p);
        for (auto _out: char_defect_out) {
            out_shapes.push_back(_out);
#ifdef DEBUG_ON
            draw_polygon(draw_img, _out["points"], cv::Scalar(0,255,0));
#endif
        }
        LOG_INFO("<<NG3>>: {}", char_defect_out.dump());

        if (img_p["dynamic_region_det"] == 1) {
            const json& ocr_det_out = m_ocr_det->forward(img_crop, img_p);
#ifdef DEBUG_ON
            for (auto _out : ocr_det_out)
            {
                if (_out["label"] == "NG2") {
                    LOG_INFO("<< NG2 >> : {}", Utf8ToAnsi(_out.dump()));
                } else {
                    LOG_INFO("[{}] : {}", Utf8ToAnsi(_out["label"]), Utf8ToAnsi(_out["result"]["text"].dump()));
                }
            }
#endif            
            
            for (auto _out: ocr_det_out) {
                out_shapes.push_back(_out);
#ifdef DEBUG_ON
                if (_out["label"] == "NG2") {
                    draw_polygon(draw_img, _out["points"], cv::Scalar(0,255,0));
                }
#endif
            }
        }

        // 异色燃油证检查
        if (ptype == RYZ_RY || ptype == RYZ_HD) {
            const json& color_check_out = m_colorcheck->forward(img_crop);
            for (auto _out: color_check_out) {
                out_shapes.push_back(_out);
            }

            if (color_check_out.size() > 0) {
                labelset.push_back(shapes_to_labelset(out_shapes));
                continue;
            }
        }

        // 打印偏移检查
        const json& offset_check_out = m_offsetcheck->forward(gray_img, img_p);
        for (auto _out: offset_check_out) {
            out_shapes.push_back(_out);
        }

        if (offset_check_out.size() > 0) {
            labelset.push_back(shapes_to_labelset(out_shapes));
            continue;
        }

        // 印章检查
        const json& stamp_det_out = m_stamp_det->forward(img_crop, img_p);
        for (auto _out: stamp_det_out) {
            out_shapes.push_back(_out);
#ifdef DEBUG_ON
            draw_polygon(draw_img, _out["points"], cv::Scalar(255,255,0));
#endif
        }
        
        // 重影检查
        // const json& double_print_check_out = m_double_print_check->forward(gray_img);
        // for (auto _out: double_print_check_out) {
        //     out_shapes.push_back(_out);
        // }

        // if (double_print_check_out.size() > 0) {
        //     labelset.push_back(shapes_to_labelset(out_shapes));
        //     // continue;
        // }

        // ######  MSAE  ##########
        MsaeInference* msae_model = get_msae_model(ptype);
        if (msae_model != NULL) {
            const json& msae_out = msae_model->forward(img_crop, img_p);

//             if (m_ref_img_tool.has_stamp()) {
//                 cv::Mat stamp = m_ref_img_tool.get_stamp_img(img_crop, img_p);
//                 const json& msae_stamp_out = m_msae_stamp->forward(stamp, img_p);
//                 for (auto _out: msae_stamp_out)
//                 {
//                     out_shapes.push_back(_out);
// #ifdef DEBUG_ON
//                     draw_polygon(draw_img, _out["points"], cv::Scalar(255,0,0));
// #endif
//                 }
//             }
            

            for (auto _out: msae_out)
            {
                out_shapes.push_back(_out);
#ifdef DEBUG_ON
                draw_polygon(draw_img, _out["points"], cv::Scalar(0,0,255));
#endif
            }
        }

#ifdef DEBUG_ON
        write_debug_img("./gtmc_debug/result.jpg", draw_img);
#endif
        labelset.push_back(shapes_to_labelset(out_shapes));
    }

    json result = {
        {"classList", {}},
        {"labelSet", labelset}
    };
    
    m_last_result = Utf8ToAnsi(result.dump());
    LOG_INFO("===========AAAAA==============");
    LOG_INFO("Result:{}", m_last_result);
    LOG_INFO("===========BBBBBBB===========");
    // LOG_INFO(">> FINAL JSON RESULT: {}", m_last_result);
    // std::ofstream("D:/run_result.json") << m_last_result<<std::endl;
    LOG_INFO("Algo::run --");
    return m_last_result.c_str();
}

MsaeInference* GtmcOcrAlgo::get_msae_model(PaperType ptype)
{
    std:: array hgz_a_types = {HGZ_A};
    std:: array hgz_b_types = {HGZ_B};
    std:: array hbz_a_types = {HBZ_A};
    std:: array hbz_b_types = {HBZ_B_RY1, HBZ_B_RY2, HBZ_B_HD1, HBZ_B_HD2, HBZ_B_CD};
    std:: array ryz_types = {RYZ_RY, RYZ_HD};
    std:: array coc_types = {COC_RY, COC_HD, COC_V4};

    MsaeInference* msae_model = NULL;
    if (std::find(hgz_a_types.begin(), hgz_a_types.end(), ptype) != hgz_a_types.end()) {
        msae_model = m_msae_hgz_a;
    } else if (std::find(hgz_b_types.begin(), hgz_b_types.end(), ptype) != hgz_b_types.end()) {
        msae_model = m_msae_hgz_b;
    } else if (std::find(hbz_a_types.begin(), hbz_a_types.end(), ptype) != hbz_a_types.end()) {
        msae_model = m_msae_hbz_a;
    } else if (std::find(hbz_b_types.begin(), hbz_b_types.end(), ptype) != hbz_b_types.end()) {
        msae_model = m_msae_hbz_b;
    } else if (std::find(ryz_types.begin(), ryz_types.end(), ptype) != ryz_types.end()) {
        msae_model = m_msae_ryz;
    } else if (std::find(coc_types.begin(), coc_types.end(), ptype) != coc_types.end()) {
        msae_model = m_msae_coc;
    }
    return msae_model;
}

std::vector<cv::Mat> GtmcOcrAlgo::load_imgs(const json& in_param) {
    std::vector<cv::Mat> imgs; 
    for (auto img_p: in_param) {
        cv::Mat img;
        if (std::string(img_p["img_path"]).empty()) {
            uint8_t *img_ptr = (uint8_t*)((uint64_t)img_p["img_ptr"]); 
            int img_w = img_p["img_w"];
            int img_h = img_p["img_h"];
            std::string format = img_p["format"];
            int type = CV_8UC1;
            if (format != "GRAY") {
                type = CV_8UC3;
            }
            img = cv::Mat(img_h, img_w, type, img_ptr);
            if (format == "GRAY") {
                cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
            } else if (format == "BGR") {
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            }
        } else {
            img = cv::imread(img_p["img_path"], cv::IMREAD_COLOR);
            if (img.channels() > 1) {
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            }
            if (img_p["rotate180"] == 1) {
                cv::Mat img_180;
                cv::rotate(img, img_180, cv::ROTATE_180);
                img = img_180;
                // LOG_INFO("******************* ROTATE 180 *******");
            }
            
        }
        imgs.push_back(img);
    }
    return imgs;
}

void GtmcOcrAlgo::package_model(std::string model_dir_path, std::string model_key)
{
    int file_len = 0;
    char *trt_file;
    file_len = Tapp::read_file(model_dir_path + "/" + model_key + ".trt.engine", &trt_file);
    LOG_INFO("package model:{}, weight size:{}", model_key, file_len);
    set_blob(model_key, trt_file, file_len);
    delete trt_file;
}

bool GtmcOcrAlgo::is_msae_a_type(PaperType ptype)
{
    std:: array msae_types = {HGZ_A, HGZ_B, COC_RY, COC_HD, COC_V4};
    return std::find(msae_types.begin(), msae_types.end(), ptype) != msae_types.end();
};

bool GtmcOcrAlgo::is_msae_b_type(PaperType ptype)
{
    std:: array msae_types = {RYZ_RY, RYZ_HD, HBZ_A, HBZ_B_RY1, HBZ_B_RY2, HBZ_B_HD1, HBZ_B_HD2, HBZ_B_CD};
    return std::find(msae_types.begin(), msae_types.end(), ptype) != msae_types.end();
};

json GtmcOcrAlgo::shapes_to_labelset(const json& shapes, const std::string& image_path)
{
    json labelset = {
        {"imageName", ""},
        {"imagePath", image_path},
        {"status", "OK"},
        {"shapes", shapes}
    };
    return labelset;
}