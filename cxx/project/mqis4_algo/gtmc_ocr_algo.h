#pragma once

#include "tapp.h"
#include "yolox_inference.h"
#include "crnn_inference.h"
#include "msae_inference.h"
#include "barcode_decoder.h"
#include "color_check.h"
#include "offset_check.h"
#include "double_print_check.h"
#include "ocr_det_algo.h"
#include "defines.h"
#include "char_defect_det.h"
#include "stamp_det.h"

class GtmcOcrAlgo :public Tapp {

public:
    GtmcOcrAlgo(std::string tapp_path, int device_id=0);
    ~GtmcOcrAlgo();

    bool load();
    void package_model(std::string model_dir_path, std::string model_key);
    void config(const char *config_json_str);
    const char* run(const char *in_param_json_str);

public:
    std::vector<cv::Mat> load_imgs(const json& in_param);
    bool is_msae_a_type(PaperType ptype);
    bool is_msae_b_type(PaperType ptype);
    json shapes_to_labelset(const json& shapes, const std::string& image_path="");
    TrtInference * load_model(std::string model_key);
    MsaeInference* get_msae_model(PaperType ptype);

    int m_device_id;
    json m_config;
    std::string m_last_result;
    CrnnInference *m_crnn=nullptr;
    DynamicalOCR *m_ocr_det = nullptr;
    MsaeInference *m_msae_hgz_a = nullptr;
    MsaeInference *m_msae_hgz_b = nullptr;
    MsaeInference *m_msae_hbz_a = nullptr;
    MsaeInference *m_msae_hbz_b = nullptr;
    MsaeInference *m_msae_ryz = nullptr;
    MsaeInference *m_msae_coc = nullptr;


    /*std::shared_ptr<CrnnInference> m_crnn = nullptr;
    std::shared_ptr<CrnnInference> m_msae_hgz_a = nullptr;
    std::shared_ptr<CrnnInference> m_msae_hgz_b = nullptr;
    std::shared_ptr<CrnnInference> m_msae_hbz_a = nullptr;
    std::shared_ptr<CrnnInference> m_msae_hbz_b = nullptr;
    std::shared_ptr<CrnnInference> m_msae_ryz = nullptr;
    std::shared_ptr<CrnnInference> m_msae_coc = nullptr;*/



    // MsaeInference *m_msae_stamp;
    RefImgTool m_ref_img_tool;
    BarcodeDecoder *m_decoder;
    ColorCheck *m_colorcheck;
    OffsetCheck *m_offsetcheck;
    DoublePrintCheck *m_double_print_check;
    StampDet *m_stamp_det;

    // Tapp* m_tapp_crnn;
    // Tapp* m_tapp_hgz_a;
    // Tapp* m_tapp_hgz_b;
    // Tapp* m_tapp_hbz_a;
    // Tapp* m_tapp_hbz_b;
    // Tapp* m_tapp_ryz;
    // Tapp* m_tapp_coc;
    CharDefectDet* m_char_defect_det;
};
