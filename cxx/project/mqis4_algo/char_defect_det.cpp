#include "char_defect_det.h"

CharDefectDet::CharDefectDet() {
    m_char_defect_algo = std::make_shared<CharDefectDetAlgo>();
}

int read_num_from_json(const json& config, std::string key) {
    if ( config.contains(key) && config[key].is_null() == false) {
        LOG_INFO("print config[{}]", key);
        LOG_INFO("config[{}]: {}", key, config[key]);
        if(config[key].is_string()) {
            std::string data = get_param<std::string>(config, key, "-1");
            LOG_INFO("<<NG TEST>> String config[{}]: {}", key, config[key]);
            LOG_INFO("<<NG TEST>> String config[{}]: {}", key, data);
            LOG_INFO("<<NG TEST>> String config[{}]: {}", key, std::stoi(data));
            return std::stoi(data);
        } else if(config[key].is_number()) {
            LOG_INFO("<<NG TEST>> Int config[{}]: {}", key, get_param<int>(config, key, -1));
            return get_param<int>(config, key, -1);
        } else{
            LOG_WARN("Due to the json type not is a number  Can not read [{}] from json!", key);
            return -1;
        }
    } else {
        LOG_WARN("The json keys not contains [{}], return -1!", key);
        return -1;
    }
}
void CharDefectDet::config_runtime(const json& config) {
    bool status = true;
    LOG_INFO("<<NG3 TEST>> runtime_config: {}", config.dump());
    m_ng3_min_area = read_num_from_json(config, "ng3_min_area");
    m_morph_size = read_num_from_json(config, "char_expand_size");
    m_bin_thr = read_num_from_json(config, "char_defect_threshold");
}


void CharDefectDet::config(json config)
{
    LOG_INFO("<<NG3 TEST>> start config....");
    m_config = config;
    json runtime_config;
    int ng3_min_area = 30;
    int char_expand_size = 7;
    int char_defect_threshold = -1;
   
    int bin_threshold = -1;
    if (m_config.contains("char_ref_img_dir"))
    {

        if(m_config.contains("char_ref_img_dir")) {
            m_ref_dir = m_config["char_ref_img_dir"];
        }

        bin_threshold = char_defect_threshold !=-1? char_defect_threshold: 200;

        m_char_defect_param["COC"] = {
            {"char_defect_threshold", bin_threshold},
            {"char_pix_std", 300},
            {"char_expand_size", char_expand_size},
            {"ng3_min_area", ng3_min_area}
            };

        bin_threshold = char_defect_threshold !=-1? char_defect_threshold: 220;
        m_char_defect_param["HGZ_B"] = {
            {"char_defect_threshold", bin_threshold},
            {"char_pix_std", 300},
            {"char_expand_size", char_expand_size},
            {"ng3_min_area", ng3_min_area}
            };


        bin_threshold = char_defect_threshold !=-1? char_defect_threshold: 100;
        m_char_defect_param["RYZ"] = {
            {"char_defect_threshold", bin_threshold},
            {"char_pix_std", 400},
            {"char_expand_size", char_expand_size},
            {"ng3_min_area", ng3_min_area}
            };

        bin_threshold = char_defect_threshold !=-1? char_defect_threshold: 200;
        m_char_defect_param["GQFT"] = {
            {"char_defect_threshold", bin_threshold},
            {"char_pix_std", 300},
            {"char_expand_size", char_expand_size},
            {"ng3_min_area", ng3_min_area}
            };

        bin_threshold = char_defect_threshold !=-1? char_defect_threshold: 180;
        m_char_defect_param["HGZ_A"] = {
            {"char_defect_threshold", bin_threshold},
            {"char_pix_std", 300},
            {"char_expand_size", char_expand_size},
            {"ng3_min_area", ng3_min_area}
            };

        bin_threshold = char_defect_threshold !=-1? char_defect_threshold: 180;
        m_char_defect_param["HBZ_A"] = {
            {"char_defect_threshold", bin_threshold},
            {"char_pix_std", 300},
            {"char_expand_size", char_expand_size},
            {"gray_scale", 4},
            {"ng3_min_area", ng3_min_area}
            };

        bin_threshold = char_defect_threshold !=-1? char_defect_threshold: 180;
        m_char_defect_param["HBZ_B"] = {
            {"char_defect_threshold", bin_threshold},
            {"char_pix_std", 300},
            {"char_expand_size", char_expand_size},
            {"ng3_min_area", ng3_min_area}
            };
        LOG_INFO("m_ref_dir: {}", m_ref_dir);
        LOG_INFO("char_defect_param: {}", m_char_defect_param.dump());
    }
    else
    {
        LOG_WARN("The config json file not contains [char_ref_img_dir]!");
    }
}

void CharDefectDet::config(json config_json, RefImgTool *ref)
{
    config(config_json);
    m_ref = ref;
}

bool CharDefectDet::read_all_json_file(const std::string &strFileName, json &ref_char_info)
{
    std::ifstream in(strFileName, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        return false;
    }
    ref_char_info = json::parse(in);
    return true;
}

json CharDefectDet::forward(cv::Mat img, json in_param)
{
    LOG_INFO("<<NG3 TEST>>Load config from in_param....");
    json all_out = json::array();
    if (img.empty())
    {
        LOG_WARN("The input image is empty!");
        return all_out;
    }

    PaperType ptype = get_paper_type(in_param);
    m_paper_name = get_paper_type_str(ptype);
    
    //获取参数
    update_paramter();
    config_runtime(in_param);    
    m_char_defect_algo->config(m_ref_dir, m_bin_thr, m_char_pix_std, m_morph_size, m_paper_name, m_ng3_min_area);
    // m_char_defect_algo->set_ref_img_type(m_paper_name);
    // m_char_defect_algo->config(m_ref_dir, 180, 200, 7);

    if (m_paper_name == "HGZ_A") {
        // HGZ条码去除下方文字
        json zxm_bbox = m_ref->get_shape_pts_by_name("hgz_barcode");

        cv::Mat barcode_img;
        json barcode_pts = bbox2polygon(zxm_bbox);
        cv::Rect barcode_pad_rc = m_ref->get_pad_roi_img(img, barcode_img, barcode_pts, 0, TFM_INFER);
        std::vector<cv::Point> zxm_char_coords;
        cv::Mat barcode_out = m_ref->barcode_extract(barcode_img, zxm_char_coords);
        if (zxm_char_coords.size() == 4) {
            json zxm_pts = json::array();
            for (auto pt : zxm_char_coords)
            {
                zxm_pts.push_back(pt.x + barcode_pad_rc.x);
                zxm_pts.push_back(pt.y + barcode_pad_rc.y);
            }
            // get_roi_img(barcode_img, m_zxm_img, m_zxm_pts, 0, 0, TFM_NONE);
            // cv::imwrite("D:/zxm.jpg", m_zxm_img);
            m_config["task"].push_back({
                {"name", "zxm_char"},
                {"type", "ZXM"},
                {"roi", zxm_pts},
                {"det_char_defect", 1}
            });
        } else {
            cv::Rect rect(0, 0, barcode_img.cols, barcode_img.rows);
            json tfm_pts = m_ref->transform_roi(barcode_pts, false);
            merge_result(rect, tfm_pts, "zxm_char", all_out);
        }
    }

    int i = 0;
    for (auto task : m_config["task"]) {
        LOG_INFO("task: {}", task.dump());
        i++;
        std::string name = task["name"];
        if (!task.contains("det_char_defect")) {
            continue;
        }
        int det_char_defect = task["det_char_defect"];
        if (det_char_defect == 0) {
            continue;
        }
        cv::Mat crop_img;
        TFM_MODE tfm_mode = task["type"] == "ZXM" ? TFM_NONE : TFM_INFER;
        json tfm_pts = m_ref->get_roi_img(img, crop_img, task["roi"], 0, 0, tfm_mode);
        std::string label = "";
#ifdef DEBUG_ON
        cv::imwrite("D:\\work_dir\\test_data\\HBZ_A\\"+name +"_"+label+".jpg", crop_img);
#endif
        if (in_param.contains(name)) {
            label = AnsiToUtf8(in_param[name]);
        } else {
            LOG_WARN("The [{}] in_param not contains [{}] info", m_paper_name, name);
            return all_out;
        }

        //  切割字符
        std::vector<cv::Mat> out_imgs;
        std::vector<cv::Rect> char_rect;
        if (name == "GQFT") {
            m_char_defect_algo->config(m_ref_dir, 180, 200, 5, "", m_ng3_min_area);
            std::vector<cv::Rect2i> gqft_result = m_char_defect_algo->forward(crop_img, "GQFT");
            if ( gqft_result.size() > 0 ) {
                cv::Rect rect(0, 0, crop_img.cols, crop_img.rows);
                merge_result(rect, tfm_pts, name, all_out);
            }
            continue;
        }

        // m_char_defect_algo->config(m_ref_dir, 180, 200, 7);
        std::vector<cv::Rect2i> gqft_result = m_char_defect_algo->forward(crop_img, label);
        if ( gqft_result.size() > 0 ) {
            cv::Rect rect(0, 0, crop_img.cols, crop_img.rows);
            merge_result(rect, tfm_pts, name, all_out);
        }
    }
    return all_out;
}

void CharDefectDet::merge_result(cv::Rect box, json tfm_pts, std::string task_name, json &all_out) {
    json tfm_roi_bbox = polygon2bbox(tfm_pts);
    int delt_x = static_cast<int>(tfm_roi_bbox[0][0]);
    int delt_y = static_cast<int>(tfm_roi_bbox[0][1]);

    json points = {
        box.x + delt_x, box.y + delt_y,
        (box.x + delt_x + box.width), box.y + delt_y,
        (box.x + delt_x + box.width), (box.y + box.height + delt_y),
        box.x + delt_x, (box.y + box.height + delt_y)};

    json out = {
        {"label", task_name == "zxm_char" ? "NG2" : "NG3"},
        {"shapeType", "polygon"},
        {"points", m_ref->transform_result(points)},
        {"result", {{"confidence", 1.0}, {"area", box.area()}, {"task_name", task_name}}},
    };
    LOG_INFO("[Character Defect det Result]: {}", Utf8ToAnsi(out.dump()));
    all_out.push_back(out);
}

bool CharDefectDet::update_paramter(const json& runtime_config) {
    bool rst1 = update_paramter();
    if(runtime_config.contains("char_expand_size")) {
        if(m_config["char_expand_size"] > 0) {
            m_morph_size = m_config["char_expand_size"];
        }
    }

    if(runtime_config.contains("char_defect_threshold")) {
        if(m_config["char_defect_threshold"] > 0) {
            m_bin_thr = m_config["char_defect_threshold"];
        }
    }

    if(runtime_config.contains("ng3_min_area")) {
        if(m_config["ng3_min_area"] > 0) {
            m_ng3_min_area = m_config["ng3_min_area"];
        }
    }
    return rst1;
}


bool CharDefectDet::update_paramter() {
    if ( m_char_defect_param.contains(m_paper_name) == false ) {
        LOG_WARN("Not found {} paramter!", m_paper_name);
        return false;
    }

    json param = m_char_defect_param[m_paper_name];
    if (m_config.contains("char_expand_size")) {
        m_morph_size = m_config["char_expand_size"];
    } else {
        m_morph_size = param["char_expand_size"];
    }

    if (m_config.contains("char_defect_threshold")) {
        m_bin_thr = m_config["char_defect_threshold"];
    } else {
        m_bin_thr = param["char_defect_threshold"];
    }

    if (param.contains("gray_scale")) {
        m_gray_scale = param["gray_scale"];
    }

    if (param.contains("char_pix_std")) {
        m_char_pix_std = param["char_pix_std"];
    }
    if (param.contains("ng3_min_area")) {
        m_ng3_min_area = param["ng3_min_area"];
    }
    return true;
}
