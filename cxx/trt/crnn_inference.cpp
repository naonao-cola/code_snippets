#include "crnn_inference.h"
#include "logger.h"
#include "utils.h"
#include <iostream>
#include <filesystem>
#include "algo_tool.h"
CrnnInference::CrnnInference(char *ptr, int size, int device_id, json info):
    TrtInference(ptr, size, device_id),
    m_info(info)
{
    m_info["width"] = 1920;
    m_info["height"] = 48;
    if (!std::filesystem::exists("D:/ocr_data/dynamic/OK")) {
        std::filesystem::create_directories("D:/ocr_data/dynamic/OK");
    }
    if (!std::filesystem::exists("D:/ocr_data/dynamic/NG")) {
        std::filesystem::create_directories("D:/ocr_data/dynamic/NG");
    }
    if (!std::filesystem::exists("D:/ocr_data/static/OK")) {
        std::filesystem::create_directories("D:/ocr_data/static/OK");
    }
    if (!std::filesystem::exists("D:/ocr_data/static/NG")) {
        std::filesystem::create_directories("D:/ocr_data/static/NG");
    }
    dynamic_ok_fs.open("D:/ocr_data/dynamic/ocr_ok_labels.txt", std::ios::app);
    dynamic_ng_fs.open("D:/ocr_data/dynamic/ocr_ng_labels.txt", std::ios::app);
    static_ok_fs.open("D:/ocr_data/static/ocr_ok_labels.txt", std::ios::app);
    static_ng_fs.open("D:/ocr_data/static/ocr_ng_labels.txt", std::ios::app);
}

CrnnInference::~CrnnInference()
{
    dynamic_ok_fs.close();
    dynamic_ng_fs.close();
    static_ok_fs.close();
    static_ng_fs.close();
}  

void CrnnInference::config(json config, RefImgTool *ref) {
    m_ref = ref;
    m_config = config;
}

void CrnnInference::preprocess(cv::Mat img, const json& in_param, TrtBufferManager &buffers) {
    auto in_dims = m_engine->getBindingDimensions(0);
    float* in_ptr = static_cast<float*>(buffers.getHostBuffer(0));
    
   /* int h = in_dims.d[2];
    int w = in_dims.d[3];*/
    //cv::Mat norm_r(h, w, CV_32F, in_ptr, 0);
    //cv::Mat norm_g(h, w, CV_32F, in_ptr + w*h, 0);
    //cv::Mat norm_b(h, w, CV_32F, in_ptr + 2*w*h, 0);

    //// LOG_INFO("img: {}x{}  info: {}x{}", img.rows, img.cols, h, w);
    //DBG_ASSERT(h == img.rows);
    //DBG_ASSERT(w == img.cols);

    //// cv::Mat img_180;
    //// cv::rotate(img, img_180, cv::ROTATE_180);

    //std::vector<cv::Mat> channels;
    //cv::split(img, channels);

    //channels[0].convertTo(norm_r, CV_32F, 1.0 / 255);
    //channels[1].convertTo(norm_g, CV_32F, 1.0 / 255);
    //channels[2].convertTo(norm_b, CV_32F, 1.0 / 255);
    cv::Mat resize_img, norm_img;
    nao::vision::crnn_resize_img(img, resize_img, (320.f / 48.f), {3, 48, 320});
    nao::vision::normalize(resize_img, norm_img, {0.5f, 0.5f, 0.5f}, {1 / 0.5f, 1 / 0.5f, 1 / 0.5f}, true);
    nao::vision::permute_batch(norm_img,in_ptr);
}

json CrnnInference::post_process(cv::Mat img, const json& in_param, TrtBufferManager &buffers) {
    auto out_dims = m_engine->getBindingDimensions(1);
    float* out_ptr = static_cast<float*>(buffers.getHostBuffer(1));
    std::string alphabet = m_info["alphabet"];
    std::wstring walphabet = Utf8ToUnicode(alphabet);
    // LOG_INFO("alphabet size:{}", walphabet.size());

    cv::Mat out_mat(out_dims.d[1], out_dims.d[2], CV_32F, out_ptr, 0);
    std::wstring wtext;

    int idx = 0;
    int lastIdx = 0;
    for (int i=0; i < out_mat.rows; i++) {
        cv::Point maxloc;
        cv::minMaxLoc(out_mat.row(i), nullptr, nullptr, nullptr, &maxloc);
        lastIdx = idx;
        idx = maxloc.x;
        if (idx == 0 || (i > 0 && lastIdx == idx)) continue;
        wtext.push_back(walphabet[idx-1]);
    }

    std::string text = UnicodeToUtf8(wtext);
    json result = {{"text", text}, {"type", 1}};
    return result;
}

std::string CrnnInference::forward_pure(cv::Mat img, const json& in_param) {
    json result = TrtInference::forward(img, in_param);
    return result["text"];
}

json CrnnInference::forward(cv::Mat img, const json& in_param) {
    json all_out = json::array();

    int i = 0;
    // m_info["width"] = 1920;
    // m_info["height"] = 32;
    for (auto task: m_config["task"]) {
        i++;
        if (task["type"] == "static_text" || task["type"] == "dynamic_text") {
            LOG_INFO("Handle task: {}", Utf8ToAnsi(task.dump()));
            cv::Mat crop_img;
            json tfm_pts = m_ref->get_roi_img(img, crop_img, task["roi"], m_info["width"], m_info["height"]);
            //json tfm_pts = m_ref->get_roi_img(img, crop_img, task["roi"], m_info["width"], m_info["height"], TFM_REF);
            // cv::Mat crop_bgr, crop_gray;
            // cv::cvtColor(crop_img, crop_bgr, cv::COLOR_RGB2BGR);
            cv::cvtColor(crop_img, crop_img, cv::COLOR_RGB2GRAY);
            cv::cvtColor(crop_img, crop_img, cv::COLOR_GRAY2RGB);
            cv::Mat crop_scale_img = gray_scale_image(crop_img, 50, 220);

            write_debug_img("./gtmc_debug/ocr_task_"+std::to_string(i)+".jpg", crop_img);
            write_debug_img("./gtmc_debug/ocr_task_scale_"+std::to_string(i)+".jpg", crop_scale_img);

            json result = TrtInference::forward(crop_img, in_param);
            json result2 = TrtInference::forward(crop_scale_img, in_param);
            result["text2"] = result2["text"];
            LOG_INFO("[Result]: {}", Utf8ToAnsi(result.dump()));

            if (task["type"] == "static_text") {
                cv::Mat ref_img = m_ref->get_ref_img();
                cv::Mat ref_crop_img;
                m_ref->get_roi_img(ref_img, ref_crop_img, task["roi"], m_info["width"], m_info["height"], TFM_REF);
                // cv::Mat ref_crop_bgr;
                // cv::cvtColor(ref_crop_img, ref_crop_bgr, cv::COLOR_RGB2BGR);
                // cv::Mat ref_crop_scale = gray_scale_image(ref_crop_img, 60, 225);
                json ref_result = TrtInference::forward(ref_crop_img, in_param);
                // LOG_INFO("ref img result:{}", ref_result.dump());

#ifdef COLLECT_OCR_DATAx
                save_ocr_data(img, task, in_param, result["text"], ref_result["text"]);
#endif
                if (result["text"] == ref_result["text"]) {
                    continue;
                }
            } else if (task["type"] == "dynamic_text") {
#ifdef COLLECT_OCR_DATAx
                if (in_param.contains(task["name"])) {
                    std::string task_name = task["name"];
                    std::string gt_txt = in_param[task_name];
                    save_ocr_data(img, task, in_param, result["text"], gt_txt);
                }
#endif
            }

            std::string name = task["name"];
            json out = {
                {"label", name},
                {"shapeType", "polygon"},
                {"points", m_ref->transform_result(tfm_pts)},
                {"result", result}
            };
            all_out.push_back(out);
        }
    }
    return all_out;
}

void CrnnInference::save_ocr_data(cv::Mat img, const json& task, const json& in_param, const std::string& result_txt, const std::string& gt_txt)
{
    std::string img_path_str = in_param["img_path"];
    std::string task_name = task["name"];
    std::filesystem::path img_pth(img_path_str.c_str());
    std::string file_name = img_pth.filename().string();
    file_name = file_name.substr(0, file_name.size()-4);
    file_name = file_name + "_" + task_name + ".jpg";
    file_name = Utf8ToAnsi(file_name);
    cv::Mat org_crop_img, org_crop_bgr;
    m_ref->get_roi_img(img, org_crop_img, task["roi"], 0, 0);
    cv::cvtColor(org_crop_img, org_crop_bgr, cv::COLOR_RGB2BGR);

    if (task["type"] == "static_text") {
        if (result_txt == gt_txt) {
            if (static_ok_fs.is_open()) {
                static_ok_fs << file_name<< "\t" << Utf8ToAnsi(gt_txt) << std::endl;
            }
            cv::imwrite("D:/ocr_data/static/OK/" + file_name, org_crop_bgr);
        } else {
            if (static_ng_fs.is_open()) {
                static_ng_fs << file_name<< "\t" << Utf8ToAnsi(gt_txt) <<"\t"<< Utf8ToAnsi(result_txt) << std::endl;
            }
            cv::imwrite("D:/ocr_data/static/NG/" + file_name, org_crop_bgr);
        }
    } else if (task["type"] == "dynamic_text") {
        if (in_param.contains(task_name)) {
            if (result_txt == gt_txt) {
                if (dynamic_ok_fs.is_open()) {
                    dynamic_ok_fs << file_name<< "\t" << Utf8ToAnsi(gt_txt) << std::endl;
                }
                cv::imwrite("D:/ocr_data/dynamic/OK/" + file_name, org_crop_bgr);
            } else {
                if (dynamic_ng_fs.is_open()) {
                    dynamic_ng_fs << file_name<< "\t" << Utf8ToAnsi(gt_txt) <<"\t"<< Utf8ToAnsi(result_txt) << std::endl;
                    cv::imwrite("D:/ocr_data/dynamic/NG/" + file_name, org_crop_bgr);
                }
            }
        }
    }
}

json CrnnInference::get_info()
{
    return m_info;
}