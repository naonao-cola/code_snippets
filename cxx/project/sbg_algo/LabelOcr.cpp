#include <windows.h>
#include "../utils/Logger.h"
/**
 * @FilePath     : /code_snippets/cxx/project/sbg_algo/LabelOcr.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-01-06 14:22:12
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-01-06 14:22:12
 * @Copyright (c) 2025 by G, All Rights Reserved.
**/
#include "../utils/Utils.h"
#include "LabelOcr.h"
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeInterface.h>
#include <AIRuntimeUtils.h>

REGISTER_ALGO(LabelOcr)

LabelOcr::LabelOcr()
{

}

LabelOcr::~LabelOcr()
{

}

void LabelOcr::FilterDetResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params)
{
    auto it = ocrdata_list.begin();
    while (it != ocrdata_list.end()) {
        OcrDataPtr ocrdata = *it;
        if (ocrdata->GetAspectRatio() < 5) {
            it = ocrdata_list.erase(it);
        } else {
            ++it;
        }
    }
}

void LabelOcr::FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params)
{
    auto it = ocrdata_list.begin();

    while (it != ocrdata_list.end()) {
        OcrDataPtr ocrdata = *it;
        std::set<char> firstCharSet = { 'A', 'B', 'C' };
        std::set<char> productCharset = { '*' };
        LOGD("LabelOcr current det result: {}, len:{}, score:{}", ocrdata->text, ocrdata->text.length(), ocrdata->conf);

         //中文判断
        if(IncludeChinese(const_cast<char*>(ocrdata->text.c_str()))){
            LOGD("LabelOcr skip result: {}, len:{}, score:{}", ocrdata->text, ocrdata->text.length(), ocrdata->conf);
            it = ocrdata_list.erase(it);
            continue;
        }

        if (productCharset.find(ocrdata->text[0]) != productCharset.end() && ocrdata->text.length() > 10) {
            // two *
            ocrdata->labelName = "product";
            std::string tmp = ocrdata->text;
            //delete *
            std::size_t startpos = tmp.find_first_not_of("*");
            std::size_t endpos = tmp.find_last_not_of("*");
            if (startpos == std::string::npos) {
                LOGD("LabelOcr skip result: {}, len:{}, score:{}", ocrdata->text, ocrdata->text.length(), ocrdata->conf);
                it = ocrdata_list.erase(it);
            }
            else {
                ocrdata->text = tmp.erase(endpos + 1).erase(0, startpos);
                ++it;
            }
            continue;
        }
        else if (ocrdata->text.length() ==11 && ocrdata->text[6] =='-') {
            // not have *
            bool is_num = true;
            for (int m = 0; m < ocrdata->text.length();m++) {
                if (m == 6) {
                    continue;
                }
                is_num = std::isalnum(ocrdata->text[m]);
                if (!is_num) {
                    break;
                }
            }
            if (!is_num) {
                it = ocrdata_list.erase(it);
            }
            else {
                ocrdata->labelName = "product";
                ++it;
            }
            continue;
        }
        else if (ocrdata->text.length() == 12 && (productCharset.find(ocrdata->text[0]) != productCharset.end() || productCharset.find(ocrdata->text[ocrdata->text.length()-1]) != productCharset.end())) {
            ocrdata->labelName = "product";
            // only one *
            std::string tmp = ocrdata->text;
            //delete *
            std::size_t startpos = tmp.find_first_not_of("*");
            std::size_t endpos = tmp.find_last_not_of("*");
            if (startpos == std::string::npos) {
                LOGD("LabelOcr skip result: {}, len:{}, score:{}", ocrdata->text, ocrdata->text.length(), ocrdata->conf);
                it = ocrdata_list.erase(it);
            }
            else {
                ocrdata->text = tmp.erase(endpos + 1).erase(0, startpos);
                ++it;
            }
            continue;
        }
        if (ocrdata->text.length() < 12 ||
            firstCharSet.find(ocrdata->text[0]) == firstCharSet.end()) {
            //LOGD("LabelOcr skip result: {}, len:{}, score:{}", StringConvert::Utf8ToAnsi(ocrdata->text), ocrdata->text.length(), ocrdata->conf);
            LOGD("LabelOcr skip result: {}, len:{}, score:{}", ocrdata->text, ocrdata->text.length(), ocrdata->conf);

            it = ocrdata_list.erase(it);
        }
        else {
            ocrdata->labelName = "catalog";
            ++it;
        }
    }
}

AlgoResultPtr LabelOcr::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("LabelOcr start run!");
    json params = GetTaskParams(task);
    bool is_debug = Utils::GetProperty(params, "debug", 0);

    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();

    std::vector<OcrDataPtr> ocrdata_list;
    json model_info;
    TextDet(task->image, ocrdata_list, model_info);
    if (model_info["modele_path"] ==0) {
        algo_result->status = ErrorCode::WRONG_PARAM;
        return algo_result;
    }
    FilterDetResult(ocrdata_list);
    TextRec(task->image, ocrdata_list, model_info);
    if (model_info["modele_path"] == 0) {
        algo_result->status = ErrorCode::WRONG_PARAM;
        return algo_result;
    }
    //
    std::vector<OcrData> watch_var;
    for (auto item : ocrdata_list) {
        OcrData sub_item = *item;
        watch_var.push_back(sub_item);
    }
    FilterRecResult(ocrdata_list);
    is_zero(task->image,ocrdata_list);

    algo_result->result_info = json::array();
    for (auto ocrdata : ocrdata_list) {
        algo_result->result_info.emplace_back(ocrdata->ToJsonResult());
    }
    align_results(algo_result);
    LOGI("LabelOcr algo end run file {}, line {} info{}", __FILE__, __LINE__, Utils::DumpJson(algo_result->result_info, false));
    LOGI("LabelOcr run finished! {}", algo_result->result_info.dump(2));
    return algo_result;
}

void LabelOcr::align_results(AlgoResultPtr algo_result) {

    bool product_flag = false;
    bool catalog_flag = false;
    for (auto& item : algo_result->result_info) {
        if (item["label"] == "product") {
            product_flag = true;
        }
        if (item["label"] == "catalog") {
            catalog_flag = true;
        }
    }
    if (!product_flag) {
        algo_result->result_info.push_back(
            {
                {"label", "product"},
                {"shapeType", "rectangle"},
                {"points", {{0, 0}, {0, 0}}},
                {"result", {{"confidence", 0}, {"str", ""}}},
            }
        );
    }
    if (!catalog_flag) {
        algo_result->result_info.push_back(
            {
                {"label", "catalog"},
                {"shapeType", "rectangle"},
                {"points", {{0, 0}, {0, 0}}},
                {"result", {{"confidence", 0}, {"str", ""}}},
            }
        );
    }
}

int LabelOcr::IncludeChinese(char *str){

    char c;
    while(1)
    {
        c=*str++;
        if (c==0) break;  //如果到字符串尾则说明该字符串没有中文字符
        if (c&0x80) {       //如果字符高位为1且下一字符高位也是1则有中文字符
            if (*str & 0x80) {
                return 1;
                }
        }
    }
    return 0;
}

int sauvola(const cv::Mat& src, cv::Mat& dst, const double& k, const int& wnd_size)
{
    cv::Mat local;
    src.convertTo(local, CV_32F);
    // 图像的平方
    cv::Mat square_local;
    square_local = local.mul(local);
    // 图像局部均值
    cv::Mat mean_local;
    // 局部均值的平方
    cv::Mat square_mean;
    // 图像局部方差
    cv::Mat var_local;
    // 局部标准差
    cv::Mat std_local;
    // 阈值图像
    cv::Mat th_img;
    // 目标图像的32F
    cv::Mat dst_f;
    // 获取局部均值
    cv::blur(local, mean_local, cv::Size(wnd_size, wnd_size));
    // 计算局部方差
    cv::blur(square_local, var_local, cv::Size(wnd_size, wnd_size));
    // 局部均值的平方
    square_mean = mean_local.mul(mean_local);
    // 标准差
    cv::sqrt(var_local - square_mean, std_local);
    th_img = mean_local.mul((1 + k * (std_local / 128 - 1)));
    cv::compare(local, th_img, dst_f, cv::CMP_GE);
    dst_f.convertTo(dst, CV_8U);
    return 0;
}

void LabelOcr::is_zero(cv::Mat img,std::vector<OcrDataPtr>& ocrdata_list){
    static int index =0;
    for(int i = 0;i<ocrdata_list.size();i++){
        OcrDataPtr ocrdata = ocrdata_list[i];
        if(ocrdata->labelName == "catalog"){
            cv::Mat itemImg = ocrdata->GetCropImage(img);
            cv::Mat itemImg_gray,dst,th_img;
            cv::cvtColor(itemImg, itemImg_gray, cv::COLOR_BGR2GRAY);

            //SaveDebugImage(itemImg_gray, "zero_origin_"+std::to_string(index)+".jpg");
            index++;
            sauvola(itemImg_gray, dst, 0.05, 15);
            //切分字符
            dst = ~dst;
            cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::dilate(dst, th_img, elementX);
            cv::erode(th_img, th_img, elementX);

            //SaveDebugImage(th_img, "zero_1_"+std::to_string(index)+".jpg");
            index++;
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> filter_hierarchy;
            cv::findContours(th_img, contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            cv::Mat gray_mask = cv::Mat::zeros(dst.size(), CV_8UC1);

            std::vector<std::pair<cv::Rect,cv::Mat>> img_vec;
            for (size_t j = 0; j < contours.size(); ++j) {
                cv::Rect rect = cv::boundingRect(contours[j]);
                double area = cv::contourArea(contours[j]);
                 if (rect.height< th_img.rows*0.29 || (rect.width /rect.height)>1.6) {
                    continue;
                }
                cv::RotatedRect r_rect = cv::minAreaRect(contours[j]);
                std::vector<std::vector<cv::Point>> draw_conts = { contours[j] };
                int width = (std::max)(r_rect.size.width, r_rect.size.height);
                int height = (std::min)(r_rect.size.width, r_rect.size.height);
                double rate = width / (height * 1.0);
                // cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
                cv::Mat tmp=th_img(rect).clone();
                img_vec.push_back(std::make_pair(rect,tmp));
            }
            std::sort(img_vec.begin(),img_vec.end(),[&](std::pair<cv::Rect,cv::Mat> a,std::pair<cv::Rect,cv::Mat>b){ return a.first.x <b.first.x;});

            if(img_vec.size() !=ocrdata->text.length() ){
                LOGI("LabelOcr is_zero end run file {}, line {} info {}", __FILE__, __LINE__,"字符数量与图片切割不一致");
                return;
            }
            //判断0 O 位置
            for (int j=0; j<img_vec.size(); j++) {
                if(ocrdata->text[j] == '0' || ocrdata->text[j] == 'O'){
                    SaveDebugImage(img_vec[j].second, "zero_"+std::to_string(index)+".jpg");
                    index++;
                }

            }

        }
    }
}