#include <windows.h>
#include "../utils/Logger.h"
#include "../utils/Utils.h"
#include "../utils/StringConvert.h"
#include "NameplateOcr.h"
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeInterface.h>
#include <AIRuntimeUtils.h>
#include <codecvt>
#include <regex>
#include <map>


REGISTER_ALGO(NameplateOcr)

#define K_CATEGORY_HEAD  "k_category_head"
#define K_AMPS_HEAD      "k_amps_head"
#define K_VOLTS_HEAD     "k_volts_head"
#define K_SYSTEM_HEAD    "k_system_head"

#define K_CATEGORY      "catalog"
#define K_AMPS          "current"
#define K_VOLTS         "voltage"
#define K_SYSTEM        "system_class"

#define K_IP_LEVEL      "iplevel"
#define K_BARCODE       "product"


NameplateOcr::NameplateOcr(){}

NameplateOcr::~NameplateOcr(){}

void NameplateOcr::FilterDetResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params){}

void NameplateOcr::CheckKeyWords(OcrDataPtr ocrdata, std::map<std::string, OcrDataPtr>& keyItems, bool is_english)
{
    std::string text = ocrdata->text;
    std::regex ip_pattern(R"(^IP\d+)");
    std::regex ip_pattern_1(R"(\\*IP\d+)");
    std::regex ip_pattern_2(R"(^1P\d+)");
    std::regex code_pattern(R"(\d{6}.?\d{4}$)");   // 10位连续数字
    std::smatch ip_match, code_match,ip_match_1;
    std::regex sys_pattern(R"(3L.*PE)");
    std::smatch sys_match;
    std::string text_ansi = StringConvert::Utf8ToAnsi(text);
    LOGD("CheckKeyWords: {}  cx:{},  cy:{}", text_ansi, ocrdata->GetCenter().x, ocrdata->GetCenter().y);

    if (std::regex_search(text_ansi, ip_match, ip_pattern) || std::regex_search(text_ansi, ip_match_1, ip_pattern_1)|| std::regex_search(text_ansi, ip_match_1, ip_pattern_2)) {
        ocrdata->labelName = K_IP_LEVEL;
        if (ip_match[0].length()>0) {
            ocrdata->text = ip_match[0];
        }
        else {
            ocrdata->text = ip_match_1[0];
        }
        //首字符维1，替换为IP
        if(ocrdata->text[0] =='1'){
            ocrdata->text.replace(0,1,"I");
        }
        ocrdata->state = 2;
    }
    else if ((text_ansi.length() == 10 || text_ansi.length() == 11) && std::regex_search(text_ansi, code_match, code_pattern)) {
        ocrdata->labelName = K_BARCODE;
        ocrdata->text = code_match[0];
        ocrdata->state = 2;
    }

    if (is_english) {
        if (text_ansi.find("Catalog") != std::string::npos) {
            ocrdata->labelName = K_CATEGORY_HEAD;
        }
        else if (text_ansi.find("Volts") != std::string::npos) {
            ocrdata->labelName = K_VOLTS_HEAD;
        }
        else if (text_ansi.find("Amps") != std::string::npos) {
            ocrdata->labelName = K_AMPS_HEAD;
        }
        else if (ocrdata->GetCenter().y > 1400 && text_ansi.find("System") != std::string::npos || (std::regex_search(text_ansi, sys_match, sys_pattern)) ) {
            ocrdata->labelName = K_SYSTEM_HEAD;
        }
    }
    else {
        if (text_ansi.find("编号") != std::string::npos || text_ansi.find("产品号") != std::string::npos) {
            ocrdata->labelName = K_CATEGORY_HEAD;
        }
        else if (text_ansi.find("电压") != std::string::npos|| text_ansi.find("压") != std::string::npos) {
            ocrdata->labelName = K_VOLTS_HEAD;
        }
        else if (text_ansi.find("电流") != std::string::npos|| text_ansi.find("流") != std::string::npos) {
            ocrdata->labelName = K_AMPS_HEAD;
        }
        else if (text_ansi.find("系统") != std::string::npos ||(std::regex_search(text_ansi, sys_match, sys_pattern))) {
            ocrdata->labelName = K_SYSTEM_HEAD;
        }
    }

    if (ocrdata->labelName.length() > 0 && ocrdata->state == 0) {
        ocrdata->state = 1;
        keyItems.insert(std::make_pair(ocrdata->labelName, ocrdata));
    }
}

// 函数用于替换字符串中的 'c' 和 'C' 为 '0'
void replaceChars(std::string& str) {
    for (char& c : str) {
        if (c == 'c' || c == 'C') {
            c = '0';
        }
    }
}

void NameplateOcr::FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params)
{
    bool is_english = true;
    //判断中英文
    for(auto ocrdata : ocrdata_list){
        std::string text_ansi = StringConvert::Utf8ToAnsi(ocrdata->text);
        ocrdata->text = RemoveSpacesAndPunctuation(ocrdata->text);
        std::size_t startpos = text_ansi.find("电压");
        std::size_t endpos = text_ansi.find("电流");
        std::size_t endpos_1 = text_ansi.find("额定");
        std::size_t endpos_2 = text_ansi.find("产品");
        if (startpos != std::string::npos || endpos != std::string::npos|| endpos_1!= std::string::npos || endpos_2!= std::string::npos) {
            is_english = false;
            break;
        }
    }
    // 找关键字，例如：产品编号、额定电流、系统等
    std::map<std::string, OcrDataPtr> keyItems;
    for (auto ocrdata : ocrdata_list) {
        CheckKeyWords(ocrdata, keyItems, is_english);
    }

    // 找检测项
    std::regex letter_pattern(R"([A-Za-z0-9]+)");
    std::vector<OcrDataPtr> catItems;
    std::vector<OcrDataPtr> ampItems;

    for (auto ocrdata : ocrdata_list) {
        int strLen = ocrdata->text.length();
        std::string text_ansi = StringConvert::Utf8ToAnsi(ocrdata->text);
        LOGD("FilterRecResult: {}  cx:{},  cy:{}", text_ansi, ocrdata->GetCenter().x, ocrdata->GetCenter().y);

        // 产品编号值
        if (keyItems.find(K_CATEGORY_HEAD) != keyItems.end()) {
            OcrDataPtr kCat = keyItems[K_CATEGORY_HEAD];
            int distY = ocrdata->GetCenter().y - kCat->GetCenter().y;
            int distX = ocrdata->GetCenter().x - kCat->GetCenter().x;
            if (ocrdata->state == 2 && std::abs(distY)>10 && std::abs(distX) > 10)continue;
            if (strLen > 0 && distX >=-500 && distX < 1000 && distY > -100 && distY < 150) {
                std::regex pattern(R"([A-Za-z0-9]+)");
                std::smatch match;
                if (std::regex_search(text_ansi, match, pattern)) {
                    catItems.push_back(ocrdata);
                    continue;
                }
            }
        }


        // 电流值
        if (keyItems.find(K_AMPS_HEAD) != keyItems.end()) {
            OcrDataPtr kAmps = keyItems[K_AMPS_HEAD];
            int distY = ocrdata->GetCenter().y - kAmps->GetCenter().y;
            int distX = ocrdata->GetCenter().x - kAmps->GetCenter().x;
            if (ocrdata->state == 2 && std::abs(distY) > 10 && std::abs(distX) > 10)continue;
            if (strLen > 0 /*&& strLen <= 30*/ && distX > -200 && distX < 500 && distY > -100 && distY < 100) {
                 //将c替换为0
                replaceChars(text_ansi);
                std::regex pattern(R"(\d+)");
                std::smatch match;
                if (std::regex_search(text_ansi, match, pattern)) {
                    if (match[0].length() == 4 || match[0].length() == 3) {
                        ocrdata->labelName = K_AMPS;
                        ocrdata->text = match[0];
                        ocrdata->state = 2;
                    }
                }
            }
        }

        // 电压值
        if (keyItems.find(K_VOLTS_HEAD) != keyItems.end()) {
            OcrDataPtr kVolts = keyItems[K_VOLTS_HEAD];
            int distY = ocrdata->GetCenter().y - kVolts->GetCenter().y;
            int distX = ocrdata->GetCenter().x - kVolts->GetCenter().x;
            if (ocrdata->state == 2 && std::abs(distY) > 10 && std::abs(distX) > 10)continue;
            //判断有几个数字,电流 电压是否在同一个检测区域
            std::regex pattern(R"(\d+)");
            std::smatch match;
            std::string::const_iterator citer = text_ansi.cbegin();
            int mat_index = 0;
            while (std::regex_search(citer, text_ansi.cend(), match, pattern))//循环匹配
            {
                citer = match[0].second;
                mat_index++;
            }
            if (strLen > 0 /*&& strLen <= 30*/ && distX > -200 && distX < 500 && distY > -100 && distY < 100) {
                if (std::regex_search(text_ansi, match, pattern)) {
                    if (match[0].length()==4) {
                        ocrdata->labelName = K_VOLTS;
                        ocrdata->text = match[0];
                        ocrdata->state = 2;
                    }
                }
            }
            //电压电流在同一行，循环匹配
            if (strLen > 20 && distY < 20 && distX < 20 && distY > -20 && distX > -20 && mat_index==2) {
                citer = text_ansi.cbegin();
                mat_index = 0;
                while (std::regex_search(citer, text_ansi.cend(), match, pattern))//循环匹配
                {
                    if (mat_index ==0) {
                        OcrDataPtr amp_data = std::make_shared<OcrData>();
                        amp_data->labelName = K_AMPS;
                        amp_data->text = match[0];
                        amp_data->state = 2;
                        amp_data->conf = 0.9;
                        ampItems.push_back(amp_data);
                    }
                    citer = match[0].second;
                    if (mat_index ==1) {
                        ocrdata->labelName = K_VOLTS;
                        ocrdata->text = match[0];
                        ocrdata->state = 2;
                    }
                    mat_index++;
                }
            }
        }

        // 系统类别
        if (keyItems.find(K_SYSTEM_HEAD) != keyItems.end()) {
            OcrDataPtr kSystem = keyItems[K_SYSTEM_HEAD];
            int distY = ocrdata->GetCenter().y - kSystem->GetCenter().y;
            int distX = ocrdata->GetCenter().x - kSystem->GetCenter().x;
            if (ocrdata->state == 2 && std::abs(distY) > 10 && std::abs(distX) > 10)continue;
            if (strLen > 0 /*&& strLen < 8*/ && distX >=0 && distX < 500 && distY > -100 && distY < 100) {
                std::regex pattern(R"(3L.*PE)");
                std::smatch match;
                if (std::regex_search(text_ansi, match, pattern)) {
                    ocrdata->labelName = K_SYSTEM;
                    ocrdata->text = match[0];
                    ocrdata->state = 2;
                }
            }
        }
    }

    // 产品编号 多行判断合并
    if (catItems.size() == 2) {
        is_zero(zero_img_,catItems[0]);
        is_zero(zero_img_,catItems[1]);
        OcrDataPtr cat_row1 = catItems[0];
        std::string cat_text = catItems[0]->text + catItems[1]->text;;
        if (catItems[1]->text.length() > catItems[0]->text.length()) {
            cat_row1 = catItems[1];
            cat_text = catItems[1]->text + catItems[0]->text;
        }
        std::string text_ansi = StringConvert::Utf8ToAnsi(cat_text);
        std::string result;
        std::copy_if(text_ansi.begin(), text_ansi.end(), std::back_inserter(result), [](char c) {return std::isalnum(c); });
        std::regex pattern(R"([A-Za-z0-9]+)");
        std::smatch match;
        if (std::regex_search(result, match, pattern)) {
            if (is_english) {
                std::string temp = RemoveSpacesAndPunctuation(result);
                std::string::size_type ch = temp.find_first_not_of("Catalog");
                if(ch>7)ch=7;
                catItems[0]->text = temp.substr(ch);
            }
            else {
                catItems[0]->text = match[0];
            }
        }
        catItems[0]->labelName = K_CATEGORY;
        catItems[0]->state = 2;
        catItems[1]->state = 0;
    }
    else if (catItems.size() == 1) {
        is_zero(zero_img_,catItems[0]);
        std::string text_ansi = StringConvert::Utf8ToAnsi(catItems[0]->text);
        std::string result;
        std::copy_if(text_ansi.begin(), text_ansi.end(), std::back_inserter(result), [](char c) {return std::isalnum(c);});
        std::regex pattern(R"([A-Za-z0-9]+)");
        std::smatch match;
        if (std::regex_search(result, match, pattern)) {
            if (is_english) {
                //去除冒号
                std::string temp = RemoveSpacesAndPunctuation(result);
                std::string::size_type ch = temp.find_first_not_of("Catalog");
                if(ch>7)ch=7;
                catItems[0]->text = temp.substr(ch);
            }
            else {
                catItems[0]->text=match[0];
            }
        }
        catItems[0]->labelName = K_CATEGORY;
        catItems[0]->state = 2;
    }

    if (ampItems.size()>0) {
        ocrdata_list.push_back(ampItems[0]);
    }
    // 删除其他字符
    auto it = ocrdata_list.begin();
    while (it != ocrdata_list.end()) {
        OcrDataPtr ocrdata = *it;
        if (ocrdata->state != 2) {
            std::string text_ansi = StringConvert::Utf8ToAnsi(ocrdata->text);
            LOGD("NameplateOcr skip result: {}, len:{}, score:{}", text_ansi, ocrdata->text.length(), ocrdata->conf);
            it = ocrdata_list.erase(it);
        }
        else {
            ++it;
        }
    }
    if (is_english) {
        OcrDataPtr ocr_data = std::make_shared<OcrData>();
        ocr_data->labelName = "is_english";
        ocr_data->text = "";
        ocr_data->conf = 1;
        ocrdata_list.push_back(ocr_data);
        LOGD("NameplateOcr is_english result: {}, ", ocr_data->conf);
    }
    else {
        OcrDataPtr ocr_data = std::make_shared<OcrData>();
        ocr_data->labelName = "is_english";
        ocr_data->text = "";
        ocr_data->conf = 0;
        ocrdata_list.push_back(ocr_data);
        LOGD("NameplateOcr is_english result: {}, ", ocr_data->conf);
    }
}

AlgoResultPtr NameplateOcr::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("NameplateOcr start run!");
    json params = GetTaskParams(task);

    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    int width = task->image.cols;
    int height = task->image.rows;
    int offset_x = (width - height) / 2;
    cv::Mat centCrop = Utils::CenterCrop(task->image, height, height);

    std::vector<OcrDataPtr> ocrdata_list;
    json model_info;
    TextDet(centCrop, ocrdata_list, model_info);
    if (model_info["modele_path"] == 0) {
        algo_result->status = ErrorCode::WRONG_PARAM;
        return algo_result;
    }

    FilterDetResult(ocrdata_list);
    std::vector<cv::Mat> watched_imgs;
    for (auto ocrdata : ocrdata_list) {
        cv::Mat itemImg = ocrdata->GetCropImage(centCrop);
        watched_imgs.push_back(itemImg);
    }
    zero_img_ = centCrop.clone();
    TextRec(centCrop, ocrdata_list, model_info);
    if (model_info["modele_path"] == 0) {
        algo_result->status = ErrorCode::WRONG_PARAM;
        return algo_result;
    }
    FilterRecResult(ocrdata_list, task->image_info);

    algo_result->result_info = json::array();
    for (auto ocrdata : ocrdata_list) {
        json item_rst = ocrdata->ToJsonResult(offset_x, 0);
        algo_result->result_info.push_back(item_rst);
    }
    align_results(algo_result);


    cv::Mat test_img = task->image.clone();
    shape_det(test_img,algo_result);
    LOGI("NameplateOcr algo end run file {}, line {} info{}", __FILE__, __LINE__, print_rec_str(algo_result).c_str());
    LOGI("NameplateOcr run finished! ");
    return algo_result;
}

std::string NameplateOcr::print_rec_str(AlgoResultPtr algo_result) {
    char*  ret = StringConvert::Utf8ToAnsi(algo_result->result_info.dump().c_str());
    std::cout << ret << std::endl;
    return std::string(ret);
}

#include <cctype>
#include <codecvt>
#include <locale>

bool is_chinese_punctuation(wchar_t c) {
    // 中文标点符号的Unicode范围大致为：3000-303F，FF00-FFEF
    return (c >= 0x3000 && c <= 0x303F) || (c >= 0xFF00 && c <= 0xFFEF);
}

std::string remove_chinese_punctuation(const std::string& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wide_string = converter.from_bytes(input);

    std::wstring filtered;
    for (wchar_t c : wide_string) {
        if (!is_chinese_punctuation(c)) {
            filtered += c;
        }
    }

    return converter.to_bytes(filtered);
}

std::string NameplateOcr::RemoveSpacesAndPunctuation(const std::string& str)
{
    std::string result;
    std::copy_if(str.begin(), str.end(), std::back_inserter(result), [](char c) {
        return !std::isspace(c) && !(std::ispunct(c) && c != '+' && c != '-');
        });
    return result;
}

void NameplateOcr::align_results(AlgoResultPtr algo_result) {
    bool product_flag = false;
    bool current_flag = false;
    bool voltage_flag = false;
    bool iplevel_flag = false;
    bool catalog_flag = false;
    bool system_class_flag = false;
    for (auto& item : algo_result->result_info) {
        //LOGI("NameplateOcr align_results label {}, txt {} ", item["label"],item["result"]["str"],);
        if (item["label"] == "product") {
            product_flag = true;
        }
        if (item["label"] == "current") {
            current_flag = true;
        }
        if (item["label"] == "voltage") {
            voltage_flag = true;
        }
        if (item["label"] == "iplevel") {
            iplevel_flag = true;
        }
        if (item["label"] == "catalog") {
            catalog_flag = true;
        }
        if (item["label"] == "system_class") {
            system_class_flag = true;
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
    if (!current_flag) {
        algo_result->result_info.push_back(
            {
                {"label", "current"},
                {"shapeType", "rectangle"},
                {"points", {{0, 0}, {0, 0}}},
                {"result", {{"confidence", 0}, {"str", ""}}},
            }
        );
    }

    if (!voltage_flag) {
        algo_result->result_info.push_back(
            {
                {"label", "voltage"},
                {"shapeType", "rectangle"},
                {"points", {{0, 0}, {0, 0}}},
                {"result", {{"confidence", 0}, {"str", ""}}},
            }
        );
    }
    if (!iplevel_flag) {
        algo_result->result_info.push_back(
            {
                {"label", "iplevel"},
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
    if (!system_class_flag) {
        algo_result->result_info.push_back(
            {
                {"label", "system_class"},
                {"shapeType", "rectangle"},
                {"points", {{0, 0}, {0, 0}}},
                {"result", {{"confidence", 0}, {"str", ""}}},
            }
        );
    }
}

 void NameplateOcr::shape_det(cv::Mat img,AlgoResultPtr algo_result){

    AIRuntimeInterface* ai_obj = GetAIRuntime();
    TaskInfoPtr _task = std::make_shared<stTaskInfo>();
    _task->imageData = { img };
    _task->modelId = 3;
    _task->taskId = 0;
    _task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(_task);

    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(_task->promiseResult);
    std::future<ModelResultPtr>   futureRst = promiseResult->get_future();
    ModelResultPtr rst = futureRst.get();

    json ccc_data = json::array();
    json asta_data = json::array();
    json ce_data = json::array();
    json kema_data = json::array();
    json ul_data = json::array();

    for (int i = 0; i < rst->itemList.size(); i++) {
        for (auto& box : rst->itemList[i]) {
            if (box.points.empty() || box.confidence == 0) continue;
#ifdef SAVE_DEBUG
            std::cout << box.confidence << "  box.code = " << box.code << "  box.shape = " << box.shape << std::endl;

            std::cout << box.points[0].x << " " << box.points[0].y << " " << box.points[1].x << " " << box.points[1].y << std::endl;
            cv::Scalar color(0, 255, 0);
            cv::rectangle(drawImg, cv::Point(box.points[0].x, box.points[0].y), cv::Point(box.points[1].x, box.points[1].y), color, 3);
#endif //draw save debug img
            json s2;
            s2["label"] = box.code;
            s2["points"] = { {box.points[0].x, box.points[0].y}, {box.points[1].x, box.points[1].y} };
            s2["shapeType"] = "rectangle";
            s2["result"] = { {"confidence", box.confidence}, {"area", (box.points[1].x - box.points[0].x) * (box.points[1].y - box.points[0].y)} };
            switch (box.code)
            {
                case CCC:
                    ccc_data.push_back(s2);
                    break;
                case ASTA:
                    asta_data.push_back(s2);
                    break;
                case CE:
                    ce_data.push_back(s2);
                    break;
                case KEMA:
                    kema_data.push_back(s2);
                    break;
                case UL:
                    ul_data.push_back(s2);
                    break;
                default:
                    break;
            }

        }
    }
    if(ccc_data.empty()){
        json s2;
        s2["label"] = "3c";
        s2["points"] = {{0, 0}, {0, 0}};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    else{
        auto it = ccc_data.begin();
        (*it)["label"] = "3c";
        algo_result->result_info.emplace_back((*it));
    }


    if(asta_data.empty()){
        json s2;
        s2["label"] = "asta";
        s2["points"] = {{0, 0}, {0, 0}};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    else{
        auto it = asta_data.begin();
        (*it)["label"] = "asta";
        algo_result->result_info.emplace_back((*it));
    }


    if(ce_data.empty()){
        json s2;
        s2["label"] = "ce";
        s2["points"] = {{0, 0}, {0, 0}};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    else{
        auto it = ce_data.begin();
        (*it)["label"] = "ce";
        algo_result->result_info.emplace_back((*it));
    }




    if(kema_data.empty()){
        json s2;
        s2["label"] = "kema";
        s2["points"] = {{0, 0}, {0, 0}};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    else{
        auto it = kema_data.begin();
        (*it)["label"] = "kema";
        algo_result->result_info.emplace_back((*it));
    }

    if(ul_data.empty()){
        json s2;
        s2["label"] = "ul";
        s2["points"] = {{0, 0}, {0, 0}};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    else{
        auto it = ul_data.begin();
        (*it)["label"] = "ul";
        algo_result->result_info.emplace_back((*it));
    }
 }

 int NameplateOcr::sauvola(const cv::Mat& src, cv::Mat& dst, const double& k, const int& wnd_size)
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

bool is_chinese_char(wchar_t c) {
    // 中文字符的Unicode范围大致为：4E00-9FFF，3400-4DBF，20000-2A6DF，2A700-2B73F，2B740-2B81F，2B820-2CEAF
    // 这里只列出了基本的中文字符范围，可以根据需要添加更多
    return (c >= 0x4E00 && c <= 0x9FFF) ||
           (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0x20000 && c <= 0x2A6DF) ||
           (c >= 0x2A700 && c <= 0x2B73F) ||
           (c >= 0x2B740 && c <= 0x2B81F) ||
           (c >= 0x2B820 && c <= 0x2CEAF);
}

int count_chinese_chars(const std::string& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wide_string = converter.from_bytes(input);

    int count = 0;
    for (wchar_t c : wide_string) {
        if (is_chinese_char(c)) {
            ++count;
        }
    }

    return count;
}

void NameplateOcr::is_zero(cv::Mat img,OcrDataPtr ocrdata){

        static int item_index =0;
        static int index =0;
        cv::Mat itemImg = ocrdata->GetCropImage(img);
        cv::Mat itemImg_gray,dst,th_img;
        cv::cvtColor(itemImg, itemImg_gray, cv::COLOR_BGR2GRAY);
        //去除冒号
        ocrdata->text = RemoveSpacesAndPunctuation(ocrdata->text);
        
        static int origin_index = 0;
        SaveDebugImage(itemImg_gray, "zero_origin_"+std::to_string(origin_index)+".jpg");
        origin_index++;
        sauvola(itemImg_gray, dst, 0.05, 15);
        //切分字符
        dst = ~dst;
        cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(dst, th_img, elementX);
        cv::erode(th_img, th_img, elementX);

        SaveDebugImage(th_img, "item_"+std::to_string(item_index)+".jpg");
        item_index++;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> filter_hierarchy;
        cv::findContours(th_img, contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::Mat gray_mask = cv::Mat::zeros(dst.size(), CV_8UC1);

        std::vector<std::pair<cv::Rect,cv::Mat>> img_vec;
        for (size_t j = 0; j < contours.size(); ++j) {
            cv::Rect rect = cv::boundingRect(contours[j]);
            double area = cv::contourArea(contours[j]);
                if (rect.height< th_img.rows*0.25 || (rect.width /rect.height)>1.6) {
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
        //中文字符的个数
        int text_size = count_chinese_chars(ocrdata->text);
        int text_lenght = ocrdata->text.length() - text_size*2;
        if(img_vec.size() ==text_lenght){
            //判断0 O 位置
            //第一行
            for (int j=0; j<ocrdata->text.length(); j++) {
                if(ocrdata->text[j] == '0' || ocrdata->text[j] == 'O'){
                    int k=j;
                    if(text_size>0){
                        k=j-text_size*2;
                    }
                    SaveDebugImage(img_vec[k].second, "zero_"+std::to_string(index)+".jpg");
                    index++;
                }
            }
        }
        else{
            LOGI("LabelOcr is_zero end run file {}, line {} info {}", __FILE__, __LINE__,"字符数量与图片切割不一致");
            return;
        }
}