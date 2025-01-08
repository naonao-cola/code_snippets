// #include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/Logger.h"
#include "sbgHl.h"
#include <typeinfo>
#include <filesystem>
#include <iostream>
#include "../utils/Utils.h"
//#include <experimental/filesystem>

namespace fs = std::filesystem;
#if USE_AI_DETECT
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeInterface.h>
#include <AIRuntimeUtils.h>
#endif // USE_AI_DETECT

REGISTER_ALGO(sbgHl)

sbgHl::sbgHl()
{

}

sbgHl::~sbgHl()
{

}

AlgoResultPtr sbgHl::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("sbgHl start run!");

    cv::Mat drawImg, drawImg2;
    json tbdata = json::array();
    json ibjdata = json::array();
    json marlydata = json::array();
    json wgsdata = json::array();
    json wbsdata = json::array();
    json screw_headdata = json::array();
    json leak_holedata = json::array();
    json bsdata = json::array();
    json gwbdata = json::array();
    json BOXdata = json::array();
    json ALLdata = json::array();

    (task->image).copyTo(drawImg);
    (task->image).copyTo(drawImg2);

    ////////////////////////////////////////////////////
#ifdef SAVE_DEBUG
write_debug_img(task, "origin_sbg_hl_object", drawImg);
#endif
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    AIRuntimeInterface* ai_obj = GetAIRuntime();
    TaskInfoPtr _task = std::make_shared<stTaskInfo>();
    _task->imageData  = { task->image };
    _task->modelId    = 4;
    _task->taskId     = 0;
    stAIModelInfo::mPtr model_info = GetAIRuntime()->GetModelInfo(_task->modelId);
    std::filesystem::path model_path(model_info->modelPath);
    if (!std::filesystem::exists(model_path)) {

        algo_result->status = ErrorCode::WRONG_PARAM;
        return algo_result;
    }

    _task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(_task);


    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(_task->promiseResult);
    std::future<ModelResultPtr>   futureRst  = promiseResult->get_future();

    ModelResultPtr rst         = futureRst.get();
    for (int i = 0; i < rst->itemList.size(); i++) {
           for (auto& box : rst->itemList[i]) {
               if (box.points.empty() || box.confidence == 0) continue;

#ifdef SAVE_DEBUG
               std::cout <<box.confidence << "  box.code = " << box.code << "  box.shape = " << box.shape << std::endl;

               std::cout << box.points[0].x << " " << box.points[0].y << " " << box.points[1].x << " "<<box.points[1].y << std::endl;
               cv::Scalar color(0, 255, 0);
               cv::rectangle(drawImg, cv::Point(box.points[0].x, box.points[0].y), cv::Point(box.points[1].x, box.points[1].y), color, 3);
#endif //draw save debug img
                json s2;
                s2["label"] = box.code;
                s2["points"] = {{box.points[0].x, box.points[0].y}, {box.points[1].x, box.points[1].y}};
                s2["shapeType"] = "rectangle";
                s2["result"] = {{"confidence", box.confidence}, {"area", (box.points[1].x - box.points[0].x)*(box.points[1].y - box.points[0].y)}};
                // algo_result->result_info.emplace_back(s2);
                ALLdata.emplace_back(s2);
                switch (box.code)
                {
                 case tb:
                    tbdata.emplace_back(s2);
                    break;
                case ibj:
                    ibjdata.emplace_back(s2);
                    break;
                case marly:
                    marlydata.emplace_back(s2);
                    break;
                case wa_gray_s:
                    wgsdata.emplace_back(s2);
                    break;
                case wa_black_s:
                    wbsdata.emplace_back(s2);
                    break;
                case screw_head:
                    screw_headdata.emplace_back(s2);
                    break;
                case leak_hole:
                    leak_holedata.emplace_back(s2);
                    break;
                case bs:
                    bsdata.emplace_back(s2);
                    break;
                case gwb:
                    gwbdata.emplace_back(s2);
                    break;
                default:
                    break;
                }
           }
    }
#ifdef SAVE_DEBUG
write_debug_img(task, "all_obj", drawImg);
#endif
    tbfilter(task, tbdata, algo_result, BOXdata, drawImg2);
    ibgfilter(task, ibjdata, algo_result, BOXdata, drawImg2);
    marlyfilter(task, marlydata, algo_result, BOXdata, drawImg2);
    wgsfilter(task, wgsdata, algo_result, BOXdata, drawImg2);
    wbsfilter(task, wbsdata, algo_result, BOXdata, drawImg2);
    screw_headfilter(task, screw_headdata, algo_result, BOXdata, drawImg2);
    leak_holefilter(task, leak_holedata, algo_result, BOXdata, drawImg2);
    bsfilter(task, bsdata, algo_result, BOXdata, drawImg2);
    gwbfilter(task, gwbdata, algo_result, BOXdata, drawImg2);
#ifdef SAVE_DEBUG
write_debug_img(task, "filter_obj", drawImg2);
#endif

    LOGI("sbgHl algo end run file {}, line {} info{}", __FILE__, __LINE__,Utils::DumpJson(algo_result->result_info,false));
    LOGI("sbgHl run finished!");

    return algo_result;
}

//条巴
void sbgHl::tbfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "tb";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
        return ;
    }

    int count =0;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];
        (*it)["label"]= "tb";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        count++;
        if(count>=4){
            //条巴大于4个中断
            break;
        }
    }
    return;
}

//绝缘黑胶   返回 ibg
void sbgHl::ibgfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "ibg";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
         return ;
    }


    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];
        (*it)["label"]= "ibg";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
    }
    return;
}

//marly 片
void sbgHl::marlyfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "marly";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
         return ;
    }
    int count=0;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];
        (*it)["label"]= "marly";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        count++;
        if(count>=1){
            break;
        }
    }

    return;
}

//灰色防水胶 gwb
void sbgHl::wgsfilter(InferTaskPtr task, json& indata,
                            AlgoResultPtr algo_result, json& BOXdata,
                            cv::Mat& img) {

    if (indata.empty()) {
        json s2;
        s2["label"] = "wa_gray_s";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }
    int count=0;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];
        (*it)["label"] = "wa_gray_s";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        count++;
        if(count>=1){
            break;
        }
    }

    return;
}

// 黑色防水胶 wa_black_s
void sbgHl::wbsfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "wa_black_s";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
         return ;
    }

    int count=0;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        (*it)["label"]= "wa_black_s";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        count++;
        if(count>=2){
            break;
        }

    }
    return;
}

// 接头螺丝标志 screw_head
void sbgHl::screw_headfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "screw_head";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
        return ;
    }

    int count=0;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        (*it)["label"]= "screw_head";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        count++;
        if(count>=2){
            break;
        }
    }
    return;
}
  //漏水孔  标志leak_hole
void sbgHl::leak_holefilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "leak_hole";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
        return ;
    }

    int count =0;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        (*it)["label"]= "leak_hole";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        cv::Mat crop = (task->image)(cv::Rect(LTx1, LTy1, (RBx1 - LTx1), (RBy1 - LTy1)));
        //write_debug_img(task, "crop", crop);
        count++;
        if(count>=1){
            break;
        }

    }

    return;
}

//黑色海绵 标志 bs
void sbgHl::bsfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "bs";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
        return ;
    }

    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];
        (*it)["label"]= "bs";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
    }
    return;
}

//地线巴 标志 gwb
void sbgHl::gwbfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img){

    if(indata.empty()){
        json s2;
        s2["label"] = "gwb";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = {{"confidence", 0}, {"area", 0}};
        algo_result->result_info.emplace_back(s2);
        return ;
    }

    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        (*it)["label"]= "gwb";
        algo_result->result_info.emplace_back((*it));
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);


    }
    return;
}

void sbgHl::write_debug_img(InferTaskPtr task, std::string name, cv::Mat img){

 //#ifdef  SAVE_DEBUG_IMG

    const std::type_info& info = typeid(*this);

    std::string imgName = task->image_info["img_name"];
    std::string fpath = fs::current_path().string() + "/debugImg/" + info.name() + "/" + imgName;
    if (!fs::exists(fpath)) {
        if (!fs::create_directories(fpath)) {
            std::cerr << "Error creating directory: " << fpath << std::endl;
            std::string fpath1 = fs::current_path().string() + "/Unkonw";
            fs::create_directories(fpath1);
        }
    }
    std::string savePath = fpath + "/" + name + ".jpg";
    cv::imwrite(savePath, img);
 //#endif //

    return;
}

bool sbgHl::isPointInsideRectangle(int pointX, int pointY, int rectLeft, int rectTop, int rectRight, int rectBottom) {
    if (pointX >= rectLeft && pointX <= rectRight && pointY >= rectTop && pointY <= rectBottom) {
        return true;
    }
    else {
        return false;
    }
}

//矩形交集，第一个矩形为螺丝孔位置，第二个矩形为大box 位置
bool sbgHl::isPointInsideRectangle(cv::Rect lhs,cv::Rect rhs){

    cv::Rect intersection = lhs & rhs;
    if(intersection.area() > 0){
        return true;
    }
    //交集为0，再判断距离
    cv::Point pt_center(lhs.x +lhs.width/2,lhs.y +lhs.height/2);
    cv::Point p1 = rhs.tl();
    cv::Point p2 = rhs.br();
    cv::Point p3(p1.x ,p1.y+rhs.height);
    cv::Point p4(p2.x,p2.y-rhs.height);

    double d1 = cv::norm(p1 - pt_center);
    double d2 = cv::norm(p2 - pt_center);
    double d3 = cv::norm(p3 - pt_center);
    double d4 = cv::norm(p1 - pt_center);

    if(d1<350 || d2<350 || d3<350 || d4<350){
        return true;
    }
    return false;
}