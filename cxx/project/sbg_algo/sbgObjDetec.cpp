#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/Logger.h"
#include "../utils/Utils.h"

#include "SbgObjDetec.h"
#include <typeinfo>
#include <filesystem>
#include <iostream>
//#include <experimental/filesystem>

namespace fs = std::filesystem;
#if USE_AI_DETECT
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeInterface.h>
#include <AIRuntimeUtils.h>
#endif // USE_AI_DETECT

REGISTER_ALGO(sbgObjDetec)

sbgObjDetec::sbgObjDetec()
{

}

sbgObjDetec::~sbgObjDetec()
{

}

AlgoResultPtr sbgObjDetec::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("sbgObjDetec start run!");

    cv::Mat drawImg, drawImg2;
    json KDdata = json::array();
    json HJdata = json::array();
    json TBdata = json::array();
    json DXBdata = json::array();
    json FSJdata = json::array();
    json AXFSJdata = json::array();
    json MLdata = json::array();
    json HSFHJdata = json::array();
    json LSdata = json::array();
    json BOXdata = json::array();
    json ALLdata = json::array();

    (task->image).copyTo(drawImg);
    (task->image).copyTo(drawImg2);
    ////////////////////////////////////////////////////

    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    AIRuntimeInterface* ai_obj = GetAIRuntime();
    TaskInfoPtr _task = std::make_shared<stTaskInfo>();
    _task->imageData = { task->image };
    _task->modelId = 2;
    _task->taskId = 0;
    stAIModelInfo::mPtr model_info = GetAIRuntime()->GetModelInfo(_task->modelId);
    std::filesystem::path model_path(model_info->modelPath);
    if (!std::filesystem::exists(model_path)) {
       
        algo_result->status = ErrorCode::WRONG_PARAM;
        return algo_result;
    }

    _task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(_task);


    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(_task->promiseResult);
    std::future<ModelResultPtr>   futureRst = promiseResult->get_future();

    ModelResultPtr rst = futureRst.get();
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
            // algo_result->result_info.emplace_back(s2);
            ALLdata.emplace_back(s2);
            switch (box.code)
            {
            case KD:
                KDdata.emplace_back(s2);
                break;
            case HJ:
                HJdata.emplace_back(s2);
                break;
            case TB:
                TBdata.emplace_back(s2);
                break;
            case DXB:
              DXBdata.emplace_back(s2);
              break;
            case FSJ:
                FSJdata.emplace_back(s2);
                break;
            case AXFSJ:
              AXFSJdata.emplace_back(s2);
              break;
            case ML:
                MLdata.emplace_back(s2);
                break;
            case HSFHJ:
                HSFHJdata.emplace_back(s2);
                break;
            case LS:
                LSdata.emplace_back(s2);
                break;
            case BOX:
                BOXdata.emplace_back(s2);
                break;
            default:
                break;
            }
        }
    }
#ifdef SAVE_DEBUG
    write_debug_img(task, "all_obj", drawImg);
#endif
    KDfilter(task, KDdata, algo_result, BOXdata, drawImg2);
    HJfilter(task, HJdata, algo_result, BOXdata, drawImg2);
    TBfilter(task, TBdata, algo_result, BOXdata, drawImg2);
    DXBfilter(task, DXBdata, algo_result, BOXdata, drawImg2);
    FSJfilter(task, FSJdata, algo_result, BOXdata, drawImg2);
    AXFSJfilter(task, AXFSJdata, algo_result, BOXdata, drawImg2);
    MLfilter(task, MLdata, algo_result, BOXdata, drawImg2);
    HSFHJfilter(task, HSFHJdata, algo_result, BOXdata, drawImg2);
    LSfilter(task, LSdata, algo_result, BOXdata, drawImg2);


#ifdef SAVE_DEBUG
    write_debug_img(task, "filter_obj", drawImg2);
#endif

    LOGI("sbgObjDetec algo end run file {}, line {} info{}", __FILE__, __LINE__, Utils::DumpJson(algo_result->result_info, false));
    LOGI("sbgObjDetec run finished!");

    return algo_result;
}

//接头螺丝孔 返回标志 screw_hole
void sbgObjDetec::KDfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;
    if (indata.empty()) {
        json s2;
        s2["label"] = "screw_hole";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }

    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;
    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */
    bool exists_flag = false;
    int screw_hole_count = 0;
    cv::Rect lhs;
    cv::Rect rhs(cv::Point(LTx, LTy), cv::Point(RBx, RBy));
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        lhs = cv::Rect(cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1));
        bool flag = isPointInsideRectangle(lhs, rhs);

        if (flag) {//
            exists_flag = true;
            (*it)["label"] = "screw_hole";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
            screw_hole_count++;
            if (screw_hole_count >= 4) {
                break;
            }
        }

    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "screw_hole";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);

    }
    return;
}

//绝缘黑胶   返回 ibg
void sbgObjDetec::HJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;
    if (indata.empty()) {
        json s2;
        s2["label"] = "ibg";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }

    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;
    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */
    bool exists_flag = false;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2, LTx, LTy, RBx, RBy);

        if (flag) {//
            exists_flag = true;
            (*it)["label"] = "ibg";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        }

    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "ibg";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);

    }
    return;
}

//条巴 标志 tb
void sbgObjDetec::TBfilter(InferTaskPtr task, json& indata,
                           AlgoResultPtr algo_result, json& BOXdata,
                           cv::Mat& img) {
  int LTx = 0;
  int LTy = 0;
  int RBx = 0;
  int RBy = 0;
  if (indata.empty()) {
    json s2;
    s2["label"] = "tb";
    s2["points"] = {};
    s2["shapeType"] = "rectangle";
    s2["result"] = {{"confidence", 0}, {"area", 0}};
    algo_result->result_info.emplace_back(s2);
    return;
  }

  if (BOXdata.empty()) {  //
    LOGW("BOX Target is empty!!! Use default BOX");
    LTx = 0;
    LTy = 0;
    RBx = task->image.cols;
    RBy = task->image.rows;
  } else if (BOXdata.size() == 1) {  //
    //
    /*
    code...
    */
    LTx = BOXdata[0]["points"][0][0];
    LTy = BOXdata[0]["points"][0][1];
    RBx = BOXdata[0]["points"][1][0];
    RBy = BOXdata[0]["points"][1][1];
  }
  //
  /*
  code...
  */
  bool exists_flag = false;
  for (auto it = indata.begin(); it != indata.end(); ++it) {
    int LTx1 = (*it)["points"][0][0];
    int LTy1 = (*it)["points"][0][1];
    int RBx1 = (*it)["points"][1][0];
    int RBy1 = (*it)["points"][1][1];

    bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2,
                                       LTx, LTy, RBx, RBy);

    if (flag) {  //
      exists_flag = true;
      (*it)["label"] = "tb";
      algo_result->result_info.emplace_back((*it));
      cv::Scalar color(0, 255, 0);
      cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color,
                    3);
    }
  }
  if (!exists_flag) {
    json s2;
    s2["label"] = "tb";
    s2["points"] = {};
    s2["shapeType"] = "rectangle";
    s2["result"] = {{"confidence", 0}, {"area", 0}};
    algo_result->result_info.emplace_back(s2);
  }
  return;
}

//地线巴 标志 gwb
void sbgObjDetec::DXBfilter(InferTaskPtr task, json& indata,
                            AlgoResultPtr algo_result, json& BOXdata,
                            cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;
    if (indata.empty()) {
        json s2;
        s2["label"] = "gwb";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }

    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;
    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */
    bool exists_flag = false;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2, LTx, LTy, RBx, RBy);

        if (flag) {//
            exists_flag = true;
            (*it)["label"] = "gwb";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        }

    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "gwb";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);

    }
    return;
}
// 条状防水胶  标志 wa_s
void sbgObjDetec::FSJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;

    if (indata.empty()) {
        json s2;
        s2["label"] = "wa_s";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }

    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;
    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */bool exists_flag = false;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2, LTx, LTy, RBx, RBy);

        if (flag) {//
            exists_flag = true;
            (*it)["label"] = "wa_s";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        }

    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "wa_s";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);

    }
    return;
}

//marly片 标志 marly
void sbgObjDetec::MLfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;

    if (indata.empty()) {
        json s2;
        s2["label"] = "marly";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }

    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;
    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */
    bool exists_flag = false;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2, LTx, LTy, RBx, RBy);

        if (flag) {//
            exists_flag = true;
            (*it)["label"] = "marly";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        }

    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "marly";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    return;
}
//黑色海绵旁边的螺丝  标志bs
void sbgObjDetec::HSFHJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;
    if (indata.empty()) {
        json s2;
        s2["label"] = "bs";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }

    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;
    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */
    bool exists_flag = false;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2, LTx, LTy, RBx, RBy);

        if (flag) {//

            exists_flag = true;
            (*it)["label"] = "bs";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
            cv::Mat crop = (task->image)(cv::Rect(LTx1, LTy1, (RBx1 - LTx1), (RBy1 - LTy1)));
            write_debug_img(task, "crop", crop);
        }
    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "bs";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    return;
}

//加强螺丝筋 标志screw
void sbgObjDetec::LSfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;
    if (indata.empty()) {
        json s2;
        s2["label"] = "screw";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }

    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;
    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */
    bool exists_flag = false;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2, LTx, LTy, RBx, RBy);

        if (flag) {//
            exists_flag = true;
            (*it)["label"] = "screw";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        }

    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "screw";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
    }
    return;
}

//凹状防水胶 标志 wa_c
void sbgObjDetec::AXFSJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img) {
    int LTx = 0;
    int LTy = 0;
    int RBx = 0;
    int RBy = 0;
    if (indata.empty()) {
        json s2;
        s2["label"] = "wa_c";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);
        return;
    }
    if (BOXdata.empty()) {//
        LOGW("BOX Target is empty!!! Use default BOX");
        LTx = 0;
        LTy = 0;
        RBx = task->image.cols;
        RBy = task->image.rows;

    }
    else if (BOXdata.size() == 1) {//
        //
        /*
        code...
        */
        LTx = BOXdata[0]["points"][0][0];
        LTy = BOXdata[0]["points"][0][1];
        RBx = BOXdata[0]["points"][1][0];
        RBy = BOXdata[0]["points"][1][1];

    }
    //
    /*
    code...
    */
    bool exists_flag = false;
    for (auto it = indata.begin(); it != indata.end(); ++it) {
        int LTx1 = (*it)["points"][0][0];
        int LTy1 = (*it)["points"][0][1];
        int RBx1 = (*it)["points"][1][0];
        int RBy1 = (*it)["points"][1][1];

        bool flag = isPointInsideRectangle((LTx1 + RBx1) / 2, (LTy1 + RBy1) / 2, LTx, LTy, RBx, RBy);

        if (flag) {//
            exists_flag = true;
            (*it)["label"] = "wa_c";
            algo_result->result_info.emplace_back((*it));
            cv::Scalar color(0, 255, 0);
            cv::rectangle(img, cv::Point(LTx1, LTy1), cv::Point(RBx1, RBy1), color, 3);
        }

    }
    if (!exists_flag) {
        json s2;
        s2["label"] = "wa_c";
        s2["points"] = {};
        s2["shapeType"] = "rectangle";
        s2["result"] = { {"confidence", 0}, {"area", 0} };
        algo_result->result_info.emplace_back(s2);

    }
    return;
}

void sbgObjDetec::write_debug_img(InferTaskPtr task, std::string name, cv::Mat img) {

    //#ifdef  SAVE_DEBUG_IMG

    const std::type_info& info = typeid(*this);

    std::string imgName = task->image_info["img_name"];
    std::string fpath = fs::current_path().string() + "\\debugImg\\" + info.name() + "\\" + imgName;
    if (!fs::exists(fpath)) {
        if (!fs::create_directories(fpath)) {
            std::cerr << "Error creating directory: " << fpath << std::endl;
            std::string fpath1 = fs::current_path().string() + "\\Unkonw";
            fs::create_directories(fpath1);
        }
    }
    std::string savePath = fpath + "\\" + name + ".jpg";
    cv::imwrite(savePath, img);
    //#endif //

    return;
}

bool sbgObjDetec::isPointInsideRectangle(int pointX, int pointY, int rectLeft, int rectTop, int rectRight, int rectBottom) {
    if (pointX >= rectLeft && pointX <= rectRight && pointY >= rectTop && pointY <= rectBottom) {
        return true;
    }
    else {
        return false;
    }
}

//矩形交集，第一个矩形为螺丝孔位置，第二个矩形为大box 位置
bool sbgObjDetec::isPointInsideRectangle(cv::Rect lhs, cv::Rect rhs) {
    cv::Rect intersection = lhs & rhs;
    if (intersection.area() > 0) {
        return true;
    }
    //交集为0，再判断距离
    cv::Point pt_center(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
    cv::Point p1 = rhs.tl();
    cv::Point p2 = rhs.br();
    cv::Point p3(p1.x, p1.y + rhs.height);
    cv::Point p4(p2.x, p2.y - rhs.height);

    double d1 = cv::norm(p1 - pt_center);
    double d2 = cv::norm(p2 - pt_center);
    double d3 = cv::norm(p3 - pt_center);
    double d4 = cv::norm(p1 - pt_center);

    if (d1 < 350 || d2 < 350 || d3 < 350 || d4 < 350) {
        return true;
    }
    return false;
}