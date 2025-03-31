#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "AlgoA.h"
#if USE_AI_DETECT
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeInterface.h>
#include <AIRuntimeUtils.h>
#endif // USE_AI_DETECT
REGISTER_ALGO(AlgoA)

AlgoA::AlgoA()
{

}

AlgoA::~AlgoA()
{

}

AlgoResultPtr AlgoA::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("AlgoA start run!");
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    AIRuntimeInterface* ai_obj = GetAIRuntime();
    TaskInfoPtr _task = std::make_shared<stTaskInfo>();
    _task->imageData  = { task->image };
    _task->modelId    = 1;
    _task->taskId     = 0;
    _task->promiseResult = new std::promise<ModelResultPtr>();
    ai_obj->CommitInferTask(_task);

    
    std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(_task->promiseResult);
    std::future<ModelResultPtr>   futureRst  = promiseResult->get_future();
   
    ModelResultPtr rst         = futureRst.get();
    for (int i = 0; i < rst->itemList.size(); i++) {
           for (auto& box : rst->itemList[i]) {
               std::cout <<box.confidence <<box.code <<box.shape << std::endl;
               std::cout << box.points[0].x << " " << box.points[0].y << " " << box.points[1].x << " "<<box.points[1].y << std::endl;
               cv::Scalar color(0, 255, 0);
               cv::rectangle(task->image, cv::Point(box.points[0].x, box.points[0].y), cv::Point(box.points[1].x, box.points[1].y), color, 3);
               cv::imshow("111", task->image);
               cv::waitKey();
               algo_result->result_info = {
                {"label", box.code},
                {"shapeType", "rectangle"},
                {"points", {{box.points[0].x, box.points[0].y}, {box.points[1].x, box.points[1].y}}},
                {"result", {{"confidence", box.confidence}, {"area", (box.points[1].x - box.points[0].x)*(box.points[1].y - box.points[0].y)}}},
               };
           }
    }
    


    //algo_result->result_info = {
    //    {"label", "AlgoA_defect"},
    //    {"shapeType", "rectangle"},
    //    {"points", {{box.points[0].x, box.points[0].y}, {150.0, 150.0}}},
    //    {"result", {{"confidence", 0.78}, {"area", 10000}}},
    //};
    
    LOGI("AlgoA run finished!");

    return algo_result;
}