
#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "PatrolCornerCut.h"
#include <AIRuntimeInterface.h>
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeUtils.h>
#include "../utils/Utils.h"
REGISTER_ALGO(PatrolCornerCut)

PatrolCornerCut::PatrolCornerCut()
{

}

PatrolCornerCut::~PatrolCornerCut()
{

}

AlgoResultPtr PatrolCornerCut::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    AlgoResultPtr algo_result                       = std::make_shared<stAlgoResult>();
    algo_result->status                             = RunStatus::OK;
    //AlgoResultPtr algoParam                         = pre_results[0];

    //std::string judgeParamsPath                     = algoParam->result_info["judgeBroken"];
    //json judgeParams                                = ReadJsonFile(judgeParamsPath);
    std::string task_type_id = task->image_info["type_id"];
    json        task_json = m_param_map[task_type_id];
    
    bool saveFet =     (bool)Utils::GetProperty(task_json["param"], "saveFet", 1); 
    std::vector<stBLOB_FEATURE>	                    BlobResultTotal;
    // BlobResultTotal.resize(300);
    RunAlgoCornerCut(task, pre_results, algo_result, task_json, BlobResultTotal);//大小角算法
    //json result_json = json::array();
    //result_to_json(BlobResultTotal, result_json, "OK");
    //algo_result->result_info = result_json;
    //saveFet = false;
    if (saveFet) {
        WriteBlobResultInfo(task, BlobResultTotal);
    }

    return algo_result;
}

void PatrolCornerCut::RunAlgoCornerCut(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal){
    START_TIMER
    const std::type_info& info = typeid(*this);
    json result_json = json::array();
    LOGI("{} start run!", info.name());
    //算法参数 json

    std::tuple<std::string, json> details_info = get_task_info(task, m_param_map);
    std::string                   task_type_id = std::get<0>(details_info);
    json                          task_param_json = std::get<1>(details_info);


    //blob特征
    std::vector<stBLOB_FEATURE>	m_BlobResult;
    ///////////////////////////AlgoParameters

    bool    saveFet = (bool)Utils::GetProperty(task_param_json["param"], "saveFet", 0);
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    std::call_once(flag, [this, task_param_json, task, saveFet]() {
        int i = 0;
        for (const auto& judgment : task_param_json["judgement"])
        {
            EdgeDefectJudgment[i].strItemName = judgment["name"].get<std::string>();
            int k = 0;
            for (const auto& params : judgment["judgeparams"])
            {
                EdgeDefectJudgment[i].Judgment[k].bUse = params["enable"].get<bool>();
                EdgeDefectJudgment[i].Judgment[k].nSign = getSignFromSymbol(params["symbol"].get<std::string>());
                EdgeDefectJudgment[i].Judgment[k].dValue = params["value"].get<double>();
                EdgeDefectJudgment[i].Judgment[k].name = params["name"].get<std::string>();

                k++;
            }
            i++;
        }
        //保存判定参数
        if (saveFet) {//多线程只保存一次
            WriteJudgeParams(task, EdgeDefectJudgment, i);
            LOGI("Save JudgeParams.txt Suceessful!");
        }

    });

    cv::Mat src, drawImg;
    try
    {

        (task->image).copyTo(src);
        (task->image).copyTo(drawImg);
        //write_debug_img(task, "src", src);
        int selectCorner = findMaxGVDifferenceCorners(task->image);
        if (selectCorner == -1) {
            write_debug_img(task, "abnorm", src);
            //return result_json;
        }
        if (selectCorner >= E_POSITION_BR && selectCorner <= E_POSITION_TL) {
            write_debug_img(task, "Corner", src);

        }

    }
    catch (const std::exception& e)
    {
        //std::cerr << e.what() << '\n';
        LOGE("broken error = {}", e.what());
    }
    for (int i = 0; i < BlobResultTotal.size(); i++)
    {
        if (BlobResultTotal[i].bFiltering == true) {
            result_to_json(BlobResultTotal, result_json, "BROKEN");
            break;
        }

    }

    algo_result->result_info = result_json;
    //const std::type_info& info = typeid(*this);
    LOGI("{} run finished!", info.name());
    END_TIMER
}

