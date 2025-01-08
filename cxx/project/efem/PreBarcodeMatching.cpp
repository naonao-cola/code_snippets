#include "PreBarcodeMatching.h"
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include <windows.h>



REGISTER_ALGO(PreBarcodeMatching)


PreBarcodeMatching::PreBarcodeMatching()
{
}

PreBarcodeMatching::~PreBarcodeMatching()
{
}

AlgoResultPtr PreBarcodeMatching::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->status       = RunStatus::OK;

    std::tuple<std::string, json> details_info      = get_task_info(task, pre_results, m_param_map);
    std::string                   task_type_id      = std::get<0>(details_info);
    json                          task_param_json   = std::get<1>(details_info);
    algo_result->result_info                        = task_param_json["param"];
     
    stAIConfigInfo ai_cfg;
    ai_cfg.preProcessThreadPriority                 = 0;
    ai_cfg.inferThreadCnt                           = 1;
    ai_cfg.inferThreadPriority                      = 0;

    AIRuntimeInterface* ai_obj                      = GetAIRuntime();
    ai_obj->InitRuntime(ai_cfg);

    //识别模型
    stAIModelInfo recModelCfg;
    recModelCfg.modelId                             = task_param_json["param"]["recModelId"];
    recModelCfg.modelName                           = "rec";
    recModelCfg.modelPath                           = task_param_json["param"]["recModelPath"];
    recModelCfg.modelVersion                        = 1;
    recModelCfg.modelBackend                        = "onnxruntime";
    recModelCfg.inferParam.confidenceThreshold      = 0.25;
    recModelCfg.modelLabelPath                      = task_param_json["param"]["modelLabelPath"];
    recModelCfg.inferParam.maxBatchSize             = 1;
    recModelCfg.algoType                            = OCR_REC;
    ai_obj->CreateModle(recModelCfg);

    //检测模型
    //stAIModelInfo detModelCfg;
    //detModelCfg.modelId                             = task_param_json["param"]["detModelId"];
    //detModelCfg.modelName                           = "det";
    //detModelCfg.modelPath                           = task_param_json["param"]["detModelPath"];
    //detModelCfg.modelVersion                        = 1;
    //detModelCfg.modelBackend                        = "onnxruntime";
    //detModelCfg.inferParam.confidenceThreshold      = 0.25;
    //detModelCfg.inferParam.maxBatchSize             = 1;
    //detModelCfg.algoType                            = OCR_DET;
    //ai_obj->CreateModle(detModelCfg);

    return algo_result;
}


 std::tuple<std::string, json> PreBarcodeMatching::get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map)
{
    std::string task_type_id = task->image_info["type_id"];
    json        task_json    = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}
