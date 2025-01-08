#include "PrePatrolEdge.h"
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include <windows.h>

#include <AIRuntimeInterface.h>
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeUtils.h>
REGISTER_ALGO(PrePatrolEdge)

PrePatrolEdge::PrePatrolEdge()
{
}

PrePatrolEdge::~PrePatrolEdge()
{
}

AlgoResultPtr PrePatrolEdge::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    START_TIMER

    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->status       = RunStatus::OK;

    std::tuple<std::string, json> details_info    = get_task_info(task, pre_results, m_param_map);
    std::string                   task_type_id    = std::get<0>(details_info);
    json                          task_param_json = std::get<1>(details_info);
    algo_result->result_info = task_param_json["param"];

    ///////////////////////////AlgoParameters
    int     hardThreshold = (int)task_param_json["param"]["hardThreshold"]; //10  明暗阈值
    int     nStepX = (int)task_param_json["param"]["nStepX"];//10  分块数量
    int     nStepY = (int)task_param_json["param"]["nStepY"];//10
    int     blockRows = (int)task_param_json["param"]["blockRows"];//240  resize  弱化
    int     blockCols = (int)task_param_json["param"]["blockCols"];//240
    int     kernelSize = (int)task_param_json["param"]["kernelSize"];//
    int     filterContArea = (int)task_param_json["param"]["filterContArea"];//
    int     bgThreshold = (int)task_param_json["param"]["bgThreshold"];//
    int		nMinSamples = (int)task_param_json["param"]["nMinSamples"];// 拟合点数
    double	distThreshold = (double)task_param_json["param"]["distThreshold"];// 点集距离
    int     minCheckEdge = (int)task_param_json["param"]["minCheckEdge"];//填充不检区min
    int     maxCheckEdge = (int)task_param_json["param"]["maxCheckEdge"];//填充不检区max
    int     cannyThresMin = (int)task_param_json["param"]["cannyThresMin"];//20;轮廓阈值
    int     cannyThresMax = (int)task_param_json["param"]["cannyThresMax"];//40;
    int     arcLength = (int)task_param_json["param"]["arcLength"];//200;  过滤轮廓周长

    //int     cropx = 182;
    //int     cropy = 119;
    //int     cropw = 342;
    //int     croph = 303;
    int     cropx = (int)task_param_json["param"]["cropx"];
    int     cropy = (int)task_param_json["param"]["cropy"];
    int     cropw = (int)task_param_json["param"]["cropw"];
    int     croph = (int)task_param_json["param"]["croph"];
    if (1) {
        int thickness = 2;
        cv::rectangle(task->image, cv::Rect(cropx - thickness, cropy - thickness, cropw + 2*thickness, croph + 2*thickness), cv::Scalar(0, 0, 255), thickness);
        write_debug_img(task, "cropImg", task->image);
        task->image = task->image(cv::Rect(cropx, cropy, cropw, croph));
    }

    try {


    cv::Mat src;
    task->image.copyTo(src);
    EnhanceContrast(src, 5, 0.2);
    cv::Mat darkBg1, brightBg, fitMask;
    BG_Subtract(src, fitMask, brightBg, nStepX, nStepY, blockRows, blockCols);
    cv::Mat lineFet(2, 2, CV_32F);
    cv::Mat mask = cv::Mat::zeros(fitMask.size(), CV_8UC1);
    int selectCorner = findMaxGVDifferenceCorners(task->image);
    if (selectCorner == -1) {
        algo_result->status = RunStatus::ABNORMAL_IMAGE;
        return algo_result;
    }
    switch (4)
    {
    case E_POSITION_BR:
        makeMask_and_obtLineVec_BR(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_BR);
        break;
    case E_POSITION_BL:
        makeMask_and_obtLineVec_BL(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_BL);
        break;
    case E_POSITION_TR:
        makeMask_and_obtLineVec_TR(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_TR);
        break;
    case E_POSITION_TL:
        makeMask_and_obtLineVec_TL(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_TL);
        break;
    case E_POSITION_B:
        makeMask_and_obtLineVec_B(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_B);
        break;
    case E_POSITION_T:
        makeMask_and_obtLineVec_T(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_T);
        break;
    case E_POSITION_R:
        makeMask_and_obtLineVec_R(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_R);
        break;
    case E_POSITION_L:
        makeMask_and_obtLineVec_L(task, task->image, fitMask, lineFet, mask, minCheckEdge, maxCheckEdge, nMinSamples, distThreshold, E_POSITION_L);
        break;
    default:
        LOGW("not found edge line!!!")
            //return;
            break;
    }

    //LOGE("Get Mask ---------------------------------------------------------");
    algo_result->result_imgs.push_back(lineFet);
    algo_result->result_imgs.push_back(mask);
     pre_results.push_back(algo_result);
    }
    catch (const std::exception& e)
    {
        //std::cerr << e.what() << '\n';
        LOGE("preEdge error = {}", e.what());
    }
    LOGI("PreBarcodeMatching run finished!");
    END_TIMER
    return algo_result;

}
std::tuple<std::string, json> PrePatrolEdge::get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map)
{
    std::string task_type_id = task->image_info["type_id"];
    json        task_json    = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}

