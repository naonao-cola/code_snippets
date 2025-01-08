
#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "PatrolEdgeSplit.h"
#include <AIRuntimeInterface.h>
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeUtils.h>
#include "../utils/Utils.h"
REGISTER_ALGO(PatrolEdgeSplit)

PatrolEdgeSplit::PatrolEdgeSplit()
{

}

PatrolEdgeSplit::~PatrolEdgeSplit()
{

}

AlgoResultPtr PatrolEdgeSplit::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->status = RunStatus::OK;
    json result_json = json::array();

    std::string task_type_id = task->image_info["type_id"];
    json        task_json = m_param_map[task_type_id];
    if (pre_results[0]->status != RunStatus::OK) {//异常返回
        algo_result->result_info = result_json;
        LOGE("Image Alarm Abmormal occurrent !!!");
        return algo_result;
    }
    bool saveFet =     (bool)Utils::GetProperty(task_json["param"], "saveFet", 1); 
    std::vector<stBLOB_FEATURE>	                    BlobResultTotal;
    // BlobResultTotal.resize(300);
    RunAlgoSplit(task, pre_results, algo_result, task_json, BlobResultTotal);//破片

    //result_to_json(BlobResultTotal, result_json, "OK");
    //algo_result->result_info = result_json;
    //saveFet = false;
    if (saveFet) {
        WriteBlobResultInfo(task, BlobResultTotal);
    }

    return algo_result;
}

void PatrolEdgeSplit::RunAlgoSplit(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal){
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
        write_debug_img(task, "src", src);

        cv::Mat removeShadowMat;
        removeShadows(task, src, removeShadowMat);


        cv::normalize(removeShadowMat, removeShadowMat, 0, 200, cv::NORM_MINMAX);
        cv::Mat matDstBufX1, matDstBufY1, matDstBuf1;
        Estimation_X(removeShadowMat, matDstBufX1, 12, 1, 0.4);
        Estimation_Y(removeShadowMat, matDstBufY1, 12, 1, 0.4);
        TwoImg_Average(matDstBufX1, matDstBufY1, matDstBuf1);
        //EnhanceContrast(src, 5, 0.2);
        cv::Mat matSub1, matSub2;
        cv::subtract(matDstBuf1, removeShadowMat, matSub1);


        cv::Mat defectMask1;
        cv::dilate(matSub1, matSub1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        Make_HardDefect_Mask(matSub1, defectMask1, hardThreshold, nStepX, nStepY);


        //cv::subtract(matDstBuf1, src, matSub2);
        //cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        //findAndDrawContours(src, binImg);

        //cv::Mat defectMask;
        //Make_HardDefect_Mask(src, defectMask, hardThreshold, nStepX, nStepY);

        //cv::erode(defectMask, defectMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        //cv::dilate(defectMask, defectMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));


        //cv::Mat fitMask = darkBg1.clone();
        //write_debug_img(task, "fitLine", fitMask);
        //cv::cvtColor(fitMask, fitMask, cv::COLOR_BGR2GRAY);

        //cv::threshold(fitMask, fitMask, 0, 255, cv::THRESH_BINARY);
        //cv::dilate(fitMask, fitMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        //cv::Mat defectLineMask;
        //Make_LineDefect_Mask(BgMask, defectLineMask, 80);

        //cv::Mat lineFet(2, 2, CV_32F);
        cv::Mat lineFet;
        cv::Mat mask;
        //cv::Mat mask;
        lineFet = pre_results[0]->result_imgs[0];

        mask = pre_results[0]->result_imgs[1];
        //if (pre_results[0]->result_imgs[1].empty()) {
        //    LOGE("Get Mask ERROR  img_id:{}\n  INFO:{}", task->image_info["img_name"], task->image_info["img_w"], task->image_info["img_h"]);
        //    mask = cv::Mat::zeros(defectMask1.size(), CV_8UC1);
        //}
        //int selectCorner = findMaxGVDifferenceCorners(task->image);






        cv::Mat result, defectEdge;
        cv::bitwise_and(defectMask1, mask, defectEdge);
        //cv::bitwise_and(src, mask, result);
        write_debug_img(task, "detectMask", defectEdge);

        ////破片提取
        stBLOB_FEATURE obviousFet;
        ////obviousBroken(result, 250, obviousFet);

        ////边缘提取
        //cv::bitwise_and(fitMask, mask, fitMask);

        ////cv::Mat darkBg, brightBg;
        ////BG_Subtract(src, darkBg, brightBg, nStepX, nStepY, blockRows, blockCols);
        //// 
        //// cv::Mat matDstBufX, matDstBufY, matDstBuf;
        //// Estimation_X(src, matDstBufX, 22, 10, 0.4);
        //// Estimation_Y(src, matDstBufY, 22, 10, 0.4);
        //// TwoImg_Average(matDstBufX, matDstBufY, matDstBuf);


        // cv::Mat matDstBufX, matDstBufY, matDstBuf;
        // Estimation_X(result, matDstBufX, 22, 1, 0.6);
        // Estimation_Y(result, matDstBufY, 22, 1, 0.6);
        // TwoImg_Average(matDstBufX, matDstBufY, matDstBuf);

        // cv::Mat matSub, matSubafter;
        // cv::subtract(matDstBuf, result, matSub);

        // cv::Mat darkBg, brightBg;
        // BG_Subtract(matSub, darkBg, brightBg, nStepX, nStepY, blockRows, blockCols);

        // cv::Mat defectMask;
        // Make_HardDefect_Mask(matSub, defectMask, hardThreshold, nStepX, nStepY);
        //// write_debug_img("matSub", matSub);
        ////cv::Mat testSub;
        ////findAndDrawContours(matSub, testSub);
        ////cv::subtract(testSub, darkBg, testSub);
        //cv::add(darkBg, brightBg, darkBg);

        ////cv::subtract(darkBg, fitMask, darkBg);

        //cv::Mat defectEdge;
        //cv::add(defectMask, darkBg, defectEdge);
        //cv::bitwise_and(defectEdge, mask, defectEdge);
        //cv::subtract(defectEdge, fitMask, defectEdge);
        //cv::subtract(defectEdge, defectMask, defectEdge);
        //
        ////cv::Mat dilateMat;
        ////cv::erode(defectEdge, dilateMat, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
        ////cv::erode(defectEdge, defectEdge, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        //write_debug_img(task, "defectEdge", defectEdge);




        //cv::subtract(matSub, defectLineMask, matSubafter, darkBg);
        //cv::Mat binDefect = cv::Mat(matSubafter.size(), CV_8UC1);
        //write_debug_img("matSubafter", matSubafter);
        ////cv::threshold(matSub, binDefect, 40, 255.0, cv::THRESH_BINARY);
        //cv::bitwise_and(matSubafter, mask, matSubafter);
        cv::Mat res, matDrawBuf, res1;
        (task->image).copyTo(matDrawBuf);
        //cv::erode(defectEdge, defectEdge, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        DoBlobCalculate(defectEdge, lineFet, matDrawBuf, 300, m_BlobResult);
        //pBlobFet pStblobResult;
        (task->image).copyTo(res1);
        DrawBlob(res1, cv::Scalar(0, 0, 255), BLOB_DRAW_BLOBS, false, m_BlobResult);
        //DrawBlob(res1, cv::Scalar(255, 0, 0), BLOB_DRAW_BLOBS_CONTOUR, false, m_BlobResult);
        DrawBlob(res1, cv::Scalar(255, 0, 0), BLOB_DRAW_ROTATED_BOX, false, m_BlobResult);
        DrawBlob(res1, cv::Scalar(255, 0, 0), BLOB_DRAW_BOUNDING_BOX, false, m_BlobResult);
        write_debug_img(task, "preResult", res1);



        if (saveFet) {
            WriteBlobResultInfo_F(task, m_BlobResult);
        }
        judgeFeature(EdgeDefectJudgment, m_BlobResult, BlobResultTotal);
        if (obviousFet.bFiltering == true) {//明显的破片缺陷
            BlobResultTotal.push_back(obviousFet);
        }
        (task->image).copyTo(res);
        DrawBlob(res, cv::Scalar(0, 255, 0), BLOB_DRAW_BOUNDING_BOX, true, BlobResultTotal);

        //DrawBlob(res, cv::Scalar(0, 0, 255), BLOB_DRAW_BLOBS, true, m_BlobResult);
        //DrawBlob(res, cv::Scalar(255, 0, 0), BLOB_DRAW_BLOBS_CONTOUR, true, m_BlobResult);
        write_debug_img(task, "Result", res);
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

