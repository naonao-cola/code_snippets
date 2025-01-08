#include <windows.h>
#include "../framework/InferenceEngine.h"
/**
 * @FilePath     : /code_snippets/cxx/project/efem/BarcodeMatching.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-01-06 14:12:44
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-01-06 14:12:45
 * @Copyright (c) 2025 by G, All Rights Reserved.
**/
#include "../utils/logger.h"
#include "BarcodeMatching.h"
#include <AIRuntimeInterface.h>
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeUtils.h>
#include <DataCode2d.h>
#include "../utils/Utils.h"
REGISTER_ALGO(BarcodeMatching)

BarcodeMatching::BarcodeMatching()
{

}

BarcodeMatching::~BarcodeMatching()
{

}

AlgoResultPtr BarcodeMatching::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("BarcodeMatching start run!");

    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->status = RunStatus::OK;
    std::vector<cv::Rect> resultRoi;
	AIRuntimeInterface* ai_obj = GetAIRuntime();
    cv::Mat cropImg;
    std::pair<float, std::string> ocrResult;//OCR识别结果{conf, resString}
    json result_json;

    LOGI("Check image abnormal start.");
    if (!isAbnormal(task)) {
        result_to_json(result_json, "OK");
        algo_result->result_info = result_json;
        LOGI(" Check image abnormal Fail!!!\n");
        return algo_result;
    }
    LOGI("Check img abnormal end.");
    json barCodeInfo;
    Tival::DataCodeResult barCode;
    try
    {
        barCode = Tival::DataCode2d::FindDataDM(task->image, barCodeInfo);
        LOGI("barCode Detect Result:\n{}", barCode.ToJson().dump());
    }
    catch (const std::exception& e)//2024/03/15 修复软件崩溃
    {
        LOGI("barCode Detect Error:\n{}", e.what());
    }

    /*
    {"aborted":false,
    "candidate_num":60,
    "data_strings":["000000253AA2"],
    "num":1,
    "reflections":[0],
    "regions":[[916.0453186035156,1759.9606323242188,-0.17650488777887208,81.36577541714922,19.435622025268877]],
    "symbol_cols":[32],
    "symbol_rows":[8],
    "undecoded_num":0
    }
    */
   barCodeInfo = barCode.ToJson();
   if (barCodeInfo["data_strings"].empty()) {
       LOGI(" Code reading Fail!!!\n");
       result_to_json(result_json, "RNG");
       algo_result->result_info = result_json;
       return algo_result;
   }


        cv::Mat taskImg, src;
        (task->image).copyTo(src);
        //src = cv::imread("E:\\projdata\\tray\\OCR\\Pic_2024_01_19_120748_50.bmp");//aa.jpg  "./Image.png"
        //TaskInfoPtr detInferTask = std::make_shared<stTaskInfo>();
        //detInferTask->imageData = { src };
        //detInferTask->modelId = pre_results[0]->result_info["detModelId"];;
        //detInferTask->taskId = 1;
        //detInferTask->promiseResult = new std::promise<ModelResultPtr>();
        //ai_obj->CommitInferTask(detInferTask);
        //// 等待结果
        //std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(detInferTask->promiseResult);
        //std::future<ModelResultPtr>   futureRst = promiseResult->get_future();
        //ModelResultPtr                rst = futureRst.get();
        //std::vector<cv::Mat>          ret_img;
        //std::vector<std::vector<int>> ocr_det;
        //
        //for (int i = 0; i < rst->itemList.size(); i++) {
        //    auto ret = rst->itemList[i];
        //    for (int j = 0; j < ret.size(); j++) {
        //        for (int k = 0; k < 4; k++) {
        //            std::vector<int> tmp_1;
        //            tmp_1.emplace_back(ret[j].points[k].x);
        //            tmp_1.emplace_back(ret[j].points[k].y);
        //            ocr_det.emplace_back(tmp_1);
        //        }
        //        cropImg = GetRotateCropImage(src, ocr_det);
        //        //cv::imshow("cropImg", cropImg);
        //        //cv::waitKey(0);
        //        break;
        //    }
        //}

        int newX = barCode.regions[0].X - barCode.regions[0].Length1*cos(barCode.regions[0].Angle)/2 - barCode.regions[0].Length2*sin(barCode.regions[0].Angle)/2;
        int newY = barCode.regions[0].Y - barCode.regions[0].Length1*sin(barCode.regions[0].Angle)/2 - barCode.regions[0].Length2*cos(barCode.regions[0].Angle)/2;
        int newW = barCode.regions[0].Length1;
        int newH = barCode.regions[0].Length2;

        int x = newX;
        int y = newY;
        int w = newW;
        int h = newH;

        cv::Mat RatationedImg(src.rows, src.cols, CV_8UC3);
        RatationedImg.setTo(0);
        cv::Point2f center = cv::Point2f(barCode.regions[0].X, barCode.regions[0].Y);
        cv::Mat M2 = getRotationMatrix2D(center, -barCode.regions[0].Angle, 1);//旋转矩阵
        cv::warpAffine(src, RatationedImg, M2, src.size(), 1, 0, cv::Scalar(0));

        cv::Rect ocrLabel = cv::Rect(center.x - w * 4.4, center.y - h * 6, w * 8.7, h * 4.5);//360 120 640 80

        //cv::rectangle(RatationedImg, ocrLabel, cv::Scalar(0, 0, 255), 2);
        cropImg = RatationedImg(ocrLabel);

        (task->image).copyTo(taskImg);
        if (cropImg.empty()) {
            result_to_json(result_json, "CNG");
            algo_result->result_info = result_json;
            LOGI("OCR Detection Fail !!!\n");
            LOGI("BarcodeMatching run interrupt. ###Check barCode!!!");

            return algo_result;
        }
        write_debug_img("ocrChar", cropImg);
#ifdef SAVE
        write_debug_img("cropImg", cropImg);
#endif
        //创建OCR识别任务
        TaskInfoPtr recInferTask = std::make_shared<stTaskInfo>();
        recInferTask->imageData = { cropImg };
        recInferTask->modelId = pre_results[0]->result_info["recModelId"];
        recInferTask->taskId = 0;
        recInferTask->promiseResult = new std::promise<ModelResultPtr>();
        //提交推理任务
        ai_obj->CommitInferTask(recInferTask);

        std::promise<ModelResultPtr>* recPromiseResult = static_cast<std::promise<ModelResultPtr>*>(recInferTask->promiseResult);
        std::future<ModelResultPtr>   recFutureRst = recPromiseResult->get_future();
        //获取推理结果
        ModelResultPtr       recRst = recFutureRst.get();


        for (int i = 0; i < recRst->itemList.size(); i++) {
            for (auto& box : recRst->itemList[i]) {
                LOGI("\t====ocr result=====");
                LOGI("conf:{}\tres: {}\n", box.confidence, box.ocr_str);
                ocrResult.first     = box.confidence;
                ocrResult.second    = box.ocr_str;
                break;
            }
        }
        if (!is_match(barCodeInfo["data_strings"][0], ocrResult.second)){
            result_to_json(result_json, "MNG");
            algo_result->result_info = result_json;
            LOGI("OCR+ReadCode Match Fail !\n");
            return algo_result;
        }
    result_to_json(result_json, ocrResult.second);
    algo_result->result_info = result_json;
    LOGI("OCR+ReadCode Match Success!");
    LOGI("BarcodeMatching run finished!");

    return algo_result;
}



void BarcodeMatching::result_to_json(json& result_info, std::string result){
    if (result == "OK") {
        result_info = json::array();
        return ;
    }
    json s2;
    s2["label"]         =   result;
    s2["points"]        =   {{0, 0},{0, 0}};
    s2["shapeType"]     =   "rectangle";
    result_info.emplace_back(s2);
    return ;
}


cv::Mat BarcodeMatching::GetRotateCropImage(const cv::Mat& srcimage, std::vector<std::vector<int>> box)
{
    std::vector<int> x_vec{ box[0][0], box[1][0], box[2][0], box[3][0] };
    std::vector<int> y_vec{ box[0][1], box[1][1], box[2][1], box[3][1] };
    int              x_min = *std::min_element(x_vec.begin(), x_vec.end());
    int              x_max = *std::max_element(x_vec.begin(), x_vec.end());

    int y_min = *std::min_element(y_vec.begin(), y_vec.end());
    int y_max = *std::max_element(y_vec.begin(), y_vec.end());
    if (x_max - x_min < 3 || y_max - y_min < 3)
        return cv::Mat();

    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;

    int x_collect[4] = { box[0][0], box[1][0], box[2][0], box[3][0] };
    int y_collect[4] = { box[0][1], box[1][1], box[2][1], box[3][1] };
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) + pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M, cv::Size(img_crop_width, img_crop_height), cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 1);
        return srcCopy;
    }
    else {
        return dst_img;
    }
}

bool BarcodeMatching::isAbnormal(InferTaskPtr task){
    if (task->image.channels() == 3)
        return true;
    if (task->image.empty())
        return false;
    if (task->image.channels() == 1)
        return false;
    return true;
}

void BarcodeMatching::write_debug_img(std::string name, cv::Mat img){
    //std::string imgName = taskA->image_info["img_name"];
    //std::string fpath = fs::current_path().string() + "\\" + imgName;
    //if (!fs::exists(fpath)) {
    //    if (!fs::create_directories(fpath)) {
    //        std::cerr << "Error creating directory: " << fpath << std::endl;
    //        std::string fpath1 = fs::current_path().string() + "\\Unkonw";
    //        fs::create_directories(fpath1);
    //    }
    //}
    //std::string savePath = fpath + "\\" + name + ".jpg";
    //cv::imwrite(savePath, img);
    return;
}

bool BarcodeMatching::is_match(std::string Rcode, std::string Ccode){

    bool matchFlag = false;
    if (Rcode == Ccode) {
        matchFlag = true;
    }
    return matchFlag;
}

