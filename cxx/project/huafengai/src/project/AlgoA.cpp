/**
 * @FilePath     : /connector_ai/src/project/AlgoA.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-10-12 11:33:46
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-11-25 16:00:20
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
#include "AlgoA.h"
#include "../../modules/tv_algo_base/src/framework/InferenceEngine.h"

#include "../../modules/tv_algo_base/src/utils/logger.h"


#include <windows.h>

#if USE_AI_DETECT
#    include <AIRuntimeDataStruct.h>
#    include <AIRuntimeInterface.h>
#    include <AIRuntimeUtils.h>
#endif   // USE_AI_DETECT

// #define DRAW 0
REGISTER_ALGO(AlgoA)

AlgoResultPtr AlgoA::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("AlgoA start run! exec update 2024/11/25");
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    getParam(task);
    CropTaskImg(task->image);
    Infer(task, algo_result);
    LOGI("AlgoA run finished!");
    return algo_result;
}

void AlgoA::getParam(InferTaskPtr task)
{
    // LOGI("exec getParam function!");
    json param_json   = GetTaskParams(task);
    batch_num_        = xx::GetProperty<int>(param_json["param"], "batch_num", 1);
    path_             = xx::GetProperty<std::string>(param_json["param"], "path", "");   //// 模板匹配
    yml_              = xx::GetProperty<std::string>(param_json["param"], "yml", "");
    center_count_     = xx::GetProperty<int>(param_json["param"], "center_count", 4);
    num_              = xx::GetProperty<int>(param_json["param"], "num", 96);
    num_levels_       = xx::GetProperty<int>(param_json["param"], "num_levels", 4);
    angle_min_        = xx::GetProperty<int>(param_json["param"], "angle_min", -10);
    angle_max_        = xx::GetProperty<int>(param_json["param"], "angle_max", 10);
    min_score_        = xx::GetProperty<double>(param_json["param"], "min_score", 0.4);
    contrast_         = xx::GetProperty<int>(param_json["param"], "contrast", 35);
    scale_min_        = xx::GetProperty<double>(param_json["param"], "scale_min", 0.9);
    scale_max_        = xx::GetProperty<double>(param_json["param"], "scale_max", 1.1);
    max_overlap_      = xx::GetProperty<double>(param_json["param"], "max_overlap", 0.2);
    strength_         = xx::GetProperty<double>(param_json["param"], "strength", 0.8);
    sort_by_score_    = xx::GetProperty<bool>(param_json["param"], "sort_by_score", false);
    detect_left_x_    = xx::GetProperty<int>(param_json["param"], "detect_left_x", 23);
    detect_left_y_    = xx::GetProperty<int>(param_json["param"], "detect_left_y", 119);
    detect_width_     = xx::GetProperty<int>(param_json["param"], "detect_width", 9132);
    detect_height_    = xx::GetProperty<int>(param_json["param"], "detect_height", 6679);
    product_rows_     = xx::GetProperty<int>(param_json["param"], "product_rows", 8);   // 产品信息
    product_cols_     = xx::GetProperty<int>(param_json["param"], "product_cols", 3);
    enable_affineImg_ = xx::GetProperty<bool>(param_json["param"], "enable_affineImg", false);


    a_                 = xx::GetProperty<double>(param_json["param"], "a", 0.8);   //// 检测规格
    b1_                = xx::GetProperty<double>(param_json["param"], "b1", 0.8);
    b2_                = xx::GetProperty<double>(param_json["param"], "b2", 0.8);
    b3_                = xx::GetProperty<double>(param_json["param"], "b3", 0.8);
    c1_                = xx::GetProperty<double>(param_json["param"], "c1", 0.8);
    c2_                = xx::GetProperty<double>(param_json["param"], "c2", 0.8);
    d1_                = xx::GetProperty<double>(param_json["param"], "d1", 0.8);
    d2_                = xx::GetProperty<double>(param_json["param"], "d2", 0.8);
    e_                 = xx::GetProperty<double>(param_json["param"], "e", 0.8);
    p_                 = xx::GetProperty<double>(param_json["param"], "p", 0.8);
    f_                 = xx::GetProperty<double>(param_json["param"], "f", 0.8);
    error_a_           = xx::GetProperty<double>(param_json["param"], "error_a", 0.8);   //// 误差规格
    error_b1_          = xx::GetProperty<double>(param_json["param"], "error_b1", 0.8);
    error_b2_          = xx::GetProperty<double>(param_json["param"], "error_b2", 0.8);
    error_b3_          = xx::GetProperty<double>(param_json["param"], "error_b3", 0.8);
    error_c1_          = xx::GetProperty<double>(param_json["param"], "error_c1", 0.8);
    error_c2_          = xx::GetProperty<double>(param_json["param"], "error_c2", 0.8);
    error_d1_          = xx::GetProperty<double>(param_json["param"], "error_d1", 0.8);
    error_d2_          = xx::GetProperty<double>(param_json["param"], "error_d2", 0.8);
    error_e_           = xx::GetProperty<double>(param_json["param"], "error_e", 0.8);
    error_p_           = xx::GetProperty<double>(param_json["param"], "error_p", 0.8);
    error_f_           = xx::GetProperty<double>(param_json["param"], "error_f", 0.8);
    pix_value_         = xx::GetProperty<double>(param_json["param"], "pix_value", 5.26);   //// 相机规格
    t_y                = xx::GetProperty<int>(param_json["param"], "t_y", 26);              //// 模板规格
    ll_x1              = xx::GetProperty<int>(param_json["param"], "ll_x1", 24);
    ll_x2              = xx::GetProperty<int>(param_json["param"], "ll_x2", 64);
    ll_y1              = xx::GetProperty<int>(param_json["param"], "ll_y1", 26);
    ll_y2              = xx::GetProperty<int>(param_json["param"], "ll_y2", 164);
    rr_x1              = xx::GetProperty<int>(param_json["param"], "rr_x1", 458);
    rr_x2              = xx::GetProperty<int>(param_json["param"], "rr_x2", 497);
    rr_y1              = xx::GetProperty<int>(param_json["param"], "rr_y1", 26);
    rr_y2              = xx::GetProperty<int>(param_json["param"], "rr_y2", 164);
    lc_x1              = xx::GetProperty<int>(param_json["param"], "lc_x1", 116);
    lc_x2              = xx::GetProperty<int>(param_json["param"], "lc_x2", 201);
    lc_y1              = xx::GetProperty<int>(param_json["param"], "lc_y1", 109);
    lc_y2              = xx::GetProperty<int>(param_json["param"], "lc_y2", 143);
    rc_x1              = xx::GetProperty<int>(param_json["param"], "rc_x1", 318);
    rc_x2              = xx::GetProperty<int>(param_json["param"], "rc_x2", 404);
    rc_y1              = xx::GetProperty<int>(param_json["param"], "rc_y1", 109);
    rc_y2              = xx::GetProperty<int>(param_json["param"], "rc_y2", 143);
    enable_saveSample_ = xx::GetProperty<bool>(param_json["param"], "enable_saveSample", false);
    sampleSavePath_    = xx::GetProperty<std::string>(param_json["param"], "sampleSavePath", {});
    maskTempImgPath_   = xx::GetProperty<std::string>(param_json["param"], "maskTempImgPath", {});

    enable_saveNGImg_ = xx::GetProperty<bool>(param_json["param"], "enable_saveNGImg", false);
    NGSavePath_       = xx::GetProperty<std::string>(param_json["param"], "NGSavePath", {});

    topClsConf_ = xx::GetProperty<double>(param_json["param"], "topClsConf", 0.5);
    lrClsConf_  = xx::GetProperty<double>(param_json["param"], "lrClsConf", 0.5);
    inClsConf_  = xx::GetProperty<double>(param_json["param"], "inClsConf", 0.5);
    // 分类开关
    enable_topClassify_ = xx::GetProperty<bool>(param_json["param"], "enable_topClassify", false);
    enable_lrClassify_  = xx::GetProperty<bool>(param_json["param"], "enable_lrClassify", false);
    enable_inClassify_  = xx::GetProperty<bool>(param_json["param"], "enable_inClassify", false);
    // 测量开关
    enable_topMeasure_ = xx::GetProperty<bool>(param_json["param"], "enable_topMeasure", true);
    enable_lrMeasure_  = xx::GetProperty<bool>(param_json["param"], "enable_lrMeasure", true);
    enable_inMeasure_  = xx::GetProperty<bool>(param_json["param"], "enable_inMeasure", true);

    rectLrInTop_0_ = param_json["param"]["rectLrInTop_0"];
    rectLrInTop_1_ = param_json["param"]["rectLrInTop_1"];

    // 组合截图框
    crop_rect_0.clear();
    crop_rect_1.clear();
    if (rectLrInTop_0_.size() == rectLrInTop_1_.size()) {
        for (int i = 0; i < rectLrInTop_0_.size(); i = i + 4) {
            crop_rect_0.emplace_back(rectLrInTop_0_[i], rectLrInTop_0_[i + 1], rectLrInTop_0_[i + 2], rectLrInTop_0_[i + 3]);
            crop_rect_1.emplace_back(rectLrInTop_1_[i], rectLrInTop_1_[i + 1], rectLrInTop_1_[i + 2], rectLrInTop_1_[i + 3]);
        }
    }
    else {
        crop_rect_0 = {};
        crop_rect_1 = {};
        LOGI("algo params crop rect error!")
    }
    imgName_ = task->image_info["img_name"];
    // maskTempImg_       = cv::imread(maskTempImgPath_, 1);
    iou_       = xx::GetProperty<float>(param_json["param"], "iou", 0.3);
    areaRatio_ = xx::GetProperty<float>(param_json["param"], "areaRatio", 0.15);
}

void AlgoA::CropTaskImg(const cv::Mat& input_img)
{
    // LOGI("exec CropTaskImg function!");
    cv::Mat gray_img;
    cv::Mat dst;
    cv::Mat dis;

    std::vector<CropSt> crop_vec;
    cv::FileStorage     fs(yml_, cv::FileStorage::READ);
    fs["template_width"] >> bw;
    fs["template_height"] >> bh;

    dis = input_img.clone();
    cv::Rect detect_rect(detect_left_x_, detect_left_y_, detect_width_, detect_height_);
    cv::Mat  detect_img = input_img(detect_rect);
    if (detect_img.channels() > 1)
        cv::cvtColor(input_img, gray_img, cv::COLOR_BGR2GRAY);
    else
        gray_img = input_img;
    cv::resize(detect_img, dst, cv::Size(detect_img.cols / 2, detect_img.rows / 2));
    nlohmann::json            match_params = {{"AngleMin", angle_min_},
                                              {"AngleMax", angle_max_},
                                              {"MinScore", min_score_},
                                              {"ScaleMin", scale_min_},
                                              {"ScaleMax", scale_max_},
                                              {"Contrast", contrast_},
                                              {"SortByScore", sort_by_score_},
                                              {"MaxOverlap", max_overlap_},
                                              {"Strength", strength_},
                                              {"Num", num_}};
    Tival::ShapeBasedMatching sbm;
    sbm.Load(path_);
    bool ret_status = sbm.IsLoaded();
    LOGI("sbm load status:{}", ret_status);
    Tival::SbmResults ret = sbm.Find(dst, match_params);
    LOGI("sbm result size:{}", ret.score.size());
    //! 坐标转换
    int retFinalNum = (std::min)(num_, int(ret.center.size()));
    for (int m = 0; m < retFinalNum; m++) {
        int             sx     = ret.center[m].x * 2 + bw / 2.0;
        int             sy     = ret.center[m].y * 2 + bh / 2.0;
        double          sangle = ret.angle[m];
        double          sscore = ret.score[m];
        double          sscale = ret.scale[m];
        cv::RotatedRect rotate_rect(cv::Point2d(sx + detect_left_x_, sy + detect_left_y_), cv::Size(bw, bh), -sangle * 180 / CV_PI);
        cv::Point2f     vertex[4];
        rotate_rect.points(vertex);
        std::vector<cv::Point2f> pt_vec;
        pt_vec.push_back(vertex[0]);
        pt_vec.push_back(vertex[1]);
        pt_vec.push_back(vertex[2]);
        pt_vec.push_back(vertex[3]);
        pt_vec = xx::order_pts(pt_vec);
#ifdef DRAW
        for (size_t j = 0; j < 4; j++)
            cv::line(dis, pt_vec[j], pt_vec[(j + 1) % 4], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
#endif
        cv::Rect box(pt_vec[0], pt_vec[2]);

        // 计算模块所属的行数
        float rowHeight = detect_height_ / product_rows_;
        int   curRow    = ((box.y + box.height / 2.0 - detect_left_y_) - 1) / rowHeight;

        cv::Mat cropImgP, invertMat;
        if (enable_affineImg_) {
            cropImgP = PerspectTransform(input_img, pt_vec, box, invertMat);
        }
        else {
            cropImgP = input_img(box);
        }
        // 用于绘制最小外接矩形
        nlohmann::json rect_json = xx::pt_json(pt_vec);
        crop_vec.emplace_back(CropSt(cropImgP, pt_vec[0], sscore, box, rect_json, curRow, invertMat));
    }
    std::sort(crop_vec.begin(), crop_vec.end(), [&](const CropSt& lhs, const CropSt& rhs) {
        return (abs(lhs.pt.y - rhs.pt.y) <= 150) ? (lhs.pt.x < rhs.pt.x ? true : false) : (lhs.pt.y < rhs.pt.y ? true : false);
    });
    for (int i = 0; i < crop_vec.size(); i++) {
        crop_vec[i].index = i;
        crop_vec[i].row   = i / product_cols_;
    }
    crop_img_vec_ = std::move(crop_vec);
    // 存小图样本
    if (enable_saveSample_) {
        for (int i = 0; i < crop_img_vec_.size(); i++) {
            // 分模块位置分类
            std::vector<cv::Rect> partRect;
            if (crop_img_vec_[i].row % 2 == 0) {
                partRect = crop_rect_0;
            }
            else {
                partRect = crop_rect_1;
            }
            int modelid = 0;
            for (int p = 0; p < partRect.size(); p++) {
                std::string savePath;
                // 截图
                cv::Rect boxCrop = partRect[p];
                cv::Mat  cropImg = crop_img_vec_[i].img(boxCrop);
                // 绘制top框
                cv::Mat drawImg = crop_img_vec_[i].img.clone();
                cv::rectangle(drawImg, boxCrop, cv::Scalar(0, 0, 255), 1, 8);
                int b = 1;
                // 模型编号: 1:top, 2:lr, 3:in
                if (p % 2 == 0) {
                    modelid++;
                }
                if (modelid == 1) {
                    savePath = sampleSavePath_ + "lr\\" + imgName_ + "_" + std::to_string(i) + "_" + std::to_string(p) + ".jpg";
                }
                else if (modelid == 2) {
                    savePath = sampleSavePath_ + "in\\" + imgName_ + "_" + std::to_string(i) + "_" + std::to_string(p) + ".jpg";
                }
                else {
                    savePath = sampleSavePath_ + "top\\" + imgName_ + "_" + std::to_string(i) + "_" + std::to_string(p) + ".jpg";
                }
                cv::imwrite(savePath, cropImg);
            }
        }
    }
    return;
}

cv::Mat draw_rst_withmask(cv::Mat image, const std::vector<std::vector<cv::Point>>& mask, cv::Scalar color, cv::Point pt_offset)
{
    cv::drawContours(image, mask, -1, color, 1, 8, {}, {}, pt_offset);
    return image;
}

void AlgoA::Infer(InferTaskPtr task, AlgoResultPtr algo_result)
{
    // LOGI("exec Infer function!");
#ifdef DRAW
    cv::Mat display = task->image.clone();
#endif
    AIRuntimeInterface*  ai_obj = GetAIRuntime();
    std::vector<lc_info> lv_vec(crop_img_vec_.size());
    // 多batch推理 batch_num_
    int lastlistNum = crop_img_vec_.size() % batch_num_;
    int listNum     = 1;

    int loopNum = std::ceil(float(crop_img_vec_.size()) / batch_num_);
    for (int loop = 0; loop < loopNum; loop++) {

        int start = loop * batch_num_;
        int end   = (std::min)(static_cast<int>(crop_img_vec_.size()), start + batch_num_);

        std::vector<cv::Mat>        cropImgList;
        std::vector<int>            cropRowList;
        std::vector<int>            cropIndexList;
        std::vector<cv::Point2d>    cropLtPointList;
        std::vector<nlohmann::json> cropRotateRectList;
        std::vector<cv::Mat>        cropInvertMatList;

        for (int k = start; k < end; k++) {
            cropImgList.push_back(crop_img_vec_[k].img);
            cropRowList.push_back(crop_img_vec_[k].row);
            cropIndexList.push_back(crop_img_vec_[k].index);
            cropLtPointList.push_back(crop_img_vec_[k].pt);
            cropRotateRectList.push_back(crop_img_vec_[k].rotateRect);
            cropInvertMatList.push_back(crop_img_vec_[k].invertMat);
        }
        // --------- 执行分类模型----------//
        std::vector<std::pair<bool, std::vector<bool>>> classify_isOKList = CheckShapeByClassify(cropImgList, cropRowList, cropIndexList);
        for (int m = 0; m < classify_isOKList.size(); m++) {
            lc_info lc;
            lc.classify_statusList = classify_isOKList[m].second;
            lc.index               = cropIndexList[m];
            lc.lt                  = cropLtPointList[m];
            lc.rotateRect          = cropRotateRectList[m];
            lc.img                 = cropImgList[m];
            lc.row                 = cropRowList[m];
            lc.invertMat           = cropInvertMatList[m];

            if (classify_isOKList[m].first == false) {
                lc.classify_status = false;
                LOGD("index: {}, classify model predict NG", cropIndexList[m]);
            }
            else {
                lc.classify_status = true;
            }
            lv_vec[cropIndexList[m]] = lc;
        }
        // -----------------------------//

        // --------执行小图分割模型----------//
        TaskInfoPtr _task    = std::make_shared<stTaskInfo>();
        _task->imageData     = {cropImgList};
        _task->modelId       = 0;
        _task->taskId        = 0;
        _task->promiseResult = new std::promise<ModelResultPtr>();
        ai_obj->CommitInferTask(_task);

        std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(_task->promiseResult);
        std::future<ModelResultPtr>   futureRst     = promiseResult->get_future();
        ModelResultPtr                rst           = futureRst.get();


        // 解析分割模型结果
        std::vector objectName = {"top", "lr", "in"};
        for (int i = 0; i < rst->itemList.size(); i++) {
            std::vector<std::vector<cv::Point>> top_pt;
            std::vector<std::vector<cv::Point>> side_pt;
            std::vector<std::vector<cv::Point>> center_pt;
            for (auto& box : rst->itemList[i]) {
                if (box.confidence <= segthreshold_) {
                    LOGD("index: {} object {} filter by confidence {}", cropIndexList[i], objectName[box.code], box.confidence);
                    continue;
                }
                if (0 == box.code) {
                    top_pt.emplace_back(box.mask[0]);
                }
                if (1 == box.code) {
                    side_pt.emplace_back(box.mask[0]);
                }
                if (2 == box.code) {
                    center_pt.emplace_back(box.mask[0]);
                }
            }
            // 逐个图像解析
            if (lv_vec[cropIndexList[i]].classify_status) {
                bool status = true;
                if (enable_topMeasure_) {
                    status &= TopComputer(top_pt, cropImgList[i], lv_vec[cropIndexList[i]]);
                }
                if (enable_lrMeasure_) {
                    status &= SideComputer(side_pt, cropImgList[i], lv_vec[cropIndexList[i]]);
                }

                if (enable_inMeasure_) {
                    status &= CenterComputer(center_pt, cropImgList[i], lv_vec[cropIndexList[i]]);
                }
                lv_vec[cropIndexList[i]].segment_status = status;

                for (int p = 0; p < top_pt.size(); p++) {
                    // mask逆变换
                    InvertMask(top_pt[p], lv_vec[cropIndexList[i]].invertMat);
                    std::vector<std::vector<cv::Point>> tmp;
                    tmp.push_back(top_pt[p]);
#ifdef DRAW
                    cv::RNG    rng(0);
                    cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                    draw_rst_withmask(display, tmp, color, crop_img_vec_[cropIndexList[i]].pt);
#endif
                    nlohmann::json top_pt_json = xx::pt_json(tmp);
                    algo_result->result_info.push_back(
                        {{"label", 0}, {"shapeType", "polygon"}, {"masks", top_pt_json}, {"result", 1}, {"points", {{int(lv_vec[cropIndexList[i]].lt.x), int(lv_vec[cropIndexList[i]].lt.y)}}}});
                }

                for (int p = 0; p < side_pt.size(); p++) {
                    InvertMask(side_pt[p], lv_vec[cropIndexList[i]].invertMat);
                    std::vector<std::vector<cv::Point>> tmp;
                    tmp.push_back(side_pt[p]);
#ifdef DRAW
                    cv::RNG    rng(1);
                    cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                    draw_rst_withmask(display, tmp, color, crop_img_vec_[cropIndexList[i]].pt);
#endif
                    nlohmann::json side_pt_json = xx::pt_json(tmp);
                    algo_result->result_info.push_back(
                        {{"label", 1}, {"shapeType", "polygon"}, {"masks", side_pt_json}, {"result", 1}, {"points", {{int(lv_vec[cropIndexList[i]].lt.x), int(lv_vec[cropIndexList[i]].lt.y)}}}});
                }
                for (int p = 0; p < center_pt.size(); p++) {
                    InvertMask(center_pt[p], lv_vec[cropIndexList[i]].invertMat);
                    std::vector<std::vector<cv::Point>> tmp;
                    tmp.push_back(center_pt[p]);
#ifdef DRAW
                    cv::RNG    rng(2);
                    cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                    draw_rst_withmask(display, tmp, color, crop_img_vec_[cropIndexList[i]].pt);
#endif
                    nlohmann::json center_pt_json = xx::pt_json(tmp);
                    algo_result->result_info.push_back(
                        {{"label", 2}, {"shapeType", "polygon"}, {"masks", center_pt_json}, {"result", 1}, {"points", {{int(lv_vec[cropIndexList[i]].lt.x), int(lv_vec[cropIndexList[i]].lt.y)}}}});
                }
            }
            // 绘制模型判断结果
#ifdef DRAW
            if (!lv_vec[cropIndexList[i]].segment_status || !lv_vec[cropIndexList[i]].classify_status) {
                cv::rectangle(display, cv::Rect(lv_vec[cropIndexList[i]].lt.x, lv_vec[cropIndexList[i]].lt.y, bw, bh), cv::Scalar(0, 0, 255), 1, 8);
            }
            else {
                cv::rectangle(display, cv::Rect(lv_vec[cropIndexList[i]].lt.x, lv_vec[cropIndexList[i]].lt.y, bw, bh), cv::Scalar(0, 255, 0), 1, 8);
            }
#endif
        }
        delete _task->promiseResult;
    }
    bool fina_ret = FinalJudge(lv_vec, algo_result);
}

float AlgoA::ComputeIoU(const cv::Rect& rect1, const cv::Rect& rect2)
{
    // 检查是否有包含关系
    if ((rect1.x <= rect2.x && rect1.y <= rect2.y && rect1.x + rect1.width >= rect2.x + rect2.width && rect1.y + rect1.height >= rect2.y + rect2.height) ||
        (rect2.x <= rect1.x && rect2.y <= rect1.y && rect2.x + rect2.width >= rect1.x + rect1.width && rect2.y + rect2.height >= rect1.y + rect1.height)) {
        // 如果存在包含关系，返回IoU为1
        return 1.0f;
    }
    // 计算交集矩形
    int x1 = max(rect1.x, rect2.x);
    int y1 = max(rect1.y, rect2.y);
    int x2 = min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = min(rect1.y + rect1.height, rect2.y + rect2.height);
    // 计算交集的宽度和高度
    int width  = max(0, x2 - x1);
    int height = max(0, y2 - y1);
    // 交集面积
    int intersectionArea = width * height;
    if (intersectionArea == 0) {
        return 0;
    }
    int area1 = rect1.width * rect1.height;
    int area2 = rect2.width * rect2.height;
    // 计算并集面积
    int unionArea = area1 + area2 - intersectionArea;
    return (float)intersectionArea / unionArea;
}

void AlgoA::InvertMask(std::vector<cv::Point>& mask, cv::Mat& invertMat)
{
    if (enable_affineImg_) {
        if (invertMat.cols != 0 && invertMat.rows != 0) {
            // mask点逆变换
            std::vector<cv::Point> rstMask;
            for (int i = 0; i < mask.size(); i++) {
                cv::Point pnt = mask[i];

                cv::Mat_<float> pointHomogeneous = (cv::Mat_<float>(3, 1) << pnt.x, pnt.y, 1);
                pointHomogeneous.convertTo(pointHomogeneous, CV_32F);
                cv::Mat transformedPoint = invertMat * pointHomogeneous;
                pnt.x                    = int(transformedPoint.at<float>(0, 0) / transformedPoint.at<float>(2, 0));
                pnt.y                    = int(transformedPoint.at<float>(1, 0) / transformedPoint.at<float>(2, 0));
                rstMask.push_back(pnt);
            }
            mask = rstMask;
        }
        else {
            return;
        }
    }
    else {
        return;
    }
}

std::vector<std::vector<cv::Point>> AlgoA::FilterMask(std::vector<std::vector<cv::Point>>& mask, lc_info& lc, int pos)
{
    std::vector<std::vector<cv::Point>> rstMask;
    // 位置  大小
    int                   row = lc.row;
    std::vector<cv::Rect> partRect;
    if (row % 2 == 0) {
        partRect = crop_rect_0;
    }
    else {
        partRect = crop_rect_1;
    }
    std::vector<cv::Rect> posRect;
    switch (pos) {
    // 1:lr 2:in 3:top
    case 1:
        posRect.push_back(partRect[0]);
        posRect.push_back(partRect[1]);
        break;
    case 2:
        posRect.push_back(partRect[2]);
        posRect.push_back(partRect[3]);
        break;
    case 3:
        posRect.push_back(partRect[4]);
        posRect.push_back(partRect[5]);
        break;
    default:
        break;
    }
    for (int i = 0; i < mask.size(); i++) {
        std::vector<cv::Point> tempMask = mask[i];
        cv::Rect               maskRect = cv::boundingRect(tempMask);
        float                  maskArea = maskRect.width * maskRect.height;
        for (int j = 0; j < posRect.size(); j++) {
            cv::Rect tempRect = posRect[j];
            float    rectArea = tempRect.width * tempRect.height;
            // 计算iou
            float iou = ComputeIoU(maskRect, tempRect);
            // 计算面积比
            float areaRatio = float(maskArea) / rectArea;
            // 满足条件
            if (iou > iou_ && areaRatio > areaRatio_) {
                rstMask.push_back(tempMask);
                LOGD("index:{} push back mask type {} ,iou:{}, areaRatio:{}", lc.index, pos, iou, areaRatio);
                break;
            }
            else {
                LOGD("index:{} filter mask type {} ,iou:{}, areaRatio:{}", lc.index, pos, iou, areaRatio);
            }
        }
    }
    return rstMask;
}


cv::Mat AlgoA::PerspectTransform(const cv::Mat& image, std::vector<cv::Point2f>& srcPoints, cv::Rect& box, cv::Mat& invertMat)
{
    cv::Mat src = image(box).clone();
    if (srcPoints.size() < 4) {
        return src;
    }
    // 源点
    std::vector<cv::Point2f> srcPnts;
    for (int i = 0; i < srcPoints.size(); i++) {
        srcPnts.emplace_back((std::max)(srcPoints[i].x - box.x, 0.f), (std::max)(srcPoints[i].y - box.y, 0.f));
    }
    // 终点
    std::vector<cv::Point2f> dstPoints;
    dstPoints.push_back(cv::Point2f(0, 0));
    dstPoints.push_back(cv::Point2f(bw, 0));
    dstPoints.push_back(cv::Point2f(bw, bh));
    dstPoints.push_back(cv::Point2f(0, bh));
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPnts, dstPoints);

    cv::Mat result;
    cv::warpPerspective(src, result, perspectiveMatrix, cv::Size(bw, bh));

    // 矩阵求逆
    try {
        perspectiveMatrix.convertTo(perspectiveMatrix, CV_32FC1);
        cv::invert(perspectiveMatrix, invertMat);
    }
    catch (...) {
        LOGI("invert mat fail");
    }
    return result;
}

std::vector<std::pair<bool, std::vector<bool>>> AlgoA::CheckShapeByClassify(std::vector<cv::Mat>& tmpImgList, std::vector<int>& rowList, std::vector<int>& indexList)
{
    // 多batch图像组合
    std::vector<std::vector<cv::Mat>> toAIImgList(int(crop_rect_0.size() / 2.0));

    for (int b = 0; b < tmpImgList.size(); b++) {
        int     row    = rowList[b];
        cv::Mat tmpImg = tmpImgList[b];
        // 分模块位置分类
        std::vector<cv::Rect> partRect;
        if (row % 2 == 0) {
            partRect = crop_rect_0;
        }
        else {
            partRect = crop_rect_1;
        }
        int modelid = 0;
        for (int i = 0; i < partRect.size(); i++) {
            // 截图
            cv::Rect box     = partRect[i];
            cv::Mat  cropImg = tmpImg(box).clone();

            // 模型编号: 1:lr, 2:in, 3:top
            if (i % 2 == 0) {
                ++modelid;
            }
            switch (modelid) {
            case 1:
                toAIImgList[0].push_back(cropImg);
                break;
            case 2:
                toAIImgList[1].push_back(cropImg);
                break;
            case 3:
                toAIImgList[2].push_back(cropImg);
                break;
            default:
                break;
            }
        }
    }
    // 多batch推理
    std::vector<std::string>       labelList = {"lr", "in", "top"};
    std::vector<std::vector<bool>> tempRstList;
    for (int k = 0; k < toAIImgList.size(); k++) {
        int               modelid         = k + 1;
        double            classifyConfThr = 1;
        std::vector<bool> isOK(toAIImgList[k].size(), true);

        switch (k) {
        case 0:
            classifyConfThr = lrClsConf_;
            if (!enable_lrClassify_)
                continue;
            break;
        case 1:
            classifyConfThr = inClsConf_;
            if (!enable_inClassify_)
                continue;
            break;
        case 2:
            classifyConfThr = topClsConf_;
            if (!enable_topClassify_)
                continue;
            break;
        default:
            break;
        }
        TaskInfoPtr _cls_task       = std::make_shared<stTaskInfo>();
        _cls_task->imageData        = {toAIImgList[k]};
        _cls_task->modelId          = modelid;
        _cls_task->taskId           = 0;
        ModelResultPtr clsResultPtr = GetAIRuntime()->RunInferTask(_cls_task);
        if (clsResultPtr->itemList.size() == 0) {
            std::fill(isOK.begin(), isOK.end(), false);
            tempRstList.push_back(isOK);
            LOGI("classify predict no result!");
            continue;
        }
        for (int m = 0; m < toAIImgList[k].size(); m++) {
            auto clsRstList = clsResultPtr->itemList[m];
            if (clsRstList.size() == 0) {
                isOK[m] = false;
                LOGI("classify predict label {} no result!", labelList[k]);
                continue;
            }
            int   code = clsRstList[0].code;   // 1: OK, 0:NG
            float conf = clsRstList[0].confidence;
            // 如果模型判断为NG, 且分数大于设定阈值（参数配置），则判为false
            if (code == 0 && conf >= classifyConfThr) {
                isOK[m] = false;
                LOGD("classify predict index:{},type:{} is NG, conf is:{}", indexList[m / 2], labelList[k], conf);
            }
            else {
                isOK[m] = true;
            }
        }
        tempRstList.push_back(isOK);
    }
    // 解析结果: 按照Batch输出
    std::vector<std::pair<bool, std::vector<bool>>> isOKList;
    std::vector<std::vector<bool>>                  batchRst(tmpImgList.size());

    for (int y = 0; y < tempRstList.size(); y++) {
        for (int z = 0; z < tempRstList[y].size(); z = z + 2) {
            batchRst[z / 2].push_back(tempRstList[y][z]);
            batchRst[z / 2].push_back(tempRstList[y][z + 1]);
        }
    }
    for (int z = 0; z < batchRst.size(); z++) {
        auto it = std::find(batchRst[z].begin(), batchRst[z].end(), false);
        if (it != batchRst[z].end()) {
            isOKList.emplace_back(false, batchRst[z]);
        }
        else {
            isOKList.emplace_back(true, batchRst[z]);
        }
    }
    return isOKList;
}

/*
根据mask 模板的取值，不同mask模板有不同的值
顶部弹片，高度基准26像素的高度
cv::Vec4i(min_x, min_y, max_x, max_y);
*/
bool AlgoA::TopComputer(std::vector<std::vector<cv::Point>>& mask, cv::Mat img, lc_info& lc)
{
    // 多余的mask
    std::vector<std::vector<cv::Point>> finalMask;
    if (mask.size() > 2) {
        finalMask = FilterMask(mask, lc, 3);
    }
    else {
        finalMask = mask;
    }
    mask = finalMask;

    if (finalMask.size() < 2) {
        LOGD("Index:{}, segment model predict top NG", lc.index);
        return false;
    }


    std::sort(finalMask.begin(), finalMask.end(), [&](const std::vector<cv::Point>& lhs, const std::vector<cv::Point>& rhs) { return xx::get_center(lhs).x < xx::get_center(rhs).x ? true : false; });
    //! 计算距离 v1 左侧 v2右侧
    if (finalMask.size() == 2) {
        cv::Vec4i v1 = xx::get_value(finalMask[0]);
        cv::Vec4i v2 = xx::get_value(finalMask[1]);
        lc.f1        = (v1[3] - t_y) * pix_value_ / 1000;
        lc.f2        = (v2[3] - t_y) * pix_value_ / 1000;
        if (lc.f1 < 0)
            lc.f1 = 0;
        if (lc.f2 < 0)
            lc.f2 = 0;
        return true;
    }
    else {
        LOGD("Index:{}, segment model predict top NG", lc.index);
        return false;
    }
}

bool AlgoA::SideComputer(std::vector<std::vector<cv::Point>> mask, cv::Mat img, lc_info& lc)
{
    if (mask.size() < 2) {
        LOGD("Index:{}, segment model predict side NG", lc.index);
        return false;
    }
    std::sort(mask.begin(), mask.end(), [&](const std::vector<cv::Point>& lhs, const std::vector<cv::Point>& rhs) { return xx::get_center(lhs).x < xx::get_center(rhs).x ? true : false; });
    //! 计算距离
    if (mask.size() == 2) {
        cv::Vec4i v1 = xx::get_value(mask[0]);
        cv::Vec4i v2 = xx::get_value(mask[1]);
        lc.e1        = (v1[1] - ll_y1) * pix_value_ / 1000;
        lc.p1        = -(v1[3] - ll_y2) * pix_value_ / 1000;
        lc.e2        = (v2[1] - rr_y1) * pix_value_ / 1000;
        lc.p2        = -(v2[3] - rr_y2) * pix_value_ / 1000;
        if (lc.e1 < 0)
            lc.e1 = 0;
        if (lc.p1 < 0)
            lc.p1 = 0;
        if (lc.e2 < 0)
            lc.e2 = 0;
        if (lc.p2 < 0)
            lc.p2 = 0;
        return true;
    }
    else {
        LOGD("Index:{}, segment model predict side NG", lc.index);
        return false;
    }
}

bool AlgoA::CenterComputer(std::vector<std::vector<cv::Point>> mask, cv::Mat img, lc_info& lc)
{
    if (mask.size() < center_count_) {
        LOGD("Index:{}, segment model predict center NG", lc.index);
        return false;
    }
    std::sort(mask.begin(), mask.end(), [&](const std::vector<cv::Point>& lhs, const std::vector<cv::Point>& rhs) { return xx::get_center(lhs).x < xx::get_center(rhs).x ? true : false; });
    if (mask.size() == 4) {
        cv::Vec4i v1 = xx::get_value(mask[0]);
        cv::Vec4i v2 = xx::get_value(mask[1]);
        cv::Vec4i v3 = xx::get_value(mask[2]);
        cv::Vec4i v4 = xx::get_value(mask[3]);
        lc.a1        = std::abs((v1[2] + v2[0]) / 2 - (lc_x1 + lc_x2) / 2) * pix_value_ / 1000;
        lc.a2        = std::abs((v3[2] + v4[0]) / 2 - (rc_x1 + rc_x2) / 2) * pix_value_ / 1000;
        lc.c11       = (v1[1] - lc_y1) * pix_value_ / 1000;
        lc.d11       = -(v1[3] - lc_y2) * pix_value_ / 1000;
        lc.c12       = (v2[1] - lc_y1) * pix_value_ / 1000;
        lc.d12       = -(v2[3] - lc_y2) * pix_value_ / 1000;
        lc.c21       = (v3[1] - rc_y1) * pix_value_ / 1000;
        lc.d21       = -(v3[3] - rc_y2) * pix_value_ / 1000;
        lc.c22       = (v4[1] - rc_y1) * pix_value_ / 1000;
        lc.d22       = -(v4[3] - rc_y2) * pix_value_ / 1000;
        lc.b11       = (v2[0] - v1[2]) * pix_value_ / 1000;
        lc.b12       = (v2[0] - v1[2]) * pix_value_ / 1000;
        lc.b13       = (v2[0] - v1[2]) * pix_value_ / 1000;
        lc.b21       = (v4[0] - v3[2]) * pix_value_ / 1000;
        lc.b22       = (v4[0] - v3[2]) * pix_value_ / 1000;
        lc.b23       = (v4[0] - v3[2]) * pix_value_ / 1000;
        if (lc.c11 < 0)
            lc.c11 = 0;
        if (lc.d11 < 0)
            lc.d11 = 0;
        if (lc.c12 < 0)
            lc.c12 = 0;
        if (lc.d12 < 0)
            lc.d12 = 0;
        if (lc.c21 < 0)
            lc.c21 = 0;
        if (lc.d21 < 0)
            lc.d21 = 0;
        if (lc.c22 < 0)
            lc.c22 = 0;
        if (lc.d22 < 0)
            lc.d22 = 0;
        if (lc.b11 < 0)
            lc.b11 = 0;
        if (lc.b12 < 0)
            lc.b12 = 0;
        if (lc.b13 < 0)
            lc.b13 = 0;
        if (lc.b21 < 0)
            lc.b21 = 0;
        if (lc.b22 < 0)
            lc.b22 = 0;
        if (lc.b23 < 0)
            lc.b23 = 0;
        return true;
    }
    else if (mask.size() == 2) {
        cv::Vec4i v1 = xx::get_value(mask[0]);
        cv::Vec4i v2 = xx::get_value(mask[1]);
        lc.c11       = (v1[1] - lc_y1) * pix_value_ / 1000;
        lc.d11       = (v1[3] - lc_y2) * pix_value_ / 1000;
        lc.c21       = (v2[1] - rc_y1) * pix_value_ / 1000;
        lc.d21       = (v2[3] - rc_y2) * pix_value_ / 1000;
        if (lc.c11 < 0)
            lc.c11 = 0;
        if (lc.d11 < 0)
            lc.d11 = 0;
        if (lc.c21 < 0)
            lc.c21 = 0;
        if (lc.d21 < 0)
            lc.d21 = 0;
    }
    else {
        LOGD("Index:{}, segment model predict center NG", lc.index);
        return false;
    }
}

bool AlgoA::FinalJudge(std::vector<lc_info> lc_vec, AlgoResultPtr algo_result)
{
    bool status_flag = true;
    for (int i = 0; i < lc_vec.size(); i++) {
        // 保存NG图
        if (enable_saveNGImg_) {
            if (!lc_vec[i].classify_status) {
                std::string saveNGPath = NGSavePath_ + "classify/" + imgName_ + "_" + std::to_string(i) + ".jpg";
                cv::imwrite(saveNGPath, lc_vec[i].img);
            }
            if (!lc_vec[i].segment_status) {
                std::string saveNGPath = NGSavePath_ + "segment/" + imgName_ + "_" + std::to_string(i) + ".jpg";
                cv::imwrite(saveNGPath, lc_vec[i].img);
            }
        }

        if (lc_vec[i].classify_status == false || lc_vec[i].segment_status == false) {
            status_flag               = false;
            algo_result->judge_result = 0;
            // 绘制NG框信息
            algo_result->result_info.push_back(
                {{"label", "W_Female_Detect_defect_1"},
                 {"shapeType", "basis"},
                 {"points", {{-1, -1}}},
                 {"result", {{"dist", {}}, {"status", {0}}, {"error", {}}, {"index", (int)lc_vec[i].index}, {"points", lc_vec[i].rotateRect}, {"width", bw}, {"height", bh}}}});

            continue;
        }

        double e_a1  = abs(lc_vec[i].a1 - a_);
        double e_b11 = abs(lc_vec[i].b11 - b1_);
        double e_b12 = abs(lc_vec[i].b12 - b2_);
        double e_b13 = abs(lc_vec[i].b13 - b3_);
        double e_c11 = abs(lc_vec[i].c11 - c1_);
        double e_c12 = abs(lc_vec[i].c12 - c2_);
        double e_d11 = abs(lc_vec[i].d11 - d1_);
        double e_d12 = abs(lc_vec[i].d12 - d2_);
        double e_e1  = abs(lc_vec[i].e1 - e_);
        double e_p1  = abs(lc_vec[i].p1 - p_);
        double e_f1  = abs(lc_vec[i].f1 - f_);

        double e_a2  = abs(lc_vec[i].a2 - a_);
        double e_b21 = abs(lc_vec[i].b21 - b1_);
        double e_b22 = abs(lc_vec[i].b22 - b2_);
        double e_b23 = abs(lc_vec[i].b23 - b3_);
        double e_c21 = abs(lc_vec[i].c21 - c1_);
        double e_c22 = abs(lc_vec[i].c22 - c2_);
        double e_d21 = abs(lc_vec[i].d21 - d1_);
        double e_d22 = abs(lc_vec[i].d22 - d2_);
        double e_e2  = abs(lc_vec[i].e2 - e_);
        double e_p2  = abs(lc_vec[i].p2 - p_);
        double e_f2  = abs(lc_vec[i].f2 - f_);

        double status_al  = e_a1 <= error_a_ ? 1 : 0;
        double status_b11 = e_b11 <= error_b1_ ? 1 : 0;
        double status_b12 = e_b12 <= error_b2_ ? 1 : 0;
        double status_b13 = e_b13 <= error_b3_ ? 1 : 0;
        double status_c11 = e_c11 <= error_c1_ ? 1 : 0;
        double status_c12 = e_c12 <= error_c2_ ? 1 : 0;
        double status_d11 = e_d11 <= error_d1_ ? 1 : 0;
        double status_d12 = e_d12 <= error_d2_ ? 1 : 0;
        double status_e1  = e_e1 <= error_e_ ? 1 : 0;
        double status_p1  = e_p1 <= error_p_ ? 1 : 0;
        double status_f1  = e_f1 <= error_f_ ? 1 : 0;

        double status_a2  = e_a2 <= error_a_ ? 1 : 0;
        double status_b21 = e_b21 <= error_b1_ ? 1 : 0;
        double status_b22 = e_b22 <= error_b2_ ? 1 : 0;
        double status_b23 = e_b23 <= error_b3_ ? 1 : 0;
        double status_c21 = e_c21 <= error_c1_ ? 1 : 0;
        double status_c22 = e_c22 <= error_c2_ ? 1 : 0;
        double status_d21 = e_d21 <= error_d1_ ? 1 : 0;
        double status_d22 = e_d22 <= error_d2_ ? 1 : 0;
        double status_e2  = e_e2 <= error_e_ ? 1 : 0;
        double status_p2  = e_p2 <= error_p_ ? 1 : 0;
        double status_f2  = e_f2 <= error_f_ ? 1 : 0;

        if (status_al < 1 || status_b11 < 1 || status_b12 < 1 || status_b13 < 1 || status_c11 < 1 || status_c12 < 1 || status_d11 < 1 || status_d12 < 1 || status_e1 < 1 || status_p1 < 1 ||
            status_f1 < 1 || status_a2 < 1 || status_b21 < 1 || status_b22 < 1 || status_b23 < 1 || status_c21 < 1 || status_c22 < 1 || status_d21 < 1 || status_d22 < 1 || status_e2 < 1 ||
            status_p2 < 1 || status_f2 < 1) {
            status_flag = false;
        }
        else {
        }
        if (!status_flag) {
            algo_result->judge_result = 0;
        }
        else {
            if (algo_result->judge_result == 0) {}
            else {
                algo_result->judge_result = 1;
            }
        }
        // 绘制数值信息
        algo_result->result_info.push_back(
            {{"label", "W_Female_Detect_defect_1"},
             {"shapeType", "basis"},
             {"points", {{-1, -1}}},
             {"result",
              {{"dist", {lc_vec[i].a1, lc_vec[i].b11, lc_vec[i].b12, lc_vec[i].b13, lc_vec[i].c11, lc_vec[i].c12, lc_vec[i].d11, lc_vec[i].d12, lc_vec[i].e1, lc_vec[i].p1, lc_vec[i].f1,
                         lc_vec[i].a2, lc_vec[i].b21, lc_vec[i].b22, lc_vec[i].b23, lc_vec[i].c21, lc_vec[i].c22, lc_vec[i].d21, lc_vec[i].d22, lc_vec[i].e2, lc_vec[i].p2, lc_vec[i].f2}},
               {"status", {status_al, status_b11, status_b12, status_b13, status_c11, status_c12, status_d11, status_d12, status_e1, status_p1, status_f1,
                           status_a2, status_b21, status_b22, status_b23, status_c21, status_c22, status_d21, status_d22, status_e2, status_p2, status_f2}},
               {"error", {e_a1, e_b11, e_b12, e_b13, e_c11, e_c12, e_d11, e_d12, e_e1, e_p1, e_f1, e_a2, e_b21, e_b22, e_b23, e_c21, e_c22, e_d21, e_d22, e_e2, e_p2, e_f2}},
               {"index", (int)lc_vec[i].index},
               {"points", {{int(lc_vec[i].lt.x), int(lc_vec[i].lt.y)}}}}}});
    }
    return status_flag;
}
