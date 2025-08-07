//
// Created by y on 24-5-27.
//
#include "alg_heamo.h"
// #include "DihLog.h"
#include "algLog.h"
#include "alg_error_code.h"
#include "imgprocess.h"
#include "project_utils.h"
#include "replace_std_string.h"

#include "timecnt.h"
#include "utils.h"

#define NET_USE_TIMECNT 1

#define VOLUME_RBC_IMG_NUMS 40            // 计算红细胞体积需要处理图像数量
#define WBC_BRIGHT_FLUO_FUSION_RATE 0.5   // WBC明暗场融合比例

extern std::string heamo_save_dir;
extern AlarmParam_t g_alarm_param;
extern bool g_pla_flag;
int         g_force_model_id = 0;
// 计数
int            g_run_count = 0;
Passageway_t   g_passageway;
CurrentCount_t g_current_count;

int Heamo_ImgNormal(AiCtxID_t                           ctx_id,
                    uint32_t                            item_id,
                    AiImg_t*                            img,
                    uint32_t                            group_idx,
                    uint32_t                            chl_idx,
                    uint32_t                            view_order,
                    uint32_t                            view_idx,
                    uint32_t                            processed_idx,
                    AiImgStage_e                        stage,
                    void*                               userdata,
                    std::list<NNetResult_t>&            result,
                    const int&                          view_pair_idx,
                    const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }

    if (stage != AI_IMG_STAGE_INTERENCE) {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "null group ptr";
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }

    AiViewReg_t* view = Heamo_FindView(group, Heamo_FindChl(group, chl_idx), view_idx);
    if (view == NULL) {
        ALGLogError << "null view ptr";
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }
    if (view->mod_id == NNET_MODID_UNKOWN) {
        return ALG_SUCC;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);
    if (g_force_model_id >= NNET_MODID_PLA) {
        mod_id = static_cast<NNetModID_e>(g_force_model_id);
    }
    int ret;
    ret = Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, mod_id, img, result, AI_INFER_TYPE_NORMAL);
    if (ret) {
        ALGLogError << "failed to inference model, group:mod " << group_id << "  " << mod_id;
        return ret;
    }
    AiModReg_t* mod = Heamo_FindMod(group, mod_id);
    if (mod == NULL) {
        ALGLogError << "failed to find model " << mod_id;
        return ALG_ERR_INVALID_MODEL;
    }
    ret = Ai_ResultCount((void*)HEAMO_CTX_CNT(ctx), *AI_GROUP_CNT_LIST(group), result, AI_MOD_MULTI_LABEL_FLAG(mod));
    if (ret) {
        ALGLogError << "failed to count result of model " << mod_id;
        return ret;
    }

    // Heamo_DoImgCallback(ctx, item_id, img, group_idx, chl_idx, view_order,
    // view_idx, processed_idx, stage, result);

    return ALG_SUCC;
}

// 将rect装入box
void TransformPolyToRectPoints(const std::list<NNetResult_t>& input_v, std::vector<std::vector<cv::Point>>& rect_points_v_v)
{
    //[category, conf, x1, y1, x2, y2,...,x4, y4]
    for (const auto& input : input_v) {
        std::vector<float> polygon_v(input.polygon_v);
        std::vector<float> horizontal_points{polygon_v[2], polygon_v[4], polygon_v[6], polygon_v[8]};
        std::vector<float> vertical_points{polygon_v[3], polygon_v[5], polygon_v[7], polygon_v[9]};
        float              left   = *std::min_element(horizontal_points.begin(), horizontal_points.end());
        float              top    = *std::min_element(vertical_points.begin(), vertical_points.end());
        float              right  = *std::max_element(horizontal_points.begin(), horizontal_points.end());
        float              bottom = *std::max_element(horizontal_points.begin(), horizontal_points.end());
        rect_points_v_v.push_back({cv::Point(left, top), cv::Point(right, bottom)});
    }
}

/**
 * @brief  :  荧光微球，判断绿色通道的均值是否 大于 小于给定的值
 * @param   src
 * @param   threshold
 * @param   type   0 的时候是大于给定的值返回 1 ，否则返回0. 1的时候小于给定的值，返回1.否则返回0
 * @return
 * @note   :
 **/

int is_error(const cv::Mat& src, int threshold, int type)
{
    if (src.empty())
        return -1;
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    // 计算每个通道的均值
    cv::Scalar meanB = cv::mean(channels[0]);
    cv::Scalar meanG = cv::mean(channels[1]);
    cv::Scalar meanR = cv::mean(channels[2]);
    // 判断绿色通道
    int value = meanG[0];
    if (type == 0) {
        if (value >= threshold) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else if (type == 1) {
        if (value <= threshold) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else {
        return -1;
    }
}

// 统计红细胞体积所需参数
int Heamo_RBCVolume(AiCtxID_t                           ctx_id,
                    uint32_t                            item_id,
                    AiImg_t*                            img,
                    uint32_t                            group_idx,
                    uint32_t                            chl_idx,
                    uint32_t                            view_order,
                    uint32_t                            view_idx,
                    uint32_t                            processed_idx,
                    AiImgStage_e                        stage,
                    void*                               userdata,
                    std::list<NNetResult_t>&            result,
                    const int&                          view_pair_idx,
                    const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }
    cv::Mat target_img;
    ResizeImg(*img, target_img, cv::Size(1920, 1920), BOTTOMPAD);
    // 仅处理指定的张数 40
    if (ctx->cnt.rbc_volume_img_counts >= VOLUME_RBC_IMG_NUMS)
        return 0;
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "null group ptr";
        return -2;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    // 目前仅人支持 倾斜体积测试
    if (!(group_id & NNET_GROUP_HUMAN))
        return 0;
    // 获取倾斜检测框
    NNetModID_e det_mod_id = NNET_MODID_RBC_INCLINE_DET;
    if (Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, det_mod_id, &target_img, result, AI_INFER_TYPE_NORMAL)) {
        ALGLogError << "failed to inference model, group:mod " << group_id << "  " << det_mod_id;
        return -3;
    }
    int ret;
    ret = MakeMidResult(ctx_id,
                        item_id,
                        &target_img,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx,
                        stage,
                        userdata,
                        result,
                        view_pair_idx,
                        call_back_params,
                        false,
                        2,
                        2);
    if (ret) {
        ALGLogError << "failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
        return -4;
    }

    // 提取倾斜红细胞
    std::vector<std::vector<cv::Point>> rect_points_v_v;
    TransformPolyToRectPoints(result, rect_points_v_v);
    std::vector<cv::Mat> processed_img_v;

    std::vector<std::vector<cv::Rect>> paste_position_v_v;
    int                                crop_nums = 0;

    RbcInclineSegPreprocess(target_img, result, 1024, 1024, 15, processed_img_v, crop_nums, paste_position_v_v);
    // 临时保存
    int incline_cell_nums = crop_nums;
    result.clear();
    // 分割
    int cell_region = 0;
    for (auto processed_img : processed_img_v) {
        // 分割出区域像素个数
        NNetModID_e seg_mod_id = NNET_MODID_RBC_INCLINE_SEG;
        result.clear();
        if (Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, seg_mod_id, &processed_img, result, AI_INFER_TYPE_NORMAL)) {
            ALGLogError << "failed to inference model, group:mod " << group_id << "  " << det_mod_id;
            return -5;
        }

        cell_region += cv::sum(result.begin()->seg_v[0])[0];
        processed_idx = processed_idx + 1;
        ret           = MakeMidResult(ctx_id,
                            item_id,
                            &processed_img,
                            group_idx,
                            chl_idx,
                            view_order,
                            view_idx,
                            processed_idx,
                            stage,
                            userdata,
                            result,
                            view_pair_idx,
                            call_back_params,
                            false,
                            2,
                            2);
        if (ret) {
            ALGLogError << "failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order
                        << " " << view_idx << " " << processed_idx;
            return -6;
        }
    }

    // 统计
    ctx->cnt.incline_cell_nums += incline_cell_nums;
    ctx->cnt.incline_pixels += cell_region;
    ctx->cnt.rbc_volume_img_counts += 1;
    return 0;
}

#define ALG_FOCAL_NET_CATE_NUMS 2
int GetFocalForeground(std::list<NNetResult_t>& seg_result, std::list<NNetResult_t>& bg_fg)
{

    if (seg_result.size() != 1 || seg_result.front().seg_v.size() != ALG_FOCAL_NET_CATE_NUMS) {
        ALGLogError << "Failed to parse focal network output";
        return ALG_ERR_FOCAL_NET;
    }
    cv::Mat back_prob(seg_result.front().seg_v[0]);
    cv::Mat cell_prob(seg_result.front().seg_v[1]);
    cv::Mat pred_mask;
    pred_mask = cell_prob > back_prob;
    pred_mask.convertTo(pred_mask, CV_8UC1, 1 / 255.0);
    NNetResult_t net_result;
    net_result.seg_v.push_back(pred_mask);
    bg_fg.push_back(net_result);
    return ALG_SUCC;
}

void GetFocalBox(std::list<NNetResult_t>& bg_fg, std::list<NNetResult_t>& box_result, std::list<NNetResult_t>& threshed_box_result)
{
    cv::Mat foreground(bg_fg.front().seg_v[0]);
    cv::Mat background = 1 - foreground;
    cv::Mat local_region;
    for (const auto& iter : box_result) {
        cv::Rect box_rect(cv::Point(iter.box.left, iter.box.top), cv::Point(iter.box.right, iter.box.bottom));
        local_region = cv::Mat(background, box_rect);
        bool include_background(cv::sum(local_region)[0] > 0);
        if (!include_background) {
            threshed_box_result.push_back(iter);
        }
    }
}

// 统计球形化红细胞体积所需参数
int Heamo_RBCVolumeSpherical(AiCtxID_t                           ctx_id,
                             uint32_t                            item_id,
                             AiImg_t*                            img,
                             uint32_t                            group_idx,
                             uint32_t                            chl_idx,
                             uint32_t                            view_order,
                             uint32_t                            view_idx,
                             uint32_t                            processed_idx,
                             AiImgStage_e                        stage,
                             void*                               userdata,
                             std::list<NNetResult_t>&            result,
                             const int&                          view_pair_idx,
                             const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }

    // 仅处理指定的张数 40
    if (ctx->cnt.rbc_volume_img_counts >= VOLUME_RBC_IMG_NUMS)
        return 0;
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "null group ptr, group id, channel id " << group_idx << " " << chl_idx;
        return -2;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    // 目前仅人支持 倾斜体积测试
    if (!(group_id & NNET_GROUP_HUMAN))
        return 0;

    // 分割出区域像素个数
    NNetModID_e seg_mod_id = NNET_MODID_SPHERICAL_FOCAL;

    std::list<NNetResult_t> seg_result;
    if (Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, seg_mod_id, img, seg_result, AI_INFER_TYPE_NORMAL)) {
        ALGLogError << "Failed to inference model, group:mod " << group_id << "  " << seg_mod_id;
        return -5;
    }

    // 查找前景
    std::list<NNetResult_t> bg_fg;
    int                     ret = 0;
    ret                         = GetFocalForeground(seg_result, bg_fg);
    if (ret) {
        return ret;
    }

    // 保存在焦中间结果
    ret = MakeMidResult(ctx_id,
                        item_id,
                        img,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx,
                        stage,
                        userdata,
                        bg_fg,
                        view_pair_idx,
                        call_back_params,
                        false,
                        2,
                        2);

    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
        return ret;
    }

    // 过滤不在焦的框
    std::list<NNetResult_t> threshed_box_result;
    GetFocalBox(bg_fg, result, threshed_box_result);

    CountVolumeParameter(ctx->cnt.rbc_volume_v, threshed_box_result, img->cols, img->rows);

    // 绘制过滤后框
    processed_idx = processed_idx + 1;
    ret           = MakeMidResult(ctx_id,
                        item_id,
                        img,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx,
                        stage,
                        userdata,
                        threshed_box_result,
                        view_pair_idx,
                        call_back_params,
                        false,
                        2,
                        2);
    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
        return -6;
    }
    return 0;
}

int Heamo_CalibCount(AiCtxID_t                           ctx_id,
                     uint32_t                            item_id,
                     AiImg_t*                            img,
                     uint32_t                            group_idx,
                     uint32_t                            chl_idx,
                     uint32_t                            view_order,
                     uint32_t                            view_idx,
                     uint32_t                            processed_idx,
                     AiImgStage_e                        stage,
                     void*                               userdata,
                     std::list<NNetResult_t>&            result,
                     const int&                          view_pair_idx,
                     const std::map<std::string, float>& call_back_params)
{
    // 校准模型仅统计明场图像中的荧光微球个数
    if (view_idx != 0) {
        return 0;
    }
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "null group ptr";
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }

    AiViewReg_t* view = Heamo_FindView(group, Heamo_FindChl(group, chl_idx), view_idx);
    if (view == NULL) {
        ALGLogError << "null view ptr, group_id chl_idx view_idx" << AI_GROUP_ID(group) << " " << chl_idx << " " << view_idx;
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }
    if (view->mod_id == NNET_MODID_UNKOWN) {
        return ALG_SUCC;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);

    int ret;
    ret = Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, NNET_MODID_CALIB_COUNT, img, result, AI_INFER_TYPE_NORMAL);
    if (ret) {
        ALGLogError << "Failed to inference model, group:mod " << group_id << "  " << mod_id;
        return ret;
    }
    switch (chl_idx) {   // 将不来自不同流道的计数结果放入相应的vector中
    case 0:
        HEAMO_CTX_CNT(ctx)->heamo_rbc_nums_v.push_back((int)result.size());
        break;
    case 1:
        HEAMO_CTX_CNT(ctx)->heamo_wbc_nums_v.push_back((int)result.size());
        break;
    case 2:
        HEAMO_CTX_CNT(ctx)->heamo_baso_nums_v.push_back((int)result.size());
        break;
    case 3:
        HEAMO_CTX_CNT(ctx)->heamo_ret_nums_v.push_back((int)result.size());
        break;
    default:
        ALGLogError << "Failed to save calib count nums. chl_idx " << chl_idx;
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }

    // Heamo_DoImgCallback(ctx, item_id, img, group_idx, chl_idx, view_order,
    // view_idx, processed_idx, stage, result);

    return 0;
}

// 红细胞流道
int Heamo_ImgRbcBriImgProc(AiCtxID_t                           ctx_id,
                           uint32_t                            item_id,
                           AiImg_t*                            img,
                           uint32_t                            group_idx,
                           uint32_t                            chl_idx,
                           uint32_t                            view_order,
                           uint32_t                            view_idx,
                           uint32_t                            processed_idx,
                           AiImgStage_e                        stage,
                           void*                               userdata,
                           std::list<NNetResult_t>&            result,
                           const int&                          view_pair_idx,
                           const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "Null group ptr";
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }

    AiViewReg_t* view = Heamo_FindView(group, Heamo_FindChl(group, chl_idx), view_idx);
    if (view == NULL) {
        ALGLogError << "null view ptr, group_id chl_idx view_idx" << AI_GROUP_ID(group) << " " << chl_idx << " " << view_idx;
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }
    if (view->mod_id == NNET_MODID_UNKOWN) {
        return ALG_SUCC;
    }

    auto checkout_flag_iter = call_back_params.find("CHECK");
    if (checkout_flag_iter != call_back_params.end() && checkout_flag_iter->second) {
        ALGLogError << "红细胞明场图进入到流道试剂错误检测，不进行检测直接返回 \n";
        std::cout << "红细胞明场图进入到流道试剂错误检测，不进行检测直接返回 \n";
        return 0;
    }
    std::cout << "红细胞明场图进入检测\n";
    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);

    int rbc_nums_p = ctx->cnt.RBC;
    int plt_nums_p = ctx->cnt.PLT;
    int ret_nums_p = ctx->cnt.RET;
    // QC红细胞,使用特殊模型
    if (HEAMO_CTX_QC(ctx) && view_idx == 0) {
        mod_id = NNET_MODID_RBC_QC;
    }
    int ret;
    ret = Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, mod_id, img, result, AI_INFER_TYPE_NORMAL);
    if (ret) {
        ALGLogError << "Failed to inference model, group:mod " << group_id << "  " << mod_id;
        return ret;
    }
    AiModReg_t* mod = Heamo_FindMod(group, mod_id);

    if (mod == NULL) {
        ALGLogError << "Failed to find model " << mod_id;
        return ALG_ERR_INVALID_MODEL;
    }
    ret = Ai_ResultCount((void*)HEAMO_CTX_CNT(ctx), *AI_GROUP_CNT_LIST(group), result, AI_MOD_MULTI_LABEL_FLAG(mod));
    if (ret) {
        ALGLogError << "Failed to count result of model " << mod_id;
        return ret;
    }
    int processed_idx_local = 0;
    // 红细胞流道,明场计算rbc底部面积

    processed_idx_local = view_idx;

    ret = MakeMidResult(ctx_id,
                        item_id,
                        img,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx_local,
                        stage,
                        userdata,
                        result,
                        view_pair_idx,
                        call_back_params,
                        false,
                        1,
                        2);
    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
        return ret;
    }

    // 保存结果数量
    if (view_idx == 0) {
        ctx->cnt.heamo_rbc_nums_v.push_back(static_cast<int>(ctx->cnt.RBC - rbc_nums_p));
    }
    else {
        ctx->cnt.heamo_plt_nums_v.push_back(static_cast<int>(ctx->cnt.PLT - plt_nums_p));
        ctx->cnt.heamo_ret_nums_v.push_back(static_cast<int>(ctx->cnt.RET - ret_nums_p));
    }

    // 红细胞流道,明场进行rbc,plt体积计算
    ALGLogInfo << "RBC NUMS " << ctx->cnt.RBC;
    processed_idx_local = 1;
    // 球形化红细胞体积
    ret = Heamo_RBCVolumeSpherical(ctx_id,
                                   item_id,
                                   img,
                                   group_idx,
                                   chl_idx,
                                   view_order,
                                   view_idx,
                                   processed_idx_local,
                                   stage,
                                   userdata,
                                   result,
                                   view_pair_idx,
                                   call_back_params);
    if (ret) {
        ALGLogError << "Failed to find spherical cell located in focal region";
        return ret;
    }

    // plt体积
    result.clear();
    mod_id = NNET_MODID_PLT_VOLUME;
    ret    = Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, mod_id, img, result, AI_INFER_TYPE_NORMAL);
    if (ret) {   // 指定推理模型
        ALGLogError << "Failed to inference model, group:mod " << group_id << "  " << mod_id;
        return ret;
    }
    CountVolumeParameter(ctx->cnt.plt_volume_v, result, img->cols, img->rows);

    // 保存结果
    processed_idx_local = 3;
    ret                 = MakeMidResult(ctx_id,
                        item_id,
                        img,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx_local,
                        stage,
                        userdata,
                        result,
                        view_pair_idx,
                        call_back_params,
                        true,
                        1,
                        2);

    g_current_count.rbc_count = ctx->cnt.RBC;
    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
        return ret;
    }

    // Heamo_DoImgCallback(ctx, item_id, img, group_idx, chl_idx, view_order,
    // view_idx, processed_idx, stage, result); 装入体积相关参数

    return 0;
}

// 红细胞流道
int Heamo_ImgRbcFluImgProc(AiCtxID_t                           ctx_id,
                           uint32_t                            item_id,
                           AiImg_t*                            img,
                           uint32_t                            group_idx,
                           uint32_t                            chl_idx,
                           uint32_t                            view_order,
                           uint32_t                            view_idx,
                           uint32_t                            processed_idx,
                           AiImgStage_e                        stage,
                           void*                               userdata,
                           std::list<NNetResult_t>&            result,
                           const int&                          view_pair_idx,
                           const std::map<std::string, float>& call_back_params)
{
    processed_idx   = 0;
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "Null group ptr";
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }

    AiViewReg_t* view = Heamo_FindView(group, Heamo_FindChl(group, chl_idx), view_idx);
    if (view == NULL) {
        ALGLogError << "null view ptr, group_id chl_idx view_idx" << AI_GROUP_ID(group) << " " << chl_idx << " " << view_idx;
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }
    if (view->mod_id == NNET_MODID_UNKOWN) {
        return ALG_SUCC;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);

    int rbc_nums_p = ctx->cnt.RBC;
    int plt_nums_p = ctx->cnt.PLT;
    int ret_nums_p = ctx->cnt.RET;

    // 取明场,暗场图
    cv::Mat* img_bri = Ai_ItemGetImg(ctx_id, item_id, 0);
    cv::Mat* img_flu = Ai_ItemGetImg(ctx_id, item_id, 1);

    // 添加流道错误检测
    for (auto call_back_iter : call_back_params) {
        ALGLogInfo << "回调函数的参数： " << call_back_iter.first << " " << call_back_iter.second << "\n";
    }
    auto checkout_flag_iter = call_back_params.find("CHECK");
    if (checkout_flag_iter != call_back_params.end() && checkout_flag_iter->second) {
        ALGLogInfo << "红细胞荧光场 进入到流道试剂错误检测 \n";
        std::cout << "红细胞荧光场 进入到流道试剂错误检测 \n";
        // 白细胞试剂加到红细胞流道 ，g1c0n0 。应该是绿的， 加错后偏黑，所以阈值设置为80 ，类型是大于
        int ret = is_error(*img_flu, 80, 0);
        if (ret) {}
        else {
            // g_passageway = -1;
            g_passageway.rbc_error++;
        }
        ALGLogInfo << "红细胞荧光场流道试剂错误检测，检测完成 \n";
        return 0;
    }
    std::cout << "红细胞荧光场 进入到检测流程\n";
    cv::Mat img_target;
    int     ret = MergePltImg(*img_bri, *img_flu, img_target);
    if (ret)
        return ret;

    // 推理
    std::list<NNetResult_t> result_primary;
    ret = Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, mod_id, &img_target, result_primary, AI_INFER_TYPE_NORMAL);
    if (ret) {
        ALGLogError << "Failed to inference model, group:mod " << group_id << "  " << mod_id;
        return ret;
    }
    // 查找当前模型box类别个数,对于plt模型,reserved_float_parmas保存的各个类别对应的conf
    std::vector<float> conf_v;
    ret = Ai_GetNetReservedFloatPrams(HEAMO_CTX_AI_CTXID(ctx), group_id, mod_id, conf_v);
    if (ret)
        return ret;
    ret = CountBoxCategoryConf(result_primary, conf_v, result);
    if (ret)
        return ret;

    AiModReg_t* mod = Heamo_FindMod(group, mod_id);
    if (mod == NULL) {
        ALGLogError << "Failed to find model " << mod_id;
        return ALG_ERR_INVALID_MODEL;
    }

    ret = Ai_ResultCount((void*)HEAMO_CTX_CNT(ctx), *AI_GROUP_CNT_LIST(group), result, AI_MOD_MULTI_LABEL_FLAG(mod));
    if (ret) {
        ALGLogError << "Failed to count result of model " << mod_id;
        return ret;
    }
    int processed_idx_local = view_idx;

    // 保存结果
    ret                       = MakeMidResult(ctx_id,
                        item_id,
                        &img_target,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx_local,
                        stage,
                        userdata,
                        result,
                        view_pair_idx,
                        call_back_params,
                        true,
                        1,
                        2);
    g_current_count.rbc_count = ctx->cnt.RBC;
    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
        return ret;
    }

    // 保存结果数量
    if (view_idx == 0) {
        ctx->cnt.heamo_rbc_nums_v.push_back(static_cast<int>(ctx->cnt.RBC - rbc_nums_p));
    }
    else {
        ctx->cnt.heamo_plt_nums_v.push_back(static_cast<int>(ctx->cnt.PLT - plt_nums_p));
        ctx->cnt.heamo_ret_nums_v.push_back(static_cast<int>(ctx->cnt.RET - ret_nums_p));
    }

    return 0;
}

// 疟原虫
int Heamo_ImgPla(AiCtxID_t                           ctx_id,
                 uint32_t                            item_id,
                 AiImg_t*                            img,
                 uint32_t                            group_idx,
                 uint32_t                            chl_idx,
                 uint32_t                            view_order,
                 uint32_t                            view_idx,
                 uint32_t                            processed_idx,
                 AiImgStage_e                        stage,
                 void*                               userdata,
                 std::list<NNetResult_t>&            result,
                 const int&                          view_pair_idx,
                 const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }

    auto checkout_flag_iter = call_back_params.find("CHECK");
    if (checkout_flag_iter != call_back_params.end() && checkout_flag_iter->second) {
        ALGLogInfo << "红细胞疟原虫 进入到流道试剂错误检测，不进行检测直接返回 \n";
        return 0;
    }


    // 疟原虫细胞五分类
    if (view_idx == 0) {
        int pla_nums_p = ctx->cnt.PLA;
        int pv1_nums_p = ctx->cnt.PV1;
        int pv2_nums_p = ctx->cnt.PV2;
        int pv3_nums_p = ctx->cnt.PV3;
        int pv4_nums_p = ctx->cnt.PV4;
        int pv5_nums_p = ctx->cnt.PV5;
        int pv6_nums_p = ctx->cnt.PV6;

        result.clear();
        // 取明场,暗场图
        cv::Mat* bright_img = Ai_ItemGetImg(ctx_id, item_id, 0);
        cv::Mat* fluo_img   = Ai_ItemGetImg(ctx_id, item_id, 1);
        // 融成蓝色图片
        cv::Mat bright_fluo_merged;
        int     ret = MergePltImg(*bright_img, *fluo_img, bright_fluo_merged);

        // 再次融合，将蓝色图像与明场图像0.5 比例融合
        cv::Mat bright_blue_img;
        MergeBrightFluoImg(*bright_img, bright_fluo_merged, WBC_BRIGHT_FLUO_FUSION_RATE, bright_blue_img);
        // 使用第一个模型
        g_force_model_id = NNET_MODID_PLA;

        ret = Heamo_ImgNormal(ctx_id,
                              item_id,
                              &bright_blue_img,
                              group_idx,
                              chl_idx,
                              view_order,
                              0,
                              processed_idx,
                              AI_IMG_STAGE_INTERENCE,
                              userdata,
                              result,
                              view_pair_idx,
                              call_back_params);

        if (ret != 0) {
            ALGLogError << "Failed to run Heamo_ImgPla inference";
            return -2;
        }

        int processed_idx_local = 0;
        ret                     = MakeMidResult(ctx_id,
                            item_id,
                            img,
                            group_idx,
                            chl_idx,
                            view_order,
                            view_idx,
                            processed_idx_local,
                            stage,
                            userdata,
                            result,
                            view_pair_idx,
                            call_back_params,
                            false,
                            2,
                            4);
        if (ret) {
            ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order
                        << " " << view_idx << " " << processed_idx << " " << g_force_model_id << "\n";
        }
        // 图片拼接，明场图与蓝色图拼接
        std::vector<cv::Mat> montageImg_v;
        if (!MergeImgPreProcess(montageImg_v, *bright_img, bright_fluo_merged, result)) {
            ALGLogError << "Failed to PreProcess Heamo_ImgPla";
            return -3;
        }
        for (auto& montageImg : montageImg_v) {
            std::list<NNetResult_t> wbc4_result;
            // 使用第二个模型
            g_force_model_id = NNET_MODID_PLA4;

            ret = Heamo_ImgNormal(ctx_id,
                                  item_id,
                                  &montageImg,
                                  group_idx,
                                  chl_idx,
                                  view_order,
                                  1,
                                  processed_idx,
                                  AI_IMG_STAGE_INTERENCE,
                                  userdata,
                                  wbc4_result,
                                  view_pair_idx,
                                  call_back_params);
            if (ret != 0) {
                ALGLogError << "Failed to run Heamo_ImgPla inference";
                return -4;
            }

            processed_idx_local = processed_idx_local + 1;
            // 保存结果
            ret = MakeMidResult(ctx_id,
                                item_id,
                                &montageImg,
                                group_idx,
                                chl_idx,
                                view_order,
                                view_idx,
                                processed_idx_local,
                                stage,
                                userdata,
                                wbc4_result,
                                view_pair_idx,
                                call_back_params,
                                true,
                                1,
                                2);
            if (ret) {
                ALGLogError << "Failed to make mid result ,group channel, order, view, "
                               "processed"
                            << group_idx << " " << chl_idx << " " << view_order << " " << view_idx << " " << processed_idx << " " << g_force_model_id
                            << "\n";
            }
        }
        result.clear();
        // 使用完成之后将疟原虫的模型标志位置为负数，避免影响后续的模型
        g_force_model_id = -5;
        ctx->cnt.heamo_pla_nums_v.push_back(static_cast<int>(ctx->cnt.PLA - pla_nums_p));
        ctx->cnt.heamo_pv1_nums_v.push_back(static_cast<int>(ctx->cnt.PV1 - pv1_nums_p));
        ctx->cnt.heamo_pv2_nums_v.push_back(static_cast<int>(ctx->cnt.PV2 - pv2_nums_p));
        ctx->cnt.heamo_pv3_nums_v.push_back(static_cast<int>(ctx->cnt.PV3 - pv3_nums_p));
        ctx->cnt.heamo_pv4_nums_v.push_back(static_cast<int>(ctx->cnt.PV4 - pv4_nums_p));
        ctx->cnt.heamo_pv5_nums_v.push_back(static_cast<int>(ctx->cnt.PV5 - pv5_nums_p));
        ctx->cnt.heamo_pv6_nums_v.push_back(static_cast<int>(ctx->cnt.PV6 - pv6_nums_p));
        return 0;
    }
    return 0;
}
int64_t g_nms_timie = 0;

// 红细胞流道
int Heamo_ImgRbc(AiCtxID_t                           ctx_id,
                 uint32_t                            item_id,
                 AiImg_t*                            img,
                 uint32_t                            group_idx,
                 uint32_t                            chl_idx,
                 uint32_t                            view_order,
                 uint32_t                            view_idx,
                 uint32_t                            processed_idx,
                 AiImgStage_e                        stage,
                 void*                               userdata,
                 std::list<NNetResult_t>&            result,
                 const int&                          view_pair_idx,
                 const std::map<std::string, float>& call_back_params)
{
    int ret = 0;
#if (NET_USE_TIMECNT)
    TimeCnt_Start("rbc");
#endif


    if (view_idx == 0) {
        ret = Heamo_ImgRbcBriImgProc(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
    else {
        ret = Heamo_ImgRbcFluImgProc(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
        g_run_count++;
    }
    if (g_pla_flag) {
        ret = Heamo_ImgPla(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }

#if (NET_USE_TIMECNT)
    g_nms_timie = TimeCnt_End("rbc");
#endif
    return ret;
}

int Heamo_ImgRbcChl(AiCtxID_t                           ctx_id,
                    uint32_t                            item_id,
                    AiImg_t*                            img,
                    uint32_t                            group_idx,
                    uint32_t                            chl_idx,
                    uint32_t                            view_order,
                    uint32_t                            view_idx,
                    uint32_t                            processed_idx,
                    AiImgStage_e                        stage,
                    void*                               userdata,
                    std::list<NNetResult_t>&            result,
                    const int&                          view_pair_idx,
                    const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }
    if (ctx->cnt.RBC > g_alarm_param.rab_num) {
        ALGLogInfo << "Heamo_ImgRbcChl 当前检测图片满足要求,不需要在推入图片" << "\n";
        ALGLogInfo << "Heamo_ImgRbcChl 当前的rbc计数: " << ctx->cnt.RBC << "  当前的wbc计数: " << ctx->cnt.WBC
                   << " rbc 阈值:  " << g_alarm_param.rab_num << " wbc 阈值:" << g_alarm_param.wbc_num << "\n";
        return 11;
    }

    if (HEAMO_CTX_CALIB(ctx)) {
        return Heamo_CalibCount(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
    else {
        return Heamo_ImgRbc(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
}

// 白细胞流道
int Heamo_ImgWBC(AiCtxID_t                           ctx_id,
                 uint32_t                            item_id,
                 AiImg_t*                            img,
                 uint32_t                            group_idx,
                 uint32_t                            chl_idx,
                 uint32_t                            view_order,
                 uint32_t                            view_idx,
                 uint32_t                            processed_idx,
                 AiImgStage_e                        stage,
                 void*                               userdata,
                 std::list<NNetResult_t>&            result,
                 const int&                          view_pair_idx,
                 const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }
#if (NET_USE_TIMECNT)
    TimeCnt_Start("wbc");
#endif

    if (stage != AI_IMG_STAGE_INTERENCE) {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        ALGLogInfo << "Heamo_ImgWBC 图片状态错误，退出 白细胞流程 " << "\n";
        return 0;
    }

    auto checkout_flag_iter = call_back_params.find("CHECK");
    if (checkout_flag_iter != call_back_params.end() && view_idx == 1 && checkout_flag_iter->second) {
        ALGLogInfo << "白细胞疟原虫 进入到流道试剂错误检测，不进行检测直接返回 \n";
        return 0;
    }

    if (view_idx == 0) {

        // 白细胞四分类
        result.clear();
        // 白细胞常规检测

        // 取明场,暗场图
        cv::Mat* bright_img = Ai_ItemGetImg(ctx_id, item_id, 0);
        cv::Mat* fluo_img   = Ai_ItemGetImg(ctx_id, item_id, 1);


        // 添加流道错误检测
        for (auto call_back_iter : call_back_params) {
            ALGLogInfo << "回调函数的参数： " << call_back_iter.first << " " << call_back_iter.second << "\n";
        }
        auto checkout_flag_iter = call_back_params.find("CHECK");
        if (checkout_flag_iter != call_back_params.end() && checkout_flag_iter->second) {
            ALGLogInfo << "白细胞明场 进入到流道试剂错误检测 \n";
            // 白细胞试剂加到红细胞流道 ，g1c0n0 。应该是绿的， 加错后偏黑，所以阈值设置为80 ，类型是大于
            int ret = is_error(*fluo_img, 45, 1);
            if (ret) {
                // g_passageway = 0;
            }
            else {
                // g_passageway = -2;
                g_passageway.wbc_error++;
            }
            ALGLogInfo << "白细胞荧光场流道试剂错误检测，检测完成 \n";
            return 0;
        }

        // 融合图像
        cv::Mat bright_fluo_merged;
        MergeBrightFluoImg(*bright_img, *fluo_img, WBC_BRIGHT_FLUO_FUSION_RATE, bright_fluo_merged);

        int wbc_nums_p = ctx->cnt.WBC;
        int ret;

        ret = Heamo_ImgNormal(ctx_id,
                              item_id,
                              &bright_fluo_merged,
                              group_idx,
                              chl_idx,
                              view_order,
                              0,
                              processed_idx,
                              AI_IMG_STAGE_INTERENCE,
                              userdata,
                              result,
                              view_pair_idx,
                              call_back_params);
        if (ret != 0) {
            ALGLogError << "Heamo_ImgWBC 白细胞流道推理错误 \n";
            return -2;
        }
        // g_current_count.wbc_count = ctx->cnt.WBC;
        //  保存结果数量
        ctx->cnt.heamo_wbc_nums_v.push_back(static_cast<int>(ctx->cnt.WBC - wbc_nums_p));

        // 保存结果
        int processed_idx_local = 0;
        ret                     = MakeMidResult(ctx_id,
                            item_id,
                            img,
                            group_idx,
                            chl_idx,
                            view_order,
                            view_idx,
                            processed_idx_local,
                            stage,
                            userdata,
                            result,
                            view_pair_idx,
                            call_back_params,
                            false,
                            2,
                            4);
        if (ret) {
            ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order
                        << " " << view_idx << " " << processed_idx;
        }

        /**
         * 实现WBC图片拼接
         */
        std::vector<cv::Mat> montageImg_v;
        if (!MergeImgPreProcess(montageImg_v, *bright_img, *fluo_img, result)) {
            ALGLogError << "Failed to PreProcess wbc4";
            return -3;
        }
        // 白细胞4分类可能切割出多张图,增加图会增加view idx,但view idx是视图,
        // 固定,因此此处直接调用模型推理,而不是增加节点
        for (auto& montageImg : montageImg_v) {
            std::list<NNetResult_t> wbc4_result;

            int neu_nums_p  = ctx->cnt.NEU;
            int lym_nums_p  = ctx->cnt.LYM;
            int mono_nums_p = ctx->cnt.MONO;
            int eos_nums_p  = ctx->cnt.EOS;
            int nrbc_nums_p = ctx->cnt.NRBC;
            int unknown_nums_p = ctx->cnt.UNKNOWN;

            ret = Heamo_ImgNormal(ctx_id,
                                item_id,
                                &montageImg,
                                group_idx,
                                chl_idx,
                                view_order,
                                1,
                                processed_idx,
                                AI_IMG_STAGE_INTERENCE,
                                userdata,
                                wbc4_result,
                                view_pair_idx,
                                call_back_params);
            if (ret != 0) {
                ALGLogError << "Failed to run wbc4 inference";
                return -4;
            }
            // 当前图像中各个类别细胞的数量
            ctx->cnt.heamo_neu_nums_v.push_back(static_cast<int>(ctx->cnt.NEU - neu_nums_p));
            ctx->cnt.heamo_lym_nums_v.push_back(static_cast<int>(ctx->cnt.LYM - lym_nums_p));
            ctx->cnt.heamo_mono_nums_v.push_back(static_cast<int>(ctx->cnt.MONO - mono_nums_p));
            ctx->cnt.heamo_eos_nums_v.push_back(static_cast<int>(ctx->cnt.EOS - eos_nums_p));
            ctx->cnt.heamo_nrbc_nums_v.push_back(static_cast<int>(ctx->cnt.NRBC - nrbc_nums_p));

            processed_idx_local = processed_idx_local + 1;
            // 保存结果
            ret = MakeMidResult(ctx_id,
                                item_id,
                                &montageImg,
                                group_idx,
                                chl_idx,
                                view_order,
                                view_idx,
                                processed_idx_local,
                                stage,
                                userdata,
                                wbc4_result,
                                view_pair_idx,
                                call_back_params,
                                true,
                                1,
                                2);
            if (ret) {
                ALGLogError << "Failed to make mid result ,group channel, order, view, "
                               "processed"
                            << group_idx << " " << chl_idx << " " << view_order << " " << view_idx << " " << processed_idx;
            }
        }
        g_current_count.wbc_count  = ctx->cnt.WBC;
        g_current_count.neu_count  = ctx->cnt.NEU;
        g_current_count.lym_count  = ctx->cnt.LYM;
        g_current_count.mono_count = ctx->cnt.MONO;
        g_current_count.eos_count  = ctx->cnt.EOS;

        result.clear();
#if (NET_USE_TIMECNT)
        TimeCnt_End("wbc");
#endif
        g_run_count++;
        return 0;
    }
#if (NET_USE_TIMECNT)
    TimeCnt_End("wbc");
#endif
    return 0;
};

int Heamo_ImgWbcChl(AiCtxID_t                           ctx_id,
                    uint32_t                            item_id,
                    AiImg_t*                            img,
                    uint32_t                            group_idx,
                    uint32_t                            chl_idx,
                    uint32_t                            view_order,
                    uint32_t                            view_idx,
                    uint32_t                            processed_idx,
                    AiImgStage_e                        stage,
                    void*                               userdata,
                    std::list<NNetResult_t>&            result,
                    const int&                          view_pair_idx,
                    const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }
    if (ctx->cnt.WBC > g_alarm_param.wbc_num) {
        ALGLogInfo << "Heamo_ImgWbcChl 当前检测图片满足要求,不需要在推入图片" << "\n";
        ALGLogInfo << "Heamo_ImgWbcChl 当前的rbc计数: " << ctx->cnt.RBC << "  当前的wbc计数: " << ctx->cnt.WBC
                   << " rbc 阈值:  " << g_alarm_param.rab_num << " wbc 阈值:" << g_alarm_param.wbc_num << "\n";
        return 11;
    }
    if (HEAMO_CTX_CALIB(ctx)) {
        return Heamo_CalibCount(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
    else {
        return Heamo_ImgWBC(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
}

double dist_p2p(const cv::Point2f& a, const cv::Point2f& b)
{
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}
// 荧光微球处理函数
int Heamo_ImgFluMicChl(AiCtxID_t                           ctx_id,
                       uint32_t                            item_id,
                       AiImg_t*                            img,
                       uint32_t                            group_idx,
                       uint32_t                            chl_idx,
                       uint32_t                            view_order,
                       uint32_t                            view_idx,
                       uint32_t                            processed_idx,
                       AiImgStage_e                        stage,
                       void*                               userdata,
                       std::list<NNetResult_t>&            result,
                       const int&                          view_pair_idx,
                       const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }
    // TODO
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "Null group ptr";
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }

    AiViewReg_t* view = Heamo_FindView(group, Heamo_FindChl(group, chl_idx), view_idx);
    if (view == NULL) {
        ALGLogError << "null view ptr, group_id chl_idx view_idx" << AI_GROUP_ID(group) << " " << chl_idx << " " << view_idx;
        return ALG_ERR_INVALID_GROUP_CHANNEL_VIEW_COMBINATION;
    }
    if (view->mod_id == NNET_MODID_UNKOWN) {
        return ALG_SUCC;
    }

    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);

    int  mic_nums_p = ctx->cnt.MIC_FLU;
    auto x_iter     = call_back_params.find("X");
    auto y_iter     = call_back_params.find("Y");
    auto z_iter     = call_back_params.find("Z");
    if (x_iter != call_back_params.end() && y_iter != call_back_params.end() && z_iter != call_back_params.end() && view_idx == 0) {
        ctx->cnt.heamo_mic_flu_x.push_back(x_iter->second);
        ctx->cnt.heamo_mic_flu_y.push_back(y_iter->second);
        ctx->cnt.heamo_mic_flu_z.push_back(z_iter->second);
    }
    int processed_idx_local=0;
    if (view_idx == 0)
    {
        //0 是明场
        cv::Mat* bright_img = Ai_ItemGetImg(ctx_id, item_id, 0);
        cv::Mat* fluo_img   = Ai_ItemGetImg(ctx_id, item_id, 1);
        if (bright_img->empty() || fluo_img->empty()) {
            std::cout<<"图片为空"<<std::endl;
        }
        // SaveImage("/mnt/user/0/16B49C13B49BF409/test/5/1.bmp", *bright_img);
        // SaveImage("/mnt/user/0/16B49C13B49BF409/test/5/2.bmp", *fluo_img);
        std::vector<cv::Rect> windows_rect;
        std::vector<cv::Mat>  windows_img;
        // 拆分图像
        crop_img(*bright_img, windows_rect, windows_img);
        // 结果
        std::vector<cv::Point2f> pts;

        for (int m = 0; m < windows_img.size(); m++) {
            int ret;
            ret = Ai_Inference(HEAMO_CTX_AI_CTXID(ctx), group_id, mod_id, &windows_img[m], result, AI_INFER_TYPE_NORMAL);
            if (ret) {
                ALGLogError << "Failed to inference model, group:mod " << group_id << "  " << mod_id;
                return ret;
            }
            AiModReg_t* mod = Heamo_FindMod(group, mod_id);

            if (mod == NULL) {
                ALGLogError << "Failed to find model " << mod_id;
                return ALG_ERR_INVALID_MODEL;
            }
            ret = Ai_ResultCount((void*)HEAMO_CTX_CNT(ctx), *AI_GROUP_CNT_LIST(group), result, AI_MOD_MULTI_LABEL_FLAG(mod));
            if (ret) {
                ALGLogError << "Failed to count result of model " << mod_id;
                return ret;
            }
            // 合并结果
            for (const auto& one_result : result) {
                double center_x = (one_result.box.left + one_result.box.right) / 2 + windows_rect[m].x;
                double center_y = (one_result.box.top + one_result.box.bottom) / 2 + windows_rect[m].y;
                pts.push_back(cv::Point2f(center_x, center_y));
            }
            // 保存结果
            processed_idx_local = processed_idx_local + 1;
            ret = MakeMidResult(ctx_id,
                                    item_id,
                                    &windows_img[m],
                                    group_idx,
                                    chl_idx,
                                    view_order,
                                    view_idx,
                                    processed_idx_local,
                                    stage,
                                    userdata,
                                    result,
                                    view_pair_idx,
                                    call_back_params,
                                    true,
                                    1,
                                    2);
            if (ret) {
                ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order
                            << " " << view_idx << " " << processed_idx;
                return ret;
            }
        }

        std::sort(pts.begin(), pts.end(), [](const cv::Point& ls, const cv::Point& rs) {
            if (std::abs(ls.y - rs.y) < 30) {
                if (ls.x < rs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (ls.y < rs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
        });
        std::vector<cv::Point2f> ret_pts;
        // 去重
        std::vector<int> index(pts.size(), 0);
        for (int i = 0; i < pts.size(); i++) {
            if (index[i] > 0)
                continue;
            double sum_x = pts[i].x;
            double sum_y = pts[i].y;
            int    count = 1;
            for (int j = i + 1; j < pts.size(); j++) {
                if (index[j] > 0)
                    continue;
                double dis = dist_p2p(pts[i], pts[j]);
                if (dis < 30) {
                    // 距离小于30的认为是同一个
                    sum_x += pts[j].x;
                    sum_y += pts[j].y;
                    count++;
                    index[i] = 1;
                    index[j] = 1;
                }
            }
            if (count > 1) {
                sum_x = sum_x / count;
                sum_y = sum_y / count;
                ret_pts.push_back(cv::Point2f(sum_x, sum_y));
            }
            else {
                ret_pts.push_back(cv::Point2f(sum_x, sum_y));
            }
        }
        ctx->cnt.heamo_mic_flu_v.push_back(static_cast<int>(ret_pts.size()));
    }

    return 0;
}
// 嗜碱细胞流道
int Heamo_ImgBaso(AiCtxID_t                           ctx_id,
                  uint32_t                            item_id,
                  AiImg_t*                            img,
                  uint32_t                            group_idx,
                  uint32_t                            chl_idx,
                  uint32_t                            view_order,
                  uint32_t                            view_idx,
                  uint32_t                            processed_idx,
                  AiImgStage_e                        stage,
                  void*                               userdata,
                  std::list<NNetResult_t>&            result,
                  const int&                          view_pair_idx,
                  const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }
#if (NET_USE_TIMECNT)
    TimeCnt_Start("baso");
#endif
    int baso_nums_p = ctx->cnt.BASO;
    int ret         = Heamo_ImgNormal(
        ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    if (ret != 0) {
        ALGLogError << "Failed to run baso inference";
        return -2;
    }
    ctx->cnt.heamo_baso_nums_v.push_back(static_cast<int>(ctx->cnt.BASO - baso_nums_p));
    // 保存结果
    int processed_idx_local = 0;
    ret                     = MakeMidResult(ctx_id,
                        item_id,
                        img,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx_local,
                        stage,
                        userdata,
                        result,
                        view_pair_idx,
                        call_back_params,
                        false,
                        2,
                        2);
    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
    }
    g_run_count++;
#if (NET_USE_TIMECNT)
    TimeCnt_End("baso");
#endif
}

int Heamo_ImgBasoChl(AiCtxID_t                           ctx_id,
                     uint32_t                            item_id,
                     AiImg_t*                            img,
                     uint32_t                            group_idx,
                     uint32_t                            chl_idx,
                     uint32_t                            view_order,
                     uint32_t                            view_idx,
                     uint32_t                            processed_idx,
                     AiImgStage_e                        stage,
                     void*                               userdata,
                     std::list<NNetResult_t>&            result,
                     const int&                          view_pair_idx,
                     const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }

    if (HEAMO_CTX_CALIB(ctx)) {
        return Heamo_CalibCount(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
    else {
        return Heamo_ImgBaso(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
}

// 网织红流道
int Heamo_ImgRet(AiCtxID_t                           ctx_id,
                 uint32_t                            item_id,
                 AiImg_t*                            img,
                 uint32_t                            group_idx,
                 uint32_t                            chl_idx,
                 uint32_t                            view_order,
                 uint32_t                            view_idx,
                 uint32_t                            processed_idx,
                 AiImgStage_e                        stage,
                 void*                               userdata,
                 std::list<NNetResult_t>&            result,
                 const int&                          view_pair_idx,
                 const std::map<std::string, float>& call_back_params)
{
    if (chl_idx == 3 || chl_idx == 7) {
        ALGLogWarning << "RET channel should not accept imgs";
        return 0;
    }
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }

#if (NET_USE_TIMECNT)
    TimeCnt_Start("ret");
#endif
    // 网织红暂时仅在暗场进行处理

    int ret = Heamo_ImgNormal(
        ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    if (ret != 0) {
        ALGLogError << "Failed to run ret inference";
        return -2;
    }
    // 保存结果
    int processed_idx_local = 0;
    ret                     = MakeMidResult(ctx_id,
                        item_id,
                        img,
                        group_idx,
                        chl_idx,
                        view_order,
                        view_idx,
                        processed_idx_local,
                        stage,
                        userdata,
                        result,
                        view_pair_idx,
                        call_back_params,
                        false,
                        2,
                        2);
    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
    }
    g_run_count++;
#if (NET_USE_TIMECNT)
    TimeCnt_End("ret");
#endif
    return 0;
}

int Heamo_ImgRetChl(AiCtxID_t                           ctx_id,
                    uint32_t                            item_id,
                    AiImg_t*                            img,
                    uint32_t                            group_idx,
                    uint32_t                            chl_idx,
                    uint32_t                            view_order,
                    uint32_t                            view_idx,
                    uint32_t                            processed_idx,
                    AiImgStage_e                        stage,
                    void*                               userdata,
                    std::list<NNetResult_t>&            result,
                    const int&                          view_pair_idx,
                    const std::map<std::string, float>& call_back_params)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return ALG_ERR_INVALID_HEAMO;
    }

    if (HEAMO_CTX_CALIB(ctx)) {
        return Heamo_CalibCount(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
    else {
        return Heamo_ImgRet(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
}


int            HeamoGetCount(bool zero_flag, int64_t& nms_time)
{

    if (zero_flag) {
        g_run_count = 0;
        nms_time=0;
    }
    ALGLogInfo << "当前传入的标志位： " << zero_flag << "   当前的计数数量为： " << g_run_count << "\n";
    nms_time = g_nms_timie;
    return g_run_count;
}
int HeamoGetCurrentCellCount(HeamoCtxID_t ctx_id, CurrentCount_t& current_count)
{

    current_count.wbc_count  = g_current_count.wbc_count;
    current_count.rbc_count  = g_current_count.rbc_count;
    current_count.neu_count  = g_current_count.neu_count;
    current_count.lym_count  = g_current_count.lym_count;
    current_count.mono_count = g_current_count.mono_count;
    current_count.eos_count  = g_current_count.eos_count;
}

int HeamoGetError(Passageway_t& passageway_info)
{
    passageway_info = g_passageway;
    ALGLogInfo << "当前检测的信息,红细胞错误： " << g_passageway.rbc_error << " 白细胞错误：  " << g_passageway.wbc_error << "\n";
    return 0;
}


int findStep(const std::vector<int>& nums)
{
    // 将数字存入集合中，自动去重并排序
    std::set<int> uniqueNums(nums.begin(), nums.end());
    // 如果所有数字都相同，步长为 0
    if (uniqueNums.size() <= 1)
        return 0;
    // 计算相邻元素之间的差值，判断是否为等差数列
    int step = *std::next(uniqueNums.begin()) - *uniqueNums.begin();
    int prev = *uniqueNums.begin();
    for (auto it = std::next(uniqueNums.begin()); it != uniqueNums.end(); ++it) {
        if (*it - prev != step)
            return -1;   // 不是等差数列，返回-1表示无有效步长
        prev = *it;
    }
    return step;
}

int HeamoGetMicFluResult(HeamoCtxID_t ctx_id, MicFluInfo_t& result)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {

        return -1;
    }

    HeamoCnt_t* cnt = HEAMO_CTX_CNT(ctx);

    int micflu__nums = cnt->MIC_FLU;
    ALGLogInfo << "HeamoGetMicFluResult,荧光微球总数 ： " << micflu__nums << "\n";
    std::cout << "HeamoGetMicFluResult,荧光微球总数 ：" << micflu__nums << std::endl;

    std::vector<int> heamo_mic_flu_v = cnt->heamo_mic_flu_v;
    std::vector<int> heamo_mic_flu_x = cnt->heamo_mic_flu_x;
    std::vector<int> heamo_mic_flu_y = cnt->heamo_mic_flu_y;
    std::vector<int> heamo_mic_flu_z = cnt->heamo_mic_flu_z;
    if (heamo_mic_flu_v.size() != heamo_mic_flu_x.size()) {
        std::cout << "heamo_mic_flu_v 与 heamo_mic_flu_x 大小不同 "<< std::endl;
        return -1;
    }

    for (int i = 0; i < heamo_mic_flu_v.size(); i++) {
        result.x_vec.push_back(heamo_mic_flu_x[i]);
        result.y_vec.push_back(heamo_mic_flu_y[i]);
        result.z_vec.push_back(heamo_mic_flu_z[i]);
        result.cell_count.push_back(heamo_mic_flu_v[i]);
    }

    //当视野数横纵向大于二的时候,才有步长
    //荧光微球禁止多流道，因为流道与流道之间的步长， 与每个流道内的步长不一致。计数会错误
    if (heamo_mic_flu_x.size() > 2 && heamo_mic_flu_y.size()>2){

        int x_min = *std::min_element(heamo_mic_flu_x.begin(), heamo_mic_flu_x.end());
        int x_max = *std::max_element(heamo_mic_flu_x.begin(), heamo_mic_flu_x.end());
        int y_min = *std::min_element(heamo_mic_flu_y.begin(), heamo_mic_flu_y.end());
        int y_max = *std::max_element(heamo_mic_flu_y.begin(), heamo_mic_flu_y.end());

        int x_step   = findStep(heamo_mic_flu_x);
        int y_step   = findStep(heamo_mic_flu_y);
        if (x_step <= 0 || y_step <= 0) {
            ALGLogError << "HeamoGetMicFluResult,荧光微球步长计算失败 " << x_step << " " << y_step << "\n";
            std::cout << "HeamoGetMicFluResult,荧光微球步长计算失败" << x_step << " " << y_step << std::endl;
            return 0;
        }
        int x_cols = 0;
        int y_rows = 0;
        if (x_step > 0 && y_step > 0) {
            x_cols = (x_max - x_min) / x_step + 1;
            y_rows = (y_max - y_min) / y_step + 1;
        }
        if (x_cols == 1 || y_rows == 1) {
            ALGLogError << "HeamoGetMicFluResult,荧光微球个数计算失败 " << "\n";
            std::cout << "HeamoGetMicFluResult,荧光微球个数计算失败" << std::endl;
            return 0;
        }
        if ((x_cols * y_rows) != heamo_mic_flu_v.size()) {
            ALGLogError << "HeamoGetMicFluResult,荧光微球 纵向 乘以 横向不等于 总数 " << x_cols << " " << y_rows << " " << heamo_mic_flu_v.size()
                        << "\n";
            std::cout << "HeamoGetMicFluResult,荧光微球 纵向 乘以 横向不等于 总数" << x_cols << " " << y_rows << " " << heamo_mic_flu_v.size()
                      << std::endl;
            return 0;
        }
        std::vector<std::vector<int>> matrix(y_rows, std::vector<int>(x_cols, 0));

        for (size_t i = 0; i < heamo_mic_flu_x.size(); ++i) {
            // 转换世界坐标为数组索引
            int col = (heamo_mic_flu_x[i] - x_min) / x_step;
            int row = (heamo_mic_flu_y[i] - y_min) / y_step;

            // 确保索引在数组范围内
            if (row >= 0 && row < y_rows && col >= 0 && col < x_cols) {
                matrix[row][col] = heamo_mic_flu_v[i];
            }
            else {
                std::cout << "坐标超出数组范围: (" << heamo_mic_flu_x[i] << ", " << heamo_mic_flu_y[i] << ")" << "(" << row << ", " << col << ")"
                          << std::endl;
                ALGLogError << "坐标超出数组范围: (" << heamo_mic_flu_x[i] << ", " << heamo_mic_flu_y[i] << ")" << "(" << row << ", " << col << ")"
                            << "\n";
            }
        }
        //横向先走,那x方向就是先从左到右
        /**
        0  1  2  3
        7  6  5  4
        8  9  10 11
        15 14 12 13
        */
        //纵向先走，那y轴就是先从上到下
        /**
       0  7  8  15
       1  6  9  14
       2  5  10 13
       3  4  11 12
       */
        for (int i = 0; i < matrix.size();i++) {
            std::vector<int> tmp_vec;
            for (int j = 0; j < matrix[i].size();j++) {
                tmp_vec.push_back(matrix[i][j]);
            }
            result.cell_vec.push_back(tmp_vec);
        }
    }
    return 0;
}
