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

#define NET_USE_TIMECNT 1

#define VOLUME_RBC_IMG_NUMS 40            // 计算红细胞体积需要处理图像数量
#define WBC_BRIGHT_FLUO_FUSION_RATE 0.5   // WBC明暗场融合比例

extern std::string heamo_save_dir;

extern bool g_pla_flag;
int         g_force_model_id = 0;
//计数
int         g_run_count=0;

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
    cv::Mat  img_target;
    int      ret = MergePltImg(*img_bri, *img_flu, img_target);
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
    ret = MakeMidResult(ctx_id,
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

    // 疟原虫细胞五分类
    if (view_idx == 0) {
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
        return 0;
    }
    return 0;
}
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


    if (view_idx == 0)
    {
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
    TimeCnt_End("rbc");
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

    if (stage != AI_IMG_STAGE_INTERENCE)
    {
        Heamo_DoImgCallback(
            ctx, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        ALGLogInfo << "Heamo_ImgWBC 图片状态错误，退出 白细胞流程 " << "\n";
        return 0;
    }

    if (view_idx == 0) {

        // 白细胞四分类
        result.clear();
        // 白细胞常规检测

        // 取明场,暗场图
        cv::Mat* bright_img = Ai_ItemGetImg(ctx_id, item_id, 0);
        cv::Mat* fluo_img   = Ai_ItemGetImg(ctx_id, item_id, 1);
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

        // 保存结果数量
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

    if (HEAMO_CTX_CALIB(ctx)) {
        return Heamo_CalibCount(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
    else {
        return Heamo_ImgWBC(
            ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    }
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

int HeamoGetCount(bool zero_flag){

    if (zero_flag) {
        g_run_count = 0;
    }
    ALGLogInfo << "当前传入的标志位： " << zero_flag << "   当前的计数数量为： " << g_run_count<<"\n";
    return g_run_count;
}

