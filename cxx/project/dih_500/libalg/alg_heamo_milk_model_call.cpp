//
// Created by y on 24-9-10.
//
#include "alg_heamo.h"
// #include "DihLog.h"
#include "algLog.h"
#include "imgprocess.h"

// 计数图片
extern int g_run_count;


int Heamo_ImgMilkGermChl(AiCtxID_t                           ctx_id,
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
    int ret = Heamo_ImgNormal(
        ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    if (ret != 0) {
        ALGLogError << "failed to run milk inference";
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
                        2,
                        cv::Scalar(0, 255, 0),
                        cv::Scalar(255, 0, 0));
    // 保存单张图数量
    ctx->cnt.milk_germ_nums_v.push_back(result.size());
    if (ret) {
        ALGLogError << "failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
    }
    if (!view_pair_idx){
        g_run_count++;
        ALGLogInfo << "当前传入的view_pair_idx： " << view_pair_idx << "   当前的计数数量为： " << g_run_count << "\n";
    }
}



int Heamo_ImgMilkCellChl(AiCtxID_t                           ctx_id,
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
    int ret = Heamo_ImgNormal(
        ctx_id, item_id, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, userdata, result, view_pair_idx, call_back_params);
    if (ret != 0) {
        ALGLogError << "failed to run milk inference";
        return -2;
    }

    auto&              channel_saved_items = ctx->cnt.element_under_view_pair_idx[MILK_CELL_STR_NAME];
    std::vector<float> flatten_boxes;
    for (const auto& iter : result) {
        flatten_boxes.emplace_back(iter.box.left);
        flatten_boxes.emplace_back(iter.box.top);
        flatten_boxes.emplace_back(iter.box.right - iter.box.left);
        flatten_boxes.emplace_back(iter.box.bottom - iter.box.top);
    }
    // 若当前流道下,该视野尚未记录,放入新数据,
    if (channel_saved_items[chl_idx].size() <= view_pair_idx) {
        channel_saved_items[chl_idx].emplace_back(flatten_boxes);
    }
    else {   // 否则将相同视野下的多个图的结果追加到同一个vector中
        channel_saved_items[chl_idx][view_pair_idx].insert(
            channel_saved_items[chl_idx][view_pair_idx].end(), flatten_boxes.begin(), flatten_boxes.end());
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
                        2,
                        cv::Scalar(0, 255, 0),
                        cv::Scalar(255, 0, 0));
    // 保存单张图数量
    ctx->cnt.milk_cell_nums_v.push_back(result.size());



    if (ret) {
        ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order << " "
                    << view_idx << " " << processed_idx;
    }

    if (!view_pair_idx) {
        g_run_count++;
        ALGLogInfo << "当前传入的view_pair_idx： " << view_pair_idx << "   当前的计数数量为： " << g_run_count << "\n";
    }
    return 0;
}
