//
// Created by y on 24-6-24.
//
#include <numeric>

#include "Calibration.h"
#include "algLog.h"
#include "alg_heamo.h"
#include "replace_std_string.h"
#include "utils.h"

#define DILUTION_RATIO_MILK_GERM 2.0   //  牛奶细菌 通道的稀释比??
#define DILUTION_RATIO_MILK_CELL 8.0   //  牛奶细胞 通道的稀释比??


#define VOLUME_MILK_GERM 150   //  HGB 单???野体积
#define VOLUME_MILK_CELL 150   //  HGB 单???野体积
#define MILK_CELL_REMOVE_DUPLICATE_IOU_THR 0.5



std::string germ_save_dir;
void        Heamo_SetGermResultDir(const std::string& save_dir)
{
    germ_save_dir = save_dir;
}


using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
/*!
 * 临时保存细菌计数结果至csv文件中
 * @param germ_nums_per_img
 * @return
 */
int TempMilkWriteResult(const std::vector<int>&   germ_nums_per_img,
                        const std::vector<int>&   cell_nums_per_img,
                        const std::vector<float>& cell_nums_per_view,
                        const int&                germ_channel_img_nums,
                        const int&                cell_channel_img_nums)
{
    if (germ_channel_img_nums == 0 && cell_channel_img_nums == 0) {
        ALGLogInfo << "Germ csv file will not be created cause zero germ imgs accepted.";
        return 0;
    }

    auto        time_now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::string csv_save_dir(germ_save_dir);
    std::string csv_save_path = csv_save_dir.append(std::to_string(time_now)).append(".csv");
    if (cell_nums_per_img.size() != cell_nums_per_view.size()) {
        ALGLogError << "view result must be consistency with each img, but " << cell_nums_per_img.size() << " " << cell_nums_per_view.size()
                    << " was given";
        return -1;
    }

    try {

        std::ofstream out_file(csv_save_path, std::ios::out);
        out_file << "germ" << "," << "cell" << "," << "cell_remove_duplicated" << std::endl;
        int max_img_nums = std::max(germ_nums_per_img.size(), cell_nums_per_img.size());
        for (int i = 0; i < max_img_nums; ++i) {
            if (i < germ_nums_per_img.size()) {
                out_file << germ_nums_per_img[i] << ",";
            }
            else {
                out_file << ",";
            }
            if (i < cell_nums_per_img.size()) {
                out_file << cell_nums_per_img[i] << ",";
            }
            else {
                out_file << ",";
            }
            if (i < cell_nums_per_view.size()) {
                out_file << cell_nums_per_view[i] << ",";
            }
            else {
                out_file << ",";
            }
            out_file << std::endl;
        }
        out_file.flush();
        out_file.close();
        return 0;
    }
    catch (std::exception& e) {
        ALGLogError << "failed to write csv data in " << csv_save_path;
        return -1;
    }
}


/*!
 * 同一个视野不同扫描平面的体细胞会根据iou进行剔重,在将剔重结果保存时,为显示当前结果为哪些扫描面
 * 的剔重,将剔重结果保存在视野第一张图对应的结果中,其余图像均填写0
 * @param accepted_view_pair_idx
 * @param milk_cell_each_view
 * @param milk_cell_each_view_expand
 * @return
 */
int Heamo_ExpandMilkCellResult(HeamoCtxID_t ctx_id, std::vector<float>& milk_cell_each_view, std::vector<float>& milk_cell_each_view_expand)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    const std::vector<std::vector<float>>& accepted_view_pair_idx = HEAMO_CTX_CNT(ctx)->accepted_view_pair_idx;
    std::vector<AiChlReg_t>                chl_list;
    int                                    ret = Heamo_GetChlReglist(chl_list, HEAMO_CTX_GROUP_IDX(ctx));
    if (ret) {
        ALGLogError << "failed to find chl ";
        return -2;
    }
    auto  milk_cell_each_view_iter = milk_cell_each_view.begin();
    float p                        = -1;
    for (int i = 0; i < chl_list.size(); ++i) {
        if (chl_list[i].chl_type == AI_CHL_TYPE_MILK_CELL_0 || chl_list[i].chl_type == AI_CHL_TYPE_MILK_CELL_1) {
            for (int j = 1; j < accepted_view_pair_idx[i].size(); ++j) {     // 第一个pair idx 为占位, 跳过
                const float& view_pair_idx = accepted_view_pair_idx[i][j];   // 当view pair idx与前一个不同时, 写入
                if (view_pair_idx != p && milk_cell_each_view_iter != milk_cell_each_view.end()) {
                    milk_cell_each_view_expand.push_back(*milk_cell_each_view_iter);
                    p = view_pair_idx;
                    milk_cell_each_view_iter++;
                }
                else {   // 相同,写入0
                    milk_cell_each_view_expand.push_back(0);
                }
            }
        }
    }
    return 0;
}

/*!
 * 牛奶体细胞根据位置去重
 * @param ctx_id
 * @param milk_cell_num
 * @return
 */
int Heamo_MilkCellRemoveDuplicates(HeamoCtxID_t ctx_id, float& milk_cell_num, std::vector<float>& milk_cell_each_view_expand)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    const auto&             channel_saved_items = ctx->cnt.element_under_view_pair_idx[MILK_CELL_STR_NAME];
    std::vector<AiChlReg_t> chl_list;
    int                     ret = Heamo_GetChlReglist(chl_list, HEAMO_CTX_GROUP_IDX(ctx));
    if (ret) {
        ALGLogError << "failed to find chl ";
        return -2;
    }
    milk_cell_num = 0;
    std::vector<float> milk_cell_each_view;
    for (int i = 0; i < chl_list.size(); ++i) {
        if (chl_list[i].chl_type == AI_CHL_TYPE_MILK_CELL_0 || chl_list[i].chl_type == AI_CHL_TYPE_MILK_CELL_1) {

            const auto& view_milk_cell_result = channel_saved_items[i];

            // 遍历每个视野, 每个视野进行nms
            for (int view_idx_under_chl = 0; view_idx_under_chl < view_milk_cell_result.size(); ++view_idx_under_chl) {
                float current_view_milk_cell_nums = 0;
                RemoveDuplicateData(view_milk_cell_result[view_idx_under_chl], MILK_CELL_REMOVE_DUPLICATE_IOU_THR, current_view_milk_cell_nums);
                // 每个通道含有一个空占位,因此需要减去
                if (view_idx_under_chl == 0) {
                    current_view_milk_cell_nums--;
                }
                milk_cell_num += current_view_milk_cell_nums;
                milk_cell_each_view.push_back(current_view_milk_cell_nums);
            }
        }
    }

    ret = Heamo_ExpandMilkCellResult(ctx_id, milk_cell_each_view, milk_cell_each_view_expand);
    if (ret) {
        ALGLogError << "failed to expand milk cell result";
    }
    return 0;
}

/*!
 * 获取视野数量
 * @param chl_save_view_nums
 * @param group_idx
 * @param germ_view_nums
 * @param cell_view_nums
 * @return
 */
int GetMilkViewNums(const std::vector<std::vector<float>>& chl_save_view_nums, const int& group_idx, float& germ_view_nums, float& cell_view_nums)
{
    germ_view_nums = 0;
    cell_view_nums = 0;
    auto group     = Heamo_FindGroup(group_idx);
    if (chl_save_view_nums.size() != group->chl_reglist->size()) {
        ALGLogError << "accept channel nums are not consistency with configured";
        return -1;
    }
    const auto& all_chls = *group->chl_reglist;
    for (int i = 0; i < all_chls.size(); ++i) {
        const auto& chl = all_chls[i];
        ALGLogInfo << "VIEW NUMS " << *(--chl_save_view_nums[i].end());
        if (chl.chl_type == AI_CHL_TYPE_MILK_CELL_0 || chl.chl_type == AI_CHL_TYPE_MILK_CELL_1) {
            cell_view_nums = cell_view_nums + (*std::max_element(chl_save_view_nums[i].begin(), chl_save_view_nums[i].end())) + 1;
        }
        else if (chl.chl_type == AI_CHL_TYPE_MILK_GERM_0 || chl.chl_type == AI_CHL_TYPE_MILK_GERM_1) {
            germ_view_nums = germ_view_nums + (*std::max_element(chl_save_view_nums[i].begin(), chl_save_view_nums[i].end())) + 1;
        }
        else {
            ALGLogError << "find unknown chl type in milk";
            return -2;
        }
    }
    return 0;
}


#define MILK_CHANNEL_NUMS 8
static int GetMilkDilutionRatio(const std::vector<float>& dilution_param_v,
                                const int                 milk_germ_channel_0_nums,
                                const int&                milk_germ_channel_1_nums,
                                const int&                milk_cell_channel_0_nums,
                                const int&                milk_cell_channel_1_nums,
                                float&                    dilution_ratio_milk_germ,
                                float&                    dilution_ratio_milk_cell)
{
    if (dilution_param_v.size() != MILK_CHANNEL_NUMS) {
        ALGLogError << "Heamo channel should be " << MILK_CHANNEL_NUMS << " but " << dilution_param_v.size() << " was given";
        return -1;
    }
    float dilution_ratio_milk_germ0, dilution_ratio_milk_germ1, dilution_ratio_milk_cell0, dilution_ratio_milk_cell1;

    dilution_ratio_milk_germ0 = dilution_param_v[0];
    dilution_ratio_milk_germ1 = dilution_param_v[1];
    dilution_ratio_milk_cell0 = dilution_param_v[2];
    dilution_ratio_milk_cell1 = dilution_param_v[3];




    // 判断开始的流道是否传入了0值
    int ret = 0;
    ret     = OpenedChannelDilutionIdentify(milk_germ_channel_0_nums, dilution_ratio_milk_germ0);
    ret |= OpenedChannelDilutionIdentify(milk_germ_channel_1_nums, dilution_ratio_milk_germ1);
    ret |= OpenedChannelDilutionIdentify(milk_cell_channel_0_nums, dilution_ratio_milk_cell0);
    ret |= OpenedChannelDilutionIdentify(milk_cell_channel_1_nums, dilution_ratio_milk_cell1);
    if (ret) {
        ALGLogError << "Opened channel configured dilution rate 0";
        ALGLogError << "accepted img nums:" << " configured dilution";
        ALGLogError << milk_germ_channel_0_nums << " " << dilution_ratio_milk_germ0;
        ALGLogError << milk_germ_channel_1_nums << " " << dilution_ratio_milk_germ1;
        ALGLogError << milk_cell_channel_0_nums << " " << dilution_ratio_milk_cell0;
        ALGLogError << milk_cell_channel_1_nums << " " << dilution_ratio_milk_cell1;
        return -4;
    }

    if (milk_germ_channel_0_nums != 0 && milk_germ_channel_1_nums != 0) {   // 两个流道都输入图像后,稀释倍数需相等
        if (dilution_ratio_milk_germ0 != dilution_ratio_milk_germ1) {
            ALGLogError << "Two germ channels have different dilution ratio";
            return -2;
        }
        dilution_ratio_milk_germ = dilution_ratio_milk_germ0;
    }
    else {   // 若仅初始化了一个流道,稀释倍数则与之相等
        if (milk_germ_channel_0_nums != 0) {
            dilution_ratio_milk_germ = dilution_ratio_milk_germ0;
        }
        else {
            dilution_ratio_milk_germ = dilution_ratio_milk_germ1;
        }
    }


    if (milk_cell_channel_0_nums != 0 && milk_cell_channel_1_nums != 0) {
        if (dilution_ratio_milk_cell0 != dilution_ratio_milk_cell1) {
            ALGLogError << "Two cell channels have different dilution ratio";
            return -3;
        }
        dilution_ratio_milk_cell = dilution_ratio_milk_cell0;
    }
    else {   //
        if (milk_cell_channel_0_nums != 0) {
            dilution_ratio_milk_cell = dilution_ratio_milk_cell0;
        }
        else {
            dilution_ratio_milk_cell = dilution_ratio_milk_cell1;
        }
    }


    ALGLogInfo << "Dilution ratio for germ cell" << dilution_ratio_milk_germ << " " << dilution_ratio_milk_cell;
    return 0;
}

/*!
 * 获取牛奶结果
 * @param ctx_id
 * @param curve_rbc
 * @param curve_plt
 * @param callback
 * @param userdata
 * @param view_param
 * @return
 */
int Heamo_GetMilkResult(HeamoCtxID_t              ctx_id,
                        std::vector<float>&       curve_rbc,
                        std::vector<float>&       curve_plt,
                        HeamoResultCallback_f     callback,
                        void*                     userdata,
                        const ResultViewParam_t&  view_param,
                        std::vector<std::string>& alarm_str_v)
{
    ALGLogInfo << "Get milk result";
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL || callback == NULL) {
        return -1;
    }
    const int& milk_germ_channel_0_nums = view_param.milk_germ_channel_0_nums;
    const int& milk_cell_channel_0_nums = view_param.milk_cell_channel_0_nums;
    const int& milk_germ_channel_1_nums = view_param.milk_germ_channel_1_nums;
    const int& milk_cell_channel_1_nums = view_param.milk_cell_channel_1_nums;

    float volume_milk_germ = VOLUME_MILK_GERM;
    float volume_milk_cell = VOLUME_MILK_CELL;

    float dilution_ratio_milk_germ, dilution_ratio_milk_cell;

    if (GetMilkDilutionRatio(HEAMO_CTX_DILUTION(ctx),
                             milk_germ_channel_0_nums,
                             milk_cell_channel_0_nums,
                             milk_germ_channel_1_nums,
                             milk_cell_channel_1_nums,
                             dilution_ratio_milk_germ,
                             dilution_ratio_milk_cell)) {
        return -1;
    }




    HeamoCnt_t* cnt = HEAMO_CTX_CNT(ctx);

    // 对视野面积进行校准
    ALG_DEPLOY::CALIBRATION::Calibration<float> calib;
    // 272
    calib.SetPhysicalSizeCalibration(HEAMO_CTX_IMG_H(ctx), HEAMO_CTX_IMG_W(ctx), HEAMO_CTX_IMG_H_UM(ctx));
    float calib_volume_milk_germ, calib_volume_milk_cell;

    // 对体积进行校准
    calib.GetVolCalibrationResult(volume_milk_germ, calib_volume_milk_germ);
    calib.GetVolCalibrationResult(volume_milk_cell, calib_volume_milk_cell);

    ALGLogInfo << "Calib view area volume germ, cell " << volume_milk_germ << " " << volume_milk_cell;

    float milk_germ_num = std::accumulate(cnt->milk_germ_nums_v.begin(), cnt->milk_germ_nums_v.end(), 0.f);
    float milk_cell_num = std::accumulate(cnt->milk_cell_nums_v.begin(), cnt->milk_cell_nums_v.end(), 0.f);
    ALGLogInfo << "Milk cell nums before iou thr " << milk_cell_num;
    // 移除重叠的体细胞
    std::vector<float> milk_cell_each_view_expand;
    int                ret = Heamo_MilkCellRemoveDuplicates(ctx_id, milk_cell_num, milk_cell_each_view_expand);
    if (ret) {
        ALGLogError << "Failed to remove duplicated milk cells";
        return -2;
    }
    ALGLogInfo << "Milk cell nums after  iou thr " << milk_cell_num;

    float milk_germ_view_nums, milk_cell_view_nums;
    ret = GetMilkViewNums(cnt->accepted_view_pair_idx, ctx->inner_group_idx, milk_germ_view_nums, milk_cell_view_nums);
    if (ret) {
        ALGLogError << "Failed to get milk view nums";
        return -2;
    }

    float milk_germ_value = (float)(milk_germ_num * dilution_ratio_milk_germ / (milk_germ_view_nums * calib_volume_milk_germ + ADD_DENOMINATOR));
    float milk_cell_value = (float)(milk_cell_num * dilution_ratio_milk_cell / (milk_cell_view_nums * calib_volume_milk_cell + ADD_DENOMINATOR));

    ALGLogInfo << "Milk germ result " << milk_germ_num << "," << dilution_ratio_milk_germ << "," << milk_germ_view_nums << ","
               << calib_volume_milk_germ;

    ALGLogInfo << "Milk cell result " << milk_cell_num << "," << dilution_ratio_milk_cell << "," << milk_cell_view_nums << ","
               << calib_volume_milk_cell;
    // unit from 1/nl -> 1000/ml
    milk_germ_value = milk_germ_value * (1e+6) / (1e+3);
    milk_cell_value = milk_cell_value * (1e+6) / (1e+3);


    // 保留位数
    std::string milk_germ_value_s, milk_cell_value_s;
    CutFloatToString(milk_germ_value, 2, milk_germ_value_s);
    CutFloatToString(milk_cell_value, 2, milk_cell_value_s);
    HEAMO_RESULT(callback, userdata, "IBC", "KIBC/ml", milk_germ_value_s.c_str())
    HEAMO_RESULT(callback, userdata, "SCC", "KSCC/ml", milk_cell_value_s.c_str())


    ALGLogInfo << "view accepted nums " << milk_germ_view_nums << " " << milk_cell_view_nums << " ";

    float milk_germ_channel_nums = milk_germ_channel_0_nums + milk_germ_channel_1_nums;
    float milk_cell_channel_nums = milk_cell_channel_0_nums + milk_cell_channel_1_nums;
    if (TempMilkWriteResult(
            cnt->milk_germ_nums_v, cnt->milk_cell_nums_v, milk_cell_each_view_expand, milk_germ_channel_nums, milk_cell_channel_nums)) {
        return -3;
    }

    return 0;
}
