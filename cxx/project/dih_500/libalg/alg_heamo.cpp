#include "alg_heamo.h"

#include <algorithm>
#include <fstream>
#include <list>
#include <numeric>
#include <string>
#include <vector>

#include "ai.h"
#include "event.h"
#include "imgprocess.h"
#include "libalgcell.h"
#include "replace_std_string.h"
#include "utils.h"
// #include "DihLog.h"
#include "Calibration.h"
#include "algLog.h"

#include "timecnt.h"

#define NET_USE_TIMECNT 1
/**********************
 * 用于换算结果的宏定义
 *********************/
#define HEAMO_DEF_PRIORITY 1        // 血球预设优先级
#define DEFAULT_DILUTION_RATE 0.0   // 默认稀释倍数,若某个流道接受到了图像,但稀释倍数0,则报错

// hgb参数设定
#define HEAMO_HGB_COEF_NUMS 4
#define HEAMO_HGB_PARAM_NUMS 8

// 疟原虫的标志位
bool g_pla_flag = false;

#define PI 3.1416

/* 血球计数器映射表 */
std::vector<AiMap_t> heamo_cnt_maplist_human = {
    AI_MAP_DEF(HeamoCnt_t, RBC, "RBC", 0.1),         AI_MAP_DEF(HeamoCnt_t, RET, "RET", 0.1),         AI_MAP_DEF(HeamoCnt_t, WBC, "WBC", 0.1),
    AI_MAP_DEF(HeamoCnt_t, NEU, "NEU", 0.1),         AI_MAP_DEF(HeamoCnt_t, LYM, "LYM", 0.1),         AI_MAP_DEF(HeamoCnt_t, MONO, "MONO", 0.1),
    AI_MAP_DEF(HeamoCnt_t, EOS, "EOS", 0.1),         AI_MAP_DEF(HeamoCnt_t, BASO, "BASO", 0.1),       AI_MAP_DEF(HeamoCnt_t, IG, "IG", 0.1),
    AI_MAP_DEF(HeamoCnt_t, UNKNOWN, "UNKNOWN", 0.1), AI_MAP_DEF(HeamoCnt_t, PLT, "PLT", 0.1),         AI_MAP_DEF(HeamoCnt_t, GAT, "GAT", 0.1),
    AI_MAP_DEF(HeamoCnt_t, NRBC, "NRBC", 0.1),       AI_MAP_DEF(HeamoCnt_t, PLT_GAT, "PLT_GAT", 0.1), AI_MAP_DEF(HeamoCnt_t, PLA, "PLA", 0.1),
    AI_MAP_DEF(HeamoCnt_t, PV1, "PV1", 0.1),         AI_MAP_DEF(HeamoCnt_t, PV2, "PV2", 0.1),         AI_MAP_DEF(HeamoCnt_t, PV3, "PV3", 0.1),
    AI_MAP_DEF(HeamoCnt_t, PV4, "PV4", 0.1),         AI_MAP_DEF(HeamoCnt_t, PV5, "PV5", 0.1),
};
std::vector<AiMap_t> heamo_cnt_maplist_cat = {AI_MAP_DEF(HeamoCnt_t, RBC, "RBC", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, RET, "RET", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, WBC, "WBC", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, NEU, "NEU", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, LYM, "LYM", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, MONO, "MONO", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, EOS, "EOS", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, BASO, "BASO", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, IG, "IG", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, UNKNOWN, "UNKNOWN", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, PLT, "PLT", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, GAT, "GAT", 0.1),
                                              AI_MAP_DEF(HeamoCnt_t, NRBC, "NRBC", 0.1)};

std::vector<AiMap_t> heamo_cnt_maplist_dog  = {AI_MAP_DEF(HeamoCnt_t, RBC, "RBC", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, RET, "RET", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, WBC, "WBC", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, NEU, "NEU", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, LYM, "LYM", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, MONO, "MONO", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, EOS, "EOS", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, BASO, "BASO", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, IG, "IG", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, UNKNOWN, "UNKNOWN", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, PLT, "PLT", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, GAT, "GAT", 0.1),
                                               AI_MAP_DEF(HeamoCnt_t, NRBC, "NRBC", 0.1)};
std::vector<AiMap_t> heamo_cnt_maplist_milk = {
    AI_MAP_DEF(HeamoCnt_t, MILK_GERM, "MILK_GERM", 0.1),
    AI_MAP_DEF(HeamoCnt_t, MILK_CELL, "MILK_CELL", 0.1),
};

//mod_id, name, multi_label_flag, fusion_rate, group_mask, letterbox, model_type_nums, \
//                   nms_nums, conf_nums,anchor_nums, label_nums, reserved_float_param_nums, reserved_string_param_nums
std::vector<AiModReg_t> heamo_mod_reglist_milk = {
    AI_MOD_DEF(NNET_MODID_MILK_GERM, "MILK_GERM", 0, 1, NNET_GROUP_MILK, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_MILK_CELL, "MILK_CELL", 0, 1, NNET_GROUP_MILK, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
};

/* 血球模型注册表 */
std::vector<AiModReg_t> heamo_mod_reglist_human = {
    AI_MOD_DEF(NNET_MODID_RBC, "RBC_VOLUME_SPHERICAL_UNQUNTIZATION", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),   // 红细胞计数模型
    AI_MOD_DEF(
        NNET_MODID_RBC_QC, "RBC_VOLUME_SPHERICAL_UNQUNTIZATION_QC", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),   // QC红细胞计数模型
    AI_MOD_DEF(NNET_MODID_WBC, "WBC", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),                     // 白细胞计数模型
    AI_MOD_DEF(NNET_MODID_WBC4, "WBC4", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, LABEL_NUMS_CUSTOM, 0, 0),   // 白细胞4分类模型
    AI_MOD_DEF(NNET_MODID_PLT, "PLT_RET", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 3, 3, 0),                 // PLT,网织红模型
    AI_MOD_DEF(NNET_MODID_BASO, "BASO", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),                   // 嗜碱模型
    //		AI_MOD_DEF(NNET_MODID_RET, "RET", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1,1,18, 1,0,0),
    AI_MOD_DEF(NNET_MODID_SPHERICAL_FOCAL, "SPHERICAL_FOCAL", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 0, 1, 0, 2, 0, 0),   // 球形化红细胞在焦面检测模型
    //        AI_MOD_DEF(NNET_MODID_RBC_INCLINE_DET, "INCLINE_RBC", 0, 1, NNET_GROUP_HUMAN, NORMAL, 1, 1,1,0, 1,0,0),
    //        AI_MOD_DEF(NNET_MODID_RBC_INCLINE_SEG, "INCLINE_RBC_SEG", 0, 1, NNET_GROUP_HUMAN, NORMAL,  1, 1,1,0, 1,0,0),
    AI_MOD_DEF(NNET_MODID_PLT_VOLUME, "PLT_VOLUME", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),   // PLT明场检测模型,用于计算PLT体积
    AI_MOD_DEF(NNET_MODID_CALIB_COUNT, "CALIB_COUNT", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),   // 校准荧光微球计数模型
    AI_MOD_DEF(NNET_MODID_PLA, "PLA", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),                   // 疟原虫模型1
    AI_MOD_DEF(NNET_MODID_PLA4, "PLA4", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 5, 0, 0),                 // 疟原虫模型2
};

std::vector<AiModReg_t> heamo_mod_reglist_cat = {
    AI_MOD_DEF(NNET_MODID_RBC, "RBC_VOLUME_SPHERICAL_UNQUNTIZATION", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_RBC_QC, "RBC_VOLUME_SPHERICAL_UNQUNTIZATION_QC", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_WBC, "WBC", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_WBC4, "WBC4", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 7, 0, 0),
    AI_MOD_DEF(NNET_MODID_PLT, "PLT_RET", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 2, 0, 0),
    AI_MOD_DEF(NNET_MODID_BASO, "BASO", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    //    AI_MOD_DEF(NNET_MODID_RET, "RET", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1,1,18, 1,0,0),
    AI_MOD_DEF(NNET_MODID_RBC_INCLINE_SEG, "SPHERICAL_FOCAL", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 2, 0, 0),
    AI_MOD_DEF(NNET_MODID_PLT_VOLUME, "PLT_VOLUME", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),


};

std::vector<AiModReg_t> heamo_mod_reglist_dog = {
    AI_MOD_DEF(NNET_MODID_RBC, "RBC_VOLUME_SPHERICAL_UNQUNTIZATION", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_RBC_QC, "RBC_VOLUME_SPHERICAL_UNQUNTIZATION_QC", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_WBC, "WBC", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_WBC4, "WBC4", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 7, 0, 0),
    AI_MOD_DEF(NNET_MODID_PLT, "PLT_RET", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 2, 0, 0),
    AI_MOD_DEF(NNET_MODID_BASO, "BASO", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    //    AI_MOD_DEF(NNET_MODID_RET, "RET", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1,1,18, 1,0,0),
    AI_MOD_DEF(NNET_MODID_RBC_INCLINE_SEG, "SPHERICAL_FOCAL", 1, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 2, 0, 0),
    AI_MOD_DEF(NNET_MODID_PLT_VOLUME, "PLT_VOLUME", 0, 1, NNET_GROUP_HUMAN, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),

};


/* RBC视图注册表 */
std::vector<AiViewReg_t> rbc_view_reglist = {
    AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_RBC, NNET_GROUP_MILK | NNET_GROUP_HUMAN | NNET_GROUP_DOG | NNET_GROUP_CAT, 0.25),
    AI_VIEW_DEF(AI_VIEW_TYPE_FLU, NNET_MODID_PLT, NNET_GROUP_MILK | NNET_GROUP_HUMAN | NNET_GROUP_DOG | NNET_GROUP_CAT, 0.25),
};
// 每一个视图绑定的数据,模型依次进入通道注册表注入的函数, 内部判断如何对数据进行处理
/* WBC视图注册表 */
std::vector<AiViewReg_t> wbc_view_reglist = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_WBC, NNET_GROUP_HUMAN | NNET_GROUP_DOG | NNET_GROUP_CAT, 1),
                                             AI_VIEW_DEF(AI_VIEW_TYPE_FLU, NNET_MODID_WBC4, NNET_GROUP_HUMAN | NNET_GROUP_DOG | NNET_GROUP_CAT, 1)};
/* BASO视图注册表 */
std::vector<AiViewReg_t> baso_view_reglist = {
    AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_BASO, NNET_GROUP_HUMAN | NNET_GROUP_DOG | NNET_GROUP_CAT, 0.125)};

/* RET视图注册表 */
std::vector<AiViewReg_t> ret_view_reglist = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_RET, NNET_GROUP_HUMAN | NNET_GROUP_DOG | NNET_GROUP_CAT, 0.25),
                                             AI_VIEW_DEF(AI_VIEW_TYPE_FLU, NNET_MODID_RET, NNET_GROUP_HUMAN | NNET_GROUP_DOG | NNET_GROUP_CAT, 0.25)};

/* 牛奶细菌 */
std::vector<AiViewReg_t> milk_view_reglist_germ = {
    AI_VIEW_DEF(AI_VIEW_TYPE_FLU, NNET_MODID_MILK_GERM, NNET_GROUP_MILK, 0.25),
};
std::vector<AiViewReg_t> milk_view_reglist_cell = {
    AI_VIEW_DEF(AI_VIEW_TYPE_FLU, NNET_MODID_MILK_CELL, NNET_GROUP_MILK, 0.25),
};

/* 血球人医通道注册表 */
std::vector<AiChlReg_t> heamo_human_chl_reglist = {
    AI_CHL_DEF(AI_CHL_TYPE_RBC, "红细胞", rbc_view_reglist, Heamo_ImgRbcChl, NNET_GROUP_HUMAN),
    AI_CHL_DEF(AI_CHL_TYPE_WBC, "白细胞", wbc_view_reglist, Heamo_ImgWbcChl, NNET_GROUP_HUMAN),
    AI_CHL_DEF(AI_CHL_TYPE_HGB, "嗜碱细胞", baso_view_reglist, Heamo_ImgBasoChl, NNET_GROUP_HUMAN),
    AI_CHL_DEF(AI_CHL_TYPE_RET, "网织红细胞", ret_view_reglist, Heamo_ImgRetChl, NNET_GROUP_HUMAN),
    AI_CHL_DEF(AI_CHL_TYPE_RBC_QC, "红细胞", rbc_view_reglist, Heamo_ImgRbcChl, NNET_GROUP_HUMAN),
    AI_CHL_DEF(AI_CHL_TYPE_WBC_QC, "白细胞", wbc_view_reglist, Heamo_ImgWbcChl, NNET_GROUP_HUMAN),
    AI_CHL_DEF(AI_CHL_TYPE_HGB_QC, "嗜碱细胞", baso_view_reglist, Heamo_ImgBasoChl, NNET_GROUP_HUMAN),
    AI_CHL_DEF(AI_CHL_TYPE_RET_QC, "网织红细胞", ret_view_reglist, Heamo_ImgRetChl, NNET_GROUP_HUMAN),

};


/* 血球动物通道注册表 */
std::vector<AiChlReg_t> heamo_animal_chl_reglist = {
    AI_CHL_DEF(AI_CHL_TYPE_RBC, "红细胞", rbc_view_reglist, Heamo_ImgRbcChl, NNET_GROUP_DOG | NNET_GROUP_CAT),
    AI_CHL_DEF(AI_CHL_TYPE_WBC, "白细胞", wbc_view_reglist, Heamo_ImgWbcChl, NNET_GROUP_DOG | NNET_GROUP_CAT),
    AI_CHL_DEF(AI_CHL_TYPE_HGB, "嗜碱细胞", baso_view_reglist, Heamo_ImgNormal, NNET_GROUP_DOG | NNET_GROUP_CAT),
    AI_CHL_DEF(AI_CHL_TYPE_RET, "网织红细胞", ret_view_reglist, Heamo_ImgNormal, NNET_GROUP_DOG | NNET_GROUP_CAT)};

/* 牛奶通道注册表 */   // 不同类别可能需要不同的视图与处理函数,因此将类别展开
std::vector<AiChlReg_t> milk_chl_reglist = {
    AI_CHL_DEF(AI_CHL_TYPE_MILK_GERM_0, "牛奶细菌", milk_view_reglist_germ, Heamo_ImgMilkGermChl, NNET_GROUP_MILK),
    AI_CHL_DEF(AI_CHL_TYPE_MILK_GERM_1, "牛奶细菌", milk_view_reglist_germ, Heamo_ImgMilkGermChl, NNET_GROUP_MILK),
    AI_CHL_DEF(AI_CHL_TYPE_MILK_CELL_0, "牛奶体细胞", milk_view_reglist_cell, Heamo_ImgMilkCellChl, NNET_GROUP_MILK),
    AI_CHL_DEF(AI_CHL_TYPE_MILK_CELL_1, "牛奶体细胞", milk_view_reglist_cell, Heamo_ImgMilkCellChl, NNET_GROUP_MILK)};

/* 血球样本分组注册表 */
std::vector<AiGroupReg_t> heamo_group_reglist = {
    AI_GROUP_DEF(AI_GROUP_TYPE_MILK, NNET_GROUP_MILK, "牛奶", "/alg/model/0", heamo_mod_reglist_milk, heamo_cnt_maplist_milk, milk_chl_reglist),
    AI_GROUP_DEF(
        AI_GROUP_TYPE_HUMAN, NNET_GROUP_HUMAN, "人", "/alg/model/1", heamo_mod_reglist_human, heamo_cnt_maplist_human, heamo_human_chl_reglist),
    AI_GROUP_DEF(AI_GROUP_TYPE_CAT, NNET_GROUP_CAT, "猫", "/alg/model/2", heamo_mod_reglist_cat, heamo_cnt_maplist_cat, heamo_animal_chl_reglist),
    AI_GROUP_DEF(AI_GROUP_TYPE_DOG, NNET_GROUP_DOG, "狗", "/alg/model/3", heamo_mod_reglist_dog, heamo_cnt_maplist_dog, heamo_animal_chl_reglist),
};

AiGroupReg_t* Heamo_FindGroup(uint32_t group_idx)
{
    Ai_FindGroup(heamo_group_reglist, group_idx);
}

AiModReg_t* Heamo_FindMod(AiGroupReg_t* group, NNetModID_e mod_id)
{
    Ai_FindMod(group, mod_id);
}

AiChlReg_t* Heamo_FindChl(AiGroupReg_t* group, uint32_t chl_idx)
{
    Ai_FindChl(group, chl_idx);
}

AiViewReg_t* Heamo_FindView(AiGroupReg_t* group, AiChlReg_t* chl, uint32_t view_idx)
{
    Ai_FindView(group, chl, view_idx);
}


/**
 * 获取血球样本分组注册表
 * @param  list			目标列表
 * @return  0 success other fail
 */
int Heamo_GetGroupReglist(std::vector<AiGroupReg_t>& list)
{
    list = heamo_group_reglist;
    return 0;
}

/**
 * 获取血球模型注册表
 * @param  list			目标列表
 * @param  group_idx		分组索引
 * @return  0 success other fail
 */
int Heamo_GetModReglist(std::vector<AiModReg_t>& list, uint32_t group_idx)
{
    return Ai_GetModReglist(list, Heamo_FindGroup(group_idx));
}

/**
 * 获取血球通道注册表
 * @param  list			目标列表
 * @param  group_idx		分组索引
 * @return  0 success other fail
 */
int Heamo_GetChlReglist(std::vector<AiChlReg_t>& list, uint32_t group_idx)
{
    return Ai_GetChlReglist(list, Heamo_FindGroup(group_idx));
}

/**
 * 获取血球视图注册表
 * @param  list			目标列表
 * @param  group_idx		分组索引
 * @param  chl_idx		通道索引
 * @return  0 success other fail
 */
int Heamo_GetViewReglist(std::vector<AiViewReg_t>& list, uint32_t group_idx, uint32_t chl_idx)
{
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    return Ai_GetViewReglist(list, group, Heamo_FindChl(group, chl_idx));
}

/**
 * 血球初始化
 * @param  none
 * @return
 */
HeamoCtxID_t Heamo_Init(AiCtxID_t ai_ctxid)
{
    HeamoCtx_t* ctx         = new HeamoCtx_t;
    HEAMO_CTX_AI_CTXID(ctx) = ai_ctxid;
    if (HEAMO_CTX_AI_CTXID(ctx) == NULL) {
        return NULL;
    }
    // 初始化血球内部拟合方程
    int ret = HEAMO_CTX_NORM_REAGENT_FIT(ctx).Init();
    if (ret) {
        ALGLogError << "Failed to init norm network.";
        return NULL;
    }

    ret = HEAMO_CTX_SPHE_REAGENT_FIT(ctx).Init();
    if (ret) {
        ALGLogError << "Failed to init sphe network.";
        return NULL;
    }

#if (NET_USE_TIMECNT)
    TimeCnt_Init("rbc", 1);
    TimeCnt_Init("wbc", 1);
    TimeCnt_Init("baso", 1);
    TimeCnt_Init("ret", 1);
    TimeCnt_Init("get_result", 1);
#endif
    return (HeamoCtxID_t)ctx;
}

int Heamo_DeInit(HeamoCtxID_t ctx_id)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    delete ctx;
    return 0;
}

int Heamo_OpenMilkCellElement(HeamoCtxID_t ctx_id, const uint32_t& group_idx)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    std::vector<AiChlReg_t> chl_list;
    int                     ret = Heamo_GetChlReglist(chl_list, group_idx);
    if (ret) {
        ALGLogError << "failed to get channel list";
        return -2;
    }

    // 索引依次为 channel_idx, view_pair_idx, box{x, y, w, h}
    std::vector<std::vector<std::vector<float>>> channel_save_elements;
    for (int i = 0; i < chl_list.size(); ++i) {

        channel_save_elements.push_back({{0, 0, 0, 0}});
    }
    ALGLogInfo << channel_save_elements.size();

    // 与检测类别绑定
    // 只能这样写,否则报错
    std::map<std::string, std::vector<std::vector<std::vector<float>>>> element_under_view_pair_idx;
    element_under_view_pair_idx[MILK_CELL_STR_NAME] = channel_save_elements;
    HEAMO_CTX_CNT(ctx)->element_under_view_pair_idx = element_under_view_pair_idx;

    std::vector<std::vector<float>> channel_save_view_pair_idx;
    for (int i = 0; i < chl_list.size(); ++i) {
        channel_save_view_pair_idx.push_back({0});
    }

    HEAMO_CTX_CNT(ctx)->accepted_view_pair_idx = channel_save_view_pair_idx;

    return 0;
}

int Heamo_OpenGroupElement(HeamoCtxID_t ctx_id, const uint32_t& group_idx)
{
    int ret = Heamo_OpenMilkCellElement(ctx_id, group_idx);
    if (ret) {
        ALGLogError << "Failed to open milk cell element";
        return ret;
    }
    return 0;
}



/*!
 * 检查是否重复close
 * @param ctx_id
 * @param op_open
 * @return
 */
int Heamo_OpenFlagCheck(HeamoCtxID_t ctx_id, const HEAMO_OPEN_TYPE& open_type, bool& open_flag_sample_first_time)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    HEAMO_OPEN_TYPE& cur_open_type = HEAMO_CTX_OPEN_FLAG(ctx);


    // 首次open, 允许不同流道重复open
    if (open_type == HEAMO_OPEN_TYPE_OPEN && cur_open_type == HEAMO_OPEN_TYPE_CLOSE) {
        open_flag_sample_first_time = true;
    }
    // 禁止重复close
    if (open_type == HEAMO_OPEN_TYPE_CLOSE && cur_open_type == HEAMO_OPEN_TYPE_CLOSE) {
        ALGLogError << "Attempt to re-close heamo";
        ALGLogError << "Heamo_OpenFlagCheck  打印关闭参数 失败 \n";
        return -2;
    }

    cur_open_type = open_type;

    return 0;
}
/*!
 * Open参数初始化
 * @param ctx_id
 * @param open_flag_sample_first_time
 * @return
 */
int Heamo_OpenSampleParam(HeamoCtxID_t              ctx_id,
                          const bool&               img_fusion,
                          const bool&               debug,
                          AiImgCallback_f           callback,
                          void*                     userdata,
                          const uint32_t&           group_idx,
                          const bool&               qc,
                          const bool&               calib,
                          const float&              img_h,
                          const float&              img_w,
                          const float&              img_h_um,
                          const std::vector<float>& alarm_param_v,
                          const std::vector<float>& dilution_param_v,
                          const std::vector<float>& task_att_v,
                          const bool&               open_flag_sample_first_time)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    ALGLogError << "Heamo_OpenSampleParam  打开参数 \n";
    // 仅对样本第一个流道的open,进行样本参数初始化
    if (open_flag_sample_first_time) {
        HEAMO_CTX_HGB_VAL(ctx)     = 0;
        HEAMO_CTX_CALLBACK(ctx)    = callback;
        HEAMO_CTX_USERDATA(ctx)    = userdata;
        HEAMO_CTX_IMG_FUSION(ctx)  = img_fusion;
        HEAMO_CTX_QC(ctx)          = qc;
        HEAMO_CTX_CALIB(ctx)       = calib;
        HEAMO_CTX_IMG_H(ctx)       = img_h;
        HEAMO_CTX_IMG_W(ctx)       = img_w;
        HEAMO_CTX_IMG_H_UM(ctx)    = img_h_um;
        HEAMO_CTX_ALARM_PARAM(ctx) = alarm_param_v;
        HEAMO_CTX_DILUTION(ctx)    = dilution_param_v;
        HEAMO_CTX_TASK_ATT(ctx)    = task_att_v;
        ALGLogInfo << "Sample open first time";

        // 获取最大通道数,以保存各个通道已接受图像对的数量
        int max_channel_nums;
        Ai_FindMaxChannelNums(heamo_group_reglist, max_channel_nums);
        HEAMO_CTX_CNT(ctx)->channel_img_nums = std::vector<int>(max_channel_nums, 0);
        int ret;
        ret = Heamo_OpenGroupElement(ctx, group_idx);
        if (ret) {
            ALGLogError << "Failed to open group element, group idx " << group_idx;
            return -3;
        }
    }

    // 不同流道可能传入不同的debug参数
    HEAMO_CTX_DEBUG(ctx) = debug;

    ALGLogInfo << "Sample open repeated";


    return 0;
}

/**
 * 血球开启计数
 * @param  ctx_id		血球上下文ID
 * @return
 */
int Heamo_Open(HeamoCtxID_t              ctx_id,
               const bool&               img_fusion,
               const bool&               debug,
               AiImgCallback_f           callback,
               void*                     userdata,
               const uint32_t&           group_idx,
               const bool&               qc,
               const bool&               calib,
               const float&              img_h,
               const float&              img_w,
               const float&              img_h_um,
               const std::vector<float>& alarm_param_v,
               const std::vector<float>& dilution_param_v,
               const std::vector<float>& task_att_v)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    Ai_WaitPriority(HEAMO_CTX_AI_CTXID(ctx), HEAMO_DEF_PRIORITY, 0xFFFF);

    // 检查是否初次open
    bool open_flag_sample_first_time = false;
    int  ret                         = Heamo_OpenFlagCheck(ctx_id, HEAMO_OPEN_TYPE_OPEN, open_flag_sample_first_time);
    if (ret) {
        return ret;
    }

    // open初始化
    ret = Heamo_OpenSampleParam(ctx_id,
                                img_fusion,
                                debug,
                                callback,
                                userdata,
                                group_idx,
                                qc,
                                calib,
                                img_h,
                                img_w,
                                img_h_um,
                                alarm_param_v,
                                dilution_param_v,
                                task_att_v,
                                open_flag_sample_first_time);
    if (ret) {
        return ret;
    }
    return 0;
}


/**
 * 血球生成图像列表
 * @param  img_list		图像列表
 * @param  group_idx		分组索引
 * @param  chl_idx		通道索引
 * @param  view_idx		视图索引
 * @param  img_array		图像缓存指针
 * @param  width			图像宽度
 * @param  heigh			图像高度
 * @return
 */
int Heamo_AddImgList(HeamoCtxID_t           ctx_id,
                     std::list<HeamoImg_t>& list,
                     uint32_t               group_idx,
                     uint32_t               chl_idx,
                     uint32_t               view_idx,
                     uint8_t*               img_data,
                     uint32_t               width,
                     uint32_t               height)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    if (img_data == NULL || !(width * height)) {
        ALGLogError << "Null img";
        return -1;
    }
    AiGroupReg_t* group = Heamo_FindGroup(group_idx);
    AiChlReg_t*   chl   = Heamo_FindChl(group, chl_idx);
    AiViewReg_t*  view  = Heamo_FindView(group, chl, view_idx);
    if (view) {
        return Ai_ConvertImage(list, img_data, width, height, HEAMO_CTX_IMG_FUSION(ctx), AI_VIEW_FUSION_RATE(view));
    }
    ALGLogError << "Null view";
    return -2;
}

/**
 * 血球计数推图
 * @param  ctx_id		血球上下文ID
 * @param  group_idx		分组索引
 * @param  chl_idx		通道索引
 * @param  img_list		图像列表
 * @return
 */

int Heamo_PushImage(HeamoCtxID_t ctx_id, std::list<HeamoImg_t>& img_list, uint32_t group_idx, uint32_t chl_idx, const int& view_pair_idx)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    ALGLogInfo << "Heamo_PushImage 血球开始推入图片,推入的组： " << group_idx << " 通道数 " << chl_idx << " 视图 " << view_pair_idx << "\n";

    AiChlReg_t* chl = Heamo_FindChl(Heamo_FindGroup(group_idx), chl_idx);
    if (chl == NULL) {
        ALGLogError << "Heamo_PushImage 血球推入图片失败, 通道为空" << "\n";
        return -2;
    }
    std::vector<AiViewReg_t> view_list;
    if (Heamo_GetViewReglist(view_list, group_idx, chl_idx)) {
        ALGLogError << "Heamo_PushImage 血球推入图片失败, 视图为空" << "\n";
        return -3;
    }
    if (!view_list.size() || img_list.size() < view_list.size()) {
        ALGLogError << "Heamo_PushImage 血球推入图片失败, 参数错误，图片数量与视图数量不符,图片数量  " << img_list.size() << " 视图数量  "
                    << view_list.size() << "\n";
        return -4;
    }
    std::map<std::string, float> call_back_params{{TASK_TYPE, TASK_TYPE_HEAMO}};
    HEAMO_CTX_GROUP_IDX(ctx) = group_idx;
    if (!Ai_ItemPush(HEAMO_CTX_AI_CTXID(ctx),
                     HEAMO_DEF_PRIORITY,
                     group_idx,
                     chl_idx,
                     ctx->cnt.channel_img_nums[chl_idx],
                     view_list.size(),
                     img_list,
                     AI_CHL_CALLBACK(chl),
                     ctx,
                     view_list,
                     view_pair_idx,
                     call_back_params)) {
        ALGLogError << "Heamo_PushImage 血球推入图片失败,推入的组： " << group_idx << " 通道数 " << chl_idx << " 视图 " << view_pair_idx << "\n";
        return -6;
    }
    ALGLogInfo << "Heamo_PushImage 血球推入图片成功,推入的组： " << group_idx << " 通道数 " << chl_idx << " 视图 " << view_pair_idx << "\n";

    ALGLogInfo << "view_list size: " << view_list.size();
    // 图像进入队列后即计数
    ctx->cnt.channel_img_nums[chl_idx] += 1;
    ctx->cnt.accepted_view_pair_idx[chl_idx].emplace_back(view_pair_idx);
    return 0;
}

//---
void GenerateUniformValue(const int& seed, const float& uniform_low, const float& uniform_high, float& value)
{
    srand(seed);
    float rand_value = float(rand());
    value            = rand_value / RAND_MAX * (uniform_high - uniform_low) + uniform_low;
}


using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

//--- temp make hgb value
void TempMakeHgbValue(float& value)
{
    auto  start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    int   hour  = int((start / 1000) / 3600);
    float main_value, second_value;
    GenerateUniformValue(hour, 0, 140, main_value);
    main_value = 140 + main_value - 80;

    float second_percent = 0.08;
    float second_low, second_high;
    second_low  = main_value * (1 - second_percent);
    second_high = main_value * (1 + second_percent);
    GenerateUniformValue(start, second_low, second_high, second_value);
    value = second_value;
    Delay(200);
}

int Heamo_PushHgb(HeamoCtxID_t ctx_id, const std::vector<HeamoHgbVal_t>& data_list, const std::vector<float>& coef_list)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }

    // hgb模块未装填,暂时放入固定值
    if (data_list.empty()) {
        ctx->cnt.hgb_data = {15110, 15097, 15193, 15107, 4825, 9692, 10309, 16850};
        ctx->cnt.hgb_coef = {1, 1, 1, 1};
        ALGLogInfo << "hgb use pseudo data and coef";
    }
    else {   // 正常逻辑
        if (coef_list.size() != HEAMO_HGB_COEF_NUMS || data_list.size() != HEAMO_HGB_PARAM_NUMS) {
            ALGLogError << "Coef and data list size should be " << HEAMO_HGB_COEF_NUMS << " and " << HEAMO_HGB_PARAM_NUMS << " respectively, but "
                        << coef_list.size() << " and " << data_list.size() << " was given";

            return -2;
        }

        ctx->cnt.hgb_data = data_list;
        ctx->cnt.hgb_coef = coef_list;
    }

    return 0;
}


/**
 * 血球等待完成
 * @param  ctx_id		血球上下文ID
 * @param  timeout		超时时间
 * @return
 */
int Heamo_WaitCplt(HeamoCtxID_t ctx_id, uint32_t timeout)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    return Ai_WaitPriority(HEAMO_CTX_AI_CTXID(ctx), HEAMO_DEF_PRIORITY, timeout);
}


/*!
 * 当进行QC但前4个流道传入了图像时,报错.未进行QC,但后四个流道传入了图像报错
 * @param group_idx
 * @param qc
 * @param channel_list
 * @return
 */
int HeamoChannelConsistyencyCheck(const int& group_idx, const bool& qc, const std::vector<int>& channel_img_nums_v)
{
    if (group_idx == ALG_CELL_GROUP_HUMAN) {
        const int& heamo_normal_imgs = std::accumulate(channel_img_nums_v.begin(), channel_img_nums_v.begin() + 4, 0);
        const int& heamo_qc_imgs     = std::accumulate(channel_img_nums_v.begin() + 4, channel_img_nums_v.end(), 0);
        ALGLogInfo << "All accepted imgs ";
        for (const auto& nums : channel_img_nums_v) {
            ALGLogInfo << nums;
        }
        if (qc && heamo_normal_imgs) {
            ALGLogError << "Running qc mode, but pushed imgs to normal channels";
            return -1;
        }
        if ((!qc) && heamo_qc_imgs) {
            ALGLogError << "Running normal mode, but pushed imgs to qc channels";
            return -1;
        }
    }
    return 0;
}

/*!
 * 获取各个通道接受的图像数量
 * @param heamo_group_type group id
 * @param channel_img_nums 各个通道idx接受的图像组数量
 * @param rbc_channel_nums 解析后的rbc通道数量
 * @param wbc_channel_nums 解析后的wbc通道数量
 * @param baso_channel_nums 解析后的baso通道数量
 * @param ret_channel_nums 解析后的ret通道数量
 */
int GetChannelNums(const uint32_t&         inner_group_idx,
                   const std::vector<int>& channel_img_nums_v,
                   int&                    rbc_channel_nums,
                   int&                    wbc_channel_nums,
                   int&                    hgb_channel_nums,
                   int&                    ret_channel_nums,
                   int&                    milk_germ_channel_0_nums,
                   int&                    milk_cell_channel_0_nums,
                   int&                    milk_germ_channel_1_nums,
                   int&                    milk_cell_channel_1_nums,
                   bool                    qc = false)
{

    std::vector<AiChlReg_t> channel_list;

    ALGLogInfo << "Accepted inner group idx " << inner_group_idx;
    // 未初始化时,赋予空值
    if (inner_group_idx > heamo_group_reglist.size()) {
        rbc_channel_nums         = 0;
        wbc_channel_nums         = 0;
        hgb_channel_nums         = 0;
        ret_channel_nums         = 0;
        milk_germ_channel_0_nums = 0;
        milk_cell_channel_0_nums = 0;
        milk_germ_channel_1_nums = 0;
        milk_cell_channel_1_nums = 0;
        ALGLogWarning << "Accepted wrong group idx in get result, the result might be wrong";
        return 0;
    }


    if (Heamo_GetChlReglist(channel_list, inner_group_idx)) {
        ALGLogError << "Failed to find channel list of inner_group_idx " << inner_group_idx;
        return -1;
    }
    if (channel_list.size() > channel_img_nums_v.size()) {
        ALGLogError << "Channel list of group " << inner_group_idx << " has more nums than "
                    << " channel_img_nums_v";
        return -2;
    }

    for (int idx = 0; idx < channel_list.size(); ++idx) {
        AiChlType_e chl_type{channel_list[idx].chl_type};
        // 预计,牛奶的两个通道与血球共用通道类型名
        switch (chl_type) {
        case AiChlType_e::AI_CHL_TYPE_RBC:
        {
            rbc_channel_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_WBC:
        {
            wbc_channel_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_HGB:
        {
            hgb_channel_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_RET:
        {
            ret_channel_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_MILK_GERM_0:
        {
            milk_germ_channel_0_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_MILK_CELL_0:
        {
            milk_cell_channel_0_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_MILK_GERM_1:
        {
            milk_germ_channel_1_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_MILK_CELL_1:
        {
            milk_cell_channel_1_nums = channel_img_nums_v[idx];
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_RBC_QC:
        {
            if (qc) {
                rbc_channel_nums = channel_img_nums_v[idx];
            }
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_WBC_QC:
        {
            if (qc) {
                wbc_channel_nums = channel_img_nums_v[idx];
            }
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_HGB_QC:
        {
            if (qc) {
                hgb_channel_nums = channel_img_nums_v[idx];
            }
            break;
        }
        case AiChlType_e::AI_CHL_TYPE_RET_QC:
        {
            if (qc) {
                ret_channel_nums = channel_img_nums_v[idx];
            }
            break;
        }
        default:
        {
            ALGLogWarning << "Unknown channel type is found in group : channel " << inner_group_idx << " : " << idx;
            break;
        }
        }
    }
    int ret = HeamoChannelConsistyencyCheck(inner_group_idx, qc, channel_img_nums_v);
    if (ret) {
        return ret;
    }
    return 0;
}



void ClipNumber(const float& src, float& dst, const float& lower, const float& upper)
{
    dst = src;
    if (src < lower) {
        dst = lower;
    }
    else if (src > upper) {
        dst = upper;
    }
}



void CutFloatToString(const float& src, const int& precision, std::string& dst)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << src;
    dst = oss.str();
}



/**
 * 血球获取结果
 * @param  ctx_id		血球上下文ID
 * @param  callback		结果回调
 * @param  userdata		用户数据
 * @return
 */
int Heamo_GetResult(HeamoCtxID_t              ctx_id,
                    std::vector<float>&       curve_rbc,
                    std::vector<float>&       curve_plt,
                    HeamoResultCallback_f     callback,
                    void*                     userdata,
                    std::vector<std::string>& alarm_str_v)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL || callback == NULL) {
        return -1;
    }
    ALGLogInfo << "开始获取结果";


#if (NET_USE_TIMECNT)
    TimeCnt_Start("get_result");
#endif
    // 各流道数量统计
    int rbc_channel_nums, wbc_channel_nums, baso_channel_nums, ret_channel_nums, milk_germ_channel_0_nums, milk_cell_channel_0_nums,
        milk_germ_channel_1_nums, milk_cell_channel_1_nums;
    if (GetChannelNums(HEAMO_CTX_GROUP_IDX(ctx),
                       ctx->cnt.channel_img_nums,
                       rbc_channel_nums,
                       wbc_channel_nums,
                       baso_channel_nums,
                       ret_channel_nums,
                       milk_germ_channel_0_nums,
                       milk_cell_channel_0_nums,
                       milk_germ_channel_1_nums,
                       milk_cell_channel_1_nums,
                       HEAMO_CTX_QC(ctx))) {
        ALGLogError << "Failed to get channel nums";
        return -2;
    }
    ResultViewParam_t view_param;
    view_param.rbc_channel_nums         = rbc_channel_nums;
    view_param.wbc_channel_nums         = wbc_channel_nums;
    view_param.baso_channel_nums        = baso_channel_nums;
    view_param.ret_channel_nums         = ret_channel_nums;
    view_param.milk_germ_channel_0_nums = milk_germ_channel_0_nums;
    view_param.milk_cell_channel_0_nums = milk_cell_channel_0_nums;
    view_param.milk_germ_channel_1_nums = milk_germ_channel_1_nums;
    view_param.milk_cell_channel_1_nums = milk_cell_channel_1_nums;


    HeamoCnt_t* cnt = HEAMO_CTX_CNT(ctx);
    // 获取相应类型
    if (HEAMO_CTX_GROUP_IDX(ctx) == 1) {   // 人医
        if (Heamo_GetHeamoResult(ctx_id, curve_rbc, curve_plt, callback, userdata, view_param, alarm_str_v)) {
            ALGLogError << "Failed to get heamo result";
            return -3;
        }
    }
    else if (HEAMO_CTX_GROUP_IDX(ctx) == 0) {   // 牛奶
        if (Heamo_GetMilkResult(ctx_id, curve_rbc, curve_plt, callback, userdata, view_param, alarm_str_v)) {
            ALGLogError << "Failed to get milk result";
            return -4;
        }
    }
    else {
        ALGLogError << "Doing unsupported type " << HEAMO_CTX_GROUP_IDX(ctx);

        return -5;
    }

#if (NET_USE_TIMECNT)
    TimeCnt_End("get_result");
#endif
#if (NET_USE_TIMECNT)
    TimeCnt_PrintResult();
#endif

    ALGLogInfo << "获取结果成功";
    return 0;
}


int Heamo_ModifyResult(const AlgCellModeID_e&    initialed_mode_id,
                       const std::string&        changed_param_key,
                       std::vector<float>&       curve_rbc,
                       std::vector<float>&       curve_plt,
                       HeamoResultCallback_f     callback,
                       void*                     userdata,
                       std::vector<std::string>& alarm_str_v)
{
    ALGLogInfo << "Running modify type " << initialed_mode_id;
    if (initialed_mode_id == AlgCellModeID_e::ALGCELL_MODE_HUMAN) {
        int ret = Heamo_ModifyHumanResult(changed_param_key, curve_rbc, curve_plt, callback, userdata, alarm_str_v);
        if (ret) {
            return ret;
        }
    }
    return 0;
}



int Heamo_Close(HeamoCtxID_t ctx_id)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    bool open_flag_sample_first_time = false;
    int  ret                         = Heamo_OpenFlagCheck(ctx_id, HEAMO_OPEN_TYPE_CLOSE, open_flag_sample_first_time);
    if (ret) {
        ALGLogInfo << "Heamo_Close 打印关闭参数 失败 \n";
        return ret;
    }
    memset(&ctx->cnt, 0, sizeof(ctx->cnt));
    HEAMO_CTX_HGB_VAL(ctx) = 0;
    return 0;
}

int Heamo_DoImgCallback(HeamoCtx_t*                         ctx,
                        uint32_t                            item_id,
                        AiImg_t*                            img,
                        uint32_t                            group_idx,
                        uint32_t                            chl_idx,
                        uint32_t                            view_order,
                        uint32_t                            view_idx,
                        uint32_t                            processed_idx,
                        AiImgStage_e                        stage,
                        std::list<NNetResult_t>&            result,
                        const int&                          view_pair_idx,
                        const std::map<std::string, float>& call_back_params)
{
    if (ctx == NULL) {
        return -1;
    }
    if (HEAMO_CTX_CALLBACK(ctx)) {
        int ret = (*HEAMO_CTX_CALLBACK(ctx))(HEAMO_CTX_AI_CTXID(ctx),
                                             item_id,
                                             img,
                                             group_idx,
                                             chl_idx,
                                             view_order,
                                             view_idx,
                                             processed_idx,
                                             stage,
                                             HEAMO_CTX_USERDATA(ctx),
                                             result,
                                             view_pair_idx,
                                             call_back_params);
        return ret;
    }
    return 0;
}


// 绘制结果并保存
int MakeMidResult(AiCtxID_t                           ctx_id,
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
                  const std::map<std::string, float>& call_back_params,
                  const bool&                         draw_name,
                  const float&                        font_scale,
                  const int&                          thickness,
                  const cv::Scalar&                   box_color,
                  const cv::Scalar&                   font_color)
{
    HeamoCtx_t* ctx = (HeamoCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }
    // 保存结果
    if (HEAMO_CTX_DEBUG(ctx)) {
        cv::Mat                 mid_result;
        std::vector<NNetResult> detect_result_v(result.begin(), result.end());
        DrawMidResult(img, img->rows, img->cols, detect_result_v, mid_result, draw_name, font_scale, thickness, box_color, font_color);
        int ret = Heamo_DoImgCallback(
            ctx, item_id, &mid_result, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);

        return ret;
    }
    return 0;
}


int OpenedChannelDilutionIdentify(const int& pushed_img_nums, const float& dilution)
{
    if (pushed_img_nums != 0 && dilution == DEFAULT_DILUTION_RATE) {
        return 1;
    }
    return 0;
}
