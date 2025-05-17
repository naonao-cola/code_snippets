
#include "alg_clarity.h"
#include <string>

#include "FocusControl.h"
#include "algLog.h"
#include "event.h"
#include "imgprocess.h"
// #include "DynoLog.h"
#define CLARITY_DEF_PRIORITY 0   // 清晰度预设优先级组

// 细聚焦
#define AI_CLARITY_FAR_NEAR_NEAR_THR 0.6
#define AI_CLARITY_FAR_NEAR_FAR_THR 0.6
#define AI_CLARITY_FAR_NEAR_CLEAR_THR 0.4
#define AI_CLARITY_FAR_NEAR_RBC_THR 0.6
#define AI_CLARITY_FAR_NEAR_COARSE_FOCUS_THR 0.6
#define AI_CLARITY_FAR_NEAR_CHANGE_VIEW_THR 0.6

// 粗聚焦
#define AI_CLARITY_COARSE_CATEGORY_NUMS 6
#define AI_CLARITY_COARSE_CLEAR_THR 0.65

// 荧光微球
#define AI_CLARITY_COARSE_FLU_MICRO_CATEGORY_NUMS 5

#define AI_USE_TIMECNT 0
#if (AI_USE_TIMECNT)

#include "timecnt.h"
// #include "dynoLog.h"

#endif

/* 血球计数器 */
typedef struct ClarityCnt
{
    int              RBC;                // 红细胞
    int              RET;                // 网织红细胞
    int              WBC;                // 白细胞
    int              NEU;                // 中性粒
    int              LYM;                // 淋巴
    int              MONO;               // 单核
    int              EOS;                // 嗜酸
    int              BASO;               // 嗜碱
    int              PLT;                // 血小板
    std::vector<int> channel_img_nums;   // 通道下已送入的图像张数
} ClarityCnt_t;

typedef struct ClarityCtx
{
    AiCtxID_t                       ai_ctxid;
    std::vector<ClarityValue_t>     value_list;
    std::vector<std::vector<float>> cls_list;
    AiImgCallback_f                 callback;
    void*                           userdata;
    int                             chl_idx;
    bool                            img_fusion;
    bool                            debug;
    ClarityCnt                      cnt;
    ALG_DEPLOY::BasoFocusControl    baso_focus_control = ALG_DEPLOY::BasoFocusControl(AI_CLARITY_FAR_NEAR_FAR,
                                                                                   AI_CLARITY_FAR_NEAR_NEAR,
                                                                                   AI_CLARITY_FAR_NEAR_COARSE_FOCUS,
                                                                                   AI_CLARITY_FAR_NEAR_CLEAR,
                                                                                   AI_CLARITY_FAR_NEAR_INDETERMINATION,
                                                                                   AI_CLARITY_FAR_NEAR_CHANGE_VIEW,
                                                                                   AI_CLARITY_FAR_NEAR_VERY_NEAR,
                                                                                   AI_CLARITY_FAR_NEAR_FORCE_CLEAR);

} ClarityCtx_t;
#define CLARITY_CTX_AI_CTXID(ctx) ((ctx)->ai_ctxid)                   // AI上下文ID
#define CLARITY_CTX_VALUE_LIST(ctx) ((ctx)->value_list)               // 清晰度值列表(梯度,嗜碱)
#define CLARITY_CTX_CALLBACK(ctx) ((ctx)->callback)                   // 清晰度回调
#define CLARITY_CTX_USERDATA(ctx) ((ctx)->userdata)                   // 用户数据
#define CLARITY_CTX_IMG_FUSION(ctx) ((ctx)->img_fusion)               // 像元融合
#define CLARITY_CTX_CLS_LIST(ctx) ((ctx)->cls_list)                   // 清晰度AI结果列表
#define CLARITY_CTX_DEBUG(ctx) ((ctx)->debug)                         // 是否启用debug
#define CLARITY_CTX_CNT(ctx) ((ctx)->cnt)                             // 是否启用debug
#define CLARITY_CTX_BASO_FOCUS_CTL(ctx) ((ctx)->baso_focus_control)   // baso 聚焦逻辑控制
/* 血球计数器映射表 */
std::vector<AiMap_t> heamo_cnt_maplist_clarity = {
    AI_MAP_DEF(ClarityCnt, BASO, "BASO", 0.1),
};

/* 清晰度模型注册表 */
std::vector<AiModReg_t> clarity_mod_reglist = {
    AI_MOD_DEF(NNET_MODID_BAS_CLARITY, "CLARITY_BASO", 0, 0, NNET_GROUP_CLARITY_AI, LETTERBOX, 1, 1, 1, 18, 1, 0, 0),
    AI_MOD_DEF(NNET_MODID_AI_CLARITY, "CLARITY_NORMAL", 0, 0, NNET_GROUP_CLARITY_AI, NORMAL, 1, 0, 0, 0, 3, 0, 0),
    AI_MOD_DEF(NNET_MODID_AI_CLARITY_FAR_NEAR, "CLARITY_FAR_NEAR", 0, 0, NNET_GROUP_CLARITY_AI, LEFT_TOP_CROP, 1, 0, 0, 0, 8, 0, 0),
    AI_MOD_DEF(NNET_MODID_AI_CLARITY_BASO_FAR_NEAR, "CLARITY_BASO_FAR_NEAR", 0, 0, NNET_GROUP_CLARITY_AI, LEFT_TOP_CROP, 1, 0, 0, 0, 8, 0, 0),
    AI_MOD_DEF(NNET_MODID_AI_CLARITY_COARSE, "CLARITY_COARSE", 0, 0, NNET_GROUP_CLARITY_AI, LEFT_TOP_CROP, 1, 0, 0, 0, 6, 0, 0),
    AI_MOD_DEF(NNET_MODID_AI_CLARITY_COARSE_FLU_MICRO, "CLARITY_FLUO", 0, 0, NNET_GROUP_CLARITY_AI, LEFT_TOP_CROP, 1, 0, 0, 0, 5, 0, 0),
    AI_MOD_DEF(NNET_MODID_AI_CLARITY_FINE_FLU_MICRO, "CLARITY_FLUO", 0, 0, NNET_GROUP_CLARITY_AI, LEFT_TOP_CROP, 1, 0, 0, 0, 5, 0, 0),
    AI_MOD_DEF(NNET_MODID_AI_CLARITY_MILK_BOARDLINE, "CLARITY_MILK_BOARDLINE", 0, 0, NNET_GROUP_CLARITY_AI, LEFT_TOP_CROP, 1, 0, 0, 0, 8, 0, 0)

};

// 视图注册表可共用
/* 梯度视图注册表 */
std::vector<AiViewReg_t> clarity1_view_reglist = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_UNKOWN, NNET_GROUP_ALL, 0.5)};

/* AI通用聚焦视图注册表 */
std::vector<AiViewReg_t> clarity_ai_normal_view_reglist = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_AI_CLARITY, NNET_GROUP_ALL, 0.5)};

/* BASO 聚焦视图注册表 */
std::vector<AiViewReg_t> clarity_baso_view_reglist = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_BAS_CLARITY, NNET_GROUP_ALL, 0.5)};

/* AI 聚焦停止版视图注册表 */
std::vector<AiViewReg_t> clarity_far_near_view_reglist = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_AI_CLARITY_FAR_NEAR, NNET_GROUP_ALL, 0.5)};

/* AI 聚焦停止版视图注册表 */
std::vector<AiViewReg_t> clarity_baso_far_near_view_reglist = {
    AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_AI_CLARITY_BASO_FAR_NEAR, NNET_GROUP_ALL, 0.5)};

/* AI 粗聚焦 */
std::vector<AiViewReg_t> clarity_coarse_view_reglist = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_AI_CLARITY_COARSE, NNET_GROUP_ALL, 0.5)};

/* AI 粗聚焦_荧光微球 */
std::vector<AiViewReg_t> clarity_coarse_flu_microsphere_view_reglist = {
    AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_AI_CLARITY_COARSE_FLU_MICRO, NNET_GROUP_ALL, 0.5)};

/* AI 细聚焦_荧光微球 */
std::vector<AiViewReg_t> clarity_fine_flu_microsphere_view_reglist = {
    AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_AI_CLARITY_FINE_FLU_MICRO, NNET_GROUP_ALL, 0.5)};

/* 牛奶底板划线--AI 粗聚焦 */
std::vector<AiViewReg_t> clarity_milk_boardline = {AI_VIEW_DEF(AI_VIEW_TYPE_BRI, NNET_MODID_AI_CLARITY_MILK_BOARDLINE, NNET_GROUP_ALL, 0.5)};

/* 清晰度图像处理函数声明 */
static int Clarity_ImgGradient(AiCtxID_t                           ctx_id,
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
                               const std::map<std::string, float>& call_back_params);

static int Clarity_ImgBaso(AiCtxID_t                           ctx_id,
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
                           const std::map<std::string, float>& call_back_params);

static int Clarity_ImgNormal(AiCtxID_t                           ctx_id,
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
                             const std::map<std::string, float>& call_back_params);

static int Clarity_ImgFarNear(AiCtxID_t                           ctx_id,
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
                              const std::map<std::string, float>& call_back_params);
/* 清晰度通道注册表 */   // 梯度聚焦两种峰梯度计算计时函数相同,只是返回的峰位置不同,二者的差异在获取清晰度最大值函数
                         // Clarity_GetBestValue
// 为保持结构一致,通道枚举类型序号与血球共用
std::vector<AiChlReg_t> clarity_chl_reglist = {
    AI_CHL_DEF(AI_CHL_TYPE_RBC, "梯度聚焦_最高峰", clarity1_view_reglist, Clarity_ImgGradient, NNET_GROUP_ALL),
    AI_CHL_DEF(AI_CHL_TYPE_WBC, "AI粗聚焦", clarity_coarse_view_reglist, Clarity_ImgFarNear, NNET_GROUP_ALL),
    AI_CHL_DEF(AI_CHL_TYPE_HGB, "AI聚焦_停止_白细胞", clarity_far_near_view_reglist, Clarity_ImgFarNear, NNET_GROUP_ALL),
    AI_CHL_DEF(AI_CHL_TYPE_RET, "AI聚焦_停止_嗜碱细胞", clarity_baso_far_near_view_reglist, Clarity_ImgFarNear, NNET_GROUP_ALL),

    AI_CHL_DEF(AI_CHL_TYPE_MILK_GERM_0, "AI粗聚焦_荧光微球", clarity_coarse_flu_microsphere_view_reglist, Clarity_ImgFarNear, NNET_GROUP_ALL),
    AI_CHL_DEF(AI_CHL_TYPE_MILK_GERM_1, "AI细聚焦_荧光微球", clarity_fine_flu_microsphere_view_reglist, Clarity_ImgFarNear, NNET_GROUP_ALL),
    AI_CHL_DEF(AI_CHL_TYPE_RET, "AI粗聚焦_嗜碱细胞", clarity_baso_far_near_view_reglist, Clarity_ImgFarNear, NNET_GROUP_ALL),
    AI_CHL_DEF(AI_CHL_TYPE_MILK_BOARDLINE, "AI粗聚焦_牛奶底板划线", clarity_milk_boardline, Clarity_ImgFarNear, NNET_GROUP_ALL),
    //		AI_CHL_DEF(AI_CHL_TYPE_RBC, "梯度聚焦_第一峰",
    // clarity1_view_reglist, Clarity_ImgGradient, NNET_GROUP_ALL),
    //		AI_CHL_DEF(AI_CHL_TYPE_WBC, "AI聚焦_通用",
    // clarity_ai_normal_view_reglist, Clarity_ImgNormal, NNET_GROUP_ALL),
    //		AI_CHL_DEF(AI_CHL_TYPE_HGB, "AI聚焦_嗜碱",
    // clarity_baso_view_reglist, Clarity_ImgBaso, NNET_GROUP_ALL),
    //          AI_CHL_DEF(AI_CHL_TYPE_MILK_CELL_0, "AI聚焦_停止_红细胞_网织红",
    //          clarity_far_near_view_reglist, Clarity_ImgFarNear,
    //          NNET_GROUP_ALL),

};
/* 清晰度分组注册表 */
std::vector<AiGroupReg_t> clarity_group_reglist = {
    AI_GROUP_DEF(AI_GROUP_TYPE_NONE, NNET_GROUP_ALL, "清晰度", "/alg/model/20", clarity_mod_reglist, heamo_cnt_maplist_clarity, clarity_chl_reglist)};

inline AiGroupReg_t* Clarity_FindGroup(uint32_t group_idx)
{
    Ai_FindGroup(clarity_group_reglist, group_idx);
}

inline AiModReg_t* Clarity_FindMod(AiGroupReg_t* group, NNetModID_e mod_id)
{
    Ai_FindMod(group, mod_id);
}

inline AiChlReg_t* Clarity_FindChl(AiGroupReg_t* group, uint32_t chl_idx)
{
    Ai_FindChl(group, chl_idx);
}

inline AiViewReg_t* Clarity_FindView(AiGroupReg_t* group, AiChlReg_t* chl, uint32_t view_idx)
{
    Ai_FindView(group, chl, view_idx);
}

/**
 * 获取清晰度样本分组注册表
 * @param  list			目标列表
 * @return  0 success other fail
 */
int Clarity_GetGroupReglist(std::vector<AiGroupReg_t>& list)
{
    list = clarity_group_reglist;
    return 0;
}

/**
 * 获取清晰度模型注册表
 * @param  list			目标列表
 * @param  group_idx		分组索引
 * @return  0 success other fail
 */
int Clarity_GetModReglist(std::vector<AiModReg_t>& list, uint32_t group_idx)
{
    return Ai_GetModReglist(list, Clarity_FindGroup(group_idx));
}

/**
 * 获取清晰度通道注册表
 * @param  list			目标列表
 * @param  group_idx		分组索引
 * @return  0 success other fail
 */
int Clarity_GetChlReglist(std::vector<AiChlReg_t>& list, uint32_t group_idx)
{
    return Ai_GetChlReglist(list, Clarity_FindGroup(group_idx));
}

/**
 * 获取清晰度视图注册表
 * @param  list			目标列表
 * @param  group_idx		分组索引
 * @param  chl_idx		通道索引
 * @return  0 success other fail
 */
int Clarity_GetViewReglist(std::vector<AiViewReg_t>& list, uint32_t group_idx, uint32_t chl_idx)
{
    AiGroupReg_t* group = Clarity_FindGroup(group_idx);
    std::cout << "Found group and channel. group_idx: " << group_idx << ", chl_idx: " << chl_idx << std::endl;
    return Ai_GetViewReglist(list, group, Clarity_FindChl(group, chl_idx));
}

/**
 * 清晰度初始化
 * @param  none
 * @return
 */
ClarityCtxID_t Clarity_Init(AiCtxID_t ai_ctxid)
{
    ClarityCtx_t* ctx         = new ClarityCtx_t;
    CLARITY_CTX_AI_CTXID(ctx) = ai_ctxid;
    if (CLARITY_CTX_AI_CTXID(ctx) == NULL) {
        return NULL;
    }
    return (ClarityCtxID_t)ctx;
}

int Clarity_DeInit(ClarityCtxID_t ctx_id)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    delete ctx;
    return 0;
}

/**
 * 清晰度开启
 * @param  ctx_id		清晰度上下文ID
 * @return
 */
int Clarity_Open(ClarityCtxID_t ctx_id, const bool& img_fusion, const bool& debug, AiImgCallback_f callback, void* userdata)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    Ai_WaitPriority(CLARITY_CTX_AI_CTXID(ctx), CLARITY_DEF_PRIORITY, 0xFFFF);
    CLARITY_CTX_CALLBACK(ctx) = callback;
    CLARITY_CTX_USERDATA(ctx) = userdata;
    CLARITY_CTX_VALUE_LIST(ctx).clear();
    CLARITY_CTX_IMG_FUSION(ctx) = img_fusion;
    CLARITY_CTX_DEBUG(ctx)      = debug;

    // 获取最大通道数,以保存各个通道已接受图像对的数量
    int max_channel_nums;
    Ai_FindMaxChannelNums(clarity_group_reglist, max_channel_nums);
    CLARITY_CTX_CNT(ctx).channel_img_nums = std::vector<int>(max_channel_nums, 0);
    return 0;
}

/**
 * 清晰度生成图像列表
 * @param  img_list		图像列表
 * @param  group_idx		分组索引
 * @param  chl_idx		通道索引
 * @param  view_idx		视图索引
 * @param  img_array		图像缓存指针
 * @param  width			图像宽度
 * @param  heigh			图像高度
 * @param  img_fusion    像元融合使能
 * @return
 **/
// 为了不将原始图像拷贝至算法队列,直接将图像resize后进行拷贝.这样能极大减少拷贝数据量

#define CLARITY_IMG_INPUT_NORAML 320
#define CLARITY_IMG_INPUT_BASO 640

int Clarity_AddImgList(ClarityCtxID_t           ctx_id,
                       std::list<ClarityImg_t>& list,
                       uint32_t                 group_idx,
                       uint32_t                 chl_idx,
                       uint32_t                 view_idx,
                       uint8_t*                 img_data,
                       uint32_t                 width,
                       uint32_t                 height)
{
    std::cout << "chl_idx in Clarity_AddImgList: " << chl_idx << std::endl;
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        ALGLogError << "Null ptr";
        return -1;
    }
    if (img_data == NULL || !(width * height)) {
        ALGLogError << "Empty img";
        return -2;
    }
    AiGroupReg_t* group           = Clarity_FindGroup(group_idx);
    AiChlReg_t*   chl             = Clarity_FindChl(group, chl_idx);
    AiViewReg_t*  view            = Clarity_FindView(group, chl, view_idx);
    ResizeType    resize_type     = ResizeType::LEFT_TOP_CROP;
    int           target_img_size = CLARITY_IMG_INPUT_NORAML;
    // 嗜碱流道图像白细胞数量可能较少,为使模型输入为整张图像,不进行crop
    if (chl_idx == 3 || chl_idx == 6) {
        resize_type     = ResizeType::NORMAL;
        target_img_size = CLARITY_IMG_INPUT_BASO;
    }

    if (view != nullptr) {

        int ret = 0;
        if (view->mod_id == NNET_MODID_UNKOWN) {
            //      ret  =  Ai_ConvertImage(list, img_data, width, height,
            //      CLARITY_CTX_IMG_FUSION(ctx), view->fusion_rate);

            ret = Ai_ConvertClarityImage(
                list, img_data, width, height, CLARITY_CTX_IMG_FUSION(ctx), view->fusion_rate, target_img_size, target_img_size, resize_type);

            return ret;
        }
        AiModReg_t* mod = Clarity_FindMod(group, AI_VIEW_MOD_ID(view));
        if (mod) {
            //      ret  =  Ai_ConvertImage(list, img_data, width, height,
            //      CLARITY_CTX_IMG_FUSION(ctx), view->fusion_rate);
            ret = Ai_ConvertClarityImage(
                list, img_data, width, height, CLARITY_CTX_IMG_FUSION(ctx), view->fusion_rate, target_img_size, target_img_size, resize_type);
            return ret;
        }
    }
    ALGLogError << "Null view, group_idx, chl_idx, view_idx " << group_idx << " " << chl_idx << " " << view_idx << " ";
    return -3;
}

/**
 * 清晰度推图
 * @param  ctx_id		清晰度上下文ID
 * @param  chl_idx		清晰度通道索引
 * @param  width			图像宽度
 * @param  height		图像高度
 * @param  img_array		图像缓存指针数组
 * @param  array_size	数组尺寸
 * @return
 */
// int Clarity_PushImage(ClarityCtxID_t ctx_id, uint32_t group_idx, uint32_t
// chl_idx, std::list<ClarityImg_t> &img_list, const int& view_pair_idx) {
//	ClarityCtx_t *ctx = (ClarityCtx_t *) ctx_id;
//	if (ctx == NULL) {
//		//    DLOG(ERROR, "null ctx");
//		return -1;
//	}
//
//	group_idx = group_idx;
//	ctx->chl_idx = int(chl_idx);
//
//	AiChlReg_t *chl = Clarity_FindChl(Clarity_FindGroup(group_idx),
// chl_idx); 	if (chl == NULL) { 		return -2;
//	}
//	std::vector<AiViewReg_t> view_list;
//	if (Clarity_GetViewReglist(view_list, group_idx, chl_idx)) {
//		return -3;
//	}
//	if (!view_list.size() || img_list.size() < view_list.size()) {
//		return -4;
//	}
//	std::map<std::string, float> call_back_params{{TASK_TYPE,
// TASK_TYPE_CLARITY}}; 	if (AI_CHL_GROUP_MASK(chl)) { 		if
//(!Ai_ItemPush(CLARITY_CTX_AI_CTXID(ctx), CLARITY_DEF_PRIORITY, group_idx,
// chl_idx, ctx->cnt.channel_img_nums[chl_idx], view_list.size(), 		                 img_list,
// AI_CHL_CALLBACK(chl), ctx, view_list, view_pair_idx, call_back_params)) {
//			return -6;
//		}
//                 ctx->cnt.channel_img_nums[chl_idx]+= 1;
//	} else {
//		if (!AI_CHL_CALLBACK(chl)) {
//			return -7;
//		}
//		Ai_WaitPriority(CLARITY_CTX_AI_CTXID(ctx), CLARITY_DEF_PRIORITY,
// 0xFFFF); 		uint32_t idx = 0; 		std::list<NNetResult_t> result; 		for (auto &img:
// img_list) { 			if (idx++ >= view_list.size()) { 				break;
//			}
//			int ret = (*AI_CHL_CALLBACK(chl))(NULL, 0, &img,
// group_idx, chl_idx, 0, idx, 0, AI_IMG_STATGE_UNDEFINED, 			                                  (void *) ctx, result,
// view_pair_idx, call_back_params);
//		}
//	}
//	return 0;
// }
#include <iostream>

int Clarity_PushImage(ClarityCtxID_t ctx_id, uint32_t group_idx, uint32_t chl_idx, std::list<ClarityImg_t>& img_list, const int& view_pair_idx)
{
    std::cout << "Entering Clarity_PushImage..." << std::endl;

    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        std::cout << "Error: ctx is NULL" << std::endl;
        return -1;
    }
    group_idx    = group_idx;
    ctx->chl_idx = int(chl_idx);
    std::cout << "group_idx: " << group_idx << ", chl_idx: " << chl_idx << std::endl;

    AiChlReg_t* chl = Clarity_FindChl(Clarity_FindGroup(group_idx), chl_idx);
    if (chl == NULL) {
        std::cout << "Error: Clarity_FindChl returned NULL" << std::endl;
        return -2;
    }

    std::vector<AiViewReg_t> view_list;
    if (Clarity_GetViewReglist(view_list, group_idx, chl_idx)) {
        std::cout << "Error: Clarity_GetViewReglist failed" << std::endl;
        return -3;
    }

    std::cout << "view_list.size(): " << view_list.size() << ", img_list.size(): " << img_list.size() << std::endl;

    if (!view_list.size() || img_list.size() < view_list.size()) {
        std::cout << "Error: Invalid view_list size or img_list size too small" << std::endl;
        return -4;
    }

    std::cout << "Processing AI_CHL_GROUP_MASK..." << std::endl;

    std::map<std::string, float> call_back_params{{TASK_TYPE, TASK_TYPE_CLARITY}};
    if (AI_CHL_GROUP_MASK(chl)) {
        std::cout << "AI_CHL_GROUP_MASK is set" << std::endl;
        if (!Ai_ItemPush(CLARITY_CTX_AI_CTXID(ctx),
                         CLARITY_DEF_PRIORITY,
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
            std::cout << "Error: Ai_ItemPush failed" << std::endl;
            return -6;
        }
        ctx->cnt.channel_img_nums[chl_idx] += 1;
    }
    else {
        std::cout << "AI_CHL_GROUP_MASK is NOT set" << std::endl;
        if (!AI_CHL_CALLBACK(chl)) {
            std::cout << "Error: AI_CHL_CALLBACK is NULL" << std::endl;
            return -7;
        }
        Ai_WaitPriority(CLARITY_CTX_AI_CTXID(ctx), CLARITY_DEF_PRIORITY, 0xFFFF);
        uint32_t                idx = 0;
        std::list<NNetResult_t> result;
        for (auto& img : img_list) {
            if (idx++ >= view_list.size()) {
                break;
            }
            std::cout << "Calling AI_CHL_CALLBACK with idx = " << idx << std::endl;
            int ret = (*AI_CHL_CALLBACK(chl))(
                NULL, 0, &img, group_idx, chl_idx, 0, idx, 0, AI_IMG_STATGE_UNDEFINED, (void*)ctx, result, view_pair_idx, call_back_params);
            std::cout << "AI_CHL_CALLBACK returned: " << ret << std::endl;
        }
    }

    std::cout << "Exiting Clarity_PushImage successfully" << std::endl;
    return 0;
}

/**
 * 清晰度等待完成
 * @param  ctx_id		清晰度上下文ID
 * @param  timeout		超时时间
 * @return
 */
int Clarity_WaitCplt(ClarityCtxID_t ctx_id, uint32_t timeout)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == nullptr) {
        ALGLogInfo << "上下文 ctx 为空";
        return -1;
    }
    return Ai_WaitPriority(CLARITY_CTX_AI_CTXID(ctx), CLARITY_DEF_PRIORITY, timeout);
}

/**
 * 获取清晰度
 * @param  ctx_id		清晰度上下文ID
 * @param  ipt_idx		输入索引
 * @param  value			数值输出指针
 * @return
 */
int Clarity_GetValue(ClarityCtxID_t ctx_id, uint32_t ipt_idx, ClarityValue_t* value)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL || value == NULL) {
        return -1;
    }
    if (true == CLARITY_CTX_VALUE_LIST(ctx).empty()) {
        return -2;
    }
    if (ipt_idx < CLARITY_CTX_VALUE_LIST(ctx).size()) {
        if (value) {
            *value = CLARITY_CTX_VALUE_LIST(ctx).at(ipt_idx);
        }
        return 0;
    }
    return -3;
}

/**
 * 获取所有清晰度
 * @param  ctx_id		清晰度上下文ID
 * @param  index			索引输出指针
 * @param  value			数值输出指针
 * @return
 */
int Clarity_GetAllValue(ClarityCtxID_t ctx_id, std::vector<ClarityValue_t>& list)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    list = CLARITY_CTX_VALUE_LIST(ctx);
    return 0;
}

/**
 * 获取最近清晰度
 * @param  ctx_id		清晰度上下文ID
 * @param  index			索引输出指针
 * @param  value			数值输出指针
 * @return
 */
int Clarity_GetLastValue(ClarityCtxID_t ctx_id, uint32_t* index, ClarityValue_t* value)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        ALGLogError << "Null ptr";
        return -1;
    }
    if (true == CLARITY_CTX_VALUE_LIST(ctx).empty()) {
        ALGLogError << "Empty value list";
        return -2;
    }
    if (index) {
        *index = CLARITY_CTX_VALUE_LIST(ctx).size() - 1;
    }
    if (value) {
        *value = CLARITY_CTX_VALUE_LIST(ctx).back();
    }
    return 0;
}

static int Clarity_PeekSeekHighest(std::vector<ClarityValue_t>& list, uint32_t* index, ClarityValue_t* value)
{
    uint32_t       oft     = 0;
    uint32_t       max_idx = 0;
    ClarityValue_t max_val = 0.0;
    for (auto temp : list) {
        if (temp > max_val) {
            max_val = temp;
            max_idx = oft;
        }
        oft++;
    }
    if (index) {
        *index = max_idx;
    }
    if (value) {
        *value = max_val;
    }
    return 0;
}

static int Clarity_CurFarNear(std::vector<ClarityValue_t>& list, uint32_t* index, ClarityValue_t* value)
{
    if (list.size() != 1) {
        ALGLogError << "For clarity far near, list size must be 1, but " << list.size() << " was given";
        return -1;
    }
    if (index) {
        *index = 0;
    }
    if (value) {
        *value = list[0];
    }
    ALGLogInfo << "Far near result in alg " << *value;
    // 远,近焦模式结果列表仅能有一个值.每次获取后清空
    list.clear();
    return 0;
}

struct ClarityItem
{
    int   index;
    float clarity;
};

// 计算第一个峰
static int Clarity_PeekSeekFirst(std::vector<ClarityValue_t>& list, uint32_t* index, ClarityValue_t* value)
{

    if (list.empty()) {
        return -1;
    }
    else if (list.size() == 1) {
        *index = 0;
        *value = list[0];
        return 0;
    }
    std::vector<int>   peakIdList;       // 峰id
    std::vector<int>   minIdList;        // 左侧波谷id
    std::vector<float> heightPeakList;   // 峰高
    std::vector<int>   widthPeakList;    // 峰宽
    // 筛选峰的阈值
    float heightValue;
    int   widthValue;
    int   focusPos     = -1;   // 区分第一个位置
    float focusClarity = 0.0;
    // 整合数据
    std::vector<ClarityItem> s_clarities;

    ClarityItem clarity_item;
    for (int idx = 0; idx < list.size(); ++idx) {
        clarity_item.index   = idx;
        clarity_item.clarity = list[idx];
        s_clarities.emplace_back(clarity_item);
    }

    std::vector<ClarityItem>::iterator it;
    // 初步寻峰
    for (it = s_clarities.begin() + 1; it != s_clarities.end() - 1; it++) {
        if (((it->clarity > (it - 1)->clarity) && (it->clarity >= (it + 1)->clarity)) ||
            ((it->clarity >= (it - 1)->clarity) && (it->clarity > (it + 1)->clarity))) {
            peakIdList.push_back(it->index);
        }
    }
    // 如果没有峰，找到区域最大值,同样适用于vector只有2个数
    if (peakIdList.empty()) {
        for (it = s_clarities.begin(); it != s_clarities.end(); it++) {
            if (it->clarity > focusClarity) {
                *index = it->index;
                *value = it->clarity;
                ALGLogInfo << "No peak";
            }
        }
    }
    else {   // 找左侧波谷
        for (unsigned int j = 0; j < peakIdList.size(); j++) {
            if (j == 0) {
                int   minimalPos     = -1;
                float minimalClarity = s_clarities.at(0).clarity;
                for (unsigned int z = 0; z < peakIdList[j]; z++) {
                    if (s_clarities.at(z).clarity < s_clarities.at(0).clarity) {
                        minimalClarity = s_clarities.at(z).clarity;
                        minimalPos     = z;
                    }
                    else {
                        minimalPos = 0;
                    }
                }
                minIdList.push_back(minimalPos);
            }
            else {
                int   minimalPos     = -1;
                float minimalClarity = s_clarities.at(peakIdList[j - 1]).clarity;
                for (unsigned int k = s_clarities.at(peakIdList[j - 1]).index; k < s_clarities.at(peakIdList[j]).index; k++) {
                    if (s_clarities.at(k).clarity < minimalClarity) {
                        minimalClarity = s_clarities.at(k).clarity;
                        minimalPos     = k;
                    }
                    //          else{
                    //            minimalPos = j - 1;
                    //          }
                }
                if (minimalPos == -1) {
                    minimalPos = j - 1;
                }
                minIdList.push_back(minimalPos);
            }
        }

        // 得到最高波谷与波峰的距离
        for (unsigned int i = 0; i < minIdList.size(); i++) {
            float height = s_clarities.at(peakIdList[i]).clarity - s_clarities.at(minIdList[i]).clarity;
            heightPeakList.push_back(height);
            int width = s_clarities.at(peakIdList[i]).index - s_clarities.at(minIdList[i]).index;
            widthPeakList.push_back(width);
        }
        // 得到排除其他峰的宽、高阈值
        float height = 0.0;
        int   width  = 0;
        for (unsigned int i = 0; i < heightPeakList.size(); i++) {
            // 综合峰高峰宽
            //			if (heightPeakList[i] > height) {
            //				height = heightPeakList[i];
            //			}
            //			heightValue = height / 2.0;
            //			if (widthPeakList[i] > width) {
            //				width = widthPeakList[i];
            //			}
            //			widthValue = width / 2.0;

            // 最宽峰

            if (widthPeakList[i] > width) {
                width       = widthPeakList[i];
                height      = heightPeakList[i];
                widthValue  = width / 2.0;
                heightValue = height / 2.0;
            }
        }
        // 删除低峰、窄峰
        // 取有意义的峰中的第一个为清晰的聚焦位置
        std::vector<int>   exact_peak_idx;
        std::vector<float> exact_peak_clarity;
        for (unsigned int i = 0; i < minIdList.size(); i++) {
            float heightPeak = s_clarities.at(peakIdList[i]).clarity - s_clarities.at(minIdList[i]).clarity;
            int   widthPeak  = s_clarities.at(peakIdList[i]).index - s_clarities.at(minIdList[i]).index;
            if (heightPeak > heightValue && widthPeak > widthValue) {
                //        *index = s_clarities.at(peakIdList[i]).index ;
                //        *value= s_clarities.at(peakIdList[i]).clarity;
                exact_peak_idx.push_back(s_clarities.at(peakIdList[i]).index);
                exact_peak_clarity.push_back(s_clarities.at(peakIdList[i]).clarity);
            }
        }
        // 根据是否有峰选择第一个峰或者第二个峰
        if (exact_peak_idx.empty()) {
            *index = 0;
            *value = 0.f;
        }
        else if (exact_peak_idx.size() <= 2) {
            *index = exact_peak_idx[0];
            *value = exact_peak_clarity[0];
        }
        else {
            *index = exact_peak_idx[1];
            *value = exact_peak_clarity[1];
        }
        ALGLogInfo << "Peak nums " << exact_peak_idx.size();
    }
    return 0;
}

/**
 * 获取最佳清晰度
 * @param  ctx_id		清晰度上下文ID
 * @param  index			索引输出指针
 * @param  value			数值输出指针
 * @return
 */
int Clarity_GetBestValue(ClarityCtxID_t ctx_id, uint32_t* index, ClarityValue_t* value)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    if (true == CLARITY_CTX_VALUE_LIST(ctx).empty()) {
        EVERROR(EVID_ERR, "CLARITY_CTX_VALUE_LIST is null");
        return -2;
    }

    switch (ctx->chl_idx) {   // 根据push的channel idx选择不同的返回类型
    case 0:
        ALGLogInfo << "Clarity_PeekSeekHighest";
        return Clarity_PeekSeekHighest(CLARITY_CTX_VALUE_LIST(ctx), index, value);
        /*case 1:
                        ALGLogInfo << "Clarity_PeekSeekFirst";
                        return Clarity_PeekSeekFirst(CLARITY_CTX_VALUE_LIST(ctx),
        index, value); case 2: ALGLogInfo << "Clarity_PeekSeekHighest"; return
        Clarity_PeekSeekHighest(CLARITY_CTX_VALUE_LIST(ctx), index, value); case 3:
                        ALGLogInfo << "Clarity_PeekSeekHighest";
                        return Clarity_PeekSeekHighest(CLARITY_CTX_VALUE_LIST(ctx),
        index, value);*/
    case 1:
        ALGLogInfo << "clarity_CoarseFarNear";
        return Clarity_CurFarNear(CLARITY_CTX_VALUE_LIST(ctx), index, value);
    case 2:
        ALGLogInfo << "Clarity_FineWbcRbc";
        return Clarity_CurFarNear(CLARITY_CTX_VALUE_LIST(ctx), index, value);
    case 3:
        ALGLogInfo << "Clarity_FineBaso";
        return Clarity_CurFarNear(CLARITY_CTX_VALUE_LIST(ctx), index, value);
    case 4:
        ALGLogInfo << "Clarity_CoarseFluMicrosphere";
        return Clarity_CurFarNear(CLARITY_CTX_VALUE_LIST(ctx), index, value);
    case 5:
        ALGLogInfo << "Clarity_FineFluMicrosphere";
        return Clarity_CurFarNear(CLARITY_CTX_VALUE_LIST(ctx), index, value);
    case 6:
        ALGLogInfo << "Clarity_CoarseBaso";
        return Clarity_CurFarNear(CLARITY_CTX_VALUE_LIST(ctx), index, value);
    case 7:
        ALGLogInfo << "Clarity_MILK_BoardLINE";
        return Clarity_CurFarNear(CLARITY_CTX_VALUE_LIST(ctx), index, value);
    default:
        ALGLogInfo << "Unknown peek seek method";
        return -3;
    }
}

/**
 * 清晰度关闭
 * @param  ctx_id		清晰度上下文ID
 * @return
 */
int Clarity_Close(ClarityCtxID_t ctx_id)
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    CLARITY_CTX_VALUE_LIST(ctx).clear();
    CLARITY_CTX_CALLBACK(ctx) = NULL;
    CLARITY_CTX_USERDATA(ctx) = NULL;
    memset(&ctx->cnt, 0, sizeof(ctx->cnt));
    return 0;
}

static int Clarity_DoImgCallback(ClarityCtx_t*                       ctx,
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
    if (CLARITY_CTX_CALLBACK(ctx)) {
        int ret = (*CLARITY_CTX_CALLBACK(ctx))(CLARITY_CTX_AI_CTXID(ctx),
                                               item_id,
                                               img,
                                               group_idx,
                                               chl_idx,
                                               view_order,
                                               view_idx,
                                               processed_idx,
                                               stage,
                                               CLARITY_CTX_USERDATA(ctx),
                                               result,
                                               view_pair_idx,
                                               call_back_params);
        return ret;
    }
    return 0;
}

static int Clarity_AddValue(ClarityCtx_t* ctx, ClarityValue_t value)
{
    if (ctx == NULL) {
        return -1;
    }
    CLARITY_CTX_VALUE_LIST(ctx).push_back(value);
    return 0;
}

static int Clarity_AddList(ClarityCtx_t* ctx, std::vector<float>& cls_list)
{
    if (ctx == NULL) {
        return -1;
    }
    //	DIHLogInfo<<"Clarity_AddValue push value";
    CLARITY_CTX_CLS_LIST(ctx).push_back(cls_list);
    return 0;
}

// 获取图像的平均梯度值
static int Clarity_GetClarity(ClarityValue_t& value, cv::Mat& img)
{
#if (AI_USE_TIMECNT)
    TimeCnt_Start("梯度计算");
#endif
    cv::Mat oriImg = img;
    cv::Mat grayImg, hFeatureMap, vFeatureMap, featureMap;
    if (oriImg.channels() != 3) {
        return -2;
    }
    cv::cvtColor(oriImg, grayImg, cv::COLOR_RGB2GRAY);
    cv::Mat hkernel = (cv::Mat_<int>(3, 3) << -1, 0, 1, 0, 0, 0, 0, 0, 0);
    cv::Mat vkernel = (cv::Mat_<int>(3, 3) << -1, 0, 0, 0, 0, 0, 1, 0, 0);

    filter2D(grayImg, hFeatureMap, -1, hkernel);
    filter2D(grayImg, vFeatureMap, -1, vkernel);
    multiply(hFeatureMap, vFeatureMap, featureMap);

    value = (ClarityValue_t)mean(featureMap)[0];
#if (AI_USE_TIMECNT)
    TimeCnt_End("梯度计算");
#endif
    return 0;
}

static int Clarity_ImgGradient(AiCtxID_t                           ctx_id,
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
    //    DLOG(INFO, "run gradient");
    if (stage != AI_IMG_STAGE_INTERENCE) {
        return 0;
    }
    ClarityCtx_t* ctx = (ClarityCtx_t*)userdata;
    if (ctx == NULL) {
        //    DLOG(ERROR, "null ctx");
        return -1;
    }
    int      cx = img->cols / 2;
    int      cy = img->rows / 2;
    int      x  = int(std::min(img->cols, img->rows) / 4);
    cv::Rect target_rect(cx - x, cy - x, 2 * x, 2 * x);
    target_rect = target_rect & cv::Rect(0, 0, img->cols, img->rows);
    cv::Mat        temp_img(*img, target_rect);
    ClarityValue_t value = 0.0;
    Clarity_GetClarity(value, temp_img);
    Clarity_AddValue(ctx, value);
    //	Clarity_DoImgCallback(ctx, 0, img, group_idx, chl_idx, view_idx,
    // AI_IMG_STAGE_CLARITY, result);
    return 0;
}

// 绘制结果并保存
static int MakeMidResult(AiCtxID_t                           ctx_id,
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
                         const cv::Scalar&                   box_color  = cv::Scalar(0, 0, 255),
                         const cv::Scalar&                   font_color = cv::Scalar(255, 0, 0))
{
    ClarityCtx_t* ctx = (ClarityCtx_t*)userdata;
    if (ctx == NULL) {
        return -1;
    }
    // 保存结果
    if (ctx->debug) {
        cv::Mat                 mid_result;
        std::vector<NNetResult> detect_result_v(result.begin(), result.end());
        DrawMidResult(img, img->rows, img->cols, detect_result_v, mid_result, draw_name, font_scale, thickness, box_color, font_color);
        return Clarity_DoImgCallback(
            ctx, item_id, &mid_result, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
    }
    return 0;
}

static int Clarity_ImgNormal(AiCtxID_t                           ctx_id,
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
    //    DLOG(INFO, "Clarity_ImgRBC");
    if (stage != AI_IMG_STAGE_INTERENCE) {
        return 0;
    }
    ClarityCtx_t* ctx = (ClarityCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return -1;
    }
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Clarity_DoImgCallback(ctx, 0, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    group_idx           = group_idx;
    AiGroupReg_t* group = Clarity_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "Null group ptr";
        return -2;
    }

    AiViewReg_t* view = Clarity_FindView(group, Clarity_FindChl(group, chl_idx), view_idx);
    if (view == nullptr) {
        ALGLogError << "Null view ptr";
        return -3;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);

    if (Ai_Inference(CLARITY_CTX_AI_CTXID(ctx), group_id, mod_id, img, result, AI_INFER_TYPE_NORMAL)) {
        ALGLogError << "Failed to inference model";
        return -4;
    }

    ClarityValue_t value = 0.0;
    if (result.size()) {

        // 存入三个类别的概率
        Clarity_AddList(ctx, result.begin()->category_v);

        float clarity_value = result.begin()->category_v[0];
        Clarity_AddValue(ctx, clarity_value);
        return 0;
    }

    //    DLOG(WARN, "clairty model do not find cells");
    //	Clarity_DoImgCallback(ctx, 0, img, group_idx, chl_idx, view_idx,
    // AI_IMG_STAGE_CLARITY, result);
    return 0;
}

void GetFarNearProb(float&                    neg_prob,
                    float&                    pos_prob,
                    float&                    focus_prob,
                    float&                    rbc_prob,
                    float&                    uncer_prob,
                    float&                    change1_prob,
                    float&                    change2_prob,
                    float&                    wbc_prob,
                    const std::vector<float>& category_v)
{
    neg_prob     = category_v[0];
    pos_prob     = category_v[1];
    focus_prob   = category_v[2];
    rbc_prob     = category_v[3];
    uncer_prob   = category_v[4];
    change1_prob = category_v[5];
    change2_prob = category_v[6];
    wbc_prob     = category_v[7];
}

int MakeRestulFarNearWbc(const std::vector<float>& category_v, float& result)
{
    if (category_v.size() != 8) {
        ALGLogError << "Category must be 8 but " << category_v.size() << " was given";
        return -1;
    }
    float neg_prob, pos_prob, focus_prob, rbc_prob, uncer_prob, change1_prob, change2_prob, wbc_prob;
    GetFarNearProb(neg_prob, pos_prob, focus_prob, rbc_prob, uncer_prob, change1_prob, change2_prob, wbc_prob, category_v);

    if (wbc_prob > AI_CLARITY_FAR_NEAR_CLEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_CLEAR;
    }
    else if (neg_prob > AI_CLARITY_FAR_NEAR_FAR_THR) {
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else if (rbc_prob > AI_CLARITY_FAR_NEAR_CLEAR_THR) {   // 将rbc清晰的面视为过焦
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (change1_prob > AI_CLARITY_FAR_NEAR_CHANGE_VIEW_THR || change2_prob > AI_CLARITY_FAR_NEAR_CHANGE_VIEW_THR) {
        result = AI_CLARITY_FAR_NEAR_CHANGE_VIEW;
    }
    else if (focus_prob > AI_CLARITY_FAR_NEAR_COARSE_FOCUS_THR) {
        result = AI_CLARITY_FAR_NEAR_COARSE_FOCUS;
    }
    else if (wbc_prob + neg_prob + rbc_prob > AI_CLARITY_FAR_NEAR_FAR_THR) {   // 当平面处于白细胞清晰面下方时,视为负焦
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (wbc_prob + pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {   // 当平面处于白细胞清晰面上方时,视为正焦
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else {
        result = AI_CLARITY_FAR_NEAR_INDETERMINATION;
    }
    return 0;
}

int MakeRestulFarNearRbc(const std::vector<float>& category_v, float& result)
{
    if (category_v.size() != 8) {
        ALGLogError << "Category must be 8 but " << category_v.size() << " was given";
        return -1;
    }

    if (category_v[3] > AI_CLARITY_FAR_NEAR_CLEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_CLEAR;
    }
    else if (category_v[0] > AI_CLARITY_FAR_NEAR_FAR_THR) {
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (category_v[1] > AI_CLARITY_FAR_NEAR_NEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else if (category_v[7] > AI_CLARITY_FAR_NEAR_CLEAR_THR) {   // 将rbc清晰的面视为过焦
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else if (category_v[5] > AI_CLARITY_FAR_NEAR_CHANGE_VIEW_THR || category_v[6] > AI_CLARITY_FAR_NEAR_CHANGE_VIEW_THR) {
        result = AI_CLARITY_FAR_NEAR_CHANGE_VIEW;
    }
    else if (category_v[2] > AI_CLARITY_FAR_NEAR_COARSE_FOCUS_THR) {
        result = AI_CLARITY_FAR_NEAR_COARSE_FOCUS;
    }
    else {
        result = AI_CLARITY_FAR_NEAR_INDETERMINATION;
    }
    return 0;
}

void GetCoarseProb(float&                    neg_prob,
                   float&                    pos_prob,
                   float&                    focus_prob,
                   float&                    change1_prob,
                   float&                    wbc_prob,
                   float&                    indetermination_prob,
                   const std::vector<float>& category_v)
{
    neg_prob   = category_v[0];
    pos_prob   = category_v[1];
    focus_prob = category_v[2];

    change1_prob         = category_v[3];
    wbc_prob             = category_v[4];
    indetermination_prob = category_v[5];
}

int MakeResultCoarse(const std::vector<float>& category_v, float& result)
{
    if (category_v.size() != AI_CLARITY_COARSE_CATEGORY_NUMS) {
        ALGLogError << "Coarse category must be " << AI_CLARITY_COARSE_CATEGORY_NUMS << " but " << category_v.size() << " was given";
        return -1;
    }
    float neg_prob, pos_prob, focus_prob, change1_prob, wbc_prob, uncer_prob;
    GetCoarseProb(neg_prob, pos_prob, focus_prob, change1_prob, wbc_prob, uncer_prob, category_v);

    result = AI_CLARITY_COARSE_INDETERMINATION;
    if ((neg_prob + pos_prob + wbc_prob) > AI_CLARITY_COARSE_CLEAR_THR) {
        result = AI_CLARITY_COARSE_CLEAR;
        ALGLogInfo << "Clarity prob" << neg_prob << " " << pos_prob << " " << focus_prob << " " << change1_prob << " " << wbc_prob << " "
                   << uncer_prob;
    }
    else {
        result = AI_CLARITY_COARSE_INDETERMINATION;
    }
    return 0;
}

void GetMilkBoardProb(float&                    focus_prob,
                      float&                    pos_prob,
                      float&                    neg_prob,
                      float&                    far_pos_prob,
                      float&                    far_neg_prob,
                      float&                    far_far_pos,
                      float&                    far_far_neg,
                      float&                    indetermination_prob,
                      const std::vector<float>& category_v)
{
    //    for (int i = 0; i < category_v.size(); i++) {
    //        std::cout << "category_v " << i << " : " << category_v[i] <<
    //        std::endl;
    //    }
    focus_prob           = category_v[0];   // F
    pos_prob             = category_v[1];   // CPF
    neg_prob             = category_v[2];   // CNF
    far_pos_prob         = category_v[3];   // FPF
    far_neg_prob         = category_v[4];   // FNF
    far_far_pos          = category_v[5];   // FFPF
    far_far_neg          = category_v[6];   // FFNF
    indetermination_prob = category_v[7];   // O
}

int MakeResultMilkboardline(const std::vector<float>& category_v, float& result)
{
    if (category_v.size() != 8) {
        ALGLogError << "Category must be 8 but" << category_v.size() << "was given";
        return -1;
    }
    float focus_prob   = 0;
    float pos_prob     = 0;
    float neg_prob     = 0;
    float far_pos_prob = 0;
    float far_neg_prob = 0;
    float far_far_pos  = 0;
    float far_far_neg  = 0;
    float uncer_prob   = 0;
    GetMilkBoardProb(focus_prob, pos_prob, neg_prob, far_pos_prob, far_neg_prob, far_far_pos, far_far_neg, uncer_prob, category_v);

    //    for (int i = 0; i < category_v.size(); i++) {
    //        std::cout << "category_v " << i << " : " << category_v[i] <<
    //        std::endl;
    //    }
    //    std::cout << "focus_prob:" << focus_prob << std::endl;
    //    std::cout << "1_prob:" << pos_prob << std::endl;
    //    std::cout << "2_prob:" << neg_prob << std::endl;
    //    std::cout << "3_prob:" << far_pos_prob << std::endl;
    //    std::cout << "4_prob:" << far_neg_prob << std::endl;
    //    std::cout << "5_prob:" << far_far_pos << std::endl;
    //    std::cout << "6_prob:" << far_far_neg << std::endl;

    if (focus_prob > AI_CLARITY_FAR_NEAR_CLEAR_THR) {
        result = AI_CLARITY_FOCUS_AM;   // 0 清晰
    }
    else if (pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {
        result = AI_CLARITY_POSITIVE_AM;   // 1 细正焦
    }
    else if (neg_prob > AI_CLARITY_FAR_NEAR_FAR_THR) {
        result = AI_CLARITY_NEGATIVE_AM;   // 2 细负焦
    }
    else if ((far_pos_prob + far_far_pos) > AI_CLARITY_FAR_NEAR_NEAR_THR) {
        result = AI_CLARITY_FAR_POSITIVE_AM;   // 3 粗正焦
    }
    else if ((far_neg_prob + far_far_neg) > AI_CLARITY_FAR_NEAR_NEAR_THR) {
        result = AI_CLARITY_FAR_NEGATIVE_AM;   // 4 粗负焦
                                               //    } else if (far_far_pos > AI_CLARITY_FAR_NEAR_CLEAR_THR && far_far_pos
                                               //    > far_pos_prob){
                                               //        result = AI_CLARITY_FAR_FAR_POSITIVE_AM;
                                               //    } else if (far_far_neg > AI_CLARITY_FAR_NEAR_CLEAR_THR && far_far_neg
                                               //    > far_neg_prob){
                                               //        result = AI_CLARITY_FAR_FAR_NEGATIVE_AM;
    }
    else {
        result = AI_CLARITY_OTHER_AM;   // 7 不确定
    }
    //    std::cout << "结果----------：" << result << std::endl;
    return 0;
}

void GetBasoFarNearProb(float&                    neg_prob,
                        float&                    pos_prob,
                        float&                    focus_prob,
                        float&                    not_know_prob,
                        float&                    far_pos_prob,
                        float&                    change1_prob,
                        float&                    change2_prob,
                        float&                    wbc_prob,
                        const std::vector<float>& category_v)
{
    neg_prob      = category_v[0];
    pos_prob      = category_v[1];
    focus_prob    = category_v[2];
    not_know_prob = category_v[3];
    far_pos_prob  = category_v[4];
    change1_prob  = category_v[5];
    change2_prob  = category_v[6];
    wbc_prob      = category_v[7];
}

int MakeRestulFineFarNearBaso(const std::vector<float>& category_v, float& result)
{

    if (category_v.size() != 8) {
        ALGLogError << "Category must be 8 but " << category_v.size() << " was given";
        return -1;
    }
    float neg_prob, pos_prob, focus_prob, not_know_prob, far_pos_prob, change1_prob, change2_prob, wbc_prob;
    GetBasoFarNearProb(neg_prob, pos_prob, focus_prob, not_know_prob, far_pos_prob, change1_prob, change2_prob, wbc_prob, category_v);

    if (wbc_prob > AI_CLARITY_FAR_NEAR_CLEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_CLEAR;
    }
    else if (neg_prob > AI_CLARITY_FAR_NEAR_FAR_THR) {
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else if (far_pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {   // 将rbc清晰的面视为过焦
        result = AI_CLARITY_FAR_NEAR_VERY_NEAR;
    }
    else if (change1_prob > AI_CLARITY_FAR_NEAR_CHANGE_VIEW_THR || change2_prob > AI_CLARITY_FAR_NEAR_CHANGE_VIEW_THR) {
        result = AI_CLARITY_FAR_NEAR_CHANGE_VIEW;
    }
    else if (focus_prob + not_know_prob > AI_CLARITY_FAR_NEAR_COARSE_FOCUS_THR) {   // 完全失焦与过于远离焦平面均进行粗聚焦
        result = AI_CLARITY_FAR_NEAR_COARSE_FOCUS;
    }
    else if (wbc_prob + neg_prob > AI_CLARITY_FAR_NEAR_FAR_THR) {   // 当平面处于白细胞清晰面下方时,视为负焦
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (wbc_prob + pos_prob + far_pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {   // 当平面处于白细胞清晰面上方时,视为正焦
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else {
        result = AI_CLARITY_FAR_NEAR_INDETERMINATION;
    }

    return 0;
}

int MakeRestulCoarseFarNearBaso(const std::vector<float>& category_v, float& result)
{

    if (category_v.size() != 8) {
        ALGLogError << "Category must be 8 but " << category_v.size() << " was given";
        return -1;
    }
    float neg_prob, pos_prob, focus_prob, not_know_prob, far_pos_prob, change1_prob, change2_prob, wbc_prob;
    GetBasoFarNearProb(neg_prob, pos_prob, focus_prob, not_know_prob, far_pos_prob, change1_prob, change2_prob, wbc_prob, category_v);

    result = AI_CLARITY_COARSE_INDETERMINATION;
    if ((pos_prob + neg_prob + wbc_prob) > AI_CLARITY_COARSE_CLEAR_THR) {
        result = AI_CLARITY_COARSE_CLEAR;
        ALGLogInfo << "Clarity prob" << neg_prob << " " << pos_prob << " " << far_pos_prob << " " << wbc_prob;
    }
    else {
        result = AI_CLARITY_COARSE_INDETERMINATION;
    }
    return 0;
}

void GetFluMicroshpereProb(
    float& neg_prob, float& pos_prob, float& focus_prob, float& wbc_prob, float& indetermination_prob, const std::vector<float>& category_v)
{
    neg_prob   = category_v[0];
    pos_prob   = category_v[1];
    focus_prob = category_v[2];

    wbc_prob             = category_v[3];
    indetermination_prob = category_v[4];
}

int MakeResultCoarseFarNearFluMicroshpere(const std::vector<float>& category_v, float& result)
{
    if (category_v.size() != AI_CLARITY_COARSE_FLU_MICRO_CATEGORY_NUMS) {
        ALGLogError << "Coarse category must be " << AI_CLARITY_COARSE_FLU_MICRO_CATEGORY_NUMS << " but " << category_v.size() << " was given";
        return -1;
    }
    float neg_prob, pos_prob, focus_prob, wbc_prob, uncer_prob;
    GetFluMicroshpereProb(neg_prob, pos_prob, focus_prob, wbc_prob, uncer_prob, category_v);

    result = AI_CLARITY_COARSE_INDETERMINATION;
    if ((neg_prob + pos_prob + wbc_prob) > AI_CLARITY_COARSE_CLEAR_THR) {
        result = AI_CLARITY_COARSE_CLEAR;
        ALGLogInfo << "Clarity prob" << neg_prob << " " << pos_prob << " " << focus_prob << " " << wbc_prob << " " << uncer_prob;
    }
    else {
        result = AI_CLARITY_COARSE_INDETERMINATION;
    }
    return 0;
}

int MakeResultFineFarNearFluMicroshpere(const std::vector<float>& category_v, float& result)
{
    if (category_v.size() != AI_CLARITY_COARSE_FLU_MICRO_CATEGORY_NUMS) {
        ALGLogError << "Coarse category must be " << AI_CLARITY_COARSE_FLU_MICRO_CATEGORY_NUMS << " but " << category_v.size() << " was given";
        return -1;
    }
    float neg_prob, pos_prob, focus_prob, wbc_prob, uncer_prob;
    GetFluMicroshpereProb(neg_prob, pos_prob, focus_prob, wbc_prob, uncer_prob, category_v);

    if (wbc_prob > AI_CLARITY_FAR_NEAR_CLEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_CLEAR;
    }
    else if (neg_prob > AI_CLARITY_FAR_NEAR_FAR_THR) {
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else if (wbc_prob + neg_prob > AI_CLARITY_FAR_NEAR_FAR_THR) {   // 当平面处于白细胞清晰面下方时,视为负焦
        result = AI_CLARITY_FAR_NEAR_FAR;
    }
    else if (wbc_prob + pos_prob > AI_CLARITY_FAR_NEAR_NEAR_THR) {   // 当平面处于白细胞清晰面上方时,视为正焦
        result = AI_CLARITY_FAR_NEAR_NEAR;
    }
    else {
        result = AI_CLARITY_FAR_NEAR_INDETERMINATION;
    }
    return 0;
}

/*!
 * 根据chl_idx绑定的不同模型,返回不同的清晰度结果
 * @param category_v    模型分类结果
 * @param chl_idx
 * @param result
 * @return
 */
int MakeResultFarNearModel(const std::vector<float>& category_v, const uint32_t& chl_idx, float& result)
{
    switch (chl_idx) {
    case 1:
        if (MakeResultCoarse(category_v, result)) {
            ALGLogError << "Failed to get coarse result";
            return -4;
        }
        return 0;
    case 2:
        if (MakeRestulFarNearWbc(category_v, result)) {
            ALGLogError << "Failed to get wbc far near result";
            return -1;
        }
        return 0;
    case 3:
        if (MakeRestulFineFarNearBaso(category_v, result)) {
            ALGLogError << "Failed to get baso far near result";
            return -2;
        }
        return 0;
    case 4:
        if (MakeResultCoarseFarNearFluMicroshpere(category_v, result)) {
            ALGLogError << "Failed to get coarse flu microsphere result";
            return -2;
        }
        return 0;
    case 5:
        if (MakeResultFineFarNearFluMicroshpere(category_v, result)) {
            ALGLogError << "Failed to get fine flu microsphere result";
            return -2;
        }
        return 0;
    case 6:
        if (MakeRestulCoarseFarNearBaso(category_v, result)) {
            ALGLogError << "Failed to get coarse baso result";
            return -2;
        }
        return 0;
    case 7:
        if (MakeResultMilkboardline(category_v, result)) {
            ALGLogError << "Failed to get milk_boardline result";
            return -2;
        }
        return 0;

    default:
        ALGLogError << "Configured wrong chl_idx for far near model, chl_idx must "
                       "in {1, 2, 3}, but "
                    << chl_idx << " was given";
        return -4;
    }
}

static int Clarity_ImgFarNear(AiCtxID_t                           ctx_id,
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
    //    DLOG(INFO, "Clarity_ImgRBC");
    if (stage != AI_IMG_STAGE_INTERENCE) {
        return 0;
    }
    ClarityCtx_t* ctx = (ClarityCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return -1;
    }
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Clarity_DoImgCallback(ctx, 0, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    group_idx           = group_idx;
    AiGroupReg_t* group = Clarity_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "Null group ptr";
        return -2;
    }

    AiViewReg_t* view = Clarity_FindView(group, Clarity_FindChl(group, chl_idx), view_idx);
    if (view == nullptr) {
        ALGLogError << "Null view ptr";
        return -3;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);

    if (Ai_Inference(CLARITY_CTX_AI_CTXID(ctx), group_id, mod_id, img, result, AI_INFER_TYPE_NORMAL)) {
        ALGLogError << "Failed to inference model";
        return -4;
    }

    if (result.size()) {

        // 存入类别的概率
        Clarity_AddList(ctx, result.begin()->category_v);
        float clarity_type;
        int   ret = MakeResultFarNearModel(result.begin()->category_v, chl_idx, clarity_type);
        if (ret) {
            ALGLogError << "Failed to make result for clarity far near version";
            return -5;
        }

        // 嗜碱流道存在特殊逻辑
        ALGLogInfo << "Used channel idx: " << chl_idx;
        if (chl_idx == 3) {
            ret = CLARITY_CTX_BASO_FOCUS_CTL(ctx).FineControl(view_pair_idx, clarity_type, result.begin()->category_v, clarity_type);
            if (ret) {
                ALGLogError << "Failed to execute baso focus control";
                return ret;
            }
        }

        Clarity_AddValue(ctx, clarity_type);

        // 保存中间结果
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
                            result,
                            view_pair_idx,
                            call_back_params,
                            false,
                            0.5,
                            1);
        if (ret) {
            ALGLogError << "Failed to make mid result ,group channel, order, view, processed" << group_idx << " " << chl_idx << " " << view_order
                        << " " << view_idx << " " << processed_idx;
            return -4;
        }

        return 0;
    }
    //  Clarity_DoImgCallback(ctx, 0, img, group_idx, chl_idx, view_idx,
    //  AI_IMG_STAGE_CLARITY, result);
}

static int Clarity_ImgBaso(AiCtxID_t                           ctx_id,
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
    //    DLOG(INFO, "Clarity_ImgRBC");
    if (stage != AI_IMG_STAGE_INTERENCE) {
        return 0;
    }
    ClarityCtx_t* ctx = (ClarityCtx_t*)userdata;
    if (ctx == NULL) {
        ALGLogError << "Null ctx";
        return -1;
    }
    if (stage != AI_IMG_STAGE_INTERENCE) {
        Clarity_DoImgCallback(ctx, 0, img, group_idx, chl_idx, view_order, view_idx, processed_idx, stage, result, view_pair_idx, call_back_params);
        return 0;
    }
    group_idx           = group_idx;
    AiGroupReg_t* group = Clarity_FindGroup(group_idx);
    if (group == NULL) {
        ALGLogError << "Null group ptr";
        return -2;
    }

    AiViewReg_t* view = Clarity_FindView(group, Clarity_FindChl(group, chl_idx), view_idx);
    if (view == nullptr) {
        ALGLogError << "Null view ptr";
        return -3;
    }
    NNetGroup_e group_id = AI_GROUP_ID(group);
    NNetModID_e mod_id   = AI_VIEW_MOD_ID(view);
    if (Ai_Inference(CLARITY_CTX_AI_CTXID(ctx), group_id, mod_id, img, result, AI_INFER_TYPE_NORMAL)) {
        ALGLogError << "Failed to inference model";
        return -4;
    }
    ClarityValue_t value = 0.0;
    if (result.size()) {
        for (const auto& data : result) {
            int            x1 = data.box.left;
            int            x2 = data.box.right;
            int            y1 = data.box.top;
            int            y2 = data.box.bottom;
            cv::Rect       rect(x1, y1, x2 - x1, y2 - y1);
            cv::Mat        segmentImg = (*img)(rect);
            ClarityValue_t temp       = 0.0;
            if (!Clarity_GetClarity(temp, segmentImg)) {
                value += temp;
            }
        }
        Clarity_AddValue(ctx, value / result.size());

        return 0;
    }
    // 若未检测到嗜碱细胞则传入0
    else {
        Clarity_AddValue(ctx, 0);
    }
    //	Clarity_DoImgCallback(ctx, 0, img, group_idx, chl_idx, view_idx,
    // AI_IMG_STAGE_CLARITY, result);
    return 0;
}
