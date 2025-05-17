#ifndef _AI_H
#define _AI_H

#include <list>
#include <vector>

#include "alg_task_flow_queue.h"
#include "libalgcell.h"
#include "neural_network.h"

#define AiCtxID_t void*
#define AiItemID_t TaskFlowItemID_t
#define AiImg_t cv::Mat
#define AiCntVal_t int
#define ViewFusionRate float

/* AI视图类型 */
typedef enum AiViewType
{
    AI_VIEW_TYPE_NONE = 0,
    AI_VIEW_TYPE_BRI,
    AI_VIEW_TYPE_FLU,
} AiViewType_e;

/* AI通道类型 */
typedef enum AiChlType
{
    AI_CHL_TYPE_NONE = 0,         // 与清晰度最高峰共用
    AI_CHL_TYPE_RBC,              // 与清晰度第一峰共用
    AI_CHL_TYPE_WBC,              // 与AI清晰度共用
    AI_CHL_TYPE_HGB,              // 与嗜碱清晰度共用
    AI_CHL_TYPE_RET,              // 与AI清晰度远近焦版本共用
    AI_CHL_TYPE_MILK_GERM_0,      // 与AI 嗜碱清晰度远近焦版本共用
    AI_CHL_TYPE_MILK_GERM_1,      // 与AI 嗜碱清晰度远近焦版本共用
    AI_CHL_TYPE_MILK_CELL_0,      //
    AI_CHL_TYPE_MILK_CELL_1,      // 与 AI粗聚焦共用
    AI_CHL_TYPE_RBC_QC,           // 与清晰度第一峰共用
    AI_CHL_TYPE_WBC_QC,           // 与AI清晰度共用
    AI_CHL_TYPE_HGB_QC,           // 与嗜碱清晰度共用
    AI_CHL_TYPE_RET_QC,           // 与AI清晰度远近焦版本共用
    AI_CHL_TYPE_MILK_BOARDLINE,   // 牛奶底板划线
} AiChlType_e;

/* AI分组类型 */
typedef enum AiGroupType
{
    AI_GROUP_TYPE_NONE = 0,
    AI_GROUP_TYPE_MILK,
    AI_GROUP_TYPE_HUMAN,
    AI_GROUP_TYPE_CAT,
    AI_GROUP_TYPE_DOG,
} AiGroupType_e;

/* AI模型注册信息 */
typedef struct AiModReg
{
    NNetModID_e              mod_id;
    const char*              name;
    uint8_t                  multi_label_flag;
    float                    fusion_rate;
    NNetGroupMask_t          group_mask;
    ResizeType               resize_type;
    float                    model_type_nums;
    float                    nms_nums;
    float                    conf_nums;
    float                    anchor_nums;
    float                    label_nums;
    float                    reserved_float_param_nums;
    float                    reserved_string_param_nums;
    std::vector<float>       model_type;
    std::vector<float>       nms;
    std::vector<float>       conf;
    std::vector<float>       anchors;
    std::vector<std::string> labels;
    std::vector<float>       reserved_float_params;
    std::vector<std::string> reserved_string_params;   // 仅用于与net保持结构一致,实际无内容,应当通过net
                                                       // ctx访问
} AiModReg_t;
#define AI_MOD_ID(reg) ((reg)->mod_id)                                   // 模型ID
#define AI_MOD_NAME(reg) ((reg)->name)                                   // 模型名称
#define AI_MOD_MULTI_LABEL_FLAG(reg) ((reg)->multi_label_flag)           // 多标签标志
#define AI_MOD_FUSION_RATE(reg) ((reg)->fusion_rate)                     // 像源融合率
#define AI_MOD_GROUP_MASK(reg) ((reg)->group_mask)                       // 分组掩码
#define AI_MOD_RESIZE_TYPE(reg) ((reg)->resize_type)                     // RESIZE类型
#define AI_MOD_NET_MODEL_TYPE_NUMS(reg) ((reg)->model_type_nums)         // 模型类型.xml读取
#define AI_MOD_NMS_CONF_NUMS(reg) ((reg)->nms_conf_nums)                 // 模型nms_thr 参数个数.xml读取
#define AI_MOD_ANCHOR_NUMS(reg) ((reg)->anchor_nums)                     // 模型anchor个数.xml读取
#define AI_MOD_LABEL_NUMS(reg) ((reg)->label_nums)                       // 标签个数.xml读取
#define AI_MOD_RS_FLOAT_NUMS(reg) ((reg)->reserved_float_param_nums)     // 预留float参数个数.xml读取
#define AI_MOD_RS_STRING_NUMS(reg) ((reg)->reserved_string_param_nums)   // 预留string数个数.xml读取

#define AI_MOD_DEF(mod_id,                                                                                                                 \
                   name,                                                                                                                   \
                   multi_label_flag,                                                                                                       \
                   fusion_rate,                                                                                                            \
                   group_mask,                                                                                                             \
                   letterbox,                                                                                                              \
                   model_type_nums,                                                                                                        \
                   nms_nums,                                                                                                               \
                   conf_nums,                                                                                                              \
                   anchor_nums,                                                                                                            \
                   label_nums,                                                                                                             \
                   reserved_float_param_nums,                                                                                              \
                   reserved_string_param_nums)                                                                                             \
    {                                                                                                                                      \
        mod_id, name, multi_label_flag, fusion_rate, group_mask, letterbox, model_type_nums, nms_nums, conf_nums, anchor_nums, label_nums, \
            reserved_float_param_nums, reserved_string_param_nums, {}, {}, {}, {}, {}, {},                                                 \
        {                                                                                                                                  \
        }                                                                                                                                  \
    }

///* AI视图类型 */
// typedef enum AiViewType
//{
//	AI_VIEW_TYPE_BRI = 1,
//	AI_VIEW_TYPE_FLU,
// }AiViewType_e;

/* AI视图注册信息 */
typedef struct AiViewReg
{
    AiViewType_e    type;
    NNetModID_e     mod_id;
    NNetGroupMask_t group_mask;
    ViewFusionRate  fusion_rate;
} AiViewReg_t;
#define AI_VIEW_TYPE(view) ((view)->type)                   // 视图类型
#define AI_VIEW_MOD_ID(view) ((view)->mod_id)               // 模型ID
#define AI_VIEW_MOD_GROUP_MASK(view) ((view)->group_mask)   // 分组掩码
#define AI_VIEW_FUSION_RATE(view) ((view)->fusion_rate)     // 像源融合率
#define AI_VIEW_DEF(type, mod_id, group_mask, fusion_rate) {type, mod_id, group_mask, fusion_rate}

/* AI图像处理阶段枚举 */
typedef enum AiImgStage
{
    AI_IMG_STATGE_UNDEFINED = 0,
    AI_IMG_STAGE_RESIZE,
    AI_IMG_STAGE_INTERENCE,
    AI_IMG_STAGE_CLARITY,
} AiImgStage_e;

/**
 * AI图像处理回调函数
 * @param  ctx_id        AI上下文ID
 * @param  item_id		项目ID
 * @param  img           图像指针
 * @param  group_idx		分组索引
 * @param  chl_idx		通道索引
 * @param  view_idx		视图索引
 * @param  stage    		处理阶段
 * @param  callback		图像回调
 * @param  userdata		用户数据
 * @return
 */
typedef int (*AiImgCallback_f)(AiCtxID_t                           ctx_id,
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

/* AI通道注册信息 */
typedef struct AiChlReg
{
    AiChlType_e               chl_type;
    const char*               name;
    std::vector<AiViewReg_t>* view_list;
    AiImgCallback_f           callback;
    NNetGroupMask_t           group_mask;
} AiChlReg_t;
#define AI_CHL_TYPE(chl) ((chl)->chl_type)           // 通道类型
#define AI_CHL_NAME(chl) ((chl)->name)               // 通道名称
#define AI_CHL_VIEW_LIST(chl) ((chl)->view_list)     // 视图列表
#define AI_CHL_CALLBACK(chl) ((chl)->callback)       // AI图像回调
#define AI_CHL_GROUP_MASK(chl) ((chl)->group_mask)   // 分组掩码
#define AI_CHL_DEF(type, name, view_list, callback, group_mask) {type, name, &(view_list), callback, group_mask}

/* AI计数器映射信息 */
typedef struct AiMap
{
    const char* labels_name;
    uint32_t    offset;
    float       min_prop;
} AiMap_t;
#define AI_MAP_LABELS_NAME(map) ((map)->labels_name)   // 标签名称
#define AI_MAP_OFFSET(map) ((map)->offset)             // 相对偏移量
#define AI_MAP_MIN_PROP(map) ((map)->min_prop)         // 最小置信度
#define AI_MAP_DEF(struct, member, labels_name, min_prop) {labels_name, offsetof(struct, member), min_prop}

/* AI样本分组注册信息 */
typedef struct AiGroupReg
{
    AiGroupType_e            group_type;
    NNetGroup_e              group_id;
    const char*              name;
    const char*              cfg_path;
    std::vector<AiModReg_t>* mod_reglist;
    std::vector<AiMap_t>*    cnt_maplist;
    std::vector<AiChlReg_t>* chl_reglist;
} AiGroupReg_t;
#define AI_GROUP_TYPE(chl) ((chl)->group_type)        // 分组类型
#define AI_GROUP_ID(chl) ((chl)->group_id)            // 分组ID
#define AI_GROUP_NAME(chl) ((chl)->name)              // 分组名称
#define AI_GROUP_CFG_PATH(chl) ((chl)->cfg_path)      // 配置路径
#define AI_GROUP_MOD_LIST(chl) ((chl)->mod_reglist)   // 模型注册表
#define AI_GROUP_CNT_LIST(chl) ((chl)->cnt_maplist)   // 计数器注册表
#define AI_GROUP_CHL_LIST(chl) ((chl)->chl_reglist)   // 通道注册表
#define AI_GROUP_DEF(type, group_id, name, cfg_path, mod_reglist, cnt_maplist, chl_reglist) \
    {type, group_id, name, cfg_path, &(mod_reglist), &(cnt_maplist), &(chl_reglist)}

/* AI推理类型 */
typedef enum AiInferType
{
    AI_INFER_TYPE_NORMAL = 0,
} AiInferType_e;

AiCtxID_t  Ai_Init(const int& item_nums);
int        Ai_DeInit(AiCtxID_t ctx_id);
int        Ai_SetNet(AiCtxID_t ctx_id, NNetCtxID_t nnet_ctxid);
int        Ai_ResetNet(AiCtxID_t ctx_id);
int        Ai_ConvertImage(std::list<AiImg_t>& list, uint8_t* img, uint32_t width, uint32_t height, const bool& img_fusion, const float& fusion_rate);
int        Ai_ConvertClarityImage(std::list<AiImg_t>& list,
                                  uint8_t*            img,
                                  uint32_t            width,
                                  uint32_t            height,
                                  const bool&         img_fusion,
                                  const float&        fusion_rate,
                                  const int&          target_width,
                                  const int&          target_height,
                                  const ResizeType&   resize_type);
AiItemID_t Ai_ItemPush(AiCtxID_t                           ctx_id,
                       uint32_t                            priority,
                       uint32_t                            group_idx,
                       uint32_t                            chl_idx,
                       uint32_t                            view_order,
                       uint32_t                            view_count,
                       std::list<AiImg_t>&                 img_list,
                       AiImgCallback_f                     callback,
                       void*                               userdata,
                       const std::vector<AiViewReg_t>&     view_list,
                       const int&                          view_pair_idx,
                       const std::map<std::string, float>& call_back_params);
int        Ai_ItemAddImg(AiCtxID_t ctx_id, AiItemID_t item_id, AiImg_t& img);
AiImg_t*   Ai_ItemGetImg(AiCtxID_t ctx_id, AiItemID_t item_id, uint32_t view_idx);
int        Ai_ItemDel(AiCtxID_t ctx_id, AiItemID_t item_id);
int        Ai_WaitPriority(AiCtxID_t ctx_id, uint32_t priority, uint32_t timeout);
int        Ai_WaitAll(AiCtxID_t ctx_id, uint32_t timeout);
int        Ai_CleanItemAll(AiCtxID_t ctx_id, uint32_t timeout);
int        Ai_ResultCount(void* map_addr, std::vector<AiMap_t>& map_list, std::list<NNetResult_t>& result, uint8_t multi_label_flag);

int Ai_AddModel(AiCtxID_t                 ctx_id,
                NNetGroup_e               group_id,
                NNetModID_e               mod_id,
                uint8_t*                  mod_data,
                uint32_t                  mod_size,
                uint8_t*                  labels_data,
                uint32_t                  labels_size,
                const ResizeType&         resize_type,
                const float&              nms_thr,
                const float&              conf_thr,
                const NNetTypeID_e&       net_type_id,
                const std::vector<float>& float_param_v);

int Ai_AddModel(AiCtxID_t                       ctx_id,
                NNetGroup_e                     group_id,
                NNetModID_e                     mod_id,
                uint8_t*                        mod_data,
                uint32_t                        mod_size,
                const ResizeType&               resize_type,
                float                           model_type_nums,
                float                           nms_nums,
                float                           conf_nums,
                float                           anchor_nums,
                float                           label_nums,
                float                           reserved_float_param_nums,
                float                           reserved_string_param_nums,
                const std::vector<float>&       model_type_v,
                const std::vector<float>&       nms,
                const std::vector<float>&       conf,
                const std::vector<float>&       anchors,
                const std::vector<std::string>& labels,
                const std::vector<float>&       reserved_float_params,
                const std::vector<std::string>& reserved_string_params);

int Ai_Inference(AiCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, AiImg_t* img, std::list<NNetResult_t>& result, AiInferType_e type);

AiGroupReg_t* Ai_FindGroup(std::vector<AiGroupReg_t>& src_list, uint32_t group_idx);
AiModReg_t*   Ai_FindMod(AiGroupReg_t* group, NNetModID_e mod_id);
AiChlReg_t*   Ai_FindChl(AiGroupReg_t* group, uint32_t chl_idx);
AiViewReg_t*  Ai_FindView(AiGroupReg_t* group, AiChlReg_t* chl, uint32_t view_idx);
int           Ai_GetModReglist(std::vector<AiModReg_t>& dst_list, AiGroupReg_t* group);
int           Ai_GetChlReglist(std::vector<AiChlReg_t>& dst_list, AiGroupReg_t* group);
int           Ai_GetViewReglist(std::vector<AiViewReg_t>& dst_list, AiGroupReg_t* group, AiChlReg_t* chl);

void Ai_FindMaxChannelNums(const std::vector<AiGroupReg_t>& group_reglist, int& channel_nums);
int  Ai_GetNetReservedFloatPrams(AiCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, std::vector<float>& reserved_float_params);

#endif /* _AI_H */
