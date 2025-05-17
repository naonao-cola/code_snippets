#ifndef _ALG_HEAMO_H_
#define _ALG_HEAMO_H_

#include <list>
#include <stdint.h>
#include <vector>

#include "ParamFitting.h"
#include "ai.h"
#include "neural_network.h"


#define HeamoCtxID_t void*
#define HeamoImg_t AiImg_t
#define HeamoHgbVal_t float
#define ADD_DENOMINATOR 1e-5   // 避免除零



// 检测项名称
#define WBC_KEY_NUM "WBC"
#define NE_KEY_PERCENTAGE "NE%"
#define LY_KEY_PERCENTAGE "LY%"
#define MO_KEY_PERCENTAGE "MO%"
#define EO_KEY_PERCENTAGE "EO%"
#define BA_KEY_PERCENTAGE "BA%"

#define NE_KEY_NUM "NE#"
#define LY_KEY_NUM "LY#"
#define MO_KEY_NUM "MO#"
#define EO_KEY_NUM "EO#"
#define BA_KEY_NUM "BA#"


#define RBC_KEY_NUM "RBC"
#define HB_KEY_VALUE "Hb"
#define HCT_KEY_VALUE "Hct"
#define MCV_KEY_VALUE "MCV"
#define MCH_KEY_VALUE "MCH"
#define MCHC_KEY_VALUE "MCHC"
#define RDW_CV_KEY_VALUE "RDW-CV"
#define RDW_SD_KEY_VALUE "RDW-SD"
#define NRBC_KEY_NUM "NRBC"
#define PLT_KEY_NUM "PLT"
#define MPV_KEY_VALUE "MPV"
#define PCT_KEY_VALUE "PCT"
#define RET_KEY_NUM "RET#"
#define RET_KEY_PERCENTAGE "RET%"

#define HEAMO_QC_ERASE_VALUE_COMIC "-"
/**********************
 * 用于指定特殊检测项的宏定义
 *********************/
#define MILK_CELL_STR_NAME "MILK_CELL"

enum HeamoModifyType
{
    HEAMO_MODIFY_TYPE_WBC = 0,
    HEAMO_MODIFY_TYPE_RBC,
    HEAMO_MODIFY_TYPE_PLT,
    HEAMO_MODIFY_TYPE_NONE
};

enum HeamoResultRequiredType
{
    HEAMO_RESULT_REQUIRED_EMPTY = 0,
    HEAMO_RESULT_REQUIRED_RET   = (1 << 0),
    HEAMO_RESULT_REQUIRED_ALL   = 0xFFFFFFFF,
};

enum HeamoResultQcValueType
{
    HEAMO_RESULT_QC_ERASE,
    HEAMO_RESULT_QC_KEEP
};

typedef struct ResultTypeMap
{
    std::string             key;
    std::string             unit;
    HeamoResultQcValueType  qc_value_type;
    float                   value;
    HeamoResultRequiredType required_mask;
    int                     precision_num;
    HeamoModifyType         type;

} ResultTypeMap_t;

#define MODIFY_TYPE_MAP_DEF(struct, key, unit, qc_value_type, value, required_mask, precision_num, type) \
    {key, unit, qc_value_type, value, required_mask, precision_num, type}





/* 血球计数器 */
typedef struct HeamoCnt
{
    int RBC       = 0;   // 红细胞
    int RET       = 0;   // 网织红细胞
    int WBC       = 0;   // 白细胞
    int NEU       = 0;   // 中性粒
    int LYM       = 0;   // 淋巴
    int MONO      = 0;   // 单核
    int EOS       = 0;   // 嗜酸
    int BASO      = 0;   // 嗜碱
    int IG        = 0;   // 未成熟细胞
    int UNKNOWN   = 0;   // 白细胞未知
    int PLT       = 0;   // 血小板
    int GAT       = 0;   // 血小板聚集
    int NRBC      = 0;   // 有核红
    int MILK_GERM = 0;   // 牛奶体细胞
    int MILK_CELL = 0;   // 牛奶细菌
    int PLT_GAT   = 0;   // 血小板聚集个数

    // 疟原虫 计数
    int PLA = 0;
    int PV1 = 0;
    int PV2 = 0;
    int PV3 = 0;
    int PV4 = 0;
    int PV5 = 0;

    int                incline_cell_nums     = 0;   // 倾斜红细胞个数
    float              incline_pixels        = 0;   // 倾斜红细胞像素个数
    int                rbc_volume_img_counts = 0;   // 白细胞流道计算红细胞体积,已处理明场红细胞图像张数
    std::vector<float> rbc_volume_v;                // 存放rbc面积
    std::vector<float> plt_volume_v;                // 存放plt面积
    NNetGroup_e        heamo_group_type;            // 样本类型
    std::vector<int>   channel_img_nums;            // 通道下已送入的图像张数


    std::vector<int> milk_germ_nums_v;   // 细菌数量
    std::vector<int> milk_cell_nums_v;   // 牛奶体细胞数量

    std::vector<int> heamo_rbc_nums_v;
    std::vector<int> heamo_plt_nums_v;
    std::vector<int> heamo_ret_nums_v;
    std::vector<int> heamo_wbc_nums_v;
    std::vector<int> heamo_neu_nums_v;
    std::vector<int> heamo_lym_nums_v;
    std::vector<int> heamo_mono_nums_v;
    std::vector<int> heamo_eos_nums_v;
    std::vector<int> heamo_baso_nums_v;
    std::vector<int> heamo_nrbc_nums_v;


    std::vector<float> hgb_data;   // hgb数据
    std::vector<float> hgb_coef;   // hgb校准系数
    std::map<std::string, std::vector<std::vector<std::vector<float>>>>
        element_under_view_pair_idx;   // 某种元素在视野下的数据
                                       // key       channel      order        box/others{x1,y1,w1,h1,x2,y2,w2,h2,...}
    std::vector<std::vector<float>> accepted_view_pair_idx;   // 某种元素在不同通道接受的视野序号
} HeamoCnt_t;

enum HEAMO_OPEN_TYPE
{
    HEAMO_OPEN_TYPE_OPEN  = 0,
    HEAMO_OPEN_TYPE_CLOSE = 1,
};

typedef ALG_DEPLOY::NormalReagentFitting    NormalReagentFitting_C;
typedef ALG_DEPLOY::SphericalReagentFitting SphericalReagentFitting_C;
typedef struct HeamoCtx
{
    AiCtxID_t                 ai_ctxid;
    HeamoCnt_t                cnt;
    HeamoHgbVal_t             hgb_value;
    AiImgCallback_f           callback;
    void*                     userdata;
    uint32_t                  inner_group_idx = 1;
    bool                      open_img_fusion;
    bool                      open_debug;
    bool                      open_qc;
    bool                      open_calib;
    float                     open_img_h;
    float                     open_img_w;
    float                     open_img_h_um;
    std::vector<float>        open_alarm_param_v;
    std::vector<float>        open_dilution_param_v;
    std::vector<float>        open_task_att_v;
    NormalReagentFitting_C    normal_reagent_fitting;
    SphericalReagentFitting_C spherical_reagent_fitting;
    HEAMO_OPEN_TYPE           open_flag = HEAMO_OPEN_TYPE_CLOSE;
} HeamoCtx_t;
#define HEAMO_CTX_AI_CTXID(ctx) ((ctx)->ai_ctxid)
#define HEAMO_CTX_CNT(ctx) (&(ctx)->cnt)
#define HEAMO_CTX_HGB_VAL(ctx) ((ctx)->hgb_value)
#define HEAMO_CTX_CALLBACK(ctx) ((ctx)->callback)
#define HEAMO_CTX_USERDATA(ctx) ((ctx)->userdata)
#define HEAMO_CTX_GROUP_IDX(ctx) ((ctx)->inner_group_idx)
#define HEAMO_CTX_IMG_FUSION(ctx) ((ctx)->open_img_fusion)
#define HEAMO_CTX_DEBUG(ctx) ((ctx)->open_debug)   // 是否进行debug
#define HEAMO_CTX_QC(ctx) ((ctx)->open_qc)         // 是否进行质控
#define HEAMO_CTX_CALIB(ctx) ((ctx)->open_calib)   // 是否进行校准计数
#define HEAMO_CTX_NORM_REAGENT_FIT(ctx) ((ctx)->normal_reagent_fitting)
#define HEAMO_CTX_SPHE_REAGENT_FIT(ctx) ((ctx)->spherical_reagent_fitting)
#define HEAMO_CTX_IMG_H(ctx) ((ctx)->open_img_h)                 // 是否进行质控
#define HEAMO_CTX_IMG_W(ctx) ((ctx)->open_img_w)                 // 是否进行质控
#define HEAMO_CTX_IMG_H_UM(ctx) ((ctx)->open_img_h_um)           // 是否进行质控
#define HEAMO_CTX_ALARM_PARAM(ctx) ((ctx)->open_alarm_param_v)   // 报警参数
#define HEAMO_CTX_DILUTION(ctx) ((ctx)->open_dilution_param_v)   // 稀释倍数参数
#define HEAMO_CTX_TASK_ATT(ctx) ((ctx)->open_task_att_v)         // 报警参数
#define HEAMO_CTX_OPEN_FLAG(ctx) ((ctx)->open_flag)              // 报警参数
// 用于统一传递参数
typedef struct ResultViewParam
{
    // 通道图像张数
    int rbc_channel_nums;
    int wbc_channel_nums;
    int baso_channel_nums;
    int ret_channel_nums;
    int milk_germ_channel_0_nums;
    int milk_cell_channel_0_nums;
    int milk_germ_channel_1_nums;
    int milk_cell_channel_1_nums;
    // 视野体积
    float volume_rbc;
    float volume_wbc;
    float volume_ret;
    float volume_baso;
    float volume_milk_germ;
    float volume_milk_cell;
    // 稀释倍数
    int dilution_ratio_rbc;
    int dilution_ratio_wbc;
    int dilution_ratio_ret;
    int dilution_ratio_baso;
    int dilution_ratio_milk_germ;
    int dilution_ratio_milk_cell;
} ResultViewParam_t;

/**
 * 血球结果回调
 * @param  name			项目名称
 * @param  unit			项目单位
 * @param  value			项目数值
 * @param  userdata		用户数据
 * @return none
 */
typedef void (*HeamoResultCallback_f)(const char* name, const char* unit, const char* value, void* userdata);

#define HEAMO_RESULT(callback, userdata, name, unit, value) (*callback)(name, unit, value, userdata);

int Heamo_GetGroupReglist(std::vector<AiGroupReg_t>& list);
int Heamo_GetModReglist(std::vector<AiModReg_t>& list, uint32_t group_idx);
int Heamo_GetChlReglist(std::vector<AiChlReg_t>& list, uint32_t group_idx);
int Heamo_GetViewReglist(std::vector<AiViewReg_t>& list, uint32_t group_idx, uint32_t chl_idx);

HeamoCtxID_t Heamo_Init(AiCtxID_t ai_ctxid);
int          Heamo_DeInit(HeamoCtxID_t ctx_id);
int          Heamo_Open(HeamoCtxID_t              ctx_id,
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
                        const std::vector<float>& task_att_v);
int          Heamo_AddImgList(HeamoCtxID_t           ctx_id,
                              std::list<HeamoImg_t>& list,
                              uint32_t               group_idx,
                              uint32_t               chl_idx,
                              uint32_t               view_idx,
                              uint8_t*               img_data,
                              uint32_t               width,
                              uint32_t               height);
int          Heamo_PushImage(HeamoCtxID_t ctx_id, std::list<HeamoImg_t>& img_list, uint32_t group_idx, uint32_t chl_idx, const int& view_pair_idx);
int          Heamo_PushHgb(HeamoCtxID_t ctx_id, const std::vector<HeamoHgbVal_t>& data_list, const std::vector<float>& coef_list);
int          Heamo_WaitCplt(HeamoCtxID_t ctx_id, uint32_t timeout);


int HeamoGetCount(bool zero_flag);

int Heamo_GetResult(HeamoCtxID_t              ctx_id,
                    std::vector<float>&       curve_rbc,
                    std::vector<float>&       curve_plt,
                    HeamoResultCallback_f     callback,
                    void*                     userdata,
                    std::vector<std::string>& alarm_str_v);


int Heamo_ModifyResult(const AlgCellModeID_e&    initialed_mode_id,
                       const std::string&        changed_param_key,
                       std::vector<float>&       curve_rbc,
                       std::vector<float>&       curve_plt,
                       HeamoResultCallback_f     callback,
                       void*                     userdata,
                       std::vector<std::string>& alarm_str_v);

int Heamo_Close(HeamoCtxID_t ctx_id);


/* 血球图像处理函数声明 */
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
                    const std::map<std::string, float>& call_back_params);
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
                    const std::map<std::string, float>& call_back_params);

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
                    const std::map<std::string, float>& call_back_params);
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
                     const std::map<std::string, float>& call_back_params);

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
                    const std::map<std::string, float>& call_back_params);



// 牛奶处理函数
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
                         const std::map<std::string, float>& call_back_params);

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
                         const std::map<std::string, float>& call_back_params);

/*!
 * 获取血球结果
 * @param ctx_id
 * @param[out] curve_rbc  rbc曲线
 * @param[out] curve_plt  plt曲线
 * @param callback        组织结果的callback
 * @param userdata
 * @param view_param      关于视野的参数
 * @return
 */
int Heamo_GetHeamoResult(HeamoCtxID_t              ctx_id,
                         std::vector<float>&       curve_rbc,
                         std::vector<float>&       curve_plt,
                         HeamoResultCallback_f     callback,
                         void*                     userdata,
                         const ResultViewParam_t&  view_param,
                         std::vector<std::string>& alarm_str_v);

int Heamo_ModifyHumanResult(const std::string&        changed_param_key,
                            std::vector<float>&       curve_rbc,
                            std::vector<float>&       curve_plt,
                            HeamoResultCallback_f     callback,
                            void*                     userdata,
                            std::vector<std::string>& alarm_str_v);

// 获取牛奶结果
int Heamo_GetMilkResult(HeamoCtxID_t              ctx_id,
                        std::vector<float>&       curve_rbc,
                        std::vector<float>&       curve_plt,
                        HeamoResultCallback_f     callback,
                        void*                     userdata,
                        const ResultViewParam_t&  view_param,
                        std::vector<std::string>& alarm_str_v);



AiGroupReg_t* Heamo_FindGroup(uint32_t group_idx);

AiModReg_t* Heamo_FindMod(AiGroupReg_t* group, NNetModID_e mod_id);

AiChlReg_t* Heamo_FindChl(AiGroupReg_t* group, uint32_t chl_idx);

AiViewReg_t* Heamo_FindView(AiGroupReg_t* group, AiChlReg_t* chl, uint32_t view_idx);


void Heamo_SetGermResultDir(const std::string& save_dir);
void Heamo_SetHeamoResultDir(const std::string& save_dir);



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
                  const cv::Scalar&                   box_color  = cv::Scalar(0, 0, 255),
                  const cv::Scalar&                   font_color = cv::Scalar(255, 0, 0));

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
                        const std::map<std::string, float>& call_back_params);



void CutFloatToString(const float& src, const int& precision, std::string& dst);
void ClipNumber(const float& src, float& dst, const float& lower, const float& upper);
int  OpenedChannelDilutionIdentify(const int& pushed_img_nums, const float& dilution);
#endif /* _ALG_HEAMO_H_ */
