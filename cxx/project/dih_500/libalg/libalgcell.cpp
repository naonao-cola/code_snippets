#include "libalgcell.h"

#include <stdio.h>
#include <algorithm>
#include <list>


#include "alg_heamo.h"
#include "alg_clarity.h"
#include "replace_std_string.h"
#include "utils.h"
#include "event.h"
//#include "DihLog.h"
#include "ModelConfig.h"
#include "algLog.h"
#include "alg_error_code.h"
#define ALGCELL_VERSION                   "V1.0.05.20250509"

extern bool g_pla_flag;

#ifndef AI_USE_TIMECNT
#define AI_USE_TIMECNT        0
#endif
#if(AI_USE_TIMECNT)

#include "timecnt.h"

#endif
#define HEAMO_TYPE      0
#define CLARITY_TYPE     1
#define HEAMO_AI_ITEM_NUMS   15
#define CLARITY_AI_ITEM_NUMS 5
typedef struct AlgCtx {
  AlgCellModeID_e work_mode;
  uint32_t func_mask;
  NNetCtxID_t nnet_ctxid;
  AiCtxID_t ai_ctxid_heamo;
  AiCtxID_t ai_ctxid_clarity;
  HeamoCtxID_t heamo_ctxid;
  ClarityCtxID_t clarity_ctxid;
  std::string cfg_path;
  AlgCellImageCallback_f heamo_callback;
  void *heamo_userdata;
  AlgCellImageCallback_f clarity_callback;
  void *clarity_userdata;
  bool img_fusion = false;

} AlgCtx_t;
#define ALGCELL_CTX_WORK_MODE(ctx)                    ((ctx)->work_mode)            // 工作模式
#define ALGCELL_CTX_FUNC_MASK(ctx)                    ((ctx)->func_mask)            // 功能掩码
#define ALGCELL_CTX_NNET_CTXID(ctx)                   ((ctx)->nnet_ctxid)           // 神经网络上下文
//#define ALGCELL_CTX_AI_CTXID(ctx)                     ((ctx)->ai_ctxid)             // AI上下文

#define ALGCELL_CTX_AI_CTXID_HEAMO(ctx)               ((ctx)->ai_ctxid_heamo)             // AI上下文
#define ALGCELL_CTX_AI_CTXID_CLARITY(ctx)             ((ctx)->ai_ctxid_clarity)             // AI上下文

#define ALGCELL_CTX_HEAMO_CTXID(ctx)                  ((ctx)->heamo_ctxid)          // 血球上下文
#define ALGCELL_CTX_CLARITY_CTXID(ctx)                ((ctx)->clarity_ctxid)        // 聚焦上下文
#define ALGCELL_CTX_HGB_CTXID(ctx)                    ((ctx)->hgb_ctx_id)           // HGB上下文
#define ALGCELL_CTX_CFG_PATH(ctx)                     ((ctx)->cfg_path)             // 配置路径
#define ALGCELL_CTX_HEAMO_CALLBACK(ctx)               ((ctx)->heamo_callback)       // 血球图像回调
#define ALGCELL_CTX_HEAMO_USERDATA(ctx)               ((ctx)->heamo_userdata)       // 血球用户数据
#define ALGCELL_CTX_CLARITY_CALLBACK(ctx)             ((ctx)->clarity_callback)     // 血球图像回调
#define ALGCELL_CTX_CLARITY_USERDATA(ctx)             ((ctx)->clarity_userdata)     // 血球用户数据
#define ALGCELL_CTX_IMG_FUSION(ctx)                   ((ctx)->img_fusion)           // 像元融合


/* 算法工作模式注册信息 */
typedef struct AlgModeReg {
  AlgCellModeID_e mode_id;
  NNetGroupMask_t group_mask;
} AlgModeReg_t;
#define ALGCELL_MODE_ID(reg)                ((reg)->mode_id)        // 工作模式ID
#define ALGCELL_MODE_GROUP_MASK(reg)        ((reg)->group_mask)        // 分组掩码
#define ALGCELL_MODE_DEF(mode_id, group_mask)        {\
    mode_id,\
    group_mask\
}

AlgCellModeID_e initialed_mode_id;

/* 算法工作模式注册表 */
std::vector<AlgModeReg_t> alg_mode_reglist = {
    ALGCELL_MODE_DEF(ALGCELL_MODE_HUMAN, NNET_GROUP_HUMAN),
    ALGCELL_MODE_DEF(ALGCELL_MODE_ANIMAL, NNET_GROUP_CAT | NNET_GROUP_DOG),
    ALGCELL_MODE_DEF(ALGCELL_MODE_MILK, NNET_GROUP_MILK),
};

int AlgCell_HeamoPushGermResultDir(const std::string &save_dir) {
  Heamo_SetGermResultDir(save_dir);
  Heamo_SetHeamoResultDir(save_dir);
  return 0;
}

// 注意区分group idx 与 group id
static int Alg_GetGroupIdsUnderMode(std::vector<uint32_t> &group_id_list, std::vector<AiGroupReg_t> &group_list,
                                std::vector<AlgModeReg_t> &mode_list, AlgCellModeID_e mode_id) {
  for (auto &mode: mode_list) {
    if (ALGCELL_MODE_ID(&mode) == mode_id) {
      for(uint32_t group_idx = 0; group_idx<group_list.size();++group_idx){
        const auto& group = group_list[group_idx];
        if (ALGCELL_MODE_GROUP_MASK(&mode) & AI_GROUP_ID(&group)) {
          group_id_list.push_back(group_idx);
        }
      }
      return 0;
    }
  }
  return -1;
}




inline int Heamo_GetGroupIdsUnderMode(std::vector<uint32_t> &group_id_list, std::vector<AiGroupReg_t> &group_list,
                                  AlgCellModeID_e mode_id) {
  Alg_GetGroupIdsUnderMode(group_id_list, group_list, alg_mode_reglist, mode_id);
}




inline int Clarity_GetGroupIdsUnderMode(std::vector<uint32_t> &group_id_list, std::vector<AiGroupReg_t> &group_list,
                                    AlgCellModeID_e mode_id) {
  Alg_GetGroupIdsUnderMode(group_id_list, group_list, alg_mode_reglist, mode_id);
}



int AlgCell_DeInit(AlgCtxID_t ctx_id) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    //    DLOG(ERROR, "null ctx");
    return -1;
  }
  if (ALGCELL_CTX_CLARITY_CTXID(ctx)) {
    Clarity_DeInit(ALGCELL_CTX_CLARITY_CTXID(ctx));
    ALGCELL_CTX_CLARITY_CTXID(ctx) = NULL;
  }
  if (ALGCELL_CTX_HEAMO_CTXID(ctx)) {
    Heamo_DeInit(ALGCELL_CTX_HEAMO_CTXID(ctx));
    ALGCELL_CTX_HEAMO_CTXID(ctx) = NULL;
  }
  // HEAMO
  if (ALGCELL_CTX_AI_CTXID_HEAMO(ctx)) {
    Ai_DeInit(ALGCELL_CTX_AI_CTXID_HEAMO(ctx));
    ALGCELL_CTX_AI_CTXID_HEAMO(ctx) = NULL;
  }
  //CLARITY
  if (ALGCELL_CTX_AI_CTXID_CLARITY(ctx)) {
    Ai_DeInit(ALGCELL_CTX_AI_CTXID_CLARITY(ctx));
    ALGCELL_CTX_AI_CTXID_CLARITY(ctx) = NULL;
  }

  if (ALGCELL_CTX_NNET_CTXID(ctx)) {
    NNet_DeInit(ALGCELL_CTX_NNET_CTXID(ctx));
    ALGCELL_CTX_NNET_CTXID(ctx) = NULL;
  }
  return 0;
}


std::string AlgCell_Version() {
  ALGLogInfo << "ALGCELL_VERSION ： " << ALGCELL_VERSION;
  const char *date_char = {__DATE__};
  const char *time_char = {__TIME__};

  char result[200] = {0};
  char dt[20] = {0};
  sprintf(dt, "%s%s", date_char, time_char);
  int month = 0;
  const char *pMonth[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                          "Dec"};
  for (int i = 0; i < 12; i++) {
    if (memcmp(pMonth[i], dt, 3) == 0) {
      month = i + 1;
      break;
    }
  }

  std::ostringstream oss;
  oss << ALGCELL_VERSION;
  // oss << dt[7] << dt[8] << dt[9] << dt[10];
  // oss << std::setw(2) << std::setfill('0') << month;
  // auto day = (dt[4] == ' ' ? (dt[5] - '0') : ((dt[4] - '0') * 10) + (dt[5] - '0'));
  // oss << std::setw(2) << std::setfill('0') << day << "_";
  // oss << __TIME__;
  // oss << result;


  return oss.str();

}


/**
* 算法初始化
* @param  none
* @return
 */
AlgCtxID_t AlgCell_Init(void) {
  ALGLogInfo << "[alg] ver: " << ALGCELL_VERSION;
  AlgCtx_t *ctx = new AlgCtx_t;
  //血球,聚焦绑定不同的ai
  ALGCELL_CTX_AI_CTXID_HEAMO(ctx) = Ai_Init(HEAMO_AI_ITEM_NUMS);
  ALGCELL_CTX_AI_CTXID_CLARITY(ctx) = Ai_Init(CLARITY_AI_ITEM_NUMS);
  if (ALGCELL_CTX_AI_CTXID_HEAMO(ctx) == NULL) {
    return NULL;
  }
  if (ALGCELL_CTX_AI_CTXID_CLARITY(ctx) == NULL) {
    return NULL;
  }


  ALGCELL_CTX_HEAMO_CTXID(ctx) = Heamo_Init(ALGCELL_CTX_AI_CTXID_HEAMO(ctx));
  if (ALGCELL_CTX_HEAMO_CTXID(ctx) == NULL) {
    return NULL;
  }
  ALGCELL_CTX_CLARITY_CTXID(ctx) = Clarity_Init(ALGCELL_CTX_AI_CTXID_CLARITY(ctx));
  if (ALGCELL_CTX_CLARITY_CTXID(ctx) == NULL) {
    return NULL;
  }
  return (AlgCtxID_t) ctx;
}

static uint8_t *Alg_ReadFile(const char *path, const char *keyword, const char *suffix, uint32_t *len) {
  FILE *fp = NULL;
  char filename[256];
  if (path == NULL || keyword == NULL) {
    return NULL;
  }
  memset(filename, 0, 256);
  snprintf(filename, 256, "%s/%s.%s", path, keyword, suffix);
  ALGLogInfo<<"Model path "<<filename;
  fp = fopen(filename, "rb");
  if (fp == NULL) {
    ALGLogError << "model init file dose not exist, " << filename;
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  uint32_t size = ftell(fp);
  if (fseek(fp, 0, SEEK_SET)) {
    return NULL;
  }
  uint8_t *data = (uint8_t *) malloc(size);
  if (data == NULL) {
    return NULL;
  }
  if (size != fread(data, 1, size, fp)) {
    free(data);
    return NULL;
  }
  if (len) {
    *len = size;
  }
  fclose(fp);
  return data;
}


static int
Alg_SyncModFile(AlgCtxID_t ctx_id, const char *cfg_path, AiGroupReg_t &group, AiModReg_t &mod, const int &ai_type) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL || cfg_path == NULL) {
    return -1;
  }
  ALGLogInfo<<"Add model "<<AI_MOD_NAME(&mod);

  std::string fullpath(cfg_path);
  fullpath = fullpath + AI_GROUP_CFG_PATH(&group);
  uint32_t mod_size = 0;
  uint32_t label_size = 0;
  uint8_t *mod_data = Alg_ReadFile(fullpath.data(), AI_MOD_NAME(&mod), "rknn", &mod_size);
  //	uint8_t *label_data = Alg_ReadFile(fullpath.data(), AI_MOD_NAME(&mod), "txt", &label_size);

  if (mod_data == nullptr) {
    ALGLogError<<"failed to add model ";
    return -2;
  }

  ALG_DEPLOY::XML::ModelConfig model_config;
  ALG_DEPLOY::XML::ConfigParams config_params;
  char filename[256];
  memset(filename, 0, 256);
  snprintf(filename, 256, "%s/%s.%s", fullpath.data(), AI_MOD_NAME(&mod), "xml");

  if(model_config.ReadXmlFile(filename, config_params)){
    ALGLogError<<"failed to parse xml";
    return -2;
  }

  //将不同模型绑定至 heamo or clarity ctx
  AiCtxID_t specific_ai_ctx = ALGCELL_CTX_AI_CTXID_HEAMO(ctx);
  if (ai_type == CLARITY_TYPE) {
    specific_ai_ctx = ALGCELL_CTX_AI_CTXID_CLARITY(ctx);
  }

  //在此处传入读取的参数及要求参数个数
  int ret = Ai_AddModel(
      specific_ai_ctx,
      group.group_id,
      mod.mod_id,
      mod_data,
      mod_size,
      mod.resize_type,
      mod.model_type_nums,
      mod.nms_nums,
      mod.conf_nums,
      mod.anchor_nums,
      mod.label_nums,
      mod.reserved_float_param_nums,
      mod.reserved_string_param_nums,
      config_params.model_type,
      config_params.nms,
      config_params.conf,
      config_params.anchors,
      config_params.labels,
      config_params.reserved_float_params,
      config_params.reserved_string_params);
  if (ret == 0) {
    ALGLogInfo << "succeed to add " << AI_GROUP_CFG_PATH(&group) << " " << AI_MOD_NAME(&mod) << " " << mod_size
               << " " << label_size;
  } else {
    ALGLogError << "failed to add " << AI_GROUP_CFG_PATH(&group) << " " << AI_MOD_NAME(&mod) << " " << mod_size
                << " " << label_size;
    return -3;
  }
  if (mod_data) {
    free(mod_data);
  }
  return 0;
}



static int Alg_ConfigHeamo(AlgCtxID_t ctx_id, AlgCellModeID_e mode_id, const char *cfg_path) {
  std::vector<AiGroupReg_t> group_list;
  if (Heamo_GetGroupReglist(group_list)) {
    return -1;
  }
  int ret = 0;
  // 获取当前机型下的group
  std::vector<uint32_t> group_id_v;
  if(Heamo_GetGroupIdsUnderMode(group_id_v, group_list, mode_id)){
    ALGLogError<<"failed to get group under mode "<<mode_id;
    return -2;
  }
  // 依次对每个组进行初始化
  for(const auto& group_idx:group_id_v){
    std::vector<AiModReg_t> mod_list;
    // 获取当前组包含的模型列表
    if (Heamo_GetModReglist(mod_list, group_idx)) {
      return -3;
    }
    ALGLogInfo <<"Init used group idx: " << group_idx;
    ALGLogInfo<<"Mod list size : "<<mod_list.size();
    for (auto &mod: mod_list) {
      ALGLogInfo<<std::endl<<std::endl;
      ret = Alg_SyncModFile(ctx_id, cfg_path, group_list.at(group_idx), mod, HEAMO_TYPE);
      if (ret != 0) {
        return -4;
      }
    }
  }

  return 0;
}




static int Alg_ConfigClarity(AlgCtxID_t ctx_id, AlgCellModeID_e mode_id, const char *cfg_path) {
  std::vector<AiGroupReg_t> group_list;
  if (Clarity_GetGroupReglist(group_list)) {
    return -1;
  }
  int ret;

  std::vector<uint32_t> group_id_v;
  if(Clarity_GetGroupIdsUnderMode(group_id_v, group_list, mode_id)){
    ALGLogError<<"Failed to get group under mode "<<mode_id;
    return -2;
  }

  // 依次对每个组进行初始化
  for(const auto& group_idx:group_id_v){
    std::vector<AiModReg_t> mod_list;
    // 获取当前组包含的模型列表
    if (Clarity_GetModReglist(mod_list, group_idx)) {
      return -3;
    }
    ALGLogInfo <<"Init used group idx: " << group_idx;
    for (auto &mod: mod_list) {
      ALGLogInfo<<std::endl<<std::endl;
      ret = Alg_SyncModFile(ctx_id, cfg_path, group_list.at(group_idx), mod, CLARITY_TYPE);
      if (ret != 0) {
        return -4;
      }
    }
  }

  return 0;
}

/**
* 算法卸载运行配置
* @param  ctx_id		算法上下文
* @return 0 success other fail
 */
int AlgCell_RunConfigUnload(AiCtxID_t ctx_id) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  Ai_ResetNet(ALGCELL_CTX_AI_CTXID_HEAMO(ctx));
  Ai_ResetNet(ALGCELL_CTX_AI_CTXID_CLARITY(ctx));
  if (ALGCELL_CTX_NNET_CTXID(ctx)) {
    NNet_DeInit(ALGCELL_CTX_NNET_CTXID(ctx));
    ALGCELL_CTX_NNET_CTXID(ctx) = NULL;
  }
  ALGCELL_CTX_CFG_PATH(ctx).clear();
  ALGCELL_CTX_WORK_MODE(ctx) = ALGCELL_MODE_NONE;
  return 0;
}

/**
* 算法加载运行配置
* @param  ctx_id		算法上下文
* @param  mode_id		工作模式
* @param  cfg_path		配置路径
* @return 0 success other fail
 */
int AlgCell_RunConfigLoad(AiCtxID_t ctx_id, AlgCellModeID_e mode_id, const char *cfg_path) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;

  EVINFO(EVTYPE_ERROR, "mode_id : %d", mode_id)
  EVINFO(EVTYPE_ERROR, "mode_id : %s", cfg_path)

  if (ctx == NULL) {
    return -1;
  }
  ALGCELL_CTX_NNET_CTXID(ctx) = NNet_Init(std::string(cfg_path));
  if (ALGCELL_CTX_NNET_CTXID(ctx) == NULL) {
    return -2;
  }
  //HEAMO CLARITY 指向同一个NET ctx
  if (Ai_SetNet(ALGCELL_CTX_AI_CTXID_HEAMO(ctx), ALGCELL_CTX_NNET_CTXID(ctx))) {

    AlgCell_RunConfigUnload(ctx_id);
    return -3;
  }
  if (Ai_SetNet(ALGCELL_CTX_AI_CTXID_CLARITY(ctx), ALGCELL_CTX_NNET_CTXID(ctx))) {

    AlgCell_RunConfigUnload(ctx_id);
    return -3;
  }
  if (Alg_ConfigHeamo(ctx_id, mode_id, cfg_path)) {
    ALGLogError << "failed to init machine " << mode_id;
    AlgCell_RunConfigUnload(ctx_id);
    return -3;
  }
  if (Alg_ConfigClarity(ctx_id, mode_id, cfg_path)) {
    ALGLogError << "failed to init machine " << mode_id;
    AlgCell_RunConfigUnload(ctx_id);
    return -4;
  }
  ALGCELL_CTX_CFG_PATH(ctx) = cfg_path;
  ALGCELL_CTX_WORK_MODE(ctx) = mode_id;
  ALGLogInfo << "succeed to init machine " << mode_id;
  initialed_mode_id = mode_id;
  return 0;
}



/**
* 获取样本测量通道注册表
* @param  ctx_id		算法上下文
* @param  list			目标列表
* @param  group_idx		分组索引
* @return 0 success other fail
 */
int AlgCell_HeamoListChl(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  std::vector<AiChlReg_t> chl_list;
  if (Heamo_GetChlReglist(chl_list, group_idx)) {
    return -3;
  }
  for (auto &chl: chl_list) {
    list.push_back(AI_CHL_TYPE(&chl));
  }
  return 0;
}

/**
* 获取样本测量视图注册表
* @param  ctx_id		算法上下文
* @param  list			目标列表
* @param  group_idx		分组索引
* @param  chl_idx		通道索引
* @return 0 success other fail
 */
int AlgCell_HeamoListView(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx, uint32_t chl_idx) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }

  std::vector<AiViewReg_t> view_list;
  if (Heamo_GetViewReglist(view_list, group_idx, chl_idx)) {
    return -3;
  }
  for (auto &view: view_list) {
    list.push_back(AI_VIEW_TYPE(&view));
  }
  return 0;
}



/**
* 获取清晰度通道注册表
* @param  ctx_id		算法上下文
* @param  list			目标列表
* @param  group_idx		分组索引
* @return 0 success other fail
 */
int AlgCell_ClarityListChl(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }

  std::vector<AiChlReg_t> chl_list;
  if (Clarity_GetChlReglist(chl_list, group_idx)) {
    return -3;
  }
  for (auto &chl: chl_list) {
    list.push_back(AI_CHL_TYPE(&chl));
  }
  return 0;
}

/**
* 获取清晰度视图注册表
* @param  ctx_id		算法上下文
* @param  list			目标列表
* @param  group_idx		分组索引
* @param  chl_idx		通道索引
* @return 0 success other fail
 */
int AlgCell_ClarityListView(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx, uint32_t chl_idx) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  std::vector<AiViewReg_t> view_list;
  if (Clarity_GetViewReglist(view_list, group_idx, chl_idx)) {
    return -3;
  }
  for (auto &view: view_list) {
    list.push_back(AI_VIEW_TYPE(&view));
  }
  return 0;
}


static int AlgSamp_ImgCallback(AiCtxID_t ctx_id, uint32_t item_id, AiImg_t *img, uint32_t group_idx, uint32_t chl_idx,
                               uint32_t view_order,
                               uint32_t view_idx, uint32_t processed_idx, AiImgStage_e stage, void *userdata,
                               std::list<NNetResult_t> &result, const int &view_pair_idx,
                               const std::map<std::string, float> &call_back_params) {
  AlgCtx_t *ctx = (AlgCtx_t *) userdata;
  if (ctx == NULL) {
    return -1;
  }

  if (ALGCELL_CTX_HEAMO_CALLBACK(ctx)) {
    cv::cvtColor(*img, *img, cv::COLOR_BGR2RGB);
    AlgCellImg_t image;
    ALGCELL_IMG_DATA(&image) = img->data;
    ALGCELL_IMG_SIZE(&image) = img->rows * img->cols * img->channels();
    ALGCELL_IMG_WIDTH(&image) = img->cols;
    ALGCELL_IMG_HEIGHT(&image) = img->rows;
    (*ALGCELL_CTX_HEAMO_CALLBACK(ctx))((AlgCtxID_t) ctx, group_idx, chl_idx, view_order, view_idx,
                                       processed_idx, (AlgCellStage_e) stage,
                                       &image, ALGCELL_CTX_HEAMO_USERDATA(ctx), view_pair_idx, call_back_params);
  }
  return 0;
}

/*!
 * 解析一类数据
 * @param open_params 接受到的数据
 * @param key         需要解析的关键字
 * @param output      解析结果
 * @return
 */
int ParseOneParams(const std::map<std::string, std::vector<float>>& open_params, const std::string& key, std::vector<float>& output){
  auto iter = open_params.find(key);
  if(iter==open_params.end()){
    ALGLogError<<"failed to find "<< key << "flag in open params";
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  if(iter->second.empty()){//每个字段必须有值
    ALGLogError<<"empty params are given for "<<key;
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  output = iter->second;
  return ALG_SUCC;
}


int AlgCell_ParseOpenParams(const std::map<std::string, std::vector<float>>& open_params,
                            bool& debug, uint32_t& group_idx, bool& qc,
                            float& img_h, float&img_w, float& img_h_um,
                            std::vector<float>& alarm_param_v,
                            std::vector<float>& dilution_param_v,
                            std::vector<float>& task_att_v,
                            bool& calib, bool clarity_open= false){

  std::vector<float> debug_v;
  if(ParseOneParams(open_params, OPEN_PARAM_DEBUG, debug_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  debug = static_cast<bool> (debug_v[0]);

  std::vector<float> group_idx_v;
  if(ParseOneParams(open_params, OPEN_PARAM_GROUP_IDX, group_idx_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  group_idx = static_cast<uint32_t>(group_idx_v[0]);

  // 疟原虫信号
  std::vector<float> pla_v;
  if (ParseOneParams(open_params, OPEN_PARAM_PLA, pla_v)) {
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  g_pla_flag = static_cast<bool>(pla_v[0]);


  std::vector<float> qc_v;
  if(ParseOneParams(open_params, OPEN_PARAM_QC, qc_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  qc = static_cast<bool>(qc_v[0]);
  if(clarity_open){//清晰度不包含之后的参数,提前退出
    ALGLogInfo<<"debug, group_idx, qc "<< debug<<" "<<group_idx<<" "<<qc;
    return ALG_SUCC;
  }

  std::vector<float> img_h_v;
  if(ParseOneParams(open_params, OPEN_PARAM_IMG_H, img_h_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  img_h = static_cast<float>(img_h_v[0]);

  std::vector<float> img_w_v;
  if(ParseOneParams(open_params, OPEN_PARAM_IMG_W, img_w_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  img_w = static_cast<float>(img_w_v[0]);

  std::vector<float> img_h_um_v;
  if(ParseOneParams(open_params, OPEN_PARAM_IMG_H_UM, img_h_um_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  img_h_um = static_cast<float>(img_h_um_v[0]);


  if(ParseOneParams(open_params, OPEN_PARAM_ALARM, alarm_param_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }

  if(ParseOneParams(open_params, OPEN_PARAM_DILUTION, dilution_param_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  if(ParseOneParams(open_params, OPEN_PARAM_TASK_APPEND_ATT, task_att_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  std::vector<float> calib_v;
  if(ParseOneParams(open_params, OPEN_PARAM_CALIB, calib_v)){
    return ALG_ERR_PARSE_OPEN_PARAM;
  }
  calib = static_cast<bool>(calib_v[0]);

  if(qc&&calib){
    ALGLogError<<"Can not set qc and clib to ture simultaneously; qc, calib: "<<qc<<" "<<calib;
    return ALG_ERR_PARSE_OPEN_PARAM;
  }

  ALGLogInfo<<"Open params:";
  ALGLogInfo<<"debug, group_idx, qc, calib: "<< debug<<" "<<group_idx<<" "<<qc<<" "<<calib;
  ALGLogInfo<<"img_h,img_w,img_h_um: "<< img_h<<" "<<img_w<<" "<<img_h_um;
  ALGLogInfo<<"alarm params:";
  for(const auto& iter:alarm_param_v){
    ALGLogInfo<<iter;
  }
  ALGLogInfo<<"dilution params:";
  for(const auto& iter:dilution_param_v){
    ALGLogInfo<<iter;
  }
  ALGLogInfo<<"ret,nrbc config:";
  for(const auto& iter:task_att_v){
    ALGLogInfo<<iter;
  }

  return ALG_SUCC;
}


/*!
 * 算法测量开启
 * @param ctx_id                算法上下文ID
 * @param func_mask             功能掩码
 * @param debug                 开启保存中间结果
 * @param callback              回调函数
 * @param userdata
 * @return
 */

int AlgCell_HeamoOpen(AlgCtxID_t ctx_id, uint32_t func_mask, AlgCellImageCallback_f callback, void *userdata, const std::map<std::string, std::vector<float>>& open_params) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return ALG_ERR_INVALID_ALG;
  }

  bool img_fusion = func_mask & ALGCELL_FUNC_FUSION;

  //解析输入参数
  uint32_t group_idx;
  bool debug, qc, calib;
  float img_h, img_w, img_h_um;
  std::vector<float> alarm_param_v, dilution_param_v, task_att_v;
  int ret = AlgCell_ParseOpenParams(open_params, debug, group_idx, qc,
                                    img_h, img_w, img_h_um, alarm_param_v,
                                    dilution_param_v,task_att_v, calib, false);
  if(ret){
    ALGLogError<<"failed to parse open params";
    return ret;
  }
  ALGLogInfo << "是否保存中间过程: " << debug;

  if (func_mask & ALGCELL_FUNC_HEAMO) {
    if (Heamo_Open(ALGCELL_CTX_HEAMO_CTXID(ctx), img_fusion, debug,
                   AlgSamp_ImgCallback,(void *) ctx, group_idx,
                   qc, calib, img_h, img_w, img_h_um, alarm_param_v, dilution_param_v,task_att_v)) {
      return -4;
    }
    ALGCELL_CTX_HEAMO_CALLBACK(ctx) = callback;
    ALGCELL_CTX_HEAMO_USERDATA(ctx) = userdata;
  }
  if (func_mask & ALGCELL_FUNC_FUSION) {
    ALGCELL_CTX_IMG_FUSION(ctx) = true;
  }
  if (func_mask & ALGCELL_FUNC_HGB) {

  }
  ALGCELL_CTX_FUNC_MASK(ctx) = func_mask;
  return ALG_SUCC;
}

/**
* 算法测量推送血球图片
* @param  ctx_id		算法上下文ID
* @param  group_idx		分组索引
* @param  chl_idx		通道索引
* @param  img_array		图像缓存数组
* @param  array_size	数组大小
* @return 0 success other fail
 */
int
AlgCell_HeamoPushImage(AlgCtxID_t ctx_id, std::vector<AlgCellImg_t> &image_list, uint32_t group_idx, uint32_t chl_idx,
                       const std::map<std::string, float> &complementary_params) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL || !image_list.size()) {
    return -1;
  }

  std::list<HeamoImg_t> data_list;
  ALGLogInfo<< " image_list size "<<image_list.size();
  for (uint32_t idx = 0; idx < image_list.size(); idx++) {
    AlgCellImg_t *img = &image_list.at(idx);
    if (Heamo_AddImgList(ALGCELL_CTX_HEAMO_CTXID(ctx), data_list, group_idx, chl_idx, idx,
                         ALGCELL_IMG_DATA(img), ALGCELL_IMG_WIDTH(img),
                         ALGCELL_IMG_HEIGHT(img))) {
      ALGLogError << "[alg/samp] add imglist err!";
      return -3;
    }
  }
  auto complementary_params_iter = complementary_params.find(VIEW_PAIR_IDX);
  if (complementary_params_iter != complementary_params.end()) {
    const int &view_pair_idx = (int) complementary_params_iter->second;
    return Heamo_PushImage(ALGCELL_CTX_HEAMO_CTXID(ctx), data_list, group_idx, chl_idx, view_pair_idx);
  }
  ALGLogError << "failed to push heamo img";
  return -4;
}


int AlgCell_HeamoGetCount(bool zero_flag){

    return HeamoGetCount(zero_flag);
}
int AlgCell_HeamoPushHgb(AlgCtxID_t ctx_id, const std::vector<AlgHgbData_t> &data_list,
                         const std::vector<float> &coef_list) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  return Heamo_PushHgb(ALGCELL_CTX_HEAMO_CTXID(ctx), data_list, coef_list);
}


static void AlgSamp_ResultCallback(const char *name, const char *unit, const char *value, void *userdata) {
  std::list<AlgCellItem_t> *list = (std::list<AlgCellItem_t> *) userdata;
  AlgCellItem_t item;
  ALGCELL_ITEM_NAME(&item) = name;
  ALGCELL_ITEM_UNIT(&item) = unit;
  ALGCELL_ITEM_VALUE(&item) = value;
  list->push_back(item);
}


void PrintHeamoResult(const AlgCellRst &result) {
  ALGLogInfo << "ALG RESULT >>>\r\n序号  名称        单位        数值\r\n";
  //血球
  uint32_t idx = 1;
  for (const auto &item: result.heamo) {
    ALGLogInfo << std::setw(10) << idx++
               << std::setw(10) << item.name.data()
               << std::setw(10) << item.value
               << std::setw(10) << item.unit.data();

  }
  //rbc曲线
  ALGLogInfo << "rbc curve:";
  std::string curve_rbc_str;
  for (const auto &float_result: result.curve_rbc) {
    curve_rbc_str.append(std::to_string(float_result)).append(" ");
  }
  ALGLogInfo << curve_rbc_str;
  //plt 曲线
  ALGLogInfo << "plt curve:" << std::endl;
  std::string curve_plt_str;
  for (const auto &float_result: result.curve_plt) {
    curve_plt_str.append(std::to_string(float_result)).append(" ");
  }
  ALGLogInfo << curve_plt_str;

  std::cout << std::endl;
}

/**
* 算法测量生成结果
* @param  ctx_id		算法上下文ID
* @param  result		算法结果列表
* @param  timeout		超时时间
* @return
 */
int AlgCell_HeamoGetResult(AlgCtxID_t ctx_id, AlgCellRst_t &result, uint32_t timeout) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  int ret = 0;
  if (ALGCELL_CTX_FUNC_MASK(ctx) & ALGCELL_FUNC_HEAMO) {
    if (Heamo_WaitCplt(ALGCELL_CTX_HEAMO_CTXID(ctx), timeout)) {
      return -2;
    }
    ret = Heamo_GetResult(ALGCELL_CTX_HEAMO_CTXID(ctx), result.curve_rbc, result.curve_plt, AlgSamp_ResultCallback,
                          (void *) &result.heamo, result.alarm_results);
    if (ret) {
	  ALGLogError<<"Failed to get heamo result, err = "<<ret;
      return ret;
    }
    PrintHeamoResult(result);

  }
  if (ALGCELL_CTX_FUNC_MASK(ctx) & ALGCELL_FUNC_HGB) {
    //hgb值在血球部分放入
  }
#if(AI_USE_TIMECNT)
  TimeCnt_PrintResult();
#endif
  return 0;
}


int AlgCell_ModifyResult(const std::string& changed_param_key, AlgCellRst_t &result){
  int ret = Heamo_ModifyResult(initialed_mode_id, changed_param_key, result.curve_rbc, result.curve_plt, AlgSamp_ResultCallback,
							   (void *) &result.heamo, result.alarm_results);

  return ret;
}


/**
* 算法测量关闭
* @param  ctx_id		算法上下文ID
* @return
 */
int AlgCell_HeamoClose(AlgCtxID_t ctx_id) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  Heamo_Close(ALGCELL_CTX_HEAMO_CTXID(ctx));
  ALGCELL_CTX_FUNC_MASK(ctx) = 0;
  return 0;
}

static int
AlgClarity_ImgCallback(AiCtxID_t ctx_id, uint32_t item_id, AiImg_t *img, uint32_t group_idx, uint32_t chl_idx,
                       uint32_t view_order,
                       uint32_t view_idx, uint32_t processed_idx, AiImgStage_e stage, void *userdata,
                       std::list<NNetResult_t> &result, const int &view_pair_idx,
                       const std::map<std::string, float> &call_back_params) {
  AlgCtx_t *ctx = (AlgCtx_t *) userdata;
  if (ctx == NULL) {
    return -1;
  }
  if (ALGCELL_CTX_CLARITY_CALLBACK(ctx)) {
    cv::cvtColor(*img, *img, cv::COLOR_BGR2RGB);
    AlgCellImg_t image;
    ALGCELL_IMG_DATA(&image) = img->data;
    ALGCELL_IMG_SIZE(&image) = img->rows * img->cols * img->channels();
    ALGCELL_IMG_WIDTH(&image) = img->cols;
    ALGCELL_IMG_HEIGHT(&image) = img->rows;
    (*ALGCELL_CTX_CLARITY_CALLBACK(ctx))((AlgCtxID_t) ctx, group_idx, chl_idx, view_order, view_idx,
                                         processed_idx,
                                         (AlgCellStage_e) stage, &image, ALGCELL_CTX_CLARITY_USERDATA(ctx),
                                         view_pair_idx, call_back_params);
  }
  return 0;
}


/**
* 算法聚焦开启
* @param  ctx_id		算法上下文ID
* @param  width			图像宽度
* @param  height		图像高度
* @return
 */

int AlgCell_ClarityOpen(AlgCtxID_t ctx_id,uint32_t func_mask, AlgCellImageCallback_f callback, void *userdata, const std::map<std::string, std::vector<float>>& open_params) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return ALG_ERR_INVALID_ALG;
  }

  //解析输入参数
  uint32_t group_idx;
  bool debug, qc, calib;
  float img_h, img_w, img_h_um;
  std::vector<float> alarm_param_v, dilution_param_v,task_att_v;
  int ret = AlgCell_ParseOpenParams(open_params, debug, group_idx, qc, img_h, img_w,
                                    img_h_um, alarm_param_v, dilution_param_v, task_att_v, calib, true);
  if(ret){
    ALGLogError<<"failed to parse open params";
    return ret;
  }
  bool img_fusion = func_mask & ALGCELL_FUNC_FUSION;
  ALGCELL_CTX_CLARITY_CALLBACK(ctx) = callback;
  ALGCELL_CTX_CLARITY_USERDATA(ctx) = userdata;
  return Clarity_Open(ALGCELL_CTX_CLARITY_CTXID(ctx), img_fusion, debug, AlgClarity_ImgCallback, (void *) ctx);
}

/**
* 算法聚焦推图
* @param  ctx_id		聚焦上下文ID
* @param  chl_idx		聚焦通道索引
* @param  img_array		图像缓存指针数组
* @param  array_size	数组尺寸
* @return
 */
int AlgCell_ClarityPushImage(AlgCtxID_t ctx_id, uint32_t group_idx, uint32_t chl_idx, AlgCellImg_t *img_array,
                             uint32_t array_size, const std::map<std::string, float> &complementary_params) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL || !array_size) {
    return -1;
  }

  std::list<ClarityImg_t> img_list;
  for (uint32_t idx = 0; idx < array_size; idx++) {
    if (Clarity_AddImgList(ALGCELL_CTX_CLARITY_CTXID(ctx), img_list, group_idx, chl_idx, idx,
                           ALGCELL_IMG_DATA(&img_array[idx]),
                           ALGCELL_IMG_WIDTH(&img_array[idx]), ALGCELL_IMG_HEIGHT(&img_array[idx]))) {
      return -3;
    }
  }
  std::cout << "chl_idx after adding images to list: " << chl_idx << std::endl;
  auto complementary_params_iter = complementary_params.find(VIEW_PAIR_IDX);
  if (complementary_params_iter != complementary_params.end()) {
    const int &view_pair_idx = (int) complementary_params_iter->second;
    std::cout << "chl_idx before calling Clarity_PushImage: " << chl_idx << std::endl;
    return Clarity_PushImage(ALGCELL_CTX_CLARITY_CTXID(ctx), group_idx, chl_idx, img_list, view_pair_idx);
  }
  ALGLogError << "failed to push clarity img";
  return -4;
}

/**
* 算法聚焦获取最近清晰度
* @param  ctx_id		聚焦上下文ID
* @param  index			索引输出指针
* @param  value			数值输出指针
* @return
 */
int AlgCell_ClarityGetResultAll(AlgCtxID_t ctx_id, std::vector<AlgClarityValue_t> &list) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == nullptr) {
    ALGLogInfo << "上下文 ctx 为空";
    return -1;
  }
  auto result = Clarity_WaitCplt(ALGCELL_CTX_CLARITY_CTXID(ctx), 0XFFFF);

  if (result) {
    ALGLogInfo << "图片任务队列处理未完成 result: " << result;
    return -2;
  }
  return Clarity_GetAllValue(ALGCELL_CTX_CLARITY_CTXID(ctx), list);
}

/**
* 算法聚焦获取最近清晰度
* @param  ctx_id		聚焦上下文ID
* @param  index			索引输出指针
* @param  value			数值输出指针
* @return
 */
int AlgCell_ClarityGetResultLast(AlgCtxID_t ctx_id, uint32_t *index, float *value) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  if (Clarity_WaitCplt(ALGCELL_CTX_CLARITY_CTXID(ctx), 0XFFFF)) {
    return -2;
  }

  return Clarity_GetLastValue(ALGCELL_CTX_CLARITY_CTXID(ctx), index, value);
}

/**
* 算法聚焦获取最佳清晰度
* @param  ctx_id		聚焦上下文ID
* @param  idx			索引输出指针
* @param  value			数值输出指针
* @return
 */
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

int AlgCell_ClarityGetResultBest(AlgCtxID_t ctx_id, uint32_t *index, float *value) {


  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }

  //auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto ret = Clarity_WaitCplt(ALGCELL_CTX_CLARITY_CTXID(ctx), 0xFFFF);
  //auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  //auto cost_time = end - start;
  //ALGLogInfo << "processing time: " << cost_time;
  if (ret != 0) {
    ALGLogInfo << "Clarity_WaitCplt err : timeOut " << ret << " : " << 0xFFFF;
    EVERROR(EVID_ERR, "Clarity_WaitCplt  err")
    return -2;
  }

  return Clarity_GetBestValue(ALGCELL_CTX_CLARITY_CTXID(ctx), index, value);

}

int AlgCell_ClarityGetResultFarNear(AlgCtxID_t ctx_id, uint32_t *index, float *value) {
  return AlgCell_ClarityGetResultBest(ctx_id, index, value);
}

int AlgCell_ClarityGetResultCoarse(AlgCtxID_t ctx_id, uint32_t *index, float *value) {
  return AlgCell_ClarityGetResultBest(ctx_id, index, value);
}

int AlgCell_ClarityGetResultFineFluMicrosphere(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value){
  return AlgCell_ClarityGetResultBest(ctx_id, index, value);

}
int AlgCell_ClarityGetResultCoarseFluMicrosphere(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value){
  return AlgCell_ClarityGetResultBest(ctx_id, index, value);
}
int AlgCell_ClarityGetResultMilkBoardLine(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value){
    return AlgCell_ClarityGetResultBest(ctx_id, index, value);
}

/**
* 算法聚焦关闭
* @param  ctx_id		聚焦上下文ID
* @return
 */
int AlgCell_ClarityClose(AlgCtxID_t ctx_id) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  return Clarity_Close(ALGCELL_CTX_CLARITY_CTXID(ctx));
}


int AlgCell_Stop(AlgCtxID_t ctx_id, uint32_t timeout) {
  AlgCtx_t *ctx = (AlgCtx_t *) ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  Ai_CleanItemAll(ALGCELL_CTX_AI_CTXID_HEAMO(ctx), timeout);
  Ai_CleanItemAll(ALGCELL_CTX_AI_CTXID_CLARITY(ctx), timeout);
  AlgCell_ClarityClose(ctx_id);
  AlgCell_HeamoClose(ctx_id);
  return 0;
}


