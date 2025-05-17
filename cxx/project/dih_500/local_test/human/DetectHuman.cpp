//
// Created by y on 23-8-4.
//
#include <iostream>
#include <algorithm>
#include <numeric>

#include "DetectHuman.h"
#include "Calibration.h"
#include "utils.h"
#include "event.h"
#include "DihLogPlog.h"
#include "TinnyNetwork.h"
#include "ModelConfig.h"
#include "timecnt.h"
#include "ai.h"
#include "alg_heamo.h"
#include "project_utils.h"

# define NET_USE_TIMECNT 1

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
namespace ALG_LOCAL{
#define WBC_BRIGHT_FLUO_FUSION_RATE 0.5  // WBC明暗场融合比例


static uint8_t *Alg_ReadFile(const char *path,  uint32_t *len)
{
  FILE *fp = NULL;
  char filename[256];
  if(path == NULL)
  {
    return NULL;
  }
  memset(filename, 0, 256);
//  snprintf(filename, 256, "%s/%s.%s", path, keyword, suffix);
  snprintf(filename, 256, "%s", path);
  //DLOG(INFO, "sample process pos = %d x=%d y=%d process=%d (%d/%d)", ctx->pos, ctx->x, ctx->y, process, ctx->currentField, ctx->totalField);
  EVINFO(EVID_INFO, "Use rknn model path: %s", filename);
  fp = fopen(filename, "rb");
  if(fp == NULL)
  {
    printf("[alg/file] unfind:%s\r\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  uint32_t size = ftell(fp);
  if(fseek(fp, 0, SEEK_SET))
  {
    return NULL;
  }
  uint8_t *data = (uint8_t*)malloc(size);
  if (data == NULL)
  {
    return NULL;
  }
  if(size != fread(data, 1, size, fp))
  {
    free(data);
    return NULL;
  }
  if(len)
  {
    *len = size;
  }
  fclose(fp);
  return data;
}

bool DetectHuman::Init(const InitParam& init_param) {
#if(NET_USE_TIMECNT)
  TimeCnt_Init("wbc_mid", 1);
  TimeCnt_Init("wbc4_mid", 1);
  TimeCnt_Init("plt_merge", 1);
#endif
  this->MapGroupId();
  //初始化算法模型
  bool guard{ false};
  this->net_ctx = NNet_Init("./data");
  if(this->net_ctx== nullptr){
    std::cout<<"Failed to init net list"<<std::endl;
    return false;
  }
  for(const auto &one_para:init_param.alg_param_v){

    //打印所有接受到的参数
    std::cout<<"sample_type: "<<one_para.alg_type<<std::endl;
    std::cout<<"alg_enable: "<<one_para.enable<<std::endl;
    if(one_para.enable){
      for(const auto& model_path:one_para.model_paths_v){
        std::cout<<model_path<<std::endl;
      }
      for(const auto& init_param_float: one_para.init_param_float_v){
        std::cout<<init_param_float<<std::endl;
      }
    }
    this->group_id = this->group_idx_to_id_m[init_param.detect_type];
    //rbc
    if(one_para.alg_type==RBC&&one_para.enable){
      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For rbc model, model path, float param should be 2 and 0 respectively, but "
                  <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }
      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }

      //要求的各个参数的个数
      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel
              (this->net_ctx, this->group_id, NNET_MODID_RBC_VOLUME_SHPERICAL, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init rbc wrong"<<std::endl;
        return false;
      }
      std::cout<<"init rbc succeed."<<std::endl;
      guard = true;
    }

    //wbc
    else if((one_para.alg_type==WBC||one_para.alg_type==WBC_SINGLE)&&one_para.enable){

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For wbc model, model path, float param should be 2 and 0 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }

      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_WBC, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init wbc wrong"<<std::endl;
        return false;
      }
      std::cout<<"init wbc succeed."<<std::endl;
      guard = true;
    }

    //wbc4
    else if(one_para.alg_type==WBC4&&one_para.enable){

      //wbc
      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=4||float_param_size!=0){
        std::cout<<"For wbc model, model path, float param should be 2 and 0 respectively, but "
                  <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }

      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_WBC, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init wbc wrong"<<std::endl;
        return false;
      }
      std::cout<<"init wbc succeed."<<std::endl;

      //wbc4
      mod_size = 0;
      label_size = 0;
      model_path = one_para.model_paths_v[2];
      config_path = one_para.model_paths_v[3];

      mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      memset(&config_params,0,sizeof(config_params));
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      required_params_nums={1, 1, 1, 18, LABEL_NUMS_CUSTOM, 0, 0};
      ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_WBC4, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);

      if(ret!=0){
        std::cout<<"init wbc4 wrong"<<std::endl;
        return false;
      }
      std::cout<<"init wbc4 succeed."<<std::endl;
      guard = true;
    }
    //pla 疟原虫,疟原虫需要2个模型，2个xml 文件
    else if (one_para.alg_type == PLA && one_para.enable) {
      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if (model_path_size != 4 || float_param_size != 0) {
        std::cout << "疟原虫需要2个模型,2个xml 文件,不需要float 参数"<< model_path_size << " " << float_param_size << " was given"
                  << std::endl;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(), &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if (model_config.ReadXmlFile(config_path, config_params)) {
        std::cout << "failed to parse xml" << std::endl;
        return false;
      }

      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      std::cout << "当前的group id:  " << this->group_id << " \n";
      int ret = NNet_AddModel(
          this->net_ctx, this->group_id, NNET_MODID_PLA, mod_data, mod_size,
          LETTERBOX, required_params_nums[0], required_params_nums[1],
          required_params_nums[2], required_params_nums[3],
          required_params_nums[4], required_params_nums[5],
          required_params_nums[6], config_params.model_type, config_params.nms,
          config_params.conf, config_params.anchors, config_params.labels,
          config_params.reserved_float_params,
          config_params.reserved_string_params);
      if (ret != 0) {
        std::cout << "初始化疟原虫模型PLA 失败" << std::endl;
        return false;
      }
      std::cout << "初始化疟原虫模型PLA 成功." << std::endl;

      // PLA4 第二个模型
      mod_size = 0;
      label_size = 0;
      model_path = one_para.model_paths_v[2];
      config_path = one_para.model_paths_v[3];

      mod_data = Alg_ReadFile(model_path.c_str(), &mod_size);
      memset(&config_params, 0, sizeof(config_params));
      if (model_config.ReadXmlFile(config_path, config_params)) {
        std::cout << "failed to parse xml" << std::endl;
        return false;
      }

      required_params_nums = {1, 1, 1, 18, 5, 0, 0};
      std::cout << "当前的group id:  " << this->group_id << " \n";
      ret = NNet_AddModel(
          this->net_ctx, this->group_id, NNET_MODID_PLA4, mod_data, mod_size,
          LETTERBOX, required_params_nums[0], required_params_nums[1],
          required_params_nums[2], required_params_nums[3],
          required_params_nums[4], required_params_nums[5],
          required_params_nums[6], config_params.model_type, config_params.nms,
          config_params.conf, config_params.anchors, config_params.labels,
          config_params.reserved_float_params,
          config_params.reserved_string_params);

      if (ret != 0) {
        std::cout << "初始化 疟原虫模型 PLA4 失败" << std::endl;
        return false;
      }
      std::cout << "初始化 疟原虫模型  PLA4 成功" << std::endl;
      guard = true;
    }
    // plt
    else if (one_para.alg_type == PLT && one_para.enable) {
      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For plt model, model path, float param should be 2 and 0 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 1, 1, 18, 3, 3, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_PLT, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init plt wrong"<<std::endl;
        return false;
      }
      std::cout<<"init plt succeed."<<std::endl;
      guard = true;
    }
    // baso
    else if (one_para.alg_type == BASO && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For baso model, model path, float param should be 2 and 0 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_BASO, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init baso wrong"<<std::endl;
        return false;
      }
      std::cout<<"init baso succeed."<<std::endl;
      guard = true;
    }
    // ret
    else if (one_para.alg_type == RET && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=2){
        std::cout<<"For ret model, model path, float param should be 2 and 2 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string label_name = one_para.model_paths_v[1];
      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      uint8_t *label_data = Alg_ReadFile(label_name.c_str(), &label_size);

      float nms_thr = one_para.init_param_float_v[0];
      float conf_thr = one_para.init_param_float_v[1];
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_RET, mod_data, mod_size, label_data, label_size,
                              LETTERBOX, nms_thr, conf_thr, NNET_TYPE_YOLO_RECT,{0});
      if(ret!=0){
        std::cout<<"init bas clarity wrong"<<std::endl;
        return false;
      }
      std::cout<<"init bas clarity succeed."<<std::endl;
      guard = true;
    }

    // baso clarity
    else if (one_para.alg_type == BASCLARITY && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=2){
        std::cout<<"For baso clarity model, model path, float param should be 2 and 2 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string label_name = one_para.model_paths_v[1];
      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      uint8_t *label_data = Alg_ReadFile(label_name.c_str(), &label_size);

      float nms_thr = one_para.init_param_float_v[0];
      float conf_thr = one_para.init_param_float_v[1];
      int ret = NNet_AddModel(net_ctx, this->group_id, NNET_MODID_BAS_CLARITY, mod_data, mod_size, label_data, label_size,
                              LETTERBOX, nms_thr, conf_thr, NNET_TYPE_YOLO_RECT,{0});
      if(ret!=0){
        std::cout<<"init bas clarity wrong"<<std::endl;
        return false;
      }
      std::cout<<"init bas clarity succeed."<<std::endl;
      guard = true;
    }

    // wbc4_single
    else if (one_para.alg_type == WBC4_SINGLE && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For wbc single model, model path, float param should be 2 and 0 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 1, 1, 18, LABEL_NUMS_CUSTOM, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_WBC4, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init wbc4_single wrong"<<std::endl;
        return false;
      }
      std::cout<<"init wbc_single succeed."<<std::endl;
      guard = true;
    }

    // rbc volume
    else if (one_para.alg_type == RBC_VOLUME && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=4||float_param_size!=3){
        std::cout<<"For volume model, model path, float param should be 4 and 3 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }
      //倾斜红细胞检测模型
      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string label_name = one_para.model_paths_v[1];
      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      uint8_t *label_data = Alg_ReadFile(label_name.c_str(), &label_size);

      float nms_thr = one_para.init_param_float_v[0];
      float conf_thr = one_para.init_param_float_v[1];
      int ret = NNet_AddModel(net_ctx, this->group_id, NNET_MODID_RBC_INCLINE_DET, mod_data, mod_size, label_data, label_size,
                              NORMAL, nms_thr, conf_thr, NNET_TYPE_PP_ROTATED_POLY,{0});

      if(ret!=0){
        std::cout<<"init incline det wrong"<<std::endl;
        return false;
      }
      //倾斜红细胞分割模型
      mod_size = 0;
      label_size = 0;
      model_path = one_para.model_paths_v[2];
      label_name = one_para.model_paths_v[3];
      mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      label_data = Alg_ReadFile(label_name.c_str(), &label_size);


      conf_thr = one_para.init_param_float_v[3];
      ret = NNet_AddModel(net_ctx, this->group_id, NNET_MODID_RBC_INCLINE_SEG, mod_data, mod_size, label_data, label_size,
                              NORMAL, nms_thr, conf_thr, NNET_TYPE_SEG_ALL,{0});

      if(ret!=0){
        std::cout<<"init incline seg wrong"<<std::endl;
        return false;
      }


      std::cout<<"init incline succeed."<<std::endl;
      guard = true;
    }

    // 梯度清晰度
    else if (one_para.alg_type == GRADCLARITY && one_para.enable) {
      std::cout<<"init grad clarity succeed."<<std::endl;
      guard = true;
    }

    // AI清晰度算法
    else if (one_para.alg_type == AI_CLARITY && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=1){
        std::cout<<"For ai clarity model, model path, float param should be 2 and 1 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string label_name = one_para.model_paths_v[1];
      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      uint8_t *label_data = Alg_ReadFile(label_name.c_str(), &label_size);

      float thr = one_para.init_param_float_v[0];

      int ret = NNet_AddModel(net_ctx, this->group_id, NNET_MODID_AI_CLARITY, mod_data, mod_size, label_data, label_size,
                              NORMAL, thr, 0, NNET_TYPE_CLS_ALL,{0});

      if(ret!=0){
        std::cout<<"init ai clarity wrong"<<std::endl;
        return false;
      }

      std::cout<<"init ai clarity succeed."<<std::endl;
      guard = true;
    }

    // plt_volume
    else if (one_para.alg_type == PLT_VOLUME && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For plt volume clarity model, model path, float param should be 2 and 0 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_PLT_VOLUME, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);

      if(ret!=0){
        std::cout<<"init plt volume wrong"<<std::endl;
        return false;
      }

      std::cout<<"init plt volume succeed."<<std::endl;
      guard = true;
    }

    // AI清晰度 远近焦算法
    else if (one_para.alg_type == AI_CLARITY_FAR_NEAR && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For ai clarity far near model, model path, float param should be 2 and 0 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 0, 0, 0, 8, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_AI_CLARITY_FAR_NEAR, mod_data, mod_size, LEFT_TOP_CROP,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);

      if(ret!=0){
        std::cout<<"init ai clarity far near wrong"<<std::endl;
        return false;
      }

      std::cout<<"init ai clarity far near succeed."<<std::endl;
      guard = true;
    }
    // GERM
    else if (one_para.alg_type == MILK_GERM && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For germ model, model path, float param should be 2 and 0 respectively, but "
                 <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }
      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;

      std::cout << "加载 XML 配置文件: " << config_path << std::endl;

      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_MILK_GERM, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);

      if(ret!=0){
        std::cout<<"init milk germ"<<std::endl;
        return false;
      }

      std::cout<<"init milk germ succeed."<<std::endl;
      guard = true;
    }

    // MILK_CELL
    else if (one_para.alg_type == MILK_CELL && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For milk cell model, model path, float param should be 2 and 0 respectively, but "
                  <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }
      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_MILK_CELL, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);

      if(ret!=0){
        std::cout<<"init milk cell wrong"<<std::endl;
        return false;
      }

      std::cout<<"init milk succeed."<<std::endl;
      guard = true;
    }

    // rbc_volume_spherical_box
    else if (one_para.alg_type == RBC_VOL_SPH_BOX && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=2){
        std::cout<<"For rbc volume spherical model, model path, float param should be 2 and 2 respectively, but "
                  <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string label_name = one_para.model_paths_v[1];
      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      uint8_t *label_data = Alg_ReadFile(label_name.c_str(), &label_size);

      float nms_thr = one_para.init_param_float_v[0];
      float conf_thr = one_para.init_param_float_v[1];

      int ret = NNet_AddModel(net_ctx, this->group_id, NNET_MODID_RBC_VOLUME_SHPERICAL, mod_data, mod_size, label_data, label_size,
                              LETTERBOX, nms_thr, conf_thr, NNET_TYPE_YOLO_RECT,{0});

      if(ret!=0){
        std::cout<<"init rbc volume spherical wrong"<<std::endl;
        return false;
      }
      std::cout<<"init rbc volume spherical succeed."<<std::endl;
      guard = true;
    }

    // rbc_volume_spherical_seg
    else if (one_para.alg_type == RBC_VOL_SPH_SEG && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=4||float_param_size!=3){
        std::cout<<"For rbc_volume_spherical_seg, model, model path, float param should be 4 and 3 respectively, but "
                  <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }
      //红细胞检测
      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 1, 1, 18, 1, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_RBC, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init wbc wrong"<<std::endl;
        return false;
      }
      //倾斜红细胞分割模型
      mod_size = 0;
      label_size = 0;
      model_path = one_para.model_paths_v[2];
      config_path = one_para.model_paths_v[3];

      mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      memset(&config_params, 0, sizeof(config_params));
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      required_params_nums = {1, 0, 1, 0, 2, 0, 0};
      ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_RBC_INCLINE_SEG, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init incline seg wrong"<<std::endl;
        return false;
      }

      std::cout<<"init rbc volume spherical seg succeed."<<std::endl;
      guard = true;
    }

    // spherical_focal
    else if (one_para.alg_type == SPHERICAL_FOCAL && one_para.enable) {
      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For spherical_focal model, model path, float param should be 2 and 0 respectively, but "
                  <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }

      std::vector<float> required_params_nums{1, 0, 1, 0, 2, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_SPHERICAL_FOCAL, mod_data, mod_size,LETTERBOX,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);
      if(ret!=0){
        std::cout<<"init spherical focal wrong"<<std::endl;
        return false;
      }
      std::cout<<"init spherical focal succeed."<<std::endl;
      guard = true;
    }

    // classification_custom
    // 完全根据model.xml对模型进行解析.启用该算法,若模型类别数与xml中配置个数不同,程序可能崩溃
    else if (one_para.alg_type == CLASSIFICATION_CUSTOM && one_para.enable) {

      int model_path_size = (int)one_para.model_paths_v.size();
      int float_param_size = (int)one_para.init_param_float_v.size();
      if(model_path_size!=2||float_param_size!=0){
        std::cout<<"For classification custom model, model path, float param should be 2 and 0 respectively, but "
                  <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
        return false;
      }

      uint32_t mod_size = 0;
      uint32_t label_size = 0;
      std::string model_path = one_para.model_paths_v[0];
      std::string config_path = one_para.model_paths_v[1];

      uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
      ALG_DEPLOY::XML::ModelConfig model_config;
      ALG_DEPLOY::XML::ConfigParams config_params;
      if(model_config.ReadXmlFile(config_path, config_params)){
        std::cout<<"failed to parse xml"<<std::endl;
        return false;
      }


      std::vector<float> required_params_nums{1, 0, 0, 0, LABEL_NUMS_CUSTOM, 0, 0};
      int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_CLASSIFICATION_CUSTOM, mod_data, mod_size, LEFT_TOP_CROP,
                              required_params_nums[0],
                              required_params_nums[1],
                              required_params_nums[2],
                              required_params_nums[3],
                              required_params_nums[4],
                              required_params_nums[5],
                              required_params_nums[6],
                              config_params.model_type,
                              config_params.nms,
                              config_params.conf,
                              config_params.anchors,
                              config_params.labels,
                              config_params.reserved_float_params,
                              config_params.reserved_string_params);

      if(ret!=0){
        std::cout<<"init classification custom wrong"<<std::endl;
        return false;
      }

      std::cout<<"init classification custom succeed."<<std::endl;
      guard = true;
    } else if (one_para.alg_type == CLARITY_MLIK_BOARDLINE && one_para.enable) {
      std::cout<<"执行CLARITY_MILK_BOARDLINE，牛奶划线"<<std::endl;

        int model_path_size = (int)one_para.model_paths_v.size();
        int float_param_size = (int)one_para.init_param_float_v.size();
        if(model_path_size!=2||float_param_size!=0){
            std::cout<<"For ai clarity far near model, model path, float param should be 2 and 0 respectively, but "
                     <<model_path_size<<" "<<float_param_size<<" was given"<<std::endl;
            return false;
        }

        uint32_t mod_size = 0;
        uint32_t label_size = 0;
        std::string model_path = one_para.model_paths_v[0];
        std::string config_path = one_para.model_paths_v[1];

        uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
        ALG_DEPLOY::XML::ModelConfig model_config;
        ALG_DEPLOY::XML::ConfigParams config_params;
        if(model_config.ReadXmlFile(config_path, config_params)){
            std::cout<<"failed to parse xml"<<std::endl;
            return false;
        }


        std::vector<float> required_params_nums{1, 0, 0, 0, LABEL_NUMS_CUSTOM, 0, 0};
        int ret = NNet_AddModel(this->net_ctx, this->group_id, NNET_MODID_AI_CLARITY_MILK_BOARDLINE, mod_data, mod_size, LEFT_TOP_CROP,
                                required_params_nums[0],
                                required_params_nums[1],
                                required_params_nums[2],
                                required_params_nums[3],
                                required_params_nums[4],
                                required_params_nums[5],
                                required_params_nums[6],
                                config_params.model_type,
                                config_params.nms,
                                config_params.conf,
                                config_params.anchors,
                                config_params.labels,
                                config_params.reserved_float_params,
                                config_params.reserved_string_params);

        if(ret!=0){
            std::cout<<"init milk board line far near wrong"<<std::endl;
            return false;
        }

        std::cout<<"init milk board line far near succeed."<<std::endl;
        guard = true;
    }
  }



  if(!guard){
      std::cout<<"Error, configured to init human, but No human algs are initialized."<<std::endl;
      return false;
  }

  //判断当前算法类型下的辅助函数是否需要测试，如human的GetRbcAndPltResult函数
  for(const auto& one_para:init_param.assit_func_param_v){
    std::cout<<one_para.fun_name<<std::endl;
    std::cout<<one_para.enable<<std::endl;

    if(one_para.fun_name=="GetRbcAndPltResult"&&one_para.enable){
      this->test_get_rbc_and_plt_result= true;
    } else if(one_para.fun_name=="GetWbcResult"&&one_para.enable){
      this->test_get_wbc_result= true;
    }
  }
  std::cout<<"Init human succeed"<<std::endl;

  if(spherical_reagent_fitting.Init()){
    ALGLogError<<"Failed to init spherical_reagent_fitting";
    return false;
  }

  return true;
}



//rbc
bool DetectHuman::ForwardRbc(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                             const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  std::cout<<"Start forward rbc."<<std::endl;
  detect_result_v.clear();
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for rbc forward."<<std::endl;
    return false;
  }
  //模型推理
  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_RBC_VOLUME_SHPERICAL, img_brightness, result);
  if(ret!=0){
    std::cout<<"rbc forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  //个数计数
  this->sample_contex.Rbc += int(detect_result_v.size());


  //面积保存
  CountVolumeParameter(this->sample_contex.rbc_volume_v, result, img_brightness->cols, img_brightness->rows );

  //绘制结果
  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw rbc result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);

  return true;
}


//WBC4 SINGLE
bool DetectHuman::ForwardWbc4Single(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                             const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  std::cout<<"Start forward wbc4 single."<<std::endl;
  detect_result_v.clear();
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for wbc4 single forward."<<std::endl;
    return false;
  }
  cv::flip(*img_brightness,*img_brightness,0);
  //模型推理
  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_WBC4, img_brightness, result);
  if(ret!=0){
    std::cout<<"wbc single  forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  //个数计数
  this->sample_contex.Rbc += int(detect_result_v.size());
  //绘制结果
  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 1, 2)){
    std::cout<<"Fail to draw wbc single result."<<std::endl;
    return false;
  }
  cv::flip(img_out,img_out,0);

  mat_bright_result_v.emplace_back(img_out);
  std::cout<<"Forward wbc single over"<<std::endl;
  return true;
}




bool DetectHuman::ForwardPlt(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                             const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){

  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for plt forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();
  int ret = 0;

  //融合图像
//  cv::Mat target_img;
//  ret = MergePltImg(*img_brightness, *img_fluorescence, target_img);
//  if(ret){
//    return ret;
//  }
  cv::Mat target_img = *img_brightness;

  std::list<NNetResult_t> result_primary, result;
  ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_PLT, &target_img, result_primary);
  if(ret!=0){
    std::cout<<"plt forward error."<<std::endl;
    return false;
  }

  //查找当前模型box类别个数,对于plt模型,reserved_float_parmas保存的各个类别对应的conf
  std::vector<float> conf_v;
  ret = NNet_GetReservedFloatPrams(this->net_ctx, this->group_id, NNET_MODID_PLT, conf_v);
  if(ret){
    return ret;
  }

  ret = CountBoxCategoryConf(result_primary, conf_v, result);
  if(ret){
    return ret;
  }

  detect_result_v.assign(result.begin(), result.end());

  this->sample_contex.Plt += int(detect_result_v.size());

  //绘制结果
  cv::Mat img_out;
  if(!DrawMidResult(&target_img, img_height, img_width, detect_result_v, img_out, true, 1, 2)){
    std::cout<<"Fail to draw plt result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);

  return true;
}

bool DetectHuman::ForwardWbc(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                             const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  std::cout<<"Start forward wbc."<<std::endl;

  detect_result_v.clear();
  if(!img_brightness || !img_fluorescence){
    std::cout<<"Emtpy brightness or fluorescence image are given for wbc forward."<<std::endl;
    return false;
  }

  detect_result_v.clear();
  std::list<NNetResult_t> result;
  cv::Mat bright_fluo_merged;



#if(NET_USE_TIMECNT)
  TimeCnt_Start("wbc_mid");
#endif
  MergeBrightFluoImg(*img_brightness, *img_fluorescence, WBC_BRIGHT_FLUO_FUSION_RATE, bright_fluo_merged);
#if(NET_USE_TIMECNT)
  TimeCnt_End("wbc_mid");
#endif
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_WBC, &bright_fluo_merged, result);
  if(ret!=0){
    std::cout<<"wbc forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());


  this->sample_contex.Wbc += int(detect_result_v.size());

  cv::Mat img_out;
  if(!DrawMidResult(&bright_fluo_merged, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw rbc result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);

  std::cout<<"Forward wbc over"<<std::endl;
  return true;
}

//疟原虫推理增加 2025年4月25日17:25:30
bool DetectHuman::ForwardPLA(std::vector<NNetResult> &detect_result_v,
                             cv::Mat *img_brightness, cv::Mat *img_fluorescence,
                             const int &img_height, const int &img_width,
                             std::vector<cv::Mat> &mat_bright_result_v,
                             std::vector<cv::Mat> &mat_fluo_result_v) {

  std::cout << "开始推理疟原虫 ." << std::endl;

  detect_result_v.clear();
  if (!img_brightness || !img_fluorescence) {
    std::cout<< "图片为空. \n";
    return false;
  }

  detect_result_v.clear();
  std::list<NNetResult_t> result;
  //将明暗场图片融合为蓝色图像
  cv::Mat bright_fluo_merged;
  int ret = MergePltImg(*img_brightness, *img_fluorescence, bright_fluo_merged);
  // 再次融合，将蓝色图像与明场图像0.5 比例融合
  cv::Mat bright_blue_img;
  MergeBrightFluoImg(*img_brightness, bright_fluo_merged,
                     WBC_BRIGHT_FLUO_FUSION_RATE, bright_blue_img);
  //明暗场图片第一次推理
  ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_PLA,
                       &bright_blue_img, result);
  if (ret != 0) {
    std::cout << "疟原虫模型PLA group_id : " << this->group_id << " 推理失败."
              << std::endl;
    return false;
  }else{
    std::cout << "疟原虫模型PLA group_id : " << this->group_id << " 推理成功."
              << std::endl;
  }
  detect_result_v.assign(result.begin(), result.end());

  int PLA_COUNT=0;
  for (const auto &one_result : result) {
    if (one_result.box.name == "PLA") {
      PLA_COUNT++;
    }
  }
  std::cout << "检测到疟原虫个数--" << PLA_COUNT << "\n";
  if (PLA_COUNT==0) {
    std::cout << "未检测到疟原虫，不进行第二次模型推理 " << PLA_COUNT << "\n";
    return true;
  }
  // 第二次推理图像准备，图片拼接
  std::vector<cv::Mat>montageImg_v;
  if (!MergeImgPreProcess(montageImg_v, *img_brightness, bright_fluo_merged,
                          result)) {
    std::cout << "疟原虫PLA4模型组图拼接 失败" << std::endl;
    return false;
  }
  std::cout << "拼接图片完成 " << montageImg_v.size() << "\n";
  //开始第二次推理
  for (auto &montageImg : montageImg_v) {
    result.clear();
    ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_PLA4,
                         &montageImg, result);
    if (ret != 0) {
      std::cout << "疟原虫 PLA4 模型 推理失败"
                << std::endl;
      return false;
    }
    detect_result_v.clear();
    detect_result_v.assign(result.begin(), result.end());

    if (!DrawMidResult(&montageImg, img_height, img_width, detect_result_v,
                       montageImg, true, 1, 2)) {
      std::cout << "Fail to draw rbc result." << std::endl;
      return false;
    }
    mat_bright_result_v.emplace_back(montageImg);
  }
  return true;
}

bool DetectHuman::ForwardWbcSingle(std::vector<NNetResult>& detect_result_v, cv::Mat* img_brightness, cv::Mat* img_fluorescence, const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v) {
  std::cout<<"Start forward wbc_single."<<std::endl;

  detect_result_v.clear();
  if(!img_brightness ){
    std::cout<<"Emtpy brightness image are given for wbc single forward."<<std::endl;
    return false;
  }

  detect_result_v.clear();
  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_WBC, img_brightness, result);
  if(ret!=0){
    std::cout<<"wbc forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());


  this->sample_contex.Wbc += int(detect_result_v.size());

  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw wbc single result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);

  std::cout<<"Forward wbc single over"<<std::endl;
  return true;
}


bool DetectHuman::ForwardWbc4(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                              const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  std::cout<<"start forward wbc4_single"<<std::endl;
  if(!img_brightness||!img_fluorescence){
    std::cout<<"Emtpy brightness or fluorescence image are given for wbc4 forward."<<std::endl;
    return false;
  }
  //推理获取白细胞
  detect_result_v.clear();
  std::list<NNetResult_t> result;
  cv::Mat bright_fluo_merged;
  MergeBrightFluoImg(*img_brightness, *img_fluorescence, WBC_BRIGHT_FLUO_FUSION_RATE, bright_fluo_merged);
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_WBC, &bright_fluo_merged, result);

  if(ret!=0){
    std::cout<<"wbc forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  std::cout<<"wbc4 wbc count "<<result.size()<<std::endl;

  // 4分类图像预处理
  std::vector<cv::Mat>  montageImg_v;
  if(!MergeImgPreProcess(montageImg_v, *img_brightness, *img_fluorescence, result)){
    std::cout<<"BloodCellClassify： failed to PreProcess WBC4."<<std::endl;
    return false;
  }

  for(auto& montageImg:montageImg_v){
    result.clear();
    ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_WBC4, &montageImg, result);
    if(ret!=0){
      std::cout<<"BloodCellClassify： failed to run WBC4 NetClassify."<<std::endl;
      return false;
    }
    detect_result_v.clear();
    detect_result_v.assign(result.begin(),result.end());

    if(!DrawMidResult(&montageImg, img_height, img_width, detect_result_v, montageImg, true, 1, 2)){
      std::cout<<"Fail to draw rbc result."<<std::endl;
      return false;
    }
    mat_bright_result_v.emplace_back(montageImg);
  }
  return true;
}
//baso
bool DetectHuman::ForwardBaso(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                              const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){

  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for baso forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();

  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_BASO, img_brightness, result);

  if(ret!=0){
    std::cout<<"bas forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  this->sample_contex.Baso += int(detect_result_v.size());
  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw baso result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);

  return true;
}


static int Clarity_GetClarity(float &value, cv::Mat &img)
{
  cv::Mat oriImg = img;
  cv::Mat grayImg, hFeatureMap, vFeatureMap, featureMap;
  if(oriImg.channels() != 3)
  {
    return -2;
  }
  cv::cvtColor(oriImg, grayImg, cv::COLOR_BGR2GRAY);
  cv::Mat hkernel = (cv::Mat_<int>(3, 3) << -1, 0, 1, 0, 0, 0, 0, 0, 0);
  cv::Mat vkernel = (cv::Mat_<int>(3, 3) << -1, 0, 0, 0, 0, 0, 1, 0, 0);

  filter2D(grayImg, hFeatureMap, grayImg.depth(), hkernel);
  filter2D(grayImg, vFeatureMap, grayImg.depth(), vkernel);
  multiply(hFeatureMap, vFeatureMap, featureMap);

  value = (float)mean(featureMap)[0];
  return 0;
}

//梯度清晰度
bool DetectHuman::ForwardGradClarity(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                        const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  float value = 0.f;
  int ret = Clarity_GetClarity(value, *img_brightness);
  if(ret!=0){
    std::cout<<"Fail to run clarity."<<std::endl;
    return false;
  }
  NNetResult  pseudo_result;
  pseudo_result.box.left = 0;
  pseudo_result.box.right = 0;
  pseudo_result.box.top = 0;
  pseudo_result.box.bottom = 0;
  pseudo_result.box.name = " ";
  pseudo_result.box.prop = value;
  detect_result_v.push_back(pseudo_result);
  return true;
}

//baso clarity
bool DetectHuman::ForwardBasoClarity(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                        const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for baso clarity forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();

  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_BAS_CLARITY, img_brightness, result);

  if(ret!=0){
    std::cout<<"baso clarity forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  this->sample_contex.Baso += int(detect_result_v.size());
  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw baso clarity result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);
  return true;
}
//ret
bool DetectHuman::ForwardRet(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_fluorescence){
    std::cout<<"Emtpy brightness image are given for ret forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();

  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_RET, img_fluorescence, result);

  if(ret!=0){
    std::cout<<"bas forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  this->sample_contex.Ret += int(detect_result_v.size());
  cv::Mat img_out;
  if(!DrawMidResult(img_fluorescence, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw baso result."<<std::endl;
    return false;
  }
  mat_fluo_result_v.emplace_back(img_out);
  return true;
}





//将rect装入box
void TransformPolyToRectPoints(const std::list<NNetResult_t> &input_v, std::vector<std::vector<cv::Point>>& rect_points_v_v){
  //[category, conf, x1, y1, x2, y2,...,x4, y4]
  for(const auto& input:input_v){
    std::vector<float> polygon_v(input.polygon_v);

    std::vector<float> horizontal_points{polygon_v[0], polygon_v[2], polygon_v[4], polygon_v[6]};
    std::vector<float> vertical_points{polygon_v[1], polygon_v[3], polygon_v[5], polygon_v[7]};
    float left = *std::min_element(horizontal_points.begin(), horizontal_points.end());
    float top = *std::min_element(vertical_points.begin(), vertical_points.end());
    float right = *std::max_element(horizontal_points.begin(), horizontal_points.end());
    float bottom = *std::max_element(vertical_points.begin(), vertical_points.end());
    rect_points_v_v.push_back({cv::Point(left, top), cv::Point(right, top), cv::Point(right, bottom), cv::Point(left, bottom)});

  }


}

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
// rbc volume
bool DetectHuman::ForwardRbcVolume(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                              const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){

  if(!img_brightness){
    std::cout<<"Emtpy brightness  image are given for rbc volume forward."<<std::endl;
    return false;
  }

  cv::Mat target_img;
  ResizeImg(*img_brightness, target_img, cv::Size(1920, 1920), BOTTOMPAD);

  //推理获取白细胞
  detect_result_v.clear();
  std::list<NNetResult_t> result;


  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_RBC_INCLINE_DET, &target_img, result);

  if(ret!=0){
    std::cout<<"rbc volume forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  //擦除背景
  std::vector<std::vector<cv::Point>> rect_points_v_v;
  TransformPolyToRectPoints(result, rect_points_v_v);

  auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

//  cv::Mat img_mask( target_img.rows, target_img.cols,CV_8UC3);
//  for (const auto& one_poly:rect_points_v_v){
//    cv::fillPoly(img_mask, std::vector<std::vector<cv::Point>>{one_poly}, cv::Scalar(1,1,1));
//  }


  std::vector<cv::Mat> processed_img_v;
  std::vector<std::vector<cv::Rect>> paste_position_v_v;
  int crop_nums = 0;

  RbcInclineSegPreprocess(target_img, result, 1024, 1024, 15, processed_img_v,
                          crop_nums, paste_position_v_v);

  auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto cost_time = end-start;
  std::cout<<"preprocess time "<<cost_time<<std::endl;


//  mat_bright_result_v.push_back(img_mask.clone()*255);
//  cv::multiply(img_mask, target_img, img_mask);
  int incline_cell_nums_cur = crop_nums;
  this->sample_contex.incline_cell_nums += incline_cell_nums_cur;
  for(auto processed_img: processed_img_v){
    mat_bright_result_v.push_back(processed_img);
    result.clear();
    ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_RBC_INCLINE_SEG, &processed_img, result);
    if(ret!=0){
      std::cout<<"incline seg error."<<std::endl;
      return false;
    }
    //获取结果

    cv::Mat pred_merge;
    std::vector<cv::Mat> channels;
    cv::Mat pred_mask(result.begin()->seg_v[0]);

    int cell_region = cv::sum(pred_mask)[0];
    this->sample_contex.incline_pixels += cell_region;

    std::cout<<"volume cell "<<this->sample_contex.incline_cell_nums<<"  "<<this->sample_contex.incline_pixels<<std::endl;
    // 绘制结果
    cv::Mat img_b;
    cv::Mat img_g;
    cv::Mat img_r;
    split(processed_img, channels);//分离色彩通道
    img_b = channels.at(0);
    img_g = channels.at(1);
    img_r = channels.at(2);

    img_b = img_b/2 + pred_mask*255/2;
    cv::Mat temp_img;
    cv::merge(std::vector<cv::Mat>{img_b, img_g, img_r}, temp_img);

    mat_bright_result_v.push_back(temp_img);

  }


  return true;
}


// ai clarity
bool DetectHuman::ForwardAiClarity(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                             const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for ai clarity forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();
  auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_AI_CLARITY, img_brightness, result);

  if(ret!=0){
    std::cout<<"ai clarity forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());


  auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto cost_time = end-start;
  std::cout<<"process time "<<cost_time<<std::endl;
  cv::Mat img_out;
  std::cout<<"961"<<std::endl;
  return true;
}





//PLT volume
bool DetectHuman::ForwardPltVolume(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                             const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for plt volume forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();

  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_PLT_VOLUME, img_brightness, result);

  if(ret!=0){
    std::cout<<"bas forward error."<<std::endl;
    return false;
  }



  detect_result_v.assign(result.begin(), result.end());
  CountVolumeParameter(this->sample_contex.plt_volume_v, result, img_brightness->cols,img_brightness->rows);

  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw baso result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);


  return true;
}


// ai clarity far near
bool DetectHuman::ForwardAiClarityFarNear(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                   const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for ai clarity far near forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();
  auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  std::list<NNetResult_t> result;
//  cv::Mat img_brightness_bgr;
//  cv::cvtColor(*img_brightness, img_brightness_bgr, cv::COLOR_RGB2BGR);
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_AI_CLARITY_FAR_NEAR, img_brightness, result);

  if(ret!=0){
    std::cout<<"ai clarity far near forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());


  auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto cost_time = end-start;
  std::cout<<"process time "<<cost_time<<std::endl;

  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw baso result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);

  return true;
}



//Germ
bool DetectHuman::ForwardMilkGerm(std::vector<NNetResult>& detect_result_v, cv::Mat* img_brightness, cv::Mat* img_fluorescence, const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v) {
  if(!img_fluorescence){
    std::cout<<"Emtpy fluorescence image are given for milk germ forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();

  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_MILK_GERM, img_fluorescence, result);

  if(ret!=0){
    std::cout<<"milk germ forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  cv::Mat img_out;
  if(!DrawMidResult(img_fluorescence, img_height, img_width, detect_result_v, img_out, true, 0.3, 1)){
    std::cout<<"Fail to draw milk germ result."<<std::endl;
    return false;
  }
  mat_fluo_result_v.emplace_back(img_out);


  return true;
}



//Germ
bool DetectHuman::ForwardMilkCell(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                              const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_fluorescence){
    std::cout<<"Emtpy fluorescence image are given for milk cell forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();

  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_MILK_CELL, img_fluorescence, result);

  if(ret!=0){
    std::cout<<"milk cell forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  cv::Mat img_out;
  if(!DrawMidResult(img_fluorescence, img_height, img_width, detect_result_v, img_out, true, 0.5, 1)){
    std::cout<<"Fail to draw milk cell result."<<std::endl;
    return false;
  }
  mat_fluo_result_v.emplace_back(img_out);


  return true;
}


//RBC volume spherical box
bool DetectHuman::ForwardRbcVolumeSphericalBox(std::vector<NNetResult>& detect_result_v, cv::Mat* img_brightness, cv::Mat* img_fluorescence, const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v) {
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for rbc volume spherical forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();

  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_RBC_VOLUME_SHPERICAL, img_brightness, result);

  if(ret!=0){
    std::cout<<"rbc volume spherical error."<<std::endl;
    return false;
  }


  detect_result_v.assign(result.begin(), result.end());
  CountVolumeParameter(this->sample_contex.rbc_volume_v, result, img_brightness->cols,img_brightness->rows);

  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to rbc volume spherical result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);


  return true;
}

//RBC volume spherical seg
bool DetectHuman::ForwardRbcVolumeSphericalSeg(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                  const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v) {
  if (!img_brightness) {
    std::cout << "Emtpy brightness  image are given for rbc volume forward."
              << std::endl;
    return false;
  }

  cv::Mat target_img(*img_brightness);
  detect_result_v.clear();
  std::list<NNetResult_t> result;
  int ret =
      NNet_Inference(this->net_ctx, this->group_id,
                     NNET_MODID_RBC_VOLUME_SHPERICAL, &target_img, result);
  if (ret != 0) {
    std::cout << "rbc volume spherical seg forward error." << std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());
  std::vector<NNetResult> threshed_box_v;
  //筛选位置合格的框
  ThrBosAccordPosition(result, img_brightness->cols,
                       img_brightness->rows, 0.03,
                       threshed_box_v);


  auto start =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();
  std::vector<cv::Mat> processed_img_v; //粘贴图像
  std::vector<std::vector<cv::Rect>> paste_position_v_v; //粘贴位置
  int crop_nums = 0;

  // 将框对应的图像粘贴至空白图
  RbcInclineSegPreprocess(target_img, std::list<NNetResult>(threshed_box_v.begin(), threshed_box_v.end()), 1024, 1024, 15, processed_img_v,
                          crop_nums, paste_position_v_v);

  auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                 .count();
  auto cost_time = end - start;
  std::cout << "preprocess time " << cost_time << std::endl;

  //处理每一张图像
  for (int i =0; i<processed_img_v.size(); ++i) {

    start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();


    cv::Mat processed_img(processed_img_v[i]);
    mat_bright_result_v.push_back(processed_img);

    result.clear();
    ret = NNet_Inference(this->net_ctx, this->group_id,
                         NNET_MODID_RBC_INCLINE_SEG, &processed_img, result);
    if (ret != 0) {
      std::cout << "incline seg error." << std::endl;
      return false;
    }
    // 获取结果
    cv::Mat pred_merge;
    std::vector<cv::Mat> channels;
    cv::Mat pred_mask(result.begin()->seg_v[0]);
    for(const auto& rect_box:paste_position_v_v[i]){
      cv::Mat rect_box_img(processed_img, rect_box);
      this->sample_contex.rbc_volume_v.push_back(cv::sum(rect_box_img)[0]);
    }
    end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    cost_time = end - start;
    std::cout << "each volume seg with statistic time " << cost_time << std::endl;


    // 绘制结果
    cv::Mat img_b;
    cv::Mat img_g;
    cv::Mat img_r;
    split(processed_img, channels);  // 分离色彩通道
    img_b = channels.at(0);
    img_g = channels.at(1);
    img_r = channels.at(2);

    img_b = img_b / 2 + pred_mask * 255 / 2;
    cv::Mat temp_img;
    cv::merge(std::vector<cv::Mat>{img_b, img_g, img_r}, temp_img);

    mat_bright_result_v.push_back(temp_img);
  }
  return true;
}


//RBC volume spherical seg
bool DetectHuman::ForwardSphericalFocal(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                               const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v) {
  std::cout<<"Start forward spherical focal "<<std::endl;
  if (!img_brightness) {
    std::cout << "Emtpy brightness  image are given for rbc volume forward."
              << std::endl;
    return false;
  }

  cv::Mat target_img(*img_brightness);
  detect_result_v.clear();
  std::list<NNetResult_t> result;
  int ret =
      NNet_Inference(this->net_ctx, this->group_id,
                     NNET_MODID_SPHERICAL_FOCAL, &target_img, result);
  if (ret != 0) {
    std::cout << "rbc volume spherical seg forward error." << std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());

  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to rbc volume spherical result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);
  return true;
}


// classification custom
bool DetectHuman::ForwardClassificationCustom(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                          const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for classification custom forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();
  auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  std::list<NNetResult_t> result;
  //  cv::Mat img_brightness_bgr;
  //  cv::cvtColor(*img_brightness, img_brightness_bgr, cv::COLOR_RGB2BGR);
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_CLASSIFICATION_CUSTOM, img_brightness, result);

  if(ret!=0){
    std::cout<<"Classification custom forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());


  auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto cost_time = end-start;
  std::cout<<"process time "<<cost_time<<std::endl;

  cv::Mat img_out;
  if(!DrawMidResult(img_brightness, img_height, img_width, detect_result_v, img_out, true, 2, 2)){
    std::cout<<"Fail to draw classification custom result."<<std::endl;
    return false;
  }
  mat_bright_result_v.emplace_back(img_out);

  return true;
}

//CLARITY_MILK_BOARDLINE 底板划线

bool DetectHuman::ForwardMILKBOARDLINE(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                             const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v){
  if(!img_brightness){
    std::cout<<"Emtpy brightness image are given for ai clarity forward."<<std::endl;
    return false;
  }
  detect_result_v.clear();
  auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  std::list<NNetResult_t> result;
  int ret = NNet_Inference(this->net_ctx, this->group_id, NNET_MODID_AI_CLARITY_MILK_BOARDLINE, img_brightness, result);

  if(ret!=0){
    std::cout<<"mlik boardline forward error."<<std::endl;
    return false;
  }
  detect_result_v.assign(result.begin(), result.end());


  auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto cost_time = end-start;
  std::cout<<"process time "<<cost_time<<std::endl;
  cv::Mat img_out;
  std::cout<<"789"<<std::endl;
  std::cout << "分类结果: " << std::endl;
    // 定义类别名称映射表
    const std::vector<std::string> class_names = {"F", "CPF", "CNF", "FPF", "FNF", "O"};

    for (size_t i = 0; i < detect_result_v.size(); ++i) {
        const auto& result = detect_result_v[i];
        std::cout << "第 " << i + 1 << " 个检测结果: " << std::endl;

        // 查找最大置信度的类别
        float max_confidence = -1;
        size_t max_index = 0;

        std::cout << "类别概率: ";
        for (size_t j = 0; j < result.category_v.size(); ++j) {
            const float confidence = result.category_v[j];

            // 输出当前类别信息
            std::cout << class_names[j] << ": " << confidence << " ";

            // 更新最大置信度信息
            if (confidence > max_confidence) {
                max_confidence = confidence;
                max_index = j;
            }
        }

        // 输出最终分类结果
        std::cout << std::endl
                  << "最终分类: " << class_names[max_index]
                  << " (置信度: " << max_confidence << ")"
                  << std::endl;
    }
//    for (size_t i = 0; i < detect_result_v.size(); ++i) {
//    const auto& result = detect_result_v[i];
//
//    std::cout << "第 " << i + 1 << " 个检测结果: " << std::endl;
//    std::cout << "类别概率: ";
//    for (size_t j = 0; j < result.category_v.size(); ++j) {
////        std::cout << "类别 " << j << " 置信度: " << result.category_v[j] << " ";
//        std::cout << j << " 置信度: " << result.category_v[j] << " ";
//    }
//    std::cout << std::endl;
//    }
  return true;
}

// 推理入口
bool DetectHuman::Forward(ForwardParam& forward_param){

  std::vector<NNetResult>& detect_results = forward_param.detect_result_v;
  cv::Mat * img_brightness = forward_param.img_brightness;
  cv::Mat * img_fluorescence = forward_param.img_fluorescence;

  if(forward_param.alg_type==AlgType::RBC&&this->net_ctx){
    if(!this->ForwardRbc(detect_results,img_brightness,img_fluorescence, forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;

  }else if(forward_param.alg_type==AlgType::WBC&& this->net_ctx){
    if(!this->ForwardWbc(detect_results,img_brightness,img_fluorescence, forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  }else if(forward_param.alg_type==AlgType::WBC4&&this->net_ctx){
    if(!this->ForwardWbc4(detect_results,img_brightness,img_fluorescence, forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;

  }
  else if (forward_param.alg_type == AlgType::PLA && this->net_ctx) {
    //疟原虫推理
    if (!this->ForwardPLA(detect_results, img_brightness, img_fluorescence,
                          forward_param.img_height, forward_param.img_width,
                          forward_param.mat_bright_result_v,
                          forward_param.mat_fluo_result_v)) {
      return false;
    }
    forward_param.processed = true;
  }

  else if (forward_param.alg_type == AlgType::PLT && this->net_ctx) {
    if(!this->ForwardPlt(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::BASO) {
    if(!this->ForwardBaso(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::BASCLARITY && this->net_ctx) {
    if(!this->ForwardBasoClarity(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::GRADCLARITY) {
    if(!this->ForwardGradClarity(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::RET) {
    if(!this->ForwardRet(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::WBC4_SINGLE) {
    if(!this->ForwardWbc4Single(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::RBC_VOLUME) {
    if(!this->ForwardRbcVolume(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::AI_CLARITY) {
    if(!this->ForwardAiClarity(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::PLT_VOLUME) {
    if(!this->ForwardPltVolume(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::AI_CLARITY_FAR_NEAR) {
    if(!this->ForwardAiClarityFarNear(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::MILK_GERM) {
    if(!this->ForwardMilkGerm(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::MILK_CELL) {
    if(!this->ForwardMilkCell(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::RBC_VOL_SPH_BOX) {
    if(!this->ForwardRbcVolumeSphericalBox(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::RBC_VOL_SPH_SEG) {
    if(!this->ForwardRbcVolumeSphericalSeg(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::SPHERICAL_FOCAL) {
    if(!this->ForwardSphericalFocal(detect_results,img_brightness,img_fluorescence,forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::WBC_SINGLE) {
    if(!this->ForwardWbcSingle(detect_results,img_brightness,img_fluorescence, forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::CLASSIFICATION_CUSTOM) {
    if(!this->ForwardClassificationCustom(detect_results,img_brightness,img_fluorescence, forward_param.img_height, forward_param.img_width, forward_param.mat_bright_result_v, forward_param.mat_fluo_result_v)){
      return false;
    }
    forward_param.processed = true;
  } else if (forward_param.alg_type == AlgType::CLARITY_MLIK_BOARDLINE) {
    if (!this->ForwardMILKBOARDLINE(detect_results, img_brightness,
                                    img_fluorescence, forward_param.img_height,
                                    forward_param.img_width,
                                    forward_param.mat_bright_result_v,
                                    forward_param.mat_fluo_result_v)) {
      return false;
    }
    forward_param.processed = true;
  }
  return true;
}

bool DetectHuman::CalculateMcv(float& mcv){
  std::vector<float> rbc_volume_v;


  int ret =spherical_reagent_fitting.SphericalMcvFitting(this->sample_contex.rbc_volume_v, rbc_volume_v, mcv);
  if(ret){
    return false;
  }
  ALGLogInfo<<"Spherical mcv "<< mcv;
  return true;
}

inline void CalculateMeanValue(const std::vector<float>& data, float& result){
  double sum = std::accumulate(data.begin(), data.end(), 0.0);
  double mean =  sum / data.size(); //均值
  result = (float)mean;
}


void DetectHuman::GetStatisticResult(){
  ALGLogInfo<<"----------rbc------------";
  float rbc_fusion_rate = 1.f;
  float mcv=0.f;
//  GetRbcVolume(this->sample_contex.rbc_volume_v, rbc_fusion_rate, this->sample_contex.incline_cell_nums,
//               this->sample_contex.incline_pixels, VOLUME_TYPE_INCLINE, mcv);
  ALGLogInfo<<"rbc incline mcv" << mcv;
  ALGLogInfo<<"incline cells "<<this->sample_contex.incline_cell_nums;
  ALGLogInfo<<"incline region size "<< this->sample_contex.incline_pixels/ (this->sample_contex.incline_cell_nums+1e-5);

  ALG_DEPLOY::CALIBRATION::Calibration<float> calib;
  calib.SetPhysicalSizeCalibration(272,3036,4024);


  //对细胞区域面积相关参数进行校准
  calib.GetAreaCalibrationResult(this->sample_contex.rbc_volume_v, this->sample_contex.rbc_volume_v);

  if(!CalculateMcv(mcv)){
    ALGLogError<<"Failed to calculate mcv";
    return ;
  }


  float rbc_mean_area = 0.f;
  CalculateMeanValue(this->sample_contex.rbc_volume_v, rbc_mean_area);
  ALGLogInfo<<"rbc bottom region size "<<rbc_mean_area;



  ALGLogInfo<<"----------plt------------";
  float mpv = 0.f;
  std::vector<float> plt_all_values;
  GetPltVolume(this->sample_contex.plt_volume_v, rbc_fusion_rate, mpv, plt_all_values);
  ALGLogInfo<<"plt mpv" << mpv;

  double sum = std::accumulate(std::begin(this->sample_contex.plt_volume_v), std::end(this->sample_contex.plt_volume_v), 0.0);
  double mean =  sum / (int)this->sample_contex.plt_volume_v.size(); //均值
  float area_mean = (float)mean;
  ALGLogInfo<<"plt bottom region" << area_mean;
}

bool DetectHuman::RunAssistFunction() {
  if(this->test_get_wbc_result){
//    this->TestGetWbcResult();
  }
  if(this->test_get_rbc_and_plt_result){
//    this->TestGetRbcAndPltResult();
  }
  return true;
}

void DetectHuman::MapGroupId(){
  this->group_idx_to_id_m[MILK_TYPE]=NNET_GROUP_MILK;
  this->group_idx_to_id_m[HUMAN_TYPE]=NNET_GROUP_HUMAN;
  this->group_idx_to_id_m[CAT_TYPE]=NNET_GROUP_CAT;
  this->group_idx_to_id_m[DOG_TYPE]=NNET_GROUP_DOG;
  this->group_idx_to_id_m[BASCLARITY_TYPE]=NNET_GROUP_CLARITY_AI;
}

void DetectHuman::MapModId(){
  this->alg_id_to_mod_id_m[RBC] = NNET_MODID_RBC;
  this->alg_id_to_mod_id_m[WBC] = NNET_MODID_WBC;
  this->alg_id_to_mod_id_m[WBC4] = NNET_MODID_WBC4;
  this->alg_id_to_mod_id_m[PLT] = NNET_MODID_PLT;
  this->alg_id_to_mod_id_m[BASO] = NNET_MODID_BASO;
  this->alg_id_to_mod_id_m[RET] = NNET_MODID_RET;
  this->alg_id_to_mod_id_m[BASCLARITY] = NNET_MODID_BAS_CLARITY;
  this->alg_id_to_mod_id_m[CLARITY_MLIK_BOARDLINE] = NNET_MODID_AI_CLARITY_MILK_BOARDLINE;

}

DetectHuman::~DetectHuman() {
  NNet_DeInit(this->net_ctx);

}
}