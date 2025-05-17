//
// Created by y on 2023/10/26.
//
#include <opencv2/opencv.hpp>
#include <functional>
#include <dirent.h>
#include <string>
#include <stdlib.h>

#include "IntegratedCode.h"
#include "replace_std_string.h"
#include "libalgcell.h"
#include "libalgimm.h"
#include "utils.h"

namespace ALG_LOCAL{
namespace INTEGRATE {
bool IntegratedCode::InitImmune(const XML::IntDetectTypeInitConfig& int_detect_type_init_config, const XML::IntTestDataDir& int_test_data_dir){

  for(const auto& sample_config: int_detect_type_init_config.sample_config_v){
    if(sample_config.sample_name=="SampleImmune"){
      this->immunectx = AlgImm_Init();
      int ret = AlgImm_RunConfigLoad(this->immunectx, "./data");
      if(ret!=0){
        std::cout<<"Failed to config Immune, error code "<<ret<<std::endl;
        return false;
      }
      this->card_info_dir = int_test_data_dir.card_info_dir;
      this->data_info_dir = int_test_data_dir.data_info_dir;
      if(sample_config.float_param_v.size()!=2){
        std::cout<<"Two prams are expected for immune init, but "+std::to_string(sample_config.float_param_v.size())+"are given"<<std::endl;
        return false;
      }
      //第一个参数
      float accepted_float_0 = sample_config.float_param_v[0];
      if(accepted_float_0==0.f) {
        this->calibration = 0;
      } else{
        this->calibration = 1;
      }
      //第二个参数
      this->coef = sample_config.float_param_v[1];
      return true;
    }
  }

  std::cout<<"Configured init immune, but none immune params are configured"<<std::endl;
  return false;
}


//读取免疫数据
bool ReadImmuneData(const std::string& data_path,std::vector<float>& data){

  FILE *file;
  uint32_t v1;
  char line[256];
  int i  = 0;
  file = fopen(data_path.c_str(), "r");
  if (file == nullptr)
  {
    std::cout<<"File not exist."<<std::endl;
    return false;
  }

  char *end_ptr;
  while (fgets(line, sizeof(line), file))
  {
    //sscanf(line, "%d", &v1);
    v1 = strtol(line,&end_ptr, 10);
    data.emplace_back(v1);

  }
  fclose(file);
  return true;
}

//读取免疫试剂卡信息
bool ReadImmuneCard(const std::string& card_path, std::string& card){
  FILE *file;
  file = fopen(card_path.c_str(), "r");
  if (file == nullptr)
  {
    return false;
  }

  do
  {
    card.push_back(fgetc(file));

  } while (!feof(file));

  fclose(file);
  card.at(card.length() - 1) = '\0';
  return true;
}



void PrintImmune(const AlgImmRst_t&  immune_result){
  std::cout<<"Print immune result"<<std::endl;
  //免疫
  //通道结果
  float immune_return_coef = 0.f;//各个通道校准系数相同
  for(int i=0;i<immune_result.channel_rst.channel_cnt;++i){
    auto channel_rst = immune_result.channel_rst.single_channel_rst[i];
    char cdata[20];
    std::string scmode=channel_rst.mode;
    int k = 0;
    //mode 内含特殊字符,将引发上层报错,需要替换
    const char ad[2]="/";
    for (k = 0; k < scmode.length(); k++)
    {
      if(cdata[k]==ad[0])
        cdata[k]=ad[0];
      else
        cdata[k] =scmode[k] ;
    }
    cdata[k] = '\0';
    printf("mode=%s  ", cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(channel_rst.validflag));
    printf("validflag=%s  ",cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f", strtod(channel_rst.signal, nullptr));
    printf("signal=%s  ",cdata);

    printf("name=%s  ",channel_rst.channel_name);

//    memset(cdata, 0, sizeof(cdata) );
//    sprintf(cdata,"%.2f",float(channel_rst.concentration_value));
//    printf("concentration_value=%s  ",cdata);

    printf("concentration=%s  ",channel_rst.concentration);
    //替换result 为unit
    std::string unit{channel_rst.unit};
    ReplaceAllDistinct(unit, "/", "/");
    printf("unit=%s  \n",unit.c_str());

    immune_return_coef = float(channel_rst.coef);

  }
  //线结果
  for (unsigned int i=0; i<immune_result.line_rst.line_cnt;++i ){
    auto line_rst = immune_result.line_rst.single_line_rst[i];
    char cdata[20];
    sprintf(cdata,"%.2f",float(line_rst.line_id));
    printf("line_id=%s  ",cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.max_point));
    printf("max_point=%s  ",cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.max_value));
    printf("max_value=%s  ",cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.signal_start));
    printf("signal_start=%s  ",cdata);


    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.signal_end));
    printf("signal_end=%s  ",cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.area));
    printf("area=%s  ",cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.base_line));
    printf("base_line=%s  ",cdata);

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(immune_return_coef));
    printf("immune_return_coef=%s  \n",cdata);

  }
  //
  char cdata[20];
  printf("immune input data:\n");
  for (unsigned int i = 0; i < immune_result.line_rst.input_length; ++i) {
    memset(cdata, 0, sizeof(cdata));
    sprintf(cdata, "%.2f", immune_result.line_rst.input_data[i]);
    printf("%s  ",cdata);
  }
  printf("\n");
  //filter
  printf("immune output data:\n");
  for (unsigned int i = 0; i < immune_result.line_rst.length; ++i) {
    memset(cdata, 0, sizeof(cdata));
    sprintf(cdata, "%.2f", immune_result.line_rst.filter_data[i]);
    printf("%s  ",cdata);
  }
  printf("\n");
}

bool IntegratedCode::TestAlgSampleImmune(){
  //读取数据路径
  //  std::vector<cv::String> card_info_path_v, data_info_path_v;
  //  cv::glob(card_info_dir, card_info_path_v);
  //  cv::glob(data_info_dir, data_info_path_v);

  std::vector<std::string> card_info_path_v, data_info_path_v;
  LoadImagePath(card_info_dir, card_info_path_v);
  LoadImagePath(data_info_dir, data_info_path_v);


  if(card_info_path_v.size()!=data_info_path_v.size()){
    std::cout<<"Immune Error, cards nums are not same as datas."<<std::endl;
  }

  std::cout<<"card size "<<card_info_path_v.size()<<std::endl;
  for(unsigned int i=0;i<card_info_path_v.size();++i){
    std::cout<<card_info_path_v[i]<<std::endl;
    std::cout<<data_info_path_v[i]<<std::endl;
    std::string card_info;
    if(!ReadImmuneCard(card_info_path_v[i], card_info)){
      std::cout<<"Immune Error, fail to read card."<<std::endl;
      return false;
    }
    std::vector<float> data;
    if(!ReadImmuneData(data_info_path_v[i], data)){
      std::cout<<"Immune Error, fail to read data."<<std::endl;
      return false;
    }

    char encodetext[MAX_ENCODE_CARD_INFO] = "";
    memset(encodetext, 0x00, MAX_ENCODE_CARD_INFO);
    strcpy(encodetext, card_info.c_str());

    char decoded_card_info_char[MAX_DECODE_CARD_INFO];
    memset(&decoded_card_info_char, 0x00, MAX_DECODE_CARD_INFO);

    //DI50 AI50
    int err = 0;
    int ret = GetCardInfo(1, 0x00000001, encodetext, sizeof(encodetext), &err, decoded_card_info_char);

    if (ret)  // 0:DI-200 1:DI-50 2:DIH-500
    {
      std::cout<<"Failed to get card content"<<std::endl;
      return false;  //
    }

    //免疫检测计算
    ret =AlgImm_Open(this->immunectx, this->calibration, std::string(decoded_card_info_char), this->coef);
    if(ret!=0){
      std::cout<<"Failed to open immune"<<std::endl;
    }

    ret = AlgImm_PushData(this->immunectx, data, 0, 0);
    if(ret!=0){
      std::cout<<"Immune Error, fail to run immune det."<<std::endl;
      return false;
    }

    //获取结果
    AlgImmRst_t immune_result;
    ret = AlgImm_GetResult(this->immunectx, immune_result);
    if(ret!=0){
      std::cout<<"Failed to get immune result, error code "<<ret<<std::endl;
    }
    ret = AlgImm_Close(this->immunectx);
    if(ret!=0){
      std::cout<<"Failed to close immune ctx, error code "<<ret<<std::endl;
    }
    //打印结果
    PrintImmune(immune_result);


  }
  return true;
}

}
}