//
// Created by y on 23-8-11.
//
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <functional>
#include <dirent.h>
#include <sys/stat.h>


#include "IntegratedCode.h"
#include "replace_std_string.h"
#include "libalgcell.h"
#include "neural_network.h"
#include "utils.h"
namespace ALG_LOCAL{
namespace INTEGRATE {
std::string int_specific_save_dir = " ";
void AlgCellImageCallback_f (AlgCtxID_t ctx_id, uint32_t group_idx, uint32_t chl_idx, uint32_t view_order, uint32_t view_idx, uint32_t processed_idx, AlgCellStage_e stage, AlgCellImg_t *img, void *userdata,
                            const int& view_pair_idx, std::map<std::string, float> call_back_params){
//  std::cout<<"Outer callback did nothing"<<std::endl;

  std::cout<<"call back with mission type "<<call_back_params[TASK_TYPE]<<std::endl;
  std::stringstream ss;
  ss<<'g';
  ss<<std::setw(1)<<std::setfill('0')<<group_idx;
  ss<<'c';
  ss<<std::setw(1)<<std::setfill('0')<<chl_idx;
  ss<<'n';
  ss<<std::setw(4)<<std::setfill('0')<<view_pair_idx;
  ss<<'v';
  ss<<std::setw(4)<<std::setfill('0')<<view_idx;
  ss<<'p';
  ss<<std::setw(2)<<std::setfill('0')<<processed_idx;
  ss<<'o';
  ss<<std::setw(4)<<std::setfill('0')<<view_order;
  std::string img_name{ss.str()};
  cv::Mat img_mat(int(img->height), int(img->width), CV_8UC3, img->data);
//  cv::cvtColor(img_mat, img_mat, cv::COLOR_RGB2BGR);
  SaveImage(int_specific_save_dir+img_name+".bmp", img_mat);

}
/*!
 * 将字符串中的指定字符串进行替换
 * @param str 需要进行处理的字符串
 * @param old_value 被替换字符串
 * @param new_value 替换为
 * @return
 */
void ReplaceAllDistinct(std::string& str, const std::string& old_value, const std::string& new_value){
  for(std::string::size_type   pos(0);   pos!=std::string::npos;   pos+=new_value.length()) {
    if(   (pos=str.find(old_value,pos))!=std::string::npos   )
      str.replace(pos,old_value.length(),new_value);
    else   break;
  }
}


//整体测试需要bgr格式的图
bool ReadImgToBuf(struct AlgCellImg *img_buf, const std::string& img_path, const bool& flip_img){
  cv::Mat img_mat;
  std::cout<<"Pushing image "+ img_path<<std::endl;
  img_mat = cv::imread(img_path);
  if(img_mat.empty()){
    std::cout<<"Error, empty image"<<std::endl;
    return false;
  }
  // 保存的图为上下翻转后的图,为模拟真实算法接受到的图,此处增加flip
  if(flip_img){
    cv::flip(img_mat,img_mat, 0);
  }

  int n_flag = img_mat.channels() * 8;//一个像素的bits
  int n_height = img_mat.rows;
  int n_width = img_mat.cols;
  int n_bytes = n_height * n_width * n_flag / 8;//图像总的字节

  img_buf->data = new unsigned char[n_bytes];
  memcpy(img_buf->data, img_mat.data, n_bytes);
  if(img_buf->data== nullptr){
    std::cout<<"Fail to transform mat to unsigned char*"<<std::endl;
  }
  img_buf->height = img_mat.rows;
  img_buf->width = img_mat.cols;
  return true;
}

bool IntegratedCode::InitMachine(const std::string& detect_type_name){
  this->algctx = AlgCell_Init(3);
  if(this->algctx== nullptr){
    std::cout<<"Fail to Init "<<detect_type_name<<std::endl;
    return false;
  }
  AlgCellModeID mode_id = this->machine_type_name_to_mode_m.at(detect_type_name);
  std::cout<<"Use mode_id "<<mode_id<<std::endl;
  std::vector<char> model_info_heamo;
  if (AlgCell_RunConfigLoad(this->algctx, mode_id, "./data", model_info_heamo) != 0) {
      std::cout << "Fail to Load work mode" << detect_type_name << std::endl;
      return false;
  }
  return true;
}

bool MakeSaveDirs(const std::string& save_dir, std::string& dst){
  DIR *mydir=opendir(save_dir.c_str()); //打开目录
  if(mydir== nullptr) {
    if (mkdir(save_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
      std::cout << "Failed to create int save dir" + save_dir << std::endl;
      return false;
    }
  }
  std::string save_dir_with_time(save_dir+"/"+ std::to_string(time(nullptr))+"/");
  if (mkdir(save_dir_with_time.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
    std::cout << "Failed to create int save time folder" << std::endl;
    return false;
  }
  dst = save_dir_with_time;
  return true;



}


bool IntegratedCode::Init(const std::vector<XML::IntDetectTypeInitConfig>& int_detect_config, const XML::IntTestDataDir& int_test_data_dir){

  this->MapAlgTypeName();
  this->MapSampleTypeName();
  this->MapMachineType();

  char        imm_alg_version[ALGIMM_LIB_VERSION_LENGTH];
  char        imm_qr_json_version[ALGIMM_LIB_VERSION_LENGTH];
  char        l_version[ALGIMM_LIB_VERSION_LENGTH];
  char        m_version[ALGIMM_LIB_VERSION_LENGTH];

  std::string alg_cell_version = AlgCell_Version();
  std::string alg_imm_version  = AlgImm_Version(imm_alg_version, imm_qr_json_version, l_version, m_version);
  std::cout<<"Alg cell version "<<alg_cell_version<<std::endl;
  std::cout<<"Alg imm version "<<alg_imm_version<<std::endl;

  for(const auto& iter: int_detect_config){
    if(iter.enable){
      // 设置open默认值
      this->debug = iter.debug;

      //初始化一种机型
      std::cout<<"Start init "+iter.detect_type_name<<std::endl;
      if(!InitMachine(iter.detect_type_name)){
        std::cout<<"Failed to init machine"<<std::endl;
        return false;
      }
      std::cout<<"Init machine succeed"<<std::endl;

      //血球
      if(!InitHeamo(iter, int_test_data_dir)){
        std::cout<<"Failed to init heamo"<<std::endl;
        return false;
      }
      std::cout<<"Init heamo succeed"<<std::endl;


      //清晰度
      if(!InitClarity(iter, int_test_data_dir)){
        std::cout<<"Failed to init clarity"<<std::endl;
        return false;
      }
      std::cout<<"Init clarity succeed"<<std::endl;

      //open 参数
      this->open_params={{OPEN_PARAM_DEBUG, std::vector<float> {static_cast<float>(this->debug)}},
                           {OPEN_PARAM_GROUP_IDX, std::vector<float>{static_cast<float>(this->group_idx)}},
                           {OPEN_PARAM_QC, std::vector<float>{static_cast<float>(this->qc)}},
                           {OPEN_PARAM_CALIB, std::vector<float>{static_cast<float>(this->calib)}},
                           {OPEN_PARAM_IMG_H, std::vector<float>{static_cast<float>(this->img_h)}},
                           {OPEN_PARAM_IMG_W, std::vector<float>{static_cast<float>(this->img_w)}},
                           {OPEN_PARAM_IMG_H_UM, std::vector<float>{static_cast<float>(this->img_h_um)}}};
      std::vector<float> alarm_parmas_v{18,2.5,18,1,4,1.5,1.5,0.3,2,
      64,22,70,110,6.5,90,29,
      600,60,
      2.5,0.5};
      std::vector<float> dilution_params_v;
      //不允许手动配置稀释倍数,麻烦.
      if(this->group_idx==AlgCellGroupID_e::ALG_CELL_GROUP_MILK){
        dilution_params_v = {2,2,8,8,0,0,0,0};
      } else{
        dilution_params_v = {50,50,10,50,50,50,10,50};
      }
      std::vector<float> task_att_v{1,1};

      this->open_params[OPEN_PARAM_ALARM] = alarm_parmas_v;
      this->open_params[OPEN_PARAM_DILUTION] = dilution_params_v;
      this->open_params[OPEN_PARAM_TASK_APPEND_ATT] = task_att_v;
      //免疫
      if(!InitImmune(iter, int_test_data_dir)){
        std::cout<<"Failed to init immune"<<std::endl;
        return false;
      }
      std::cout<<"Init immune succeed"<<std::endl;

      //保存当前机型下需要检测的检测算法;如人医下的计数、免疫算法
      for(const auto& sample_type:iter.sample_config_v){
        if(sample_type.enable){
          std::cout<<"Alg will run "+sample_type.sample_name<<std::endl;
          this->test_type.emplace_back(this->alg_type_name_to_type_m.at(sample_type.sample_name));
        }
      }

      //结果保存相关
      this->save_dir = int_test_data_dir.save_dir;
      if(this->debug){
        if(!MakeSaveDirs(this->save_dir, int_specific_save_dir)){
          return false;
        }
      }
      std::cout<<"Init AlgSampleInit of "+iter.detect_type_name+" succeed."<<std::endl;
      break ;
    }
  }
  return true;
}


bool IntegratedCode::TestSample(){
  for(const auto iter: this->test_type){
    switch (iter) {
      case  ALG_SAMPLE_FIELD:{
        //计算结果
        std::cout<<"Start SampleField."<<std::endl;
        if(!this->TestAlgSampleField()){
          return false;
        }
        break;
      }
      case ALG_SAMPLE_HGB:{
        if(!this->TestAlgSampleHgb()){
          return false;
        }
        break;
      }
      case ALG_SAMPLE_CLARITY:{
        std::cout<<"Start clarity"<<std::endl;
        if(!this->TestAlgSampleClarity()){
          return false;
        }
        break;
      }
      case ALG_SAMPLE_IMMUNE:{
        std::cout<<"Start SampleImmune."<<std::endl;
        if(!this->TestAlgSampleImmune()){
          return false;
        }
        break;
      }
      case ALG_SAMPLE_HYBRID:{
        std::cout<<"Start SampleHybrid."<<std::endl;
        if(!this->TestAlgSampleHybrid()){
          return false;
        }
        break;
      }
      default:{
        std::cout<<"Error, Configured wrong test type in IntegratedCode test."<<std::endl;
        break;
      }
    }
  }
  //释放所有image buf
  AlgCell_DeInit(this->algctx);
  std::cout<<"Succeed to release"<<std::endl;

  return true;
}
void IntegratedCode::MapAlgTypeName(){
  this->alg_type_name_to_type_m["SampleField"]=INTEGRATE::IntegratedTestType::ALG_SAMPLE_FIELD;
  this->alg_type_name_to_type_m["SampleClarity"]=INTEGRATE::IntegratedTestType::ALG_SAMPLE_CLARITY;
  this->alg_type_name_to_type_m["SampleHGB"]=INTEGRATE::IntegratedTestType::ALG_SAMPLE_HGB;
  this->alg_type_name_to_type_m["SampleImmune"]=INTEGRATE::IntegratedTestType::ALG_SAMPLE_IMMUNE;
  this->alg_type_name_to_type_m["SampleHybrid"]=INTEGRATE::IntegratedTestType::ALG_SAMPLE_HYBRID;

}

void IntegratedCode::MapSampleTypeName(){
/*  this->sample_type_name_to_type_m["Milk"]= NNetGroup::NNET_GROUP_MILK;
  this->sample_type_name_to_type_m["Human"]= NNetGroup::NNET_GROUP_HUMAN;
  this->sample_type_name_to_type_m["Cat"]= NNetGroup::NNET_GROUP_CAT;
  this->sample_type_name_to_type_m["Dog"]= NNetGroup::NNET_GROUP_DOG;*/

  this->sample_type_name_to_type_m["Milk"]= AlgCellGroupID::ALG_CELL_GROUP_MILK;
  this->sample_type_name_to_type_m["Human"]= AlgCellGroupID::ALG_CELL_GROUP_HUMAN;
  this->sample_type_name_to_type_m["Cat"]= AlgCellGroupID::ALG_CELL_GROUP_CAT;
  this->sample_type_name_to_type_m["Dog"]= AlgCellGroupID::ALG_CELL_GROUP_DOG;
}
//动物同属一种机型
void IntegratedCode::MapMachineType(){
  this->machine_type_name_to_mode_m["Human"]=AlgCellModeID::ALGCELL_MODE_HUMAN;
  this->machine_type_name_to_mode_m["Cat"]=AlgCellModeID::ALGCELL_MODE_ANIMAL;
  this->machine_type_name_to_mode_m["Dog"]=AlgCellModeID::ALGCELL_MODE_ANIMAL;
  this->machine_type_name_to_mode_m["Milk"]=AlgCellModeID::ALGCELL_MODE_MILK;
}


//释放内存
IntegratedCode::~IntegratedCode(){

}

}
}
