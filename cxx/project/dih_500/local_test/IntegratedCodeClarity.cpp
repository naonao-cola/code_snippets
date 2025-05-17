//
// Created by y on 2023/10/26.
//

#include <functional>
#include <dirent.h>
#include "IntegratedCode.h"
#include "replace_std_string.h"
#include "libalgcell.h"
#include "utils.h"
#define CLARITY_TRADITION_HIGHEST_PEAK           0
#define CLARITY_AI_COARSE                        1
#define CLARITY_AI_FINE_WBC_FAR_NEAR             2
#define CLARITY_AI_FINE_BASO_FAR_NEAR            3
#define CLARITY_AI_FINE_FLU_MICRO_FAR_NEAR       4
#define CLARITY_AI_COARSE_FLU_MICRO_FAR_NEAR     5
#define CLARITY_AI_CARSE_BASO_FAR_NEAR           6
#define CLARITY_AI_MILK_BOARDLINE                7
namespace ALG_LOCAL{
namespace INTEGRATE {


bool IntegratedCode::InitClarity(const XML::IntDetectTypeInitConfig& int_detect_type_init_config, const XML::IntTestDataDir& int_test_data_dir){

  //xml中最后一个通道为清晰度图像目录
  int int_channel_size = (int)int_test_data_dir.channel_img_dir.size();
  std::vector<std::string> channel_dir = int_test_data_dir.channel_img_dir[int_channel_size-1];
  if(channel_dir.size()!=1){
    std::cout<<"Dir for Clarity must be 1, but "<<channel_dir.size()<<" was given"<<std::endl;
    return false;
  }
  std::string clarity_dir = channel_dir[0];
  std::cout<<"Clarity use dir "<<clarity_dir<<std::endl;
  //读取目录
  DIR *dir = opendir(clarity_dir.c_str());
  if(dir == nullptr){
    std::cout<< "dir dose not exist. "<<clarity_dir << std::endl;
    return true;
  }
  struct dirent* entry;
  while(((entry = readdir(dir)) != nullptr)){
    if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0){
      continue;
    }
    std::cout<<"clarity img dir "<<entry->d_name<<std::endl;
    this->clarity_dir_v.emplace_back(clarity_dir + "/"+entry->d_name);
  }


  //获取清晰度检测类型-梯度最高峰,梯度第一峰,AI
  for(const auto& sample_config:int_detect_type_init_config.sample_config_v){
    if(sample_config.sample_name=="SampleClarity"){
      if(sample_config.float_param_v.size()!=1){
        std::cout<<"Clarity init need one float param, but "<< sample_config.float_param_v.size()<<" was configured"<<std::endl;
        return false;
      }
      float clarity_type_float = sample_config.float_param_v[0];

      if(clarity_type_float==float(CLARITY_TRADITION_HIGHEST_PEAK)||clarity_type_float==float(CLARITY_AI_FINE_WBC_FAR_NEAR)||
          clarity_type_float==float(CLARITY_AI_FINE_BASO_FAR_NEAR)||clarity_type_float==float(CLARITY_AI_COARSE)||
          clarity_type_float==float(CLARITY_AI_COARSE_FLU_MICRO_FAR_NEAR)||clarity_type_float==float(CLARITY_AI_FINE_FLU_MICRO_FAR_NEAR)||
          clarity_type_float == (CLARITY_AI_CARSE_BASO_FAR_NEAR)||clarity_type_float == CLARITY_AI_MILK_BOARDLINE){
        this->clarity_channel_id = int(clarity_type_float);
        std::cout<<"clarity channel id: "<<this->clarity_channel_id<<std::endl;
      } else{
        std::cout<<"Clarity init only accept params among {0.0, 1.0, 2.0, 3.0, 4.0, 5.0  6.0}, but "
                  <<clarity_type_float<<" was given"<<std::endl;
        return false;
      }

      return true;
    }
  }
  //未找到 SampleClarity配置项
  return false;
}


bool IntegratedCode::ClarityFindClearest(const std::vector<std::string>& img_path_v,
                                         std::vector<uint32_t>&index_v,
                                         std::vector<float>& clarity_v,
                                         const int& view_pair_idx){
  std::cout<<"run clearest version";
  std::map<std::string, float> complementary_params;
  complementary_params[VIEW_PAIR_IDX] = (float)view_pair_idx;
  int ret;
  for (int idx = 0; idx < img_path_v.size(); ++idx) {
    AlgCellImg img[MAX_IMG_NUMS_UNDER_PAIR];

    if (!ReadImgToBuf(&img[0], img_path_v[idx])) {
      std::cout << "Failed to read img" << std::endl;
      return false;
    }

    ret =
        AlgCell_ClarityPushImage(this->algctx, this->clarity_outer_group_idx,
                                 this->clarity_channel_id, img, 1, complementary_params);
    delete [] img->data;
    if (ret != 0) {
      std::cout << "Failed to push clarity images, error code " << ret
                << std::endl;
      return false;
    }
  }
  //获取结果
  uint32_t index = 0;
  float clarity = 0.f;
  ret =
      AlgCell_ClarityGetResultBest(this->algctx, &index, &clarity);
  if(ret!=0){
    std::cout << "Failed to get clarity result, error code "<< ret << std::endl;
    return false;
  }
  std::cout<<"Best clarity idx "<< index
            <<" value "<< clarity<<std::endl;
  index_v.emplace_back(index);
  clarity_v.emplace_back(clarity);

  return true;
}
bool IntegratedCode::ClarityCheckEach(const std::vector<std::string>& img_path_v,
                                      std::vector<uint32_t>&index_v,
                                      std::vector<float>& clarity_v, const int& view_pair_idx){
  std::cout<<"run farnear version";
  int ret;
  uint32_t index = 0;
  float clarity = 0.f;
  std::map<std::string, float> complementary_params;
  complementary_params[VIEW_PAIR_IDX] = (float)view_pair_idx;
  std::cout<<"VIEW PAIR IDX "<<complementary_params[VIEW_PAIR_IDX];
  for (int idx = 0; idx < img_path_v.size(); ++idx) {
    AlgCellImg img[MAX_IMG_NUMS_UNDER_PAIR];

    if (!ReadImgToBuf(&img[0], img_path_v[idx])) {
      std::cout << "Failed to read img" << std::endl;
      return false;
    }
    ret =
        AlgCell_ClarityPushImage(this->algctx, this->clarity_outer_group_idx,
                                 this->clarity_channel_id, img, 1, complementary_params);
    delete [] img->data;
    if (ret != 0) {
      std::cout << "Failed to push clarity images, error code " << ret
                << std::endl;
      return false;
    }
      std::cout << "Current clarity_channel_id: " << this->clarity_channel_id << std::endl;
    //获取结果
    if(this->clarity_channel_id==CLARITY_AI_COARSE){
      ret =
          AlgCell_ClarityGetResultCoarse(this->algctx, &index, &clarity);
    } else if(this->clarity_channel_id ==CLARITY_AI_MILK_BOARDLINE){
      ret =
              AlgCell_ClarityGetResultMilkBoardLine(this->algctx, &index, &clarity);
    } else{
      ret =
          AlgCell_ClarityGetResultFarNear(this->algctx, &index, &clarity);
    }


    if(ret!=0){
      std::cout << "Failed to get clarity result, error code "<< ret << std::endl;
      return false;
    }
    index_v.emplace_back(index);
    clarity_v.emplace_back(clarity);
  }
  return true;
}


bool IntegratedCode::TestAlgSampleClarity(int times){
  //循环测试
  for(int temp_iter=0; temp_iter<times; ++temp_iter){
    std::cout<<"------------------------------------"<<temp_iter<<std::endl;
    std::vector<std::vector<uint32_t>> index_v_v;
    std::vector<std::vector<float>> clarity_v_v;
    std::vector<std::vector<std::string>> img_path_v_v;
    for(int i =0; i<this->clarity_dir_v.size(); ++i) {
      std::vector<uint32_t> index_v;
      std::vector<float> clarity_v;
      //装载数据
      int ret=
          AlgCell_ClarityOpen(this->algctx, (1 << 0), AlgCellImageCallback_f, nullptr, this->open_params);
      if(ret!=0){
        std::cout << "Failed to open clarity, error code "<< ret << std::endl;
        return false;
      }
      std::string img_dir(this->clarity_dir_v[i]);
      //一组图像
      std::vector<std::string> img_path_v;
      std::cout<<"The group used img dir "<<img_dir<<std::endl;
      LoadImagePath(img_dir, img_path_v);
      img_path_v_v.push_back(img_path_v);
      bool local_ret;
      // 根据配置的清晰度id调用相应清晰度算法
      if(this->clarity_channel_id==CLARITY_AI_FINE_WBC_FAR_NEAR||this->clarity_channel_id==CLARITY_AI_FINE_BASO_FAR_NEAR||
          this->clarity_channel_id==CLARITY_AI_COARSE||CLARITY_AI_COARSE_FLU_MICRO_FAR_NEAR||CLARITY_AI_FINE_FLU_MICRO_FAR_NEAR||
          this->clarity_channel_id == CLARITY_AI_CARSE_BASO_FAR_NEAR||this->clarity_channel_id == CLARITY_AI_MILK_BOARDLINE){
        local_ret = this->ClarityCheckEach(img_path_v,index_v,clarity_v, i);
      }else{
        local_ret = this->ClarityFindClearest(img_path_v,index_v,clarity_v, i);
      }
      if(!local_ret){
        std::cout << "Failed to find clarity with type "<< this->clarity_channel_id << std::endl;
        return false;
      }
      index_v_v.push_back(index_v);
      clarity_v_v.push_back(clarity_v);

      //关闭ctx
      AlgCell_ClarityClose(this->algctx);

    }

    //所有结果打印
    std::cout<<"Clarity result"<<std::endl;
    for(int file_idx=0; file_idx<img_path_v_v.size(); ++file_idx){
      if(this->clarity_channel_id==CLARITY_AI_FINE_WBC_FAR_NEAR||this->clarity_channel_id==CLARITY_AI_FINE_BASO_FAR_NEAR||
          this->clarity_channel_id==CLARITY_AI_COARSE||CLARITY_AI_COARSE_FLU_MICRO_FAR_NEAR||CLARITY_AI_FINE_FLU_MICRO_FAR_NEAR||
          this->clarity_channel_id == CLARITY_AI_CARSE_BASO_FAR_NEAR|| this->clarity_channel_id == CLARITY_AI_MILK_BOARDLINE){
        for(int img_idx=0;img_idx<img_path_v_v[file_idx].size();++img_idx){
          std::cout<<"img "<<img_path_v_v[file_idx][img_idx]
                    <<" clarity "<<clarity_v_v[file_idx][img_idx]<<std::endl;
        }
      }else{
        for(int img_idx=0;img_idx<index_v_v[0].size();++img_idx){
          std::cout<<"file "<<this->clarity_dir_v[file_idx]
                    <<" idx" << index_v_v[file_idx][img_idx]
                    <<" clarity "<<clarity_v_v[file_idx][img_idx]<<std::endl;
        }
      }
    }

  }
  std::cout<<"temp_iter over"<<std::endl;
  return true;
}

}
}