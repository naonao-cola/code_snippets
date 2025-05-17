//
// Created by y on 2023/10/26.
//
#include <functional>
#include <dirent.h>
#include <random>

#include "IntegratedCode.h"
#include "replace_std_string.h"
#include "libalgcell.h"
#include "utils.h"
#include "alg_heamo.h"

namespace ALG_LOCAL{
namespace INTEGRATE {

/*!
 * 将内部枚举型group idx 转换为序号
 * @param algctx ctx
 * @param inner_group_idx
 * @param group_idx
 * @param func 转换函数
 * @return
 */
bool ReverseGroupToIdx(AlgCtxID_t algctx, const int& inner_group_idx, int& outer_group_idx, const std::function<int(AlgCtxID_t, std::vector<uint32_t> &)>& func_list_group){
  std::vector<uint32_t> list;
  int ret = func_list_group(algctx, list);
  if(ret!=0){
    std::cout<<"failed to get group list, error code "<<ret<<std::endl;
    return false;
  }
  std::cout<<"inner_group_idx "<<inner_group_idx
            <<" list size "<<list.size()
            <<std::endl;
  for(auto iter:list){
    std::cout<<"group idx in list :"<<iter<<std::endl;
  }
  for(int i=0;i< list.size(); ++i){
    std::cout<<"reversed "<<list[i]<<std::endl;
    if(list[i]==inner_group_idx){
      outer_group_idx = i;
      std::cout<<"outer_group_idx "<<outer_group_idx<<std::endl;
      return true;
    }
  }
  return true;
}

/*!
 * 根据传入的通道图像目录总表及算法通道信息确认真实需要处理的图像目录
 * @param algctx ctx
 * @param outer_group_idx 外部group idx
 * @param channel_img_dir 多个通道对应的图像路径
 * @param[out] identified_channel_config 根据ctx算法启用的通道,及channel_img_dir确认真实需要处理的图像目录
 * @return
 */
//传入的通道目录配置
bool IdentifyChannels(AlgCtxID_t algctx, int outer_group_idx, const std::vector<std::vector<std::string>>& channel_img_dir,
                      std::vector<std::vector<std::string>>& identified_channel_config){

  std::vector<uint32_t> channel_list;
  int ret = 0;
  ret = AlgCell_HeamoListChl(algctx, channel_list, outer_group_idx);
  std::cout<<"channel_list size "<<channel_list.size()<<std::endl;

  if(ret!=0){
    std::cout<<"failed to get channel list, error code "<<ret<<" outer group idx "<< outer_group_idx<<std::endl;
    return false;
  }
  //血球
  for(int channel_idx=0;channel_idx<channel_list.size();++channel_idx){
    //检查当前配置通道数是否小于算法通道数
    if(channel_idx>=channel_img_dir.size()){
      std::cout<<"Configured channels are smaller than required"<<std::endl;
      return false;
    }
    std::vector<uint32_t> view_list;
    ret = AlgCell_HeamoListView(algctx, view_list, outer_group_idx, channel_idx);
    if(ret!=0){
      std::cout<<"failed to get view list, error code "<<ret<<std::endl;
      return false;
    }
    std::vector<std::string> identified_view_list;
    //检查当前配置通道的视图数是否小于算法视图数
    if(channel_img_dir[channel_idx].size()<view_list.size()){
      std::cout<<"Configured view list are smaller than required"<<std::endl;
      std::cout<<"channel idx "<< channel_idx<<" should have "<<view_list.size()<<
          " but "<<channel_img_dir[channel_idx].size()<<" was given"<<std::endl;
      return false;
    }
    for(int view_idx=0;view_idx<view_list.size();++view_idx){
      identified_view_list.emplace_back(channel_img_dir[channel_idx][view_idx]);
    }
    //存入确认后的表中
    identified_channel_config.emplace_back(identified_view_list);
  }
  return true;
  //清晰度

}


/*!
 * 读取单个通道下的图像路径
 * @param channel 通道路径,数据包含通道下多个视图对应的图像目录
 * @param[out] img_path_v_v 当前通道下图像路径,内部vector数据依次为当前通道下拍摄的一个视野的图像路径,包含明场,荧光场... 外部vector为多个视野
 * @param[out] img_nums_each_dir 视图目录下图像数量,要求单个通道内配置的多个目录下图像数量相等
 * @return
 */
bool LoadChannelImgDirs(const std::vector<std::string>& channel,std::vector<std::vector<std::string>>& img_path_v_v,
                        int& img_nums_each_dir ){
  //视图
  for(int view_idx =0;view_idx<channel.size(); ++view_idx){

    std::vector<std::string> img_path_v;
    std::string img_dir = channel[view_idx];
    LoadImagePath(img_dir, img_path_v);
    if(img_nums_each_dir!=0&&img_nums_each_dir!=img_path_v.size()){
      std::cout<<"img nums with regard to different views under the save channel are different"<<std::endl;
      return false;
    }
    for(int img_order=0; img_order<img_path_v.size(); ++img_order){

      img_path_v_v[img_order][view_idx]= img_path_v[img_order];
    }
    img_nums_each_dir = int(img_path_v.size());
  }

  return true;
}

void PrintHeamo(const AlgCellRst &result){
  printf("ALG RESULT >>>\r\n序号  名称        单位        数值\r\n");
  //血球
  uint32_t idx = 1;
  for(const auto& item:result.heamo)
  {
    printf("%4.4d  %-10.10s  %-10.10s  %-10.10s\r\n", idx++, item.name.data(), item.unit.data(), item.value.data());
  }
  std::cout<<"rbc curve size: "<<result.curve_rbc.size()<<std::endl;
  //rbc曲线
  std::cout<<"rbc curve:"<<std::endl;
  for(const auto& float_result: result.curve_rbc){
    std::cout<<float_result<<" ";
  }
  std::cout<<std::endl;

  //plt 曲线
  std::cout<<"plt curve:"<<std::endl;
  for(const auto& float_result: result.curve_plt){
    std::cout<<float_result<<" ";
  }
  std::cout<<std::endl;

  std::cout<<"alarm"<<std::endl;
  for(const auto& iter:result.alarm_results){
    std::cout<<iter<<std::endl;
  }
}


void PrintHGB(const float& hgb_result){
  std::cout<<"Print hgb result"<<std::endl;
  std::cout<<hgb_result<<std::endl;
}
std::uniform_real_distribution<> dis(0, 1);
std::default_random_engine e;    // 生成无符号随机整数

void ChangeValue(AlgCellItem_t& item){

  // 特殊值不转换
  if(item.unit== HEAMO_QC_ERASE_VALUE_COMIC){
	return;
  }

  //生成随即数用于测试
  float rand_num = (float)dis(e);
  std::cout<<"Temp num "<<rand_num<<std::endl;
  std::stringstream ss(item.value);
  float value_f;
  ss>>value_f;
  float new_value = value_f * rand_num;
  item.value = std::to_string(new_value);

}

// 将上次测试结果,单次随机修改一项后,显示
int ChangeResult(AlgCellRst_t heamo_result){
  std::vector<std::string> change_item_names{WBC_KEY_NUM, NE_KEY_PERCENTAGE, LY_KEY_PERCENTAGE, MO_KEY_PERCENTAGE,
											EO_KEY_PERCENTAGE, BA_KEY_PERCENTAGE, RBC_KEY_NUM, MCV_KEY_VALUE, HB_KEY_VALUE, PLT_KEY_NUM};

  for(const auto& item_name: change_item_names){
	for(auto& item: heamo_result.heamo){
	  if(item.name == item_name){
		// 每修改依一次,打印依次
		ChangeValue(item);
		std::cout<<"--------------------"<<std::endl;
		std::cout<<"Modified"<<std::endl;;
		std::cout<<"--------------------"<<std::endl;

		int ret = AlgCell_ModifyResult(std::string(item_name) , heamo_result);
		if(ret){
		  std::cout<<"Failed to modify result."<<std::endl;
		  return false;
		}
		PrintHeamo(heamo_result);
		break;
	  }

  }
  }

  PrintHeamo(heamo_result);
  return true;
}


void IntegratedCode::PrintResult(const IntegratedTestType& test_type){
  int ret=0;
  switch (test_type) {
    case ALG_SAMPLE_FIELD:{
      AlgCellRst_t heamo_result;
      AlgCell_HeamoPushGermResultDir("/data/alg_test/2reconstruct/");
      ret = AlgCell_HeamoGetResult(this->algctx, heamo_result, 0XFFFFFF);
      if(ret!=0){
        std::cout<<"Failed to get heamo result, error code "<<ret<<std::endl;
      }
      PrintHeamo(heamo_result);

      AlgCell_HeamoClose(this->algctx);

	  const bool test_modify = false; // 是否进行修改结果测试
	  if(test_modify){

		int local_ret = ChangeResult(heamo_result);
		if(!local_ret){
		  return ;
		}
	  }





      break;
    }
    case ALG_SAMPLE_HGB:{
      //打印结果
      AlgCellRst_t heamo_result;
      ret = AlgCell_HeamoGetResult(this->algctx, heamo_result, 0XFFFFFF);
      if(ret!=0){
        std::cout<<"Failed to get heamo result, error code "<<ret<<std::endl;
      }
      PrintHeamo(heamo_result);
      AlgCell_HeamoClose(this->algctx);
      break;
    }
    default:{
      std::cout<<"Warning, tested none type"<<std::endl;
    }
  }

}

bool IntegratedCode::InitHeamo(const XML::IntDetectTypeInitConfig& int_detect_type_init_config, const XML::IntTestDataDir& int_test_data_dir){

  //打印流道配置
  std::cout<<"Accepted channel"<<std::endl;
  for(const auto&  channel:int_test_data_dir.channel_img_dir){
    std::cout<<"channel"<<std::endl;
    for(const auto& view:channel){
      std::cout<<view<<std::endl;
    }
  }

  this->group_idx = this->sample_type_name_to_type_m.at(int_detect_type_init_config.detect_type_name);


  if(!IdentifyChannels(this->algctx, this->group_idx, int_test_data_dir.channel_img_dir, this->identified_heamo_channel_config)){
    std::cout<<"Fail to identify channel"<<std::endl;
    return false;
  }
  std::cout<<"Identified channel config:"<<std::endl;
  //打印真实流道
  for(const auto&  channel:this->identified_heamo_channel_config){
    std::cout<<"channel"<<std::endl;
    for(const auto& view:channel){
      std::cout<<view<<std::endl;
    }
  }

  for(const auto& sample_config:int_detect_type_init_config.sample_config_v){
    //获取整体测试参数
    if(sample_config.sample_name=="SampleField") {
      if (sample_config.float_param_v.size() != 6) {
        std::cout << "Heamo init need 6 float param, but "
                  << sample_config.float_param_v.size() << " was configured"
                  << std::endl;
        return false;
      }
      if (sample_config.float_param_v[0] == 0) {
        std::cout << "Heamo param first can not be set to 0 " << std::endl;
        return false;
      }
      this->img_pair_nums_deemed_same_view = sample_config.float_param_v[0];
      this->qc = static_cast<bool>(sample_config.float_param_v[1]);
      this->img_h = sample_config.float_param_v[2];
      this->img_w = sample_config.float_param_v[3];
      this->img_h_um = sample_config.float_param_v[4];
      this->calib = static_cast<bool>(sample_config.float_param_v[5]);
    }
    //获取hgb校准参数
    if(sample_config.sample_name=="SampleHGB"){
      auto float_param_v = sample_config.float_param_v;
      if(float_param_v.size()!=4){
        std::cout<<"HGB init need 4 params but "<<float_param_v.size()<<" was given"<<std::endl;
        return false;
      }
      this->hgb_coef =float_param_v;
    }
  }
  //hgb 未配置成功
  return true;
}

bool IntegratedCode::TestAlgSampleHgb(){
  //HGB与血球属于一个整体,但数据形式不同,为方便测试, 将二者分离


  //生成随即数用于测试
  std::uniform_int_distribution<unsigned> u(0,100);
  std::default_random_engine e;    // 生成无符号随机整数
  int times = 50;
  int random_num = 8;//measureI0, measureIs, referenceI0, referenceIs
  for(int time =0; time< times; ++time){
    std::vector<float> data_list;
    std::cout<<"hgb value"<<std::endl;
    for(int num_idx=0; num_idx<random_num; ++num_idx){
      int nums = u(e);
      data_list.emplace_back(nums);
      std::cout<<"  "<<nums;
    }
    std::cout<<std::endl;

    int ret = AlgCell_HeamoOpen(this->algctx, ALGCELL_FUNC_HEAMO,   AlgCellImageCallback_f, nullptr, this->open_params);
    if(ret!=0){
      std::cout<<"Failed to open HGB, error code: "<<ret<<std::endl;
      return false;
    }

    ret = AlgCell_HeamoPushHgb(this->algctx, data_list, this->hgb_coef);

    if(ret!=0){
      std::cout<<"Failed to push hgb, error code "<<ret<<std::endl;
      return false;
    }
    this->PrintResult(ALG_SAMPLE_HGB);
  }
  return true;
}


bool IntegratedCode::TestAlgSampleField(int times) {
  //读取图像路径
  /*  std::vector<cv::String> img_bright_path_v, img_fluo_path_v;
    cv::glob(input_bright_dir, img_bright_path_v);
    cv::glob(input_fluo_dir, img_fluo_path_v);*/
  //依流道进行数据读取
  for (int temp_iter=0; temp_iter<times; ++temp_iter){
    int ret = AlgCell_HeamoOpen(this->algctx, ALGCELL_FUNC_HEAMO,  AlgCellImageCallback_f, nullptr, this->open_params);
    if(ret!=0){
      std::cout<<"Failed to open heamo, error code: "<<ret<<std::endl;
      return false;
    }

    for(int channel_idx=0;channel_idx<this->identified_heamo_channel_config.size();++channel_idx){
      //流道
      const auto channel = this->identified_heamo_channel_config[channel_idx];
      int view_nums = int(channel.size());

      //{img_bright_path, img_fluo_path, img_brigh_path...}
      std::vector<std::string> img_path_unit(view_nums ,"");
      std::vector<std::vector<std::string>> img_path_v_v(MAX_IMG_PAIRS, img_path_unit);
      int img_nums_each_dir = 0;

      //读取图像路径
      if(!LoadChannelImgDirs(channel, img_path_v_v, img_nums_each_dir)){
        std::cout<<"Failed to load img path"<<std::endl;
        return false;
      }
      std::map<std::string, float> complementary_params;
      for(int idx=0;idx<img_nums_each_dir;++idx){
        auto img_pairs_v = img_path_v_v[idx];
        std::vector<AlgCellImg> img_v;
        //读取单个视图的数据数据   {img_bright_path, img_fluo_path, img_brigh_path...}
        for(int pair_idx=0;pair_idx<img_pairs_v.size();++pair_idx){

          auto img = new AlgCellImg;
          if(!ReadImgToBuf(img, img_pairs_v[pair_idx])){
            std::cout<<"Failed to read img"<<std::endl;
            return false;
          }
          img_v.emplace_back(*img);
          delete img;
        }
        complementary_params[VIEW_PAIR_IDX] = (int) (idx/(img_pair_nums_deemed_same_view));
        //推理
        int ret =AlgCell_HeamoPushImage(this->algctx,img_v, this->group_idx, channel_idx, complementary_params);
        if(ret!=0){
          std::cout<<"Failed to push heamo images, error code "<<ret<<std::endl;
          return false;
        }
        //回收图像
        for(auto img: img_v){
          delete [] img.data;
        }

      }
    }
    ret = AlgCell_HeamoPushHgb(this->algctx, {13784,	13340,	13597,	13597,	2737,	6520,	7784,	13597}, this->hgb_coef);

    if(ret!=0){
      std::cout<<"Failed to push hgb, error code "<<ret<<std::endl;
      return false;
    }
    this->PrintResult(ALG_SAMPLE_FIELD);
    return true;
  }

}


}
}