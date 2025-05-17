//
// Created by y on 24-4-3.
//
#include <numeric>

#include "ParamFitting.h"
//#include "DihLog.h"
#include "algLog.h"
#include "utils.h"

#define ADD_DENOMINATOR       1e-5   //为避免除0，对分母进行增加
namespace ALG_DEPLOY{

int NormalReagentFitting::Init() {
  int ret = 0;
  ret = (ret || mcv_network.Init());
  ret = (ret || mpv_network.Init());
  ret = (ret || rdw_cv_network.Init());
  ret = (ret || rdw_sd_network.Init());
  ret = (ret || hgbNetwork.Init());
  ret = (ret || cell_count_network.Init());
  if(ret){
    ALGLogError<<"Failed to init NormalReagentFitting";
    return -1;
  }
}






int NormalReagentFitting::Forward(const std::vector<float>& area_rbc_v, const float& incline_rbc_nums,
                                  const float& incline_rbc_region, const std::vector<float>& area_plt_v,
                                  const float& relative_line_ratio_to_standard,
                                  const std::vector<float> &data_v, const std::vector<float> &coef_v,
                                  float& mcv, float& rdw_cv, float& rdw_sd, float& mpv,
                                  std::vector<float>& curve_rbc, std::vector<float>& curve_plt,
                                  float& hgb, float& rbc, float& ret_, float&plt,
                                  float&neu, float&lym, float& mono, float&eos, float& baso){
  int ret = 0;
  // mcv
  ret = NormalMcvFitting(area_rbc_v, incline_rbc_nums, incline_rbc_region, mcv);
  if(ret){
    ALGLogError<<"Failed to get normal reagent mcv";
    return -1;
  }


  // rbc line
  ret = NormalGetRbcVolumeLine(area_rbc_v, curve_rbc);
  if(ret){
    ALGLogError<<"Failed to get normal reagent rbc volume line";
    return -2;
  }

  // mpv
  std::vector<float> plt_volume_v;
  ret = NormalMpvFitting(area_plt_v, plt_volume_v, mpv);
  if(ret){
    ALGLogError<<"Failed to get normal reagent rbc mpv";
    return -3;
  }

  // plt line
  ret = NormalGetPltVolumeLine(plt_volume_v, curve_plt);
  if(ret){
    ALGLogError<<"Failed to get normal reagent plt volume line";
    return -4;
  }

  // rdw_cv
  ret = NormalRdwCvFitting(area_rbc_v, mcv, rdw_cv);
  if(ret){
    ALGLogError<<"Failed to get normal reagent rdw_cv";
    return -5;
  }

  // rdw_sd
  ret = NormalRdwSdFitting(curve_rbc, rdw_sd);
  if(ret){
    ALGLogError<<"Failed to get normal reagent rdw_sd";
    return -6;
  }

  // hgb
  ret = HgbFitting(data_v, coef_v, hgb);
  if(ret){
    ALGLogError<<"Failed to get normal reagent hgb";
    return -7;
  }


  // count
  ret = AllCellCountFitting(rbc, ret_, plt,
                            neu, lym,  mono, eos, baso);
  if(ret){
    ALGLogError<<"Failed to get spherical reagent count";
    return -8;
  }
  return 0;

}
/////////////////////////
////mcv
/////////////////////////
int NormalReagentFitting::NormalMcvFitting(const std::vector<float>& area_rbc_v, const float& incline_rbc_nums,
                                           const float& incline_rbc_region, float& mcv){
  if(area_rbc_v.empty()){
    mcv=0;
    return 0;
  }
  //for get params from rk35xx
  //  std::cout<<"rbc ori value"<<std::endl;
  //  for(auto i :area_rbc_v){
  //    std::cout<<" "<<i<<",";
  //  }
  std::cout<<std::endl;

  double sum = std::accumulate(std::begin(area_rbc_v), std::end(area_rbc_v), 0.0);
  double mean =  sum / ((int)area_rbc_v.size()+ADD_DENOMINATOR); //均值
  ALGLogInfo<<"SUM SIZE " <<sum<< " "<<area_rbc_v.size();
  float area_mean = (float)mean;

  float incline_mean = incline_rbc_region/(incline_rbc_nums+ADD_DENOMINATOR);
  std::vector<float> network_input{area_mean, incline_mean};
  std::vector<float> network_output;
  int ret = mcv_network.Forward(network_input, network_output);
  if(ret){
    return -1;
  }
  mcv = network_output[0];
  //避免白细胞流道测试数据过少导致结果为负数
  if(mcv<0){
    mcv = 0;
  }
  return 0;

}
/////////////////////////
////mpv
/////////////////////////
int NormalReagentFitting::NormalMpvFitting(const std::vector<float>& area_plt_v, std::vector<float>& plt_volume_v, float& mpv){
  if(area_plt_v.empty()){
    mpv=0;
    return 0;
  }
  //  std::cout<<"----------plt ori value--------------"<<std::endl;
  //  for(auto i :area_plt_v){
  //    std::cout<<" "<<i<<",";
  //  }
  //  std::cout<<std::endl;

  //计算单个plt体积
  std::vector<float> network_input(area_plt_v);
  std::vector<float> network_output;
  int ret = mpv_network.Forward(network_input,  network_output);
  if(ret){
    return -1;
  }
  plt_volume_v = network_output;
  //计算平均体积
  float sum = std::accumulate(plt_volume_v.begin(), plt_volume_v.end(), 0.0);
  float mean = sum/plt_volume_v.size();
  mpv = mean;

  //  std::cout<<"----------------plt volume value-------------"<<std::endl;
  //  for(int i =0; i< plt_volume_v.size(); ++i){
  //    std::cout<<", "<<plt_volume_v[i];
  //  }
  std::cout<<std::endl;
  return 0;
}
/////////////////////////
////rdw_cv
/////////////////////////
int NormalReagentFitting::NormalRdwCvFitting(const std::vector<float>& vol_v, const float& mcv, float& rdw_cv){
  if(vol_v.empty()){
    rdw_cv=0;
    return 0;
  }

  float std_value;
  CalculateStd(vol_v, std_value);
  ALGLogInfo<<"std, mcv " <<std_value<<" "<<mcv;
  std::vector<float> network_input{std_value/mcv*100};
  std::vector<float> network_output;
  int ret = rdw_cv_network.Forward(network_input, network_output);
  if(ret){
    return -1;
  }
  rdw_cv = network_output[0];
  return 0;

}
/////////////////////////
////rdw sd
/////////////////////////
int NormalReagentFitting::NormalRdwSdFitting(const std::vector<float>& vol_v, float& rdw_sd){
  if(vol_v.empty()){
    rdw_sd=0;
    return 0;
  }

  int max_idx = PseudoArgMax(vol_v.begin(), vol_v.end());
  std::vector<float> left_data(vol_v.begin(), vol_v.begin()+max_idx+1);
  std::vector<float> right_data(vol_v.begin()+max_idx, vol_v.end());
  float percentage_value = vol_v[max_idx]*NORMAL_RDW_SD_PERCENTAGE;
  int left_min_idx, right_min_idx;
  FindMinDifference(left_data, percentage_value, left_min_idx);
  FindMinDifference(right_data, percentage_value, right_min_idx);
  float raw_rdw_ds = right_min_idx+max_idx-left_min_idx;
  std::vector<float> network_input{raw_rdw_ds};
  std::vector<float> network_output;

  int ret = rdw_sd_network.Forward(network_input, network_output);
  if(ret){
    return -1;
  }
  rdw_sd = network_output[0];
  return 0;
}
/////////////////////////
////rbc line
/////////////////////////
int NormalReagentFitting::NormalGetRbcVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result){
  //避免对空mat进行操作
  if(vol_v.empty()){
    result = std::vector<float>(NORMAL_VOLUME_SIZE, 0);
    return 0;
  }
  auto temp_v = std::vector<float>(NORMAL_VOLUME_SIZE, 0);
  std::vector<float> vol_v_copy(vol_v);
  cv::Mat vol{vol_v_copy};

  //排除异常大的值
  MatClip(vol, NORMAL_VOLUME_MAX_AREA, vol);

  //面积缩放
  vol = vol/NORMAL_VOLUME_RBC_DOWN_RATIO;

  //计数
  std::vector<int> v = vol.reshape(1, 1);
  for(int i=0;i<vol_v.size();++i){
    //    int idx = (v[i]==0)*1+v[i]-1;
    int idx = v[i];
    temp_v[idx] +=1;
  }

  //中值滤波
  std::vector<float> v_median;
  MedianFilter(temp_v, v_median, int(NORMAL_VOLUME_RBC_KERNEL_MEDIUM/2));

  //均值滤波
  cv::Mat temp_mat{v_median};
  cv::blur(temp_mat, temp_mat, cv::Size(NORMAL_VOLUME_RBC_KERNEL_BLUR,NORMAL_VOLUME_RBC_KERNEL_BLUR));
  result = temp_mat.reshape(1,1);
  return 0;
}
/////////////////////////
////plt line
/////////////////////////
int NormalReagentFitting::NormalGetPltVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result_v){

  int ret = LocalMeanVolumeLine(vol_v, NORMAL_VOLUME_SIZE,
                                NORMAL_VOLUME_PLT_DILATE_RATIO, NORMAL_VOLUME_PLT_DOWN_RATIO,
                                NORMAL_VOLUME_MAX_AREA,
                                NORMAL_VOLUME_PLT_SAMPLE_WIDTH, NORMAL_VOLUME_PLT_KERNEL_BLUR,
                                result_v);
  return ret;
}
}