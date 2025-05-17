//
// Created by y on 24-3-28.
//
#include <numeric>

#include "ParamFitting.h"
//#include "DihLog.h"
#include "algLog.h"
#include "utils.h"

namespace ALG_DEPLOY {



//其他

int LocalMeanVolumeLine(const std::vector<float>& vol_v, const int& volume_size,
                  const int& dilate_ratio, const int& down_ratio,
                  const int& max_volume_confine,
                  const int& sample_window_size, const int& mean_kernel_size,
                  std::vector<float>& result_v){

  ALGLogInfo<<"V D D M S M "<<volume_size<<" "<<dilate_ratio<<" "<<down_ratio<<" "<<max_volume_confine<<" "<<sample_window_size<<" "<<mean_kernel_size;
  //避免对空mat进行操作
  if(vol_v.empty()){
    result_v = std::vector<float>(volume_size, 0);
    return 0;
  }

  cv::Mat vol{vol_v};
  vol = vol * dilate_ratio;
  //排除异常大的值
  MatClip(vol, max_volume_confine, vol);

  //面积缩放
  vol = vol/down_ratio;

  //计数
  auto plt_nums_under_interval = std::vector<float>(volume_size, 0);

  std::vector<float> v_f = vol.reshape(1, 1);
  std::for_each(v_f.begin(), v_f.end(), [](float & value){value = std::round(value);});
  std::vector<int> v(v_f.begin(), v_f.end());

  for(int i=0;i<vol_v.size();++i){
    //    int idx = (v[i]==0)*1+v[i]-1;
    int idx = v[i];
    plt_nums_under_interval[idx] +=1;
  }


  //统计局部均值
  std::vector<float> local_mean_v;
  for(int idx=0;idx<plt_nums_under_interval.size();idx=idx+sample_window_size){
//    float max_value = *std::max_element(plt_nums_under_interval.begin()+idx, plt_nums_under_interval.begin()+idx+sample_window_size);

    float sum = std::accumulate(plt_nums_under_interval.begin()+idx, plt_nums_under_interval.begin()+idx+sample_window_size, 0.f);
    float mean = sum/sample_window_size;
    local_mean_v.push_back(mean);
  }

  cv::Mat local_mean_mat{local_mean_v};
  local_mean_mat = local_mean_mat.reshape(0, 1);
  cv::resize(local_mean_mat, local_mean_mat, cv::Size(volume_size, 1), cv::INTER_LINEAR);


  //均值滤波
  cv::blur(local_mean_mat, local_mean_mat, cv::Size(mean_kernel_size,mean_kernel_size));
  result_v = local_mean_mat.reshape(1,1);
  return 0;



}


int LocalMaxVolumeLine(const std::vector<float>& vol_v, const int& volume_size,
                        const int& dilate_ratio, const int& down_ratio,
                        const int& max_volume_confine,
                        const int& sample_window_size, const int& mean_kernel_size,
                        std::vector<float>& result_v){

  ALGLogInfo<<"V D D M S M "<<volume_size<<" "<<dilate_ratio<<" "<<down_ratio<<" "<<max_volume_confine<<" "<<sample_window_size<<" "<<mean_kernel_size;
  //避免对空mat进行操作
  if(vol_v.empty()){
    result_v = std::vector<float>(volume_size, 0);
    return 0;
  }

  cv::Mat vol{vol_v};
  vol = vol * dilate_ratio;
  //排除异常大的值
  MatClip(vol, max_volume_confine, vol);

  //面积缩放
  vol = vol/down_ratio;

  //计数
  auto plt_nums_under_interval = std::vector<float>(volume_size, 0);
  std::vector<float> v_f = vol.reshape(1, 1);
  std::for_each(v_f.begin(), v_f.end(), [](float & value){value = std::round(value);});
  std::vector<int> v(v_f.begin(), v_f.end());
  for(int i=0;i<vol_v.size();++i){
    //    int idx = (v[i]==0)*1+v[i]-1;
    int idx = v[i];
    plt_nums_under_interval[idx] +=1;
  }
//  std::cout<<"rdw ds intervel -----------------"<<std::endl;
//  for(const auto& iter:plt_nums_under_interval){
//    std::cout<<iter<<" ";
//  }
//  std::cout<<std::endl;

  //统计局部最大值
  std::vector<float> local_mean_v{0};
  int idx = 0;
  for(;idx+sample_window_size<plt_nums_under_interval.size();idx=idx+sample_window_size){
    float max_value = *std::max_element(plt_nums_under_interval.begin()+idx, plt_nums_under_interval.begin()+idx+sample_window_size);

    local_mean_v.push_back(max_value);

  }
  if(idx<plt_nums_under_interval.size()){//余留数据
    float max_value = *std::max_element(plt_nums_under_interval.begin()+idx, plt_nums_under_interval.end());
    local_mean_v.push_back(max_value);
  }



//  std::cout<<"rdw sd max -----------------"<<std::endl;
//  for(const auto& iter:local_mean_v){
//    std::cout<<iter<<" ";
//  }
  // resize 至指定大小
  std::cout<<std::endl;
  cv::Mat local_mean_mat{local_mean_v};
  local_mean_mat = local_mean_mat.reshape(0, 1);
  cv::resize(local_mean_mat, local_mean_mat, cv::Size(volume_size, 1), cv::INTER_LINEAR);


  //均值滤波
  cv::blur(local_mean_mat, local_mean_mat, cv::Size(mean_kernel_size,mean_kernel_size));
  result_v = local_mean_mat.reshape(1,1);
  return 0;



}


int MediumMeanVolumeLine(const std::vector<float>& vol_v, const int& volume_size,
                         const int& dilate_ratio, const int& down_ratio,
                         const int& max_volume_confine,
                         const int& medium_kernel_size, const int& mean_kernel_size,
                         std::vector<float>& result_v){
  ALGLogInfo<<"Medium Mean: V D D M S M "<<volume_size<<" "<<dilate_ratio<<" "<<down_ratio
             <<" "<<max_volume_confine<<" "<<medium_kernel_size<<" "<<mean_kernel_size;
  //避免对空mat进行操作
  if(vol_v.empty()){
    result_v = std::vector<float>(volume_size, 0);
    return 0;
  }

  cv::Mat vol{vol_v};
  vol = vol * dilate_ratio;
  //排除异常大的值
  MatClip(vol, max_volume_confine, vol);

  //面积缩放
  vol = vol/down_ratio;

  //计数
  auto nums_under_interval = std::vector<float>(volume_size, 0);
  std::vector<float> v_f = vol.reshape(1, 1);
  std::for_each(v_f.begin(), v_f.end(), [](float & value){value = std::round(value);});
  std::vector<int> v(v_f.begin(), v_f.end());
  for(int i=0;i<vol_v.size();++i){
    //    int idx = (v[i]==0)*1+v[i]-1;
    int idx = v[i];
    nums_under_interval[idx] +=1;
  }


  //中值滤波
  std::vector<float> vol_medium;
  MedianFilter(nums_under_interval, vol_medium, medium_kernel_size/2);

  //均值滤波
  cv::Mat mean_mat{vol_medium};
  cv::blur(mean_mat, mean_mat, cv::Size(mean_kernel_size,mean_kernel_size));
  result_v = mean_mat.reshape(1,1);
  return 0;
}




int ParamFitting::HgbFitting(const std::vector<float> &data_v, const std::vector<float> &coef_v, float& hgb){


  if (coef_v.size() != HEAMO_HGB_COEF_NUMS || data_v.size() != HEAMO_HGB_PARAM_NUMS) {
    ALGLogError<<"Coef and data list size should be "<< HEAMO_HGB_COEF_NUMS
                <<" and "<<HEAMO_HGB_PARAM_NUMS<<" respectively, but "
                <<coef_v.size()<<" and "<<data_v.size() <<" was given";

    return -1;
  }

  float r0 = data_v[0];
  float r1 = data_v[1];
  float r2 = data_v[2];
  float m0 = data_v[4];
  float m1 = data_v[5];
  float m2 = data_v[6];


  float a = coef_v[0];
  float b = coef_v[1];
  float c = coef_v[2];
  float d = coef_v[3];

  ALGLogInfo<<"R "<<r0<<" "<<r1<<" "<<r2;
  ALGLogInfo<<"M "<<m0<<" "<<m1<<" "<<m2;
  ALGLogInfo<<"C "<<a<<" "<<b<<" "<<c<<" "<<d;

  float input_param1 = log10(r0/m0);
  float input_param2 = log10(r1/m1);
  float input_param3 = log10(r2/m2);
/*  std::vector<float> network_input{input_param1, input_param2, input_param3};
  std::vector<float> network_output;

  int ret = hgbNetwork.Forward(network_input, network_output);
  if(ret){
    return -2;
  }
  hgb = network_output[0];*/
  hgb = (a*input_param1+b*input_param2+c*input_param3+d)*100;
  if(std::isnan(hgb)){
    ALGLogWarning<<"Hb produced nan value, please check the input, hgb is forced set to 0 ";
    hgb = 0.f;
  }
  return 0;
}
int ParamFitting::CellCountFitting(const float& src, float& dst){

  std::vector<float> network_input{src};
  std::vector<float> network_output;

  int ret = cell_count_network.Forward(network_input, network_output);
  if(ret){
    return -2;
  }
  dst = network_output[0];




  return 0;
}

int ParamFitting::AllCellCountFitting(float& rbc, float& ret_, float&plt,
                        float&neu, float&lym, float& mono, float&eos, float& baso){
/*  int ret = CellCountFitting(rbc, rbc);
  if(ret){
    ALGLogError<<"Failed to get rbc count";
    return -1;
  }*/
  /*ret = CellCountFitting(ret_, ret_);
  if(ret){
    ALGLogError<<"Failed to get ret_ count";
    return -2;
  }
  ret = CellCountFitting(neu, neu);
  if(ret){
    ALGLogError<<"Failed to get neu count";
    return -3;
  }

  ret = CellCountFitting(lym, lym);
  if(ret){
    ALGLogError<<"Failed to get lym count";
    return -4;
  }

  ret = CellCountFitting(mono, mono);
  if(ret){
    ALGLogError<<"Failed to get mono count";
    return -5;
  }

  ret = CellCountFitting(eos, eos);
  if(ret){
    ALGLogError<<"Failed to get eos count";
    return -6;
  }

  ret = CellCountFitting(baso, baso);
  if(ret){
    ALGLogError<<"Failed to get baso count";
    return -7;
  }
  ret = CellCountFitting(plt, plt);
  if(ret){
    ALGLogError<<"Failed to get plt count";
    return -8;
  }*/
  return 0;
}





}