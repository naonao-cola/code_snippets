//
// Created by y on 24-12-12.
//
#include <iostream>

#include "FocusControl.h"
#include "algLog.h"

#define  OUTPUT_POSITION std::cout

namespace ALG_DEPLOY {

FocusControl::FocusControl(const float& far,
                           const float& type_near,
                           const float& type_coarse_focus,
                           const float& type_clear,
                           const float& type_indetermination,
                           const float& type_change_view,
                           const float& type_very_near,
                           const float& type_force_clear) {
  res_type_far             =far;
  res_type_near            =type_near;
  res_type_coarse_focus    =type_coarse_focus;
  res_type_clear           =type_clear;
  res_type_indetermination =type_indetermination;
  res_type_change_view     =type_change_view;
  res_type_very_near       =type_very_near;
  res_type_force_clear     =type_force_clear;
}

BasoFocusControl::BasoFocusControl(const float& far, const float& type_near, const float& type_coarse_focus,
                               const float& type_clear, const float& type_indetermination, const float& type_change_view,
                               const float& type_very_near, const float& type_force_clear): FocusControl(far, type_near, type_coarse_focus,
                                                                                                              type_clear, type_indetermination, type_change_view,
                                                                                                              type_very_near, type_force_clear){

}

int BasoFocusControl::FineControl(const int& view_pair_idx, const float& src, const std::vector<float>& pred_prob, float& dst){
  ALGLogInfo<<"View pair idx "<<view_pair_idx;
  if(view_pair_idx!=previous_view_idx){
    fine_proc_res.clear();
    previous_view_idx = view_pair_idx;
  }

  //无白细胞检查
  int ret = NonWbcCheck(src,pred_prob, dst);
  if(ret){return ret;}
  if(src!=dst){
    fine_proc_res.push_back(src);
    return 0;
  }
  //分层检查
  HierarchyCheck(src,dst);

  fine_proc_res.push_back(src);
  return 0;
}

void ExistResTimeCheck(const float& src, const std::vector<float>& res, int& times){
  times = 0;
  for (const auto& iter: res){
    if(src==iter){
      times ++;
    }
  }
}

#define CLARITY_BASO_CATEGORY_NUMS 9
/*!
 *视野无白细胞检查
 * @param src
 * @param dst
 * @return
 */
int BasoFocusControl::NonWbcCheck(const float& src, const std::vector<float>& pred_prob, float& dst){

  if(fine_proc_res.size()<OSCILLATION_LENGTH){//当前视野聚焦次数超过限定才进行无白细胞检查逻辑
    dst = src;
    ALGLogInfo << " NonWbcCheck 超过限定才进行无白细胞检查逻辑 :  " << fine_proc_res.size();
    return 0;
  }

  if(pred_prob.size()!=CLARITY_BASO_CATEGORY_NUMS){
    ALGLogError<<"Clarity category nums for baso must be "<<CLARITY_BASO_CATEGORY_NUMS<<" , but "<<pred_prob.size()<<" was given";
    return -1;
  }
  ALGLogInfo << " NonWbcCheck:  " << pred_prob.size();

  float neg_prob = pred_prob[0];
  float pos_prob = pred_prob[1];
  float far_pos_prob = pred_prob[4];
  float wbc_prob          = pred_prob[7] + pred_prob[8];
  float pseudo_clear_prob = neg_prob+pos_prob+far_pos_prob+wbc_prob;

  int far_pos_times = 0;
  ExistResTimeCheck(res_type_very_near, fine_proc_res, far_pos_times);
  int neg_times = 0;
  ExistResTimeCheck(res_type_far,fine_proc_res, neg_times);
  int pos_times = 0;
  ExistResTimeCheck(res_type_near,fine_proc_res, pos_times);

  ALGLogInfo<<"Focus condition, clear prob "<<pseudo_clear_prob<<" , far, pos, neg time "<<far_pos_times<<" "<<pos_times<<" "<<neg_times;
  if((float)far_pos_times+pos_times>0
      &&(neg_times>0)&&(pseudo_clear_prob)>NON_WBC_VERY_THR){//远离焦平面出现百分比>thr, 且至少出现1次负焦或失焦(细聚焦不会从失焦开始)
    dst = res_type_force_clear;
    return 0;
  }
  dst = src;
  return 0;
}

/*
 * 分层现象检查
 */
int BasoFocusControl::HierarchyCheck(const float& src, float& dst){
  if(fine_proc_res.size()<OSCILLATION_LENGTH){//当前视野聚焦次数超过限定才进行分层检查逻辑
    dst = src;
    return 0;
  }

  int neg_times = 0;
  ExistResTimeCheck(res_type_far,fine_proc_res, neg_times);
  int pos_times = 0;
  ExistResTimeCheck(res_type_near,fine_proc_res, pos_times);
  if((float)neg_times/fine_proc_res.size()>=HIERARCHY_PERCENTAGE_THR
      &&(float)pos_times/fine_proc_res.size()>=HIERARCHY_PERCENTAGE_THR){//远离焦平面出现百分比>thr, 且至少出现1次负焦或失焦(细聚焦不会从失焦开始)
    dst = res_type_force_clear;

    std::cout<<"  "<<neg_times<<" "<<pos_times<<" "<< (float)neg_times/fine_proc_res.size()<<" "<<(float)pos_times/fine_proc_res.size()<<std::endl;
    for(const auto& iter: fine_proc_res){
      OUTPUT_POSITION<<" "<<iter;
    }
    OUTPUT_POSITION<<std::endl;

    return 0;
  }
  dst = src;
  return 0;

}



}
