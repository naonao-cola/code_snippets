//
// Created by y on 24-12-12.
//

#ifndef DIH_ALG_FOCUSCONTROL_H
#define DIH_ALG_FOCUSCONTROL_H
#include <vector>
namespace ALG_DEPLOY {



class FocusControl {
 public:
  FocusControl(const float& far,
               const float& type_near,
               const float& type_coarse_focus,
               const float& type_clear,
               const float& type_indetermination,
               const float& type_change_view,
               const float& type_very_near,
               const float& type_force_clear);
  virtual ~FocusControl() = default;
  virtual int FineControl(const int& viw_idx, const float& src, const std::vector<float>& pred_prob, float& dst)=0;
  int previous_view_idx;
  float res_type_far             =0   ;
  float res_type_near            =1   ;
  float res_type_coarse_focus    =2   ;
  float res_type_clear          =3    ;
  float res_type_indetermination =4   ;
  float res_type_change_view     =5   ;
  float res_type_very_near       =6   ;
  float res_type_force_clear     =7   ;

};

class BasoFocusControl:public FocusControl{
 public:
  BasoFocusControl(const float& far,
                   const float& type_near,
                   const float& type_coarse_focus,
                   const float& type_clear,
                   const float& type_indetermination,
                   const float& type_change_view,
                   const float& type_very_near,
                   const float& type_force_clear);
  virtual ~BasoFocusControl()=default;
  int FineControl(const int& view_pair_idx, const float& src, const std::vector<float>& pred_prob, float& dst) override;
 private:
  int NonWbcCheck(const float& src, const std::vector<float>& pred_prob, float& dst);
  int HierarchyCheck(const float& src, float& dst);

  const int OSCILLATION_LENGTH = 20;
  const float NON_WBC_VERY_THR = 0.6;
  const float HIERARCHY_PERCENTAGE_THR = 0.2;
  std::vector<float> fine_proc_res;

};

}
#endif  // DIH_ALG_FOCUSCONTROL_H
