//
// Created by y on 24-7-22.
//
#include <vector>
#include <string>
#ifndef TEST_LIBALG_ALARMMANAGER_H
#define TEST_LIBALG_ALARMMANAGER_H

#define ALARM_ALL_PSEUDO_MIN_THR                      -999999
#define ALARM_ALL_PSEUDO_MAX_THR                      999999
#define ALARM_ALL_EXIST                               0
#define ALARM_ALL_PEAK_THR                            1

class AlarmManager{
 public:
  AlarmManager()=default;
  ~AlarmManager()=default;
  int SetCustomAlarmParams(const std::vector<float>& alarm_params_v){return 0;};
};
class HeamoAlarmManager:public AlarmManager{
 public:
  HeamoAlarmManager()=default;
  ~HeamoAlarmManager()=default;

  int SetCustomAlarmParams(const std::vector<float>& alarm_params_v);
  int GetAlarmResults(const float &wbc, const float &neu, const float &lym, const float &mono, const float &eos,
                  const float &baso,const float &nrbc, const float &ig, const float &wbc_agglomeration,//wbc
                  const std::vector<float> &rbc_curve, const float &rdw_cv, const float &rdw_sd, const float &mcv,
                  const float &rbc, const float &hgb, const float &mchc, const float &rbc_agglomeration,//rbc
                  const float &plt, const std::vector<float> &plt_curve, const float &plt_agglomeration,//plt
                  const float &ret,//ret
                  std::vector<std::string>& output);
  int ModifyAlarmResults(const std::vector<std::string>& previous_alarm,const float &wbc, const float &neu, const float &lym, const float &mono, const float &eos,
					  const float &baso,const float &nrbc, const float &ig, const float &wbc_agglomeration,//wbc
					  const std::vector<float> &rbc_curve, const float &rdw_cv, const float &rdw_sd, const float &mcv,
					  const float &rbc, const float &hgb, const float &mchc, const float &rbc_agglomeration,//rbc
					  const float &plt, const std::vector<float> &plt_curve, const float &plt_agglomeration,//plt
					  const float &ret,//ret
					  std::vector<std::string>& output);
 private:
  float wbc_num_high_thr=18, wbc_num_low_thr=2.5, neu_num_high_thr=18, neu_num_low_thr=1,
      lym_num_high_thr=4, mon_num_high_thr=1.5, eos_num_high_thr=1.5, baso_num_high_thr=0.3,
	  ig_num_upper_thr=2, rbc_rdw_sd_upper_thr=64, rbc_rdw_cv_upper_thr=22, rbc_mcv_low_thr=70,
	  rbc_mcv_high_thr=110, rbc_num_high_thr=6.5, rbc_hgb_low_thr=90, rbc_mchc_low_thr=29,
	  plt_num_high_thr=600, plt_num_low_thr=60, ret_num_high_thr=2.5, ret_num_low_thr=0.5;
  float nrbc_num_high_thr = ALARM_ALL_EXIST;
  float wbc_aggregation_num_upper_thr = ALARM_ALL_EXIST;

  float rbc_histogram_abnormal_thr = ALARM_ALL_EXIST;
  float rbc_histogram_double_peak_thr = ALARM_ALL_PEAK_THR;
  float rbc_aggregation_num_upper = ALARM_ALL_EXIST;

  float plt_histogram_abnormal_thr = ALARM_ALL_EXIST;
  float plt_aggregation_num_upper_thr = ALARM_ALL_EXIST;

  float cst_alarm_nums = 20;

};

#endif  // TEST_LIBALG_ALARMMANAGER_H
