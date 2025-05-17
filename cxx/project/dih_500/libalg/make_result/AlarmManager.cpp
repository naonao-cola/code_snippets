//
// Created by y on 24-7-22.
//
#include "AlarmManager.h"
//#include "DihLog.h"
#include "algLog.h"
#include "utils.h"
/*!
 * 通用单个参数报警检测,当值小于low或大于high时,报警
 * @param num 当前值
 * @param low 低值
 * @param high 高值
 * @param name 名称
 */

void Heamo_AlarmCheckSingleNormal(const float &num, const float &low, const float &high,
                                  const std::string &name, std::vector<std::string>& alarm_str_v,
                                  bool check_repeat=false) {

  if (num < low || num > high) {
    if(check_repeat){// rdw_cv, rw_sd报警相同,若有单个报警,第二个报警不填入
      auto find_idx = std::find(alarm_str_v.begin(),alarm_str_v.end(),name);
      if(find_idx!=alarm_str_v.end()){
        return;
      } else{
        alarm_str_v.push_back(name);
        return;
      }
    }
    alarm_str_v.push_back(name);
  }

}

/*!
 * 检验是否存在双峰
 * @param data
 * @param peak_nums
 */
void DoublePeakCheck(std::vector<float>& data, float& peak_nums){
  peak_nums = 0;
  uint32_t index;
  float value;
  int peak_nums_int;
  PeekSeekFirst(data, &index, &value, peak_nums_int);
  peak_nums = (float)peak_nums_int;

}


int HeamoAlarmManager::SetCustomAlarmParams(const std::vector<float>& alarm_params_v){
  if(alarm_params_v.size()!=cst_alarm_nums){
    ALGLogError<<"For heamo, alarm params should be "<<cst_alarm_nums<<" but "<<alarm_params_v.size()<<" was given";
    return -1;
  }
  wbc_num_high_thr=alarm_params_v[0];wbc_num_low_thr=alarm_params_v[1];neu_num_high_thr=alarm_params_v[2];neu_num_low_thr=alarm_params_v[3];
  lym_num_high_thr=alarm_params_v[4];mon_num_high_thr=alarm_params_v[5];eos_num_high_thr=alarm_params_v[6];baso_num_high_thr=alarm_params_v[7];
  ig_num_upper_thr=alarm_params_v[8];rbc_rdw_sd_upper_thr=alarm_params_v[9];rbc_rdw_cv_upper_thr=alarm_params_v[10];rbc_mcv_low_thr=alarm_params_v[11];
  rbc_mcv_high_thr=alarm_params_v[12];rbc_num_high_thr=alarm_params_v[13];rbc_hgb_low_thr=alarm_params_v[14];rbc_mchc_low_thr=alarm_params_v[15];
  plt_num_high_thr=alarm_params_v[16];plt_num_low_thr=alarm_params_v[17];ret_num_high_thr=alarm_params_v[18];ret_num_low_thr=alarm_params_v[19];

  return 0;
}
int HeamoAlarmManager::GetAlarmResults(const float &wbc, const float &neu, const float &lym, const float &mono, const float &eos,
                                  const float &baso,
                                  const float &nrbc, const float &ig, const float &wbc_agglomeration,//wbc
                                  const std::vector<float> &rbc_curve, const float &rdw_cv, const float &rdw_sd, const float &mcv,
                                  const float &rbc, const float &hgb, const float &mchc, const float &rbc_agglomeration,//rbc
                                  const float &plt, const std::vector<float> &plt_curve, const float &plt_agglomeration,//plt
                                  const float &ret,//ret
                                  std::vector<std::string>& output){

  // WBC
  Heamo_AlarmCheckSingleNormal(wbc, wbc_num_low_thr, ALARM_ALL_PSEUDO_MAX_THR, "wbc_num_low_warning", output);
  Heamo_AlarmCheckSingleNormal(wbc, ALARM_ALL_PSEUDO_MIN_THR, wbc_num_high_thr, "wbc_num_high_warning", output);
  Heamo_AlarmCheckSingleNormal(neu, neu_num_low_thr, ALARM_ALL_PSEUDO_MAX_THR, "neu_num_low_warning", output);
  Heamo_AlarmCheckSingleNormal(neu, ALARM_ALL_PSEUDO_MIN_THR, neu_num_high_thr, "neu_num_high_warning", output);
  Heamo_AlarmCheckSingleNormal(lym, ALARM_ALL_PSEUDO_MIN_THR, lym_num_high_thr, "lym_num_high_warning", output);
  Heamo_AlarmCheckSingleNormal(mono, ALARM_ALL_PSEUDO_MIN_THR, mon_num_high_thr, "mon_num_high_warning", output);
  Heamo_AlarmCheckSingleNormal(eos, ALARM_ALL_PSEUDO_MIN_THR, eos_num_high_thr, "eos_num_high_warning", output);
  Heamo_AlarmCheckSingleNormal(baso, ALARM_ALL_PSEUDO_MIN_THR, baso_num_high_thr, "baso_num_high_warning", output);
  Heamo_AlarmCheckSingleNormal(nrbc, ALARM_ALL_PSEUDO_MIN_THR, nrbc_num_high_thr, "nrbc_num_high_warning", output);
  const float ig_percentage = ig/wbc;
  Heamo_AlarmCheckSingleNormal(ig_percentage, ALARM_ALL_PSEUDO_MIN_THR, ig_num_upper_thr, "ig_num_upper_warning", output);
  Heamo_AlarmCheckSingleNormal(wbc_agglomeration, ALARM_ALL_PSEUDO_MIN_THR, wbc_aggregation_num_upper_thr,
                               "wbc_aggregation_num_upper_warning", output);

  // RBC

  Heamo_AlarmCheckSingleNormal(rdw_sd, ALARM_ALL_PSEUDO_MIN_THR, rbc_rdw_sd_upper_thr,
                               "rbc_rdw_sd_upper_warning", output, true);
  Heamo_AlarmCheckSingleNormal(rdw_cv, ALARM_ALL_PSEUDO_MIN_THR, rbc_rdw_cv_upper_thr,
                               "rbc_rdw_cv_upper_warning", output, true);
  Heamo_AlarmCheckSingleNormal(mcv, rbc_mcv_low_thr, ALARM_ALL_PSEUDO_MAX_THR, "rbc_mcv_low_warning", output);
  Heamo_AlarmCheckSingleNormal(mcv, ALARM_ALL_PSEUDO_MIN_THR, rbc_mcv_high_thr, "rbc_mcv_high_warning", output);
  Heamo_AlarmCheckSingleNormal(rbc, ALARM_ALL_PSEUDO_MIN_THR, rbc_num_high_thr, "rbc_num_high_warning", output);
  Heamo_AlarmCheckSingleNormal(hgb, rbc_hgb_low_thr, ALARM_ALL_PSEUDO_MAX_THR, "rbc_hgb_low_warning", output);
  Heamo_AlarmCheckSingleNormal(mchc, rbc_mchc_low_thr, ALARM_ALL_PSEUDO_MAX_THR, "rbc_mchc_low_warning", output);

  Heamo_AlarmCheckSingleNormal(rbc_agglomeration, ALARM_ALL_PSEUDO_MIN_THR, rbc_aggregation_num_upper,
                               "rbc_aggregation_num_upper_warning", output);
  // 双峰检测
  float peak_nums_rbc = 0;
  std::vector<float> rbc_curve_modify (rbc_curve.begin(), rbc_curve.end());
  DoublePeakCheck(rbc_curve_modify, peak_nums_rbc);
  Heamo_AlarmCheckSingleNormal(peak_nums_rbc, ALARM_ALL_PSEUDO_MIN_THR, rbc_histogram_double_peak_thr, "rbc_histogram_double_peak_warning", output);


  // PLT
  Heamo_AlarmCheckSingleNormal(plt, plt_num_low_thr, ALARM_ALL_PSEUDO_MAX_THR, "plt_num_low_warning", output);
  Heamo_AlarmCheckSingleNormal(plt, ALARM_ALL_PSEUDO_MIN_THR, plt_num_high_thr, "plt_num_high_warning", output);

  Heamo_AlarmCheckSingleNormal(plt_agglomeration, ALARM_ALL_PSEUDO_MIN_THR, plt_aggregation_num_upper_thr,
                               "plt_aggregation_num_upper_warning", output);
  // RET
  Heamo_AlarmCheckSingleNormal(ret, ret_num_high_thr, ALARM_ALL_PSEUDO_MAX_THR, "ret_num_low_warning", output);
  Heamo_AlarmCheckSingleNormal(ret, ALARM_ALL_PSEUDO_MIN_THR, ret_num_low_thr, "ret_num_high_warning", output);



  return 0;
}

// 这些key对应的值无法用25项报告单结果重新计算. 因此需使用下发下来的值
std::vector<std::string> alarm_param_not_in_result{"ig_num_upper_warning","wbc_aggregation_num_upper_warning", "rbc_aggregation_num_upper_warning",
											   "plt_aggregation_num_upper_warning"};

int HeamoAlarmManager::ModifyAlarmResults(const std::vector<std::string>& previous_alarm,const float &wbc, const float &neu,
										  const float &lym, const float &mono, const float &eos,
										  const float &baso, const float &nrbc, const float &ig, const float &wbc_agglomeration,//wbc
										  const std::vector<float> &rbc_curve, const float &rdw_cv, const float &rdw_sd, const float &mcv,
										  const float &rbc, const float &hgb, const float &mchc, const float &rbc_agglomeration,//rbc
										  const float &plt, const std::vector<float> &plt_curve, const float &plt_agglomeration,//plt
										  const float &ret,//ret
										  std::vector<std::string>& output){
  output.clear();
  // 根据下发参数填入无法计算的参数
  for(const auto& alarm_string: previous_alarm){
	for(const auto& iter:alarm_param_not_in_result){
	  if(alarm_string == iter){
		output.push_back(alarm_string);
	  }
	}
  }

  // 计算报警参数
  GetAlarmResults(wbc, neu, lym, mono, eos,
				  baso,nrbc, ig, wbc_agglomeration,//wbc
				  rbc_curve, rdw_cv, rdw_sd, mcv,
				  rbc, hgb, mchc, rbc_agglomeration,//rbc
				  plt, plt_curve, plt_agglomeration,//plt
				  ret,//ret
				  output);
  return 0;
}

