//
// Created by y on 24-3-28.
//

#ifndef TEST_LIBALG_PARAMFITTING_H
#define TEST_LIBALG_PARAMFITTING_H

#include <vector>

#include "TinnyNetwork.h"

namespace ALG_DEPLOY {

int LocalMeanVolumeLine(const std::vector<float>& vol_v, const int& volume_size,
                        const int& dilate_ratio, const int& down_ratio,
                        const int& max_volume_confine,
                        const int& sample_window_size, const int& mean_kernel_size,
                        std::vector<float>& result_v);
int LocalMaxVolumeLine(const std::vector<float>& vol_v, const int& volume_size,
                       const int& dilate_ratio, const int& down_ratio,
                       const int& max_volume_confine,
                       const int& sample_window_size, const int& mean_kernel_size,
                       std::vector<float>& result_v);
int MediumMeanVolumeLine(const std::vector<float>& vol_v, const int& volume_size,
                        const int& dilate_ratio, const int& down_ratio,
                        const int& max_volume_confine,
                        const int& medium_kernel_size, const int& mean_kernel_size,
                        std::vector<float>& result_v);
class ParamFitting {
 public:
  ParamFitting() = default;
  virtual ~ParamFitting() = default;
  /*!
   * 初始化接口,用于组织参数格式
   * @return
   */
  virtual int Init() = 0;

  /*!
   * 推理接口,用于推理并获取结果
   * @param input_param_v 输入数据
   * @param result_v
   * @return
   */
  virtual int Forward(const std::vector<float>& area_rbc_v, const float& incline_rbc_nums,
                      const float& incline_rbc_region, const std::vector<float>& area_plt_v,
                      const float& relative_line_ratio_to_standard,
                      const std::vector<float> &data_v, const std::vector<float> &coef_v,
                      float& mcv, float& rdw_cv, float& rdw_sd, float& mpv,
                      std::vector<float>& curve_rbc, std::vector<float>& curve_plt,
                      float& hgb, float& rbc, float& ret_, float&plt,
                      float&neu, float&lym, float& mono, float&eos, float& baso) = 0;

  int HgbFitting(const std::vector<float> &data_v, const std::vector<float> &coef_v, float& hgb);


  int AllCellCountFitting(float& rbc, float& ret_, float&plt,
                          float&neu, float&lym, float& mono, float&eos, float& baso);


  int CellCountFitting(const float& src, float& dst);




  PSEUDO_NETWORK::HgbNetwork hgbNetwork;
  PSEUDO_NETWORK::CellCountNetwork cell_count_network;
  int HEAMO_HGB_COEF_NUMS   = 4;
  int HEAMO_HGB_PARAM_NUMS  = 8;

};


class NormalReagentFitting: public ParamFitting{
 public:
  NormalReagentFitting() = default;
  ~NormalReagentFitting() override{};
  /*!
   * 初始化接口,用于初始化参数
   * @return
   */
  int Init() override;

/*!
 * 推理接口,用于推理并获取结果
 * @param area_rbc_v            rbc面积
 * @param incline_rbc_nums_v    倾斜红细胞数量
 * @param incline_rbc_region_v  倾斜红细胞像素总面积
 * @param area_plt_v            plt面积
 * @param mcv[out]
 * @param rdw_cv[out]
 * @param rdw_sd[out]
 * @param mpv[out]
 * @param curve_rbc_v[out]      rbc直方图
 * @param curve_plt_v[out]      plt直方图
 * @return
 */
  int Forward(const std::vector<float>& area_rbc_v, const float& incline_rbc_nums,
              const float& incline_rbc_region, const std::vector<float>& area_plt_v,
              const float& relative_line_ratio_to_standard,
              const std::vector<float> &data_v, const std::vector<float> &coef_v,
              float& mcv, float& rdw_cv, float& rdw_sd, float& mpv,
              std::vector<float>& curve_rbc, std::vector<float>& curve_plt,
              float& hgb, float& rbc, float& ret_, float&plt,
              float&neu, float&lym, float& mono, float&eos, float& baso);
 private:
  int NormalMcvFitting(const std::vector<float>& area_rbc_v, const float& incline_rbc_nums,
                       const float& incline_rbc_region, float& mcv);
  int NormalMpvFitting(const std::vector<float>& area_plt_v, std::vector<float>& plt_volume_v, float& mpv);
  int NormalRdwCvFitting(const std::vector<float>& vol_v, const float& mcv, float& rdw_cv);
  int NormalRdwSdFitting(const std::vector<float>& vol_v, float& rdw_sd);

  int NormalGetRbcVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result);
  int NormalGetPltVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result);


  PSEUDO_NETWORK::NormalMcvNetwork mcv_network;
  PSEUDO_NETWORK::NormalMpvNetwork mpv_network;
  PSEUDO_NETWORK::NormalRdwCvNetwork rdw_cv_network;
  PSEUDO_NETWORK::NormalRdwSdNetwork rdw_sd_network;


  //体积通用设置
  const int NORMAL_VOLUME_SIZE                  = 300;
  const int NORMAL_VOLUME_MAX_AREA              = 30000;

  //rbc设置
  const int NORMAL_VOLUME_RBC_KERNEL_MEDIUM     = 17;
  const int NORMAL_VOLUME_RBC_KERNEL_BLUR       = 23;
  const int NORMAL_VOLUME_RBC_DOWN_RATIO        = 100;

  //plt设置
  const int NORMAL_VOLUME_PLT_DOWN_RATIO        = 100;
  const int NORMAL_VOLUME_PLT_SAMPLE_WIDTH      = 20;
  const int NORMAL_VOLUME_PLT_DILATE_RATIO      = 1000;
  const int NORMAL_VOLUME_PLT_KERNEL_BLUR       = 15;

  //rdw设置
  const float NORMAL_RDW_SD_PERCENTAGE            = 0.2;

};






class SphericalReagentFitting: public ParamFitting{
 public:
  SphericalReagentFitting() = default;
  ~SphericalReagentFitting() override = default;
  /*!
   * 初始化接口,用于初始化参数
   * @return
   */
  int Init() override;

  /*!
 * 推理接口,用于推理并获取结果
 * @param area_rbc_v            rbc面积
 * @param incline_rbc_nums_v    倾斜红细胞数量
 * @param incline_rbc_region_v  倾斜红细胞像素总面积
 * @param area_plt_v            plt面积
 * @param mcv[out]
 * @param rdw_cv[out]
 * @param rdw_sd[out]
 * @param mpv[out]
 * @param curve_rbc_v[out]      rbc直方图
 * @param curve_plt_v[out]      plt直方图
 * @return
   */
  int Forward(const std::vector<float>& area_rbc_v, const float& incline_rbc_nums,
              const float& incline_rbc_region, const std::vector<float>& area_plt_v,
              const float& relative_line_ratio_to_standard,
              const std::vector<float> &data_v, const std::vector<float> &coef_v,
              float& mcv, float& rdw_cv, float& rdw_sd, float& mpv,
              std::vector<float>& curve_rbc, std::vector<float>& curve_plt,
              float& hgb, float& rbc, float& ret_, float&plt,
              float&neu, float&lym, float& mono, float&eos, float& baso);

 public:
  int SphericalMcvFitting(const std::vector<float>& area_rbc_v,
                          std::vector<float>& rbc_volume_v, float& mcv);
  int SphericalMpvFitting(const std::vector<float>& area_plt_v, std::vector<float>& plt_volume_v, float& mpv);
  int SphericalRdwCvFitting(const std::vector<float>& vol_v, const float& mcv, float& rdw_cv);
  int SphericalRdwSdFitting(const std::vector<float>& vol_v, const float& mcv, float& rdw_sd);

  int SphericalGetRbcVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result);
  int SphericalGetPltVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result);


  PSEUDO_NETWORK::SphericalMcvNetwork mcv_network;
  PSEUDO_NETWORK::SphericalMpvNetwork mpv_network;
  PSEUDO_NETWORK::SphericalRdwCvNetwork rdw_cv_network;
  PSEUDO_NETWORK::SphericalPdwCvNetwork pdw_cv_network;
  PSEUDO_NETWORK::SphericalRdwSdNetwork rdw_sd_network;

  float pdw = 0.0;

  private:
  //体积通用设置
  const int SPHERICAL_VOLUME_SIZE                  = 300;
  const int SPHERICAL_VOLUME_MAX_AREA              = 30000;

  //plt设置
  const int SPHERICAL_VOLUME_PLT_DOWN_RATIO        = 100;
  const int SPHERICAL_VOLUME_PLT_SAMPLE_WIDTH      = 31;
  const int SPHERICAL_VOLUME_PLT_DILATE_RATIO      = 1000;
  const int SPHERICAL_VOLUME_PLT_KERNEL_BLUR       = 17;

  //rdw cv设置
  const int SPHERICAL_VOLUME_RDW_CV_KERNEL_MEDIUM  = 1;
  const int SPHERICAL_VOLUME_RDW_CV_KERNEL_BLUR    = 1;
  const float SPHERICAL_RDW_CV_PERCENTAGE          = 0.6286;

  //rdw sd设置
  const int SPHERICAL_VOLUME_RDW_SD_KERNEL_LOCAL   = 5;
  const int SPHERICAL_VOLUME_RDW_SD_KERNEL_BLUR    = 5;
  const float SPHERICAL_RDW_SD_PERCENTAGE          = 0.2;

  //rbc line
  const int SPHERICAL_VOLUME_RBC_LINE_KERNEL_LOCAL   = 9;
  const int SPHERICAL_VOLUME_RBC_LINE_KERNEL_BLUR    = 9;
};


}







#endif  // TEST_LIBALG_PARAMFITTING_H
