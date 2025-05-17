//
// Created by y on 23-8-4.
//

#ifndef RKNN_ALG_DEMO_DETECTHUMAN_H
#define RKNN_ALG_DEMO_DETECTHUMAN_H
//#include "DetectType.h"
#include "rknn_api.h"

#include "DetectType.h"
#include "imgprocess.h"
#include "replace_std_string.h"
#include "ParamFitting.h"

namespace ALG_LOCAL {
class DetectHuman: public DetectType {
 public:
  DetectHuman() {};

  ~DetectHuman();

  bool Init(const InitParam& init_param) override;
  //为模拟真实图片获取情况,图片以
  bool Forward(ForwardParam& forward_param) override;

  void GetStatisticResult() override ;
  bool RunAssistFunction() override;
  bool CalculateMcv(float& mcv);
  bool InitOneAlg(NNetCtxID_t *net_ctx, const std::vector<std::string>& model_paths_v,
                  const std::vector<float>& float_param_v);
  //rbc
  bool ForwardRbc(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                  const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);

  bool ForwardPlt(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                  const int& img_height, const  int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);

  bool ForwardWbc(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                  const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  //增加疟原虫检测
  bool ForwardPLA(std::vector<NNetResult> &detect_result_v,
                  cv::Mat *img_brightness, cv::Mat *img_fluorescence,
                  const int &img_height, const int &img_width,
                  std::vector<cv::Mat> &mat_bright_result_v,
                  std::vector<cv::Mat> &mat_fluo_result_v);

  bool ForwardWbc4(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                  const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);

  bool ForwardBaso(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                   const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardGradClarity(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
          const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);

  bool ForwardBasoClarity(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                          const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardRet(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                          const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);

  bool ForwardWbc4Single(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                     const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardWbcSingle(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                         const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardRbcVolume(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                     const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardAiClarity(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                        const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardPltVolume(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                      const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardAiClarityFarNear(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                        const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardMilkGerm(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                        const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardMilkCell(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                       const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardRbcVolumeSphericalBox(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                           const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardRbcVolumeSphericalSeg(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                    const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardSphericalFocal(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                          const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardClassificationCustom(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                                                const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  bool ForwardMILKBOARDLINE(std::vector<NNetResult>& detect_result_v, cv::Mat * img_brightness, cv::Mat * img_fluorescence,
                               const int& img_height, const int& img_width, std::vector<cv::Mat>& mat_bright_result_v, std::vector<cv::Mat>& mat_fluo_result_v);
  void TestGetRbcAndPltResult();
  void TestGetWbcResult();
  void MapGroupId();
  //将算法id映射至相关的模型id
  void MapModId();
 private:
  //rknn推理配置指针

  NNetCtxID_t net_ctx = nullptr;

  //图像高宽
//  std::vector<RstItem> result;
  bool test_get_rbc_and_plt_result = false;//是否进行rbc,plt计数
  bool test_get_wbc_result = false;//是否进行wbc计数
  std::map<DetectTypeName, NNetGroup> group_idx_to_id_m;
  std::map<AlgType, NNetModID> alg_id_to_mod_id_m;

  NNetGroup group_id = NNET_GROUP_HUMAN;
  AlgSampleContex sample_contex;
  std::vector<RstItem> result_count_v;

  ALG_DEPLOY::SphericalReagentFitting spherical_reagent_fitting;

};
}






#endif //RKNN_ALG_DEMO_DETECTHUMAN_H
