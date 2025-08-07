//
// Created by y on 24-12-9.
//
#include "project_utils.h"
#include "algLog.h"
/*!
 * 对模型box输出的conf依据配置进行筛选
 * @param src
 * @param conf_v
 * @param dst
 * @return
 */
int CountBoxCategoryConf(const std::list<NNetResult_t>& src, const std::vector<float>& conf_v,
                         std::list<NNetResult_t>& dst) {
  for(const auto& one_result: src){
    if(one_result.write_rect_box){
      if(one_result.box.label>=conf_v.size()){
        ALGLogError<<"Model output category is greater than max. cur, max "<<one_result.box.label<<" "<<conf_v.size();
        return -1;
      }
      int cur_pred_category = one_result.box.label;
      if(one_result.box.prop >= conf_v[cur_pred_category]){
        dst.push_back(one_result);
      }

    }
  }
  return 0;

}


/*!
 * 将明场转换为灰度图,替换荧光场图像的b通道
 * @param img_bri
 * @param img_flu
 * @param[out] output
 * @return
 */
int MergePltImg(const cv::Mat& img_bri, const cv::Mat& img_flu, cv::Mat& output){
  if(img_bri.empty()||img_flu.empty()){
    std::cout<<"Emtpy brightness or fluorescence image are given for plt merge."<<std::endl;
    return -1;
  }
  std::vector<cv::Mat>  channels_flu;
  split(img_flu, channels_flu);//分离色彩通道
  cv::Mat img_bri_gray;
  cv::cvtColor(img_bri, img_bri_gray, CV_RGB2GRAY);
  cv::merge(std::vector<cv::Mat>{channels_flu[0],channels_flu[1], img_bri_gray}, output);
  return 0;
}