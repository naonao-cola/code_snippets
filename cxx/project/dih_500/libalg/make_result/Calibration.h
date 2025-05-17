//
// Created by y on 24-4-2.
//
#ifndef TEST_LIBALG_CALIBRATION_H
#define TEST_LIBALG_CALIBRATION_H
# include <vector>

#include "utils.h"
//#include "DihLog.h"
#include "algLog.h"
namespace ALG_DEPLOY {
namespace CALIBRATION{
template <typename T>
class Calibration{
 public:
  Calibration()=default;
  ~Calibration()=default;
  /*!
   * 设置当前相机拍摄图像高对应的物理尺寸,单位um
   * @param current_img_height_um
   */
  int SetPhysicalSizeCalibration(const T& img_h, const T& img_w, const T& img_h_um);


  void GetVolCalibrationResult(T src, T& dst);


  void GetAreaCalibrationResult(std::vector<T> src, std::vector<T>& dst);


 private:
   float view_area;
   float pixel_um;

};

/*template <typename T>
int GetViewArea(const T& img_h, const T& img_w, const T& img_h_um,
                T& view_area, T& pixel_um){
  ALGLogInfo<<"camera img info "<<img_h<<" "<<img_w<<" "<<img_h_um;
  if(img_h*img_w*img_h_um==0){//任意值为0,均不正常
    ALGLogError<<"get error camera img info";
    return -1;
  }
  pixel_um = img_h_um/img_h;
  view_area = pixel_um* img_h*pixel_um*img_w;
  return 0;
}*/

/*!
 * 获取相机 物理尺寸-像素 校准系数
 * @param calibrated_area_percentage
 */
template <typename T>
int Calibration<T>::SetPhysicalSizeCalibration(const T& img_h, const T& img_w, const T& img_h_um){
  ALGLogInfo<<"Camera img info "<<img_h<<" "<<img_w<<" "<<img_h_um;
  if(img_h*img_w*img_h_um==0){//任意值为0,均不正常
    ALGLogError<<"Get error camera img info";
    return -1;
  }
  this->pixel_um = img_h_um/img_h;
  this->view_area = pixel_um* img_h*pixel_um*img_w;


  ALGLogInfo<<"Pixel um, view_area "<<this->pixel_um<<" "<<this->view_area;

  return 0;

}





/*!
 * 获取视野体积
 * @tparam T
 * @param src
 * @param dst
 */
template <typename T>
void Calibration<T>::GetVolCalibrationResult(T src, T& dst){
  dst = src * view_area/1e+6;
}

/*!
 * 将像素个数校正为物理面积
 * @tparam T
 * @param src
 * @param dst
 */
template <typename T>
void Calibration<T>::GetAreaCalibrationResult(std::vector<T> src, std::vector<T>& dst){
  float mean_plt_pixel = PseudoMean(src.begin(), src.end());
  ALGLogInfo<<"mean_plt_pixel "<<mean_plt_pixel;
  dst.clear();
  for(const auto& iter:src){
    dst.push_back(iter*pixel_um*pixel_um);
  }
}



}
}

#endif  // TEST_LIBALG_CALIBRATION_H
