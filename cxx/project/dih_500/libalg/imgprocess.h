#ifndef _IMGPROCESS_H_
#define _IMGPROCESS_H_
#include <opencv2/opencv.hpp>

#include "neural_network.h"
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
 // ��ϸ���ķ���ӫ�ⳡ������ƴ��
 bool MergeImgPreProcess(std::vector<cv::Mat> &montageImg,
                         const cv::Mat &bright_img, const cv::Mat &fluo_img,
                         std::list<NNetResult> &vecRst);

 bool RbcInclineSegPreprocess(
     const cv::Mat &img, const std::list<NNetResult> &vecRst, const int &height,
     const int &width, const int &interval, std::vector<cv::Mat> &montageImg,
     int &crop_nums, std::vector<std::vector<cv::Rect>> &paste_position_v_v);

 void ResizeImg(cv::Mat &img, cv::Mat &target_img,
                const cv::Size &target_size = cv::Size(640, 640),
                const ResizeType &resize_type = NORMAL,
                const cv::Scalar &pad_value = cv::Scalar(114, 114, 114),
                cv::InterpolationFlags interpolation = cv::INTER_LINEAR);
 /*!
  * 将box恢复至原图大小
  * @param origin_size       图像处理后尺寸
  * @param processed_size    图像原图尺寸
  * @param src_x1            box的left
  * @param src_y1            box的top
  * @param src_x2            box的right
  * @param src_y2            box的bottom
  * @param x1                恢复后box的left
  * @param y1                恢复后box的top
  * @param x2                恢复后box的right
  * @param y2                恢复后box的bottom
  * @param letterbox         图像处理时是否采用letterbox
  */
 void RecoverBox(const cv::Size &origin_size, const cv::Size &processed_size,
                 const float &src_x1, const float &src_y1, const float &src_x2,
                 const float &src_y2, int &x1, int &y1, int &x2, int &y2,
                 bool letterbox); //--img ����const

 void MergeBrightFluoImg(const cv::Mat &img_brightness,
                         const cv::Mat &img_fluorescence,
                         const float &fusion_rate, cv::Mat &img_target);
 // �����м���
 bool DrawMidResult(cv::Mat *img, const int &img_height, const int &img_width,
                    const std::vector<NNetResult> &detect_result_v,
                    cv::Mat &img_mat, const bool &draw_name,
                    const float &font_scale, const int &thickness,
                    const cv::Scalar &box_color = cv::Scalar(0, 0, 255),
                    const cv::Scalar &font_color = cv::Scalar(255, 0, 0));

 // rga resize
 int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image,
                cv::Mat &resized_image, const cv::Size &target_size);

 int resize_rga_output_buf(void **resize_buf, const cv::Mat &img,
                           const cv::Size &target_size);

 /*!
  * ɸѡ�Ǳ߽紦�Ŀ�
  * @param vec_rst                   �����
  * @param img_width
  * @param img_height
  * @param box_position_confine
  * ����߽�ðٷֱ����ڵĿ��޳�
  * @param vec_thr_rst               �����
  */
 void ThrBosAccordPosition(const std::list<NNetResult_t> &vec_rst,
                           const int &img_width, const int &img_height,
                           const float &box_position_confine,
                           std::vector<NNetResult_t> &vec_thr_rst);

 /*!
  * ����ϸ�Ŀ�����
  * @param ctx_volume_v ɸѡ��ÿ��������
  * @param vec_thr_rst  ɸѡ���
  * @param vec_rst      �����
  * @param img_width    ԭʼͼ���
  * @param img_height   ԭʼͼ���
  * @param box_ratio_thr
  * @param box_position_confine
  */
 void CountVolumeParameter(std::vector<float> &ctx_volume_v,
                           const std::list<NNetResult_t> &vec_rst,
                           const int &img_width, const int &img_height,
                           const float &box_position_confine = 0.03);

 /*!
  * �ָ��ָ�����
  * @param origin_size       ԭʼͼ���С
  * @param processed_size    �����ͼ���С
  * @param src               �����ͼ��
  * @param dst               �ָ����
  */
 void RecoverSeg(const cv::Size &origin_size, const cv::Size &processed_size,
                 const cv::Mat &src, cv::Mat &dst);


 void crop_img(cv::Mat img, std::vector<cv::Rect>& windows_rect, std::vector<cv::Mat>& windows_img);

#endif   // _IMGPROCESS_H_
