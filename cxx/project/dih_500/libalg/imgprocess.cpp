#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "utils.h"
#include "imgprocess.h"
#include "replace_std_string.h"
#include "algLog.h"
#include "libalgcell.h"
#include "opencv2/imgcodecs.hpp"

std::map<std::string, std::vector<ImageRet_t>> g_image_result;
using namespace std;



#define HSVHLOW 26		//HSV空间H低值(绿色荧光) ori 35
#define HSVHTALL 77		//HSV空间H高值
#define HSVSLOW 43		//HSV空间S低值
#define HSVSTALL 255	//HSV空间S高值
#define HSVVLOW 46		//HSV空间V低值
#define HSVVTALL 255	//HSV空间V高值

#define INCLINE_SEG_START 15 //倾斜红细胞间隔像素个数

/// <summary>
/// 通过RBC转换为HSV图像进行颜色筛选，对应颜色在mask会显示黑色背景上面有白色的前景
/// </summary>
/// <param name="img"></param> RBC图像d
/// <returns></returns>mask图像
void imgfeature_color(const cv::Mat& img, cv::Mat& mask)
{

  cv::Mat imgHSV;
  cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);
  //白细胞的绿色荧光
  /*Scalar lower(35, 43, 46);
  Scalar upper(77, 255, 255);*/
  cv::Scalar lower(HSVHLOW, HSVSLOW, HSVVLOW);
  cv::Scalar upper(HSVHTALL, HSVSTALL, HSVVTALL);
  inRange(imgHSV, lower, upper, mask);

}


/// <summary>
/// 使用迭代器逐像素访问,在颜色筛选后找到有白色像素的图像
/// </summary>
/// <param name="img"></param> hsv处理后的mask
/// <returns></returns>白色前景像素大于1的图像
int Iterator( cv::Mat& img)
{

/*  Mat_<uchar>::iterator it = img.begin<uchar>();
  Mat_<uchar>::iterator itend = img.end<uchar>();
  int CountWhite = 0;
  for (; it != itend; it++)
  {
    if ((*it) == 255)
    {
      CountWhite++;
    }
  }*/
  cv::Mat ret = (img-244)>0;
  int CountWhite = (int) cv::sum(ret)[0];
  if (CountWhite > 1){
//    cout << "total white pixel nums:" << CountWhite << endl;
    return 1;
  }
  return 0;
}



//白细胞四分类荧光场和明场拼接
bool MergeImgPreProcess(std::vector<cv::Mat> &montageImg,
                        const cv::Mat &bright_img, const cv::Mat &fluo_img,
                        std::list<NNetResult> &vecRst) {
  if (bright_img.empty()|| fluo_img.empty())
  {
    std::cout<<"empty img."<<std::endl;
    return false;
  }
  int width = bright_img.cols/2;
  int height = bright_img.rows/2;
  cv::Mat dstimg = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);


  //根据框的坐标将目标裁剪出来
  int copyImgWidth = 0;
  int copyImgHeight = 0;
  int imgHeight = 0;
  unsigned int i = -1;
  cv::Rect img_rect(cv::Size(0, 0), cv::Size(bright_img.cols, bright_img.rows));

  for (const auto& one_result:vecRst)
  {

    if (one_result.box.name != "WBC" && one_result.box.name != "PLA") {
      continue;
    }
    i = i + 1;
    cv::Mat pieceImg = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
    int x1 = one_result.box.left;
    int x2 = one_result.box.right;
    int y1 = one_result.box.top;
    int y2 = one_result.box.bottom;

    cv::Rect temp_rect(cv::Size(x1, y1), cv::Size(x2-x1,y2-y1));
    cv::Rect filter_rect = (temp_rect&img_rect);


    x1 = filter_rect.x;
    y1 = filter_rect.y;
    int w = filter_rect.width;
    int h = filter_rect.height;


    x2 = x1 + w ;
    y2 = y2 + h ;


    cv::Rect rect(x1, y1, w, h);
    //cout << "rect:\n" << rect << endl;
    cv::Mat segmentBright = bright_img(rect);
    cv::Mat segmentFluo = fluo_img(rect);
    /*imshow("荧光图像",segmentFluo);
    waitKey(20);*/

    //通过荧光场白细胞的荧光进行初步筛选
/*    cv::Mat hsvFluo;
    imgfeature_color(segmentFluo, hsvFluo);
//    imshow("hsv",hsvFluo);
//    waitKey(20);
    int ret = Iterator(hsvFluo);
    if (ret == 0)
    {
      std::cout<<"warning, bright img do not find counterparts"<<std::endl;
      continue;
    }*/
    if (copyImgWidth + w * 2 > width)
    {
      copyImgHeight += imgHeight+50;
      copyImgWidth = 0;
    }
    if (copyImgHeight + h > height)
    {
      pieceImg = dstimg.clone();
      montageImg.push_back(pieceImg);
      dstimg = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
      copyImgWidth = 0;
      copyImgHeight = 0;
      imgHeight = 0;
      i = i - 1;
      continue;
    }
    segmentBright.copyTo(dstimg(cv::Rect(copyImgWidth, copyImgHeight, segmentBright.cols, segmentBright.rows)));
    copyImgWidth += w;
    segmentFluo.copyTo(dstimg(cv::Rect(copyImgWidth, copyImgHeight, segmentFluo.cols, segmentFluo.rows)));
    copyImgWidth += w+50;
    if (imgHeight < h)
    {
      imgHeight = h;
    }

  }
  montageImg.push_back(dstimg);
  return true;
}

//预处理倾斜红细胞分割图像
bool RbcInclineSegPreprocess( const cv::Mat& img, const std::list<NNetResult> &vecRst,
                             const int& height, const int& width, const int& interval,
                             std::vector<cv::Mat>& montageImg, int& crop_nums,
                             std::vector<std::vector<cv::Rect>>& paste_position_v_v)
{
  if (img.empty())
  {
    std::cout<<"empty img."<<std::endl;
    return false;
  }

  cv::Mat dstimg = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
  std::vector<cv::Rect> dst_position_v;

  //根据框的坐标将目标裁剪出来
  int copyImgWidth = INCLINE_SEG_START;
  int copyImgHeight = INCLINE_SEG_START;
  int imgHeight = INCLINE_SEG_START;
  unsigned int i = -1;
  cv::Rect img_rect(cv::Size(0, 0), cv::Size(img.cols, img.rows));
  for (const auto& one_result:vecRst)
  {
    i = i+1;
    cv::Mat pieceImg = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
    int x1 = one_result.box.left;
    int x2 = one_result.box.right;
    int y1 = one_result.box.top;
    int y2 = one_result.box.bottom;
// 排除边缘框
    if(x1<0||x2>width||y1<0||y2>height) continue;
    crop_nums +=1;
    cv::Rect temp_rect(cv::Size(x1, y1), cv::Size(x2-x1,y2-y1));
    cv::Rect filter_rect = (temp_rect&img_rect);


    x1 = filter_rect.x;
    y1 = filter_rect.y;
    int w = filter_rect.width;
    int h = filter_rect.height;


    x2 = x1 + w ;
    y2 = y2 + h ;


    cv::Rect rect(x1, y1, w, h);
    cv::Mat segmentBright = img(rect);


    if (copyImgWidth + w * 2 > width)
    {
      copyImgHeight += imgHeight+interval;
      copyImgWidth = INCLINE_SEG_START;
    }
    if (copyImgHeight + h > height)
    {
      pieceImg = dstimg.clone();
      montageImg.push_back(pieceImg);
      paste_position_v_v.push_back(dst_position_v);
      //清空目标图
      dstimg = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
      dst_position_v.clear();
      copyImgWidth = INCLINE_SEG_START;
      copyImgHeight = INCLINE_SEG_START;
      imgHeight = INCLINE_SEG_START;
      i = i - 1;
      continue;
    }
    cv::Rect crop_paste_position(copyImgWidth, copyImgHeight, segmentBright.cols, segmentBright.rows);
    segmentBright.copyTo(dstimg(crop_paste_position));
    dst_position_v.push_back(crop_paste_position);
//    copyImgWidth += w;
    copyImgWidth += w+interval;
    //最高像素位置
    if (imgHeight < h+INCLINE_SEG_START)
    {
      imgHeight = h+INCLINE_SEG_START;
    }

  }
  montageImg.push_back(dstimg);
  paste_position_v_v.push_back(dst_position_v);
  return true;
}

//resize 图像
void ResizeImg(cv::Mat& img, cv::Mat& target_img,
               const cv::Size& target_size,
               const ResizeType& resize_type,
               const cv::Scalar& pad_value,
               cv::InterpolationFlags interpolation) {
    std::cout << " 262 ResizeImg 传入的 缩放类型 " << resize_type << std::endl;
    std::cout << " 262 ResizeImg 传入的 图片宽高 " << img.cols << " " << img.rows << std::endl;
    std::cout << " 262 ResizeImg 传入的 target_size " << target_size.width << " " << target_size.height << std::endl;


    if(cv::Size(img.cols, img.rows)==target_size){
      target_img = img;
      return;
    }
    const int in_w = img.cols;
    const int in_h = img.rows;
    int tar_w = target_size.width;
    int tar_h = target_size.height;
    if (resize_type==LETTERBOX) {
        float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
        int inside_w = int(round(in_w * r));
        int inside_h = int(round(in_h * r));
        int padd_w = tar_w - inside_w;
        int padd_h = tar_h - inside_h;
        cv::resize(img, target_img, cv::Size(inside_w, inside_h), 0, 0, interpolation);
        int top = int(round(padd_h / 2.f - 0.1));
        int bottom = int(round(padd_h - top + 0.1));
        int left = int(round(padd_w / 2.f - 0.1));
        int right = int(round(padd_w - left + 0.1));
        cv::copyMakeBorder(target_img, target_img, top, bottom, left, right,
                           cv::BORDER_CONSTANT, pad_value);
    } else if(resize_type==BOTTOMPAD){
        float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
        int inside_w = int(round(in_w * r));
        int inside_h = int(round(in_h * r));
        int padd_w = tar_w - inside_w;
        int padd_h = tar_h - inside_h;
        cv::resize(img, target_img, cv::Size(inside_w, inside_h), 0, 0, interpolation);
        int bottom = int(round(padd_h + 0.1));
        int right = int(round(padd_w + 0.1));
        cv::copyMakeBorder(target_img, target_img, 0, bottom, 0, right,
                           cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    } else if(resize_type==LEFT_TOP_CROP){
        float r = std::max(float(tar_h) / in_h, float(tar_w) / in_w);
        int inside_w = int(in_w * r);
        int inside_h = int(in_h * r);
        std::cout << " inside_w " << inside_w << " inside_h " << inside_h << std::endl;
        cv::resize(img, target_img, cv::Size(inside_w, inside_h), 0, 0, interpolation);
        int start_x = (target_img.cols-target_size.width)/2;
        int start_y = (target_img.rows-target_size.height)/2;
        target_img = cv::Mat(target_img, cv::Rect(start_x,start_y, target_size.width, target_size.height)).clone();
        cv::flip(target_img, target_img, 0);//由于python端未做上下翻转,此处对原图进行上下翻转
    }
    // else if (resize_type == NORMAL) {
    //     target_img = img;
    // }
    else if (resize_type == RGA_NORMAL) {
    }
    else {
        // float r        = std::max(float(tar_h) / in_h, float(tar_w) / in_w);
        // int   inside_w = int(in_w * r);
        // int   inside_h = int(in_h * r);
        // cv::resize(img, target_img, cv::Size(inside_w, inside_h), 0, 0, interpolation);
        // std::cout << " inside_w " << inside_w << " inside_h " << inside_h << std::endl;
        //target_img = img;
        std::cout << " inside_w " << target_size.width << " inside_h " << target_size.height << std::endl;
        cv::resize(img, target_img, target_size, 0, 0, interpolation);
    }
}


int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size)
{
  im_rect src_rect;
  im_rect dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  size_t img_width = image.cols;
  size_t img_height = image.rows;
  size_t target_width = target_size.width;
  size_t target_height = target_size.height;
  src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
  dst = wrapbuffer_virtualaddr((void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);
  int ret = imcheck(src, dst, src_rect, dst_rect);
  if (IM_STATUS_NOERROR != ret)
  {
    std::cout<<"rga check error! "<<imStrError((IM_STATUS)ret)<<std::endl;
    return -2;
  }
  IM_STATUS STATUS = imresize(src, dst);
  return 0;
}



int resize_rga_output_buf(void **resize_buf, const cv::Mat &img,  const cv::Size &target_size)
{

  rga_buffer_t src;
  rga_buffer_t dst;
  im_rect src_rect;
  im_rect dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  memset(&src, 0, sizeof(src));
  memset(&dst, 0, sizeof(dst));


  //rga ori version
  size_t target_width = target_size.width;
  size_t target_height = target_size.height;
  size_t channel = img.channels();
  *resize_buf          = malloc(target_height * target_width * channel);
  memset(*resize_buf, 0x00, target_height * target_width * channel);



  src = wrapbuffer_virtualaddr((void *)img.data, img.cols, img.rows, RK_FORMAT_RGB_888);
  dst = wrapbuffer_virtualaddr((void *)*resize_buf, target_width , target_height, RK_FORMAT_RGB_888);


  int ret = imcheck(src, dst, src_rect, dst_rect);

  if (IM_STATUS_NOERROR != ret)
  {
    printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
    return -1;
  }

  // 按照模型大小进行原图片的resize
  IM_STATUS STATUS = imresize(src, dst);

  if (STATUS != IM_STATUS_SUCCESS)
  {

    printf("running failed, %s\r\n", imStrError((IM_STATUS)STATUS));
    return -1;
  }
  return 0;
}



void RecoverBox(const cv::Size& origin_size, const cv::Size& processed_size,
                const float& src_x1, const float& src_y1,
                const float& src_x2, const float& src_y2,
                int& x1, int& y1,
                int& x2, int& y2, bool letterbox)  //--img 增加const
{
  float x1_f, x2_f, y1_f, y2_f;
  if (letterbox) {
    const float in_w = origin_size.width;
    const float in_h = origin_size.height;
    float tar_w = processed_size.width;
    float tar_h = processed_size.height;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    float inside_w = in_w * r;
    float inside_h = in_h * r;
    float padd_w = tar_w - inside_w;
    float padd_h = tar_h - inside_h;
    float top = (padd_h / 2 - 0.0000001f);
    float bottom = (padd_h - top + 0.1f);
    float left = (padd_w / 2 - 0.0000001f);
    float right = (padd_w - left + 0.1f);

    x1_f = src_x1 - left;
    x2_f = src_x2 - left;
    y1_f = src_y1 - top;
    y2_f = src_y2 - top;
    x1 = std::round(x1_f / r);
    x2 = std::round(x2_f / r);
    y1 = std::round(y1_f / r);
    y2 = std::round(y2_f / r);
  }
  else {
    x1_f = src_x1 / (float)processed_size.width;
    x2_f = src_x2 / (float)processed_size.width;
    y1_f = src_y1 / (float)processed_size.height;
    y2_f = src_y2 / (float)processed_size.height;
    x1 = x1_f * (float)origin_size.width;
    x2 = x2_f * (float)origin_size.width;
    y1 = y1_f * (float)origin_size.height;
    y2 = y2_f * (float)origin_size.height;
  }
  cv::Rect RectCLX;
  cv::Point c1(x1, y1);
  cv::Point c2(x2, y2);
  RectCLX = cv::Rect(c1, c2) & cv::Rect(0, 0, origin_size.width, origin_size.height);
  x1 = RectCLX.x;
  y1 = RectCLX.y;
  x2 = RectCLX.x+RectCLX.width;
  y2 = RectCLX.y+RectCLX.height;
}

void RecoverSeg(const cv::Size& origin_size, const cv::Size& processed_size,
                const cv::Mat& src, cv::Mat& dst){
  const int in_w = origin_size.width;
  const int in_h = origin_size.height;
  int tar_w = processed_size.width;
  int tar_h = processed_size.height;
  float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
  int inside_w = int(round(in_w * r));
  int inside_h = int(round(in_h * r));
  int padd_w = tar_w - inside_w;
  int padd_h = tar_h - inside_h;
  cv::Point left_top(padd_w, int(round(padd_h / 2.f - 0.1)));
  cv::Point right_bottom(padd_w+tar_w, int(round(padd_h / 2.f - 0.1)+tar_h));

  dst = cv::Mat(src, cv::Rect(left_top, right_bottom)&cv::Rect(cv::Point(0,0),cv::Point(src.cols, src.rows)));
  cv::resize(dst, dst, origin_size, 0,0, cv::InterpolationFlags::INTER_CUBIC);  // 仅使用双线形插值
}

void MergeBrightFluoImg(const cv::Mat& img_brightness, const cv::Mat& img_fluorescence,
                        const float& fusion_rate, cv::Mat& img_target){
  cv::addWeighted(img_brightness, fusion_rate, img_fluorescence, 1-fusion_rate, 0, img_target);
}


//将数组中的点组织成成对的点
void ConstructPoint(const std::vector<float>& xy_v, std::vector<cv::Point>& point){
  if(xy_v.size()%2!=0) return;
  int point_num = xy_v.size()/2;
  for(int i =0; i<point_num; ++i){
    point.emplace_back(xy_v[i*2], xy_v[i*2+1]);
  }
}

std::vector<std::vector<char>>       g_img_name_vec;
std::vector<std::vector<unsigned char>> g_img_buff_vec;

//绘制box
#include "timecnt.h"
void DrawBox(cv::Mat& img, const std::vector<NNetResult>& detect_result_v,
             const bool& draw_name, const float& font_scale, const int& thickness,
             const cv::Scalar& box_color, const cv::Scalar& font_color){
    cv::Mat save_dih_img = img.clone();
    //绘制rect box
  for(const auto& one_result:detect_result_v){

    if(!one_result.write_rect_box){
      continue;
    }


    //std::cout << "496 DrawBox name : " << one_result.box.name << std::endl;
    auto img_item = g_image_result.find(one_result.box.name);
    if (img_item != g_image_result.end()){
      //已经有的情况下
      std::vector<ImageRet_t>&  img_ret= img_item->second;
      if (img_ret.size()>=10){
        //大于10个,什么都不做
      }
      else{
          //名字
          std::vector<char> name_vec(one_result.box.name.begin(),one_result.box.name.end());
          g_img_name_vec.push_back(std::move(name_vec));
          ImageRet_t signal_ret;
          signal_ret.image_name.assign(g_img_name_vec[g_img_name_vec.size() - 1].begin(), g_img_name_vec[g_img_name_vec.size() - 1].end());
          // 小于10个往里面添加
          //图片
          cv::Mat cropped_image =
              save_dih_img(cv::Range(one_result.box.top, one_result.box.bottom), cv::Range(one_result.box.left, one_result.box.right)).clone();
          std::vector<unsigned char> buff;
          svpng(buff, cropped_image.cols, cropped_image.rows, cropped_image.data, 0);
          g_img_buff_vec.push_back(buff);
          signal_ret.base64_buff.assign(g_img_buff_vec[g_img_buff_vec.size() - 1].begin(), g_img_buff_vec[g_img_buff_vec.size() - 1].end());
          //存储
          img_ret.push_back(std::move(signal_ret));
      }
    }
    else{
        // 新增的情况下
        // 名字
        ImageRet_t        img_ret;
        std::vector<char> name_vec(one_result.box.name.begin(), one_result.box.name.end());
        g_img_name_vec.push_back(std::move(name_vec));
        img_ret.image_name.assign(g_img_name_vec[g_img_name_vec.size() - 1].begin(), g_img_name_vec[g_img_name_vec.size() - 1].end());
        // 图片数据
        cv::Mat cropped_image =
            save_dih_img(cv::Range(one_result.box.top, one_result.box.bottom), cv::Range(one_result.box.left, one_result.box.right)).clone();
        std::vector<unsigned char> buff;
        svpng(buff, cropped_image.cols, cropped_image.rows, cropped_image.data, 0);
        g_img_buff_vec.push_back(buff);
        img_ret.base64_buff.assign(g_img_buff_vec[g_img_buff_vec.size() - 1].begin(), g_img_buff_vec[g_img_buff_vec.size() - 1].end());
        // 存储
        std::vector<ImageRet_t> image_info_vec;
        image_info_vec.push_back(std::move(img_ret));
        std::string _cellName = one_result.box.name;
        g_image_result.insert(std::pair<std::string, std::vector<ImageRet_t>>(std::move(_cellName), std::move(image_info_vec)));
    }
    cv::rectangle(img,cv::Point(one_result.box.left, one_result.box.top),
                  cv::Point(one_result.box.right,one_result.box.bottom),
                  box_color, thickness);
    int put_text_left = (int)(one_result.box.left);
    int put_text_top = (int)(one_result.box.bottom);
    std::string box_message;
    if(draw_name){
      std::stringstream  ss;
      ss<<one_result.box.name<<": "<<one_result.box.prop;
      //box_message = one_result.name+(": ") +std::to_string(one_result.prop);
      box_message = ss.str();
    }
    cv::putText(img, box_message, cv::Point(put_text_left, put_text_top), cv::FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness);
  }


  //绘制polygon
  for(const auto& iter:detect_result_v){
    std::vector<cv::Point> poly_cv_point_v;
    ConstructPoint(iter.polygon_v, poly_cv_point_v);
    cv::polylines(img, poly_cv_point_v,true,box_color);
  }
}




//绘制概率
void DrawCategory(cv::Mat& img, const std::vector<NNetResult>& detect_result_v,
                  const float& font_scale, const int& thickness){
  for(const auto& one_result:detect_result_v){
    //一列绘制20个结果
    int cols_nums = 20;
    for(int i=0; i<one_result.category_v.size(); ++i){
      int horizon_position = i/cols_nums+1;
      int vertical_position = i%cols_nums+1;
      int put_text_left = (int)(img.cols*(1.f/cols_nums)*horizon_position);
      int put_text_top = (int)(img.rows*(1.f/cols_nums)*vertical_position);
      std::cout<<"draw category: "<< i <<" "<<one_result.category_v[i]<<std::endl;

      std::stringstream  ss;
      ss<<i<<": "<<one_result.category_v[i];
//      cv::putText(img,std::to_string(i)+ ": "+std::to_string(one_result.category_v[i]),cv::Point(put_text_left, put_text_top),
//                  cv::FONT_HERSHEY_SIMPLEX,font_scale, cv::Scalar(255,0,0), thickness);
      cv::putText(img,ss.str(),cv::Point(put_text_left, put_text_top),
                  cv::FONT_HERSHEY_SIMPLEX,font_scale, cv::Scalar(255,0,0), thickness);
    }


  }
}

void DrawSegmentation(cv::Mat& img, const std::vector<NNetResult>& detect_result_v){
  for(const auto& one_result:detect_result_v){
    for(int i=0; i<one_result.seg_v.size(); ++i){
      std::vector<cv::Mat> channels;
      cv::Mat pred_mask(one_result.seg_v[0]);//暂时仅支持绘制单个类别的分割结果,后续增加不同类别对颜色的映射
      cv::Mat img_b;
      cv::Mat img_g;
      cv::Mat img_r;
      img = img/2;
      split(img, channels);//分离色彩通道
      img_r = channels.at(0);
      img_g = channels.at(1);
      img_b = channels.at(2);

      img_r = img_r + pred_mask*255/7;
      //      cv::Mat temp_img;
      cv::merge(std::vector<cv::Mat>{img_r, img_g, img_b}, img);

    }

  }

}


bool DrawMidResult(cv::Mat* img, const int&img_height, const int& img_width,
                   const std::vector<NNetResult>& detect_result_v, cv::Mat& img_mat,
                   const bool& draw_name, const float& font_scale, const int& thickness,
                   const cv::Scalar& box_color, const cv::Scalar& font_color){
//  img_mat = cv::Mat(img_height, img_width, CV_8UC3, img).clone();
  img_mat = img->clone();
  DrawBox(img_mat, detect_result_v, draw_name, font_scale, thickness, box_color, font_color);
  DrawCategory(img_mat, detect_result_v, font_scale, thickness);
  // 新加
  //cv::flip(img_mat, img_mat, -1);
  DrawSegmentation(img_mat, detect_result_v);
  return true;
}

/*!
 * 根据box位置进行筛选
 * @param vec_rst
 * @param img_width
 * @param img_height
 * @param box_position_confine
 * @param vec_thr_rst
 */
void ThrBosAccordPosition(const std::list<NNetResult_t> &vec_rst, const int &img_width,
                          const int &img_height, const float& box_position_confine,
                          std::vector<NNetResult_t>& vec_thr_rst){
  cv::Rect img_rect(cv::Size(0, 0), cv::Size(img_width, img_height));
  int x1 = box_position_confine * (float) img_width;
  int y1 = box_position_confine * (float) img_height;
  int x2 = x1 + (1-2*box_position_confine) * (float) img_width;
  int y2 = y1 + (1-2*box_position_confine) * (float) img_height;
  for (const auto &i: vec_rst) {
    cv::Rect temp_rect(cv::Size(i.box.left, i.box.top),
                       cv::Size(i.box.right - i.box.left, i.box.bottom - i.box.top));
    cv::Rect filter_rect = (temp_rect & img_rect);
    int x = (i.box.left + i.box.right)/2;
    int y = (i.box.bottom + i.box.top)/2;

    //位于非边缘的细胞才用于计算体积
    if ((x1-x)*(x2-x)<0 && (y1-y)*(y2-y)<0) {
      vec_thr_rst.push_back(i);
    }

  }
}

//暂时计算细胞面积
void
CountVolumeParameter(std::vector<float> &ctx_volume_v,  const std::list<NNetResult_t> &vec_rst, const int &img_width,
                     const int &img_height, const float& box_position_confine) {
  std::vector<NNetResult_t> vec_thr_rst;
  ThrBosAccordPosition(vec_rst, img_width, img_height, box_position_confine, vec_thr_rst);
  for(const auto& threshed_box:vec_thr_rst){
    ctx_volume_v.push_back((float)((threshed_box.box.right-threshed_box.box.left)*(threshed_box.box.bottom-threshed_box.box.top)));
  }
}


void crop_img(cv::Mat img, std::vector<cv::Rect>& windows_rect, std::vector<cv::Mat>& windows_img)
{

    int   window_w_      = 800;
    int   window_h_      = 800;
    float overlap_ratio_ = 0.2;
    int   step_x         = int(window_w_ * (1 - overlap_ratio_));
    int   step_y         = int(window_h_ * (1 - overlap_ratio_));

    int x = 0;
    int y = 0;
    while (x < img.cols) {
        y          = 0;
        int x_flag = 0;
        while (y < img.rows) {
            int y_flag = 0;
            int x2     = x + window_w_;
            int y2     = y + window_h_;
            if (x2 > img.cols) {
                x_flag = 1;
                x2     = img.cols;
                x      = x2 - window_w_;
            }
            if (y2 > img.rows) {
                y_flag = 1;
                y2     = img.rows;
                y      = y2 - window_h_;
            }
            cv::Rect mask_rect = cv::Rect(cv::Point(x, y), cv::Point(x2, y2));
            cv::Mat  mask_img  = img(mask_rect).clone();
            windows_img.emplace_back(mask_img);
            windows_rect.emplace_back(mask_rect);
            y += step_y;
            if (y_flag) {
                break;
            }
        }
        x += step_x;
        if (x_flag) {
            break;
        }
    }
}


