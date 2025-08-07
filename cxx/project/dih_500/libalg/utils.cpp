
//
// Created by y on 23-8-11.
//

#include "utils.h"
#include <string.h>
#include <algorithm>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>


#pragma pack(1) // 让编译器做1字节对齐

//code from SaveImageThread
typedef struct tagBITMAP_FILE_HEADER
{
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;
} BITMAP_FILE_HEADER;

//code from SaveImageThread
typedef struct tag_BITMAP_INFO_HEADER
{
    unsigned int biSize;
    unsigned int biWidth;
    unsigned int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    unsigned int biXPelsPerMeter;
    unsigned int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
} BITMAP_INFO_HEADER;


/*!
 * 将字符串中的指定字符串进行替换
 * @param str 需要进行处理的字符串
 * @param old_value 被替换字符串
 * @param new_value 替换为
 * @return
 */
void ReplaceAllDistinct(std::string& str, const std::string& old_value, const std::string& new_value){
  for(std::string::size_type   pos(0);   pos!=std::string::npos;   pos+=new_value.length()) {
    if(   (pos=str.find(old_value,pos))!=std::string::npos   )
      str.replace(pos,old_value.length(),new_value);
    else   break;
  }
}


//将cv::mat类型转为 unsigned char * 型
void MatToData(const cv::Mat &srcImg, unsigned char*data)
{
    int nFlag = srcImg.channels() * 8;//一个像素的bits
    int nHeight = srcImg.rows;
    int nWidth = srcImg.cols;

    int nBytes = nHeight * nWidth * nFlag / 8;//图像总的字节
/*    if (data){
        delete[] data;
    }*/

    data = new unsigned char[nBytes];//new的单位为字节
    memcpy(data, srcImg.data, nBytes);//转化函数,注意Mat的data成员
}

//将cunsigned char*型转为 cv::mat类型
bool DataToMat(void* data, const int& nH, const int& nW, const int& nFlag, cv::Mat& outImg)//nH,nW为BYTE*类型图像的高和宽,nFlag为通道数
{
    if (data == nullptr)
    {
        return false;
    }
    int nByte = nH * nW * nFlag / 8;//字节计算
    int nType = nFlag == 8 ? CV_8UC1 : CV_8UC3;
    outImg = cv::Mat::zeros(nH, nW, nType);
    memcpy(outImg.data, (unsigned char*)data, nByte);
    return true;
}
long get_memory_usage()
{
    long page_size = sysconf (_SC_PAGESIZE);
    long num_pages = sysconf (_SC_PHYS_PAGES);
    long mem = (num_pages/1024) * (page_size/1024);
    long long free_pages = sysconf (_SC_AVPHYS_PAGES);
    long long free_mem = (free_pages/1024) * (page_size/1024);
//    fprintf(stderr,"Memory %lld MB\\%lld MB.\n", mem, free_mem);

    return (long)(free_mem);
}

void LoadImagePath(std::string imgDirPath,std::vector<std::string> &vimgPath)
{

    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(imgDirPath.c_str())))
    {
        std::cout<<"Folder doesn't Exist! "<<imgDirPath<<std::endl;
        return;
    }else{
      std::cout<<"Read "<<imgDirPath<<" succeed."<<std::endl;
    }

    while((ptr = readdir(pDir))!=0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            vimgPath.push_back(imgDirPath + "/" + ptr->d_name);
        }
    }
    sort(vimgPath.begin(),vimgPath.end());

    closedir(pDir);
}
//main code from SaveImageThread

struct ImageBuf
{
    unsigned char* rgbBuf;
    int bufLen;
    int width;
    int height;
    int bitCount;
};

void SaveImage(const std::string& save_path, const cv::Mat& img)
{
    cv::Mat dst;
    cv::flip(img, dst, 0);
    int n_bytes = img.rows*img.cols*img.channels();

    ImageBuf buf;
    buf.rgbBuf = dst.data;
    buf.bufLen = n_bytes;
    buf.height = dst.rows;
    buf.width = dst.cols;

    BITMAP_FILE_HEADER	stBfh = { 0 };
    BITMAP_INFO_HEADER  stBih = { 0 };
    unsigned long				dwBytesRead = 0;
    FILE* file;

    stBfh.bfType = (unsigned short)'M' << 8 | 'B';			 //定义文件类型
    stBfh.bfOffBits = sizeof(BITMAP_FILE_HEADER) + sizeof(BITMAP_INFO_HEADER);
    stBfh.bfSize = stBfh.bfOffBits + buf.bufLen; //文件大小

    stBih.biSize = sizeof(BITMAP_INFO_HEADER);
    stBih.biWidth = buf.width;
    stBih.biHeight = buf.height;
    stBih.biPlanes = 1;
    stBih.biBitCount = 24;
    stBih.biCompression = 0L;
    stBih.biSizeImage = 0;
    stBih.biXPelsPerMeter = 0;
    stBih.biYPelsPerMeter = 0;
    stBih.biClrUsed = 0;
    stBih.biClrImportant = 0;

    unsigned long dwBitmapInfoHeader =(unsigned long)40UL;

    file = fopen(save_path.c_str(), "wb");
    if (file)
    {
        fwrite(&stBfh, sizeof(BITMAP_FILE_HEADER), 1, file);
        fwrite(&stBih, sizeof(BITMAP_INFO_HEADER), 1, file);
        fwrite(buf.rgbBuf, buf.bufLen, 1, file);
        fclose(file);
    }
}
void SaveImageBuff(const cv::Mat& img, std::vector<unsigned char>& memory_buffer){
    cv::Mat dst;
    cv::flip(img, dst, 0);
    int n_bytes = img.rows * img.cols * img.channels();

    ImageBuf buf;
    buf.rgbBuf = dst.data;
    buf.bufLen = n_bytes;
    buf.height = dst.rows;
    buf.width  = dst.cols;

    BITMAP_FILE_HEADER stBfh       = {0};
    BITMAP_INFO_HEADER stBih       = {0};
    unsigned long      dwBytesRead = 0;
    FILE*              file;

    stBfh.bfType    = (unsigned short)'M' << 8 | 'B';   // 定义文件类型
    stBfh.bfOffBits = sizeof(BITMAP_FILE_HEADER) + sizeof(BITMAP_INFO_HEADER);
    stBfh.bfSize    = stBfh.bfOffBits + buf.bufLen;   // 文件大小

    stBih.biSize          = sizeof(BITMAP_INFO_HEADER);
    stBih.biWidth         = buf.width;
    stBih.biHeight        = buf.height;
    stBih.biPlanes        = 1;
    stBih.biBitCount      = 24;
    stBih.biCompression   = 0L;
    stBih.biSizeImage     = 0;
    stBih.biXPelsPerMeter = 0;
    stBih.biYPelsPerMeter = 0;
    stBih.biClrUsed       = 0;
    stBih.biClrImportant  = 0;

    unsigned long dwBitmapInfoHeader = (unsigned long)40UL;

    memory_buffer.clear();
    memory_buffer.insert(
        memory_buffer.end(), reinterpret_cast<unsigned char*>(&stBfh), reinterpret_cast<unsigned char*>(&stBfh) + sizeof(BITMAP_FILE_HEADER));
    memory_buffer.insert(
        memory_buffer.end(), reinterpret_cast<unsigned char*>(&stBih), reinterpret_cast<unsigned char*>(&stBih) + sizeof(BITMAP_INFO_HEADER));
    memory_buffer.insert(memory_buffer.end(), buf.rgbBuf, buf.rgbBuf + buf.bufLen);
}

//中值滤波
void MedianFilter(const std::vector<float>& src , std::vector<float>& dst, int half_wind_size)
{
    dst = std::vector<float>(src.size(), 0);
    // 数据长度
    int data_len = src.size();
    if(src.empty()){
        return ;
    }

    if (data_len <= half_wind_size * 2)
    {
        int mid_idx = int(src.size()/2) ;
        dst[0] = src[mid_idx];
        return ;
    }
    // 中值滤波
    int len = 2 * half_wind_size + 1;
    float* buff = NULL;
    int isize = (2*half_wind_size+1) * sizeof(float);
    buff = (float*)malloc(isize);
    float medain_value = 0.0;

    int m = 0;
    int n = 0;
    int tmp = 0;
    int index = 0;
    for (int i = half_wind_size; i < (data_len - half_wind_size); i++)
    {
        memset(buff, 0x00, isize);
        index = 0;
        for (int j = (i - half_wind_size); j <= (i + half_wind_size); j++)
        {
            // 数据存入
            buff[index] = src[j];
            index++;
        }

        // 冒泡排序
        m = 0;
        n = 0;
        tmp = 0;
        medain_value = 0;

        for (m = 0; m < len - 1; m++)
        {
            for (n = 0; n < (len - m - 1); n++)
            {
                if (buff[n] > buff[n + 1])
                {
                    tmp = buff[n];
                    buff[n] = buff[n + 1];
                    buff[n + 1] = tmp;
                }
            }
        }
        medain_value = buff[half_wind_size];
        dst[i] = medain_value;
    }

    // 释放内存
    if (NULL != buff)
    {
        free(buff);
        buff = NULL;
    }
}

//将超过max_thr的值设为max_thr,
void MatClip(const cv::Mat& src, const int& max_thr, cv::Mat& dst){
    //{88, 30001}->{0,30000}
    cv::Mat vol_max_threshed;
    cv::threshold(src, vol_max_threshed, max_thr, max_thr,cv::THRESH_BINARY);

    //{88, 30001}->{30000, 0}
    cv::Mat vol_under_thresh;
    cv::threshold(src, vol_under_thresh, max_thr, max_thr, cv::THRESH_BINARY_INV);

    //{30000, 0}->{1, 0}
    vol_under_thresh = vol_under_thresh>0;
    vol_under_thresh.convertTo(vol_under_thresh, CV_32F, 1.0/255);

    //{1, 0}->{88, 0}
    cv::multiply(vol_under_thresh, src,vol_under_thresh);

    //{88, 30000}
    dst = vol_under_thresh + vol_max_threshed;

}



void GetRbcVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result){


  //避免对空mat进行操作
  if(vol_v.empty()){
    result = std::vector<float>(VOLUME_SIZE, 0);
    return;
  }
  auto temp_v = std::vector<float>(VOLUME_SIZE, 0);
  std::vector<float> vol_v_copy(vol_v);
  cv::Mat vol{vol_v_copy};

  //排除异常大的值
  MatClip(vol, VOLUME_MAX_AREA, vol);

  //面积缩放
  vol = vol/VOLUME_RBC_DOWN_RATIO;

  //计数
  std::vector<int> v = vol.reshape(1, 1);
  for(int i=0;i<vol_v.size();++i){
//    int idx = (v[i]==0)*1+v[i]-1;
    int idx = v[i];
    temp_v[idx] +=1;
  }

  //中值滤波
  std::vector<float> v_median;
  MedianFilter(temp_v, v_median, int(VOLUME_RBC_KERNEL_MEDIUM/2));

  //均值滤波
  cv::Mat temp_mat{v_median};
  cv::blur(temp_mat, temp_mat, cv::Size(VOLUME_RBC_KERNEL_BLUR,VOLUME_RBC_KERNEL_BLUR));
  result = temp_mat.reshape(1,1);


}

/*!
 * 计算Plt体积曲线
 * @param vol_v plt体积数据
 * @param result 曲线数据
 * @param volume_down_ratio 体积缩小比例
 */

void GetPltVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result){

  //避免对空mat进行操作
  if(vol_v.empty()){
    result = std::vector<float>(VOLUME_SIZE, 0);
    return;
  }

  cv::Mat vol{vol_v};
  vol = vol * VOLUME_PLT_DILATE_RATIO;
  //排除异常大的值
  MatClip(vol, VOLUME_MAX_AREA, vol);

  //面积缩放
  vol = vol/VOLUME_PLT_DOWN_RATIO;

  //计数
  auto plt_nums_under_interval = std::vector<float>(VOLUME_SIZE, 0);
  std::vector<int> v = vol.reshape(1, 1);
  for(int i=0;i<vol_v.size();++i){
//    int idx = (v[i]==0)*1+v[i]-1;
    int idx = v[i];
    plt_nums_under_interval[idx] +=1;
  }
  //统计局部最大值
  std::vector<float> local_max_v;
  for(int idx=0;idx<plt_nums_under_interval.size();idx=idx+VOLUME_PLT_SAMPLE_WIDTH){
    float max_value = *std::max_element(plt_nums_under_interval.begin()+idx, plt_nums_under_interval.begin()+idx+VOLUME_PLT_SAMPLE_WIDTH);
    local_max_v.push_back(max_value);
  }

  cv::Mat local_max_mat{local_max_v};
  local_max_mat = local_max_mat.reshape(0, 1);
  cv::resize(local_max_mat, local_max_mat, cv::Size(VOLUME_SIZE, 1), cv::INTER_LINEAR);

  //均值滤波
  cv::blur(local_max_mat, local_max_mat, cv::Size(VOLUME_PLT_KERNEL_BLUR,VOLUME_PLT_KERNEL_BLUR));
  result = local_max_mat.reshape(1,1);

}

void sigmoid( cv::Mat src, cv::Mat& dst) {
  src = -1 * src;
  exp(src, src);
  src = 1 + src;
  dst = 1 / src;
}
//RBC面积拟合体积
void RbcAreaFitting(const std::vector<float>& vol_v, const float& fusion_rate, float& value) {
  if(vol_v.empty()){
    value=0;
    return;
  }
  double sum = std::accumulate(std::begin(vol_v), std::end(vol_v), 0.0);
  double mean =  sum / (int)vol_v.size(); //均值
  value = (float)mean;
  //依据像元融合倍率进行区域缩放
  value = value* (1/fusion_rate)*(1/fusion_rate);
  //以下值由计算得到
  float input = value / 10000;//10000为fusion_rate==1下的正则化值
  cv::Mat conv1_weight{ -1.8741,-1.0094,1.2848,1.7440 };
  conv1_weight = conv1_weight.reshape(0, 1);//1*4
  cv::Mat conv1_bias{ 0.5225, -0.7483, -0.0753, -0.5706 };
  conv1_bias = conv1_bias.reshape(0, 1);//1*4

  cv::Mat conv3_weight{ -1.9995, -0.9613,  1.2691,  1.6077 };
  conv3_weight = conv3_weight.reshape(0, 4);//4*1

  cv::Mat x = input * conv1_weight;
  x = x + conv1_bias;
  sigmoid(x, x);
  x = x * conv3_weight;

//  (int)(*(b.data + b.step[0] * row + b.step[1]* col + channel));
  value  = (float)x.at<double>(0)*100;
}




//plt new
void PltTinyNetwork(const std::vector<float>& input_v, std::vector<float>& all_values){

  cv::Mat conv1_weight{ -2.6802,  0.5613,  0.8142,  0.8959};
  conv1_weight.convertTo(conv1_weight, CV_32F);
  conv1_weight = conv1_weight.reshape(0, 1);//1*4

  cv::Mat conv1_bias{2.6108, -0.4391, -0.8432, -0.5225};
  conv1_bias = conv1_bias.reshape(0, 1);//1*4
  conv1_bias.convertTo(conv1_bias, CV_32F);
  cv::resize(conv1_bias, conv1_bias, cv::Size(4,input_v.size()));


  cv::Mat conv2_weight{-2.4015, -1.9613, -1.8608,  0.1832,
                       0.0777,  0.4762, -0.0383, -0.5789,
                       0.4758,  0.2316,  0.0537, -0.5859,
                       0.9161,  0.0721,  0.3345, -0.1040,
  };
  conv2_weight = conv2_weight.reshape(0, 4);//4*4
  conv2_weight.convertTo(conv2_weight, CV_32F);

  cv::Mat conv2_bias{ -0.5795, -0.2853, -0.1254, -0.1728 };
  conv2_bias = conv2_bias.reshape(0, 1);//1*4
  conv2_bias.convertTo(conv2_bias, CV_32F);
  cv::resize(conv2_bias, conv2_bias, cv::Size(4, input_v.size()));

  cv::Mat conv3_weight{1.8864,  1.3334, 1.1922,  -0.4971};
  conv3_weight = conv3_weight.reshape(0, 4);//4*1
  conv3_weight.convertTo(conv3_weight, CV_32F);


  cv::Mat input{input_v};
  input = input.reshape(0,input_v.size());//n*1

  cv::Mat x = input*conv1_weight;//n*1 1*4 -> n*4

  x = x + conv1_bias;
  sigmoid(x,x);
  x = x*conv2_weight;//n*4 4*4 -> n*4
  x = x+conv2_bias;
  sigmoid(x,x);
  x = x*conv3_weight;//n*4 4*1 -> n*1
  all_values = x.reshape(0,1);
  //value = x.at<float>(input_v.size()-1)*10;

}


void RdwCvTinyNetwork(const std::vector<float>& input_v, std::vector<float>& all_values){

  cv::Mat conv1_weight{ 1.2374, -1.4347, -1.1412,  0.9278};
  conv1_weight.convertTo(conv1_weight, CV_32F);
  conv1_weight = conv1_weight.reshape(0, 1);//1*4

  cv::Mat conv1_bias{-1.0896,  1.2564,  0.1870, -0.7893};
  conv1_bias = conv1_bias.reshape(0, 1);//1*4
  conv1_bias.convertTo(conv1_bias, CV_32F);
  cv::resize(conv1_bias, conv1_bias, cv::Size(4,input_v.size()));


  cv::Mat conv2_weight{-0.8819, -0.2063,  0.5225,  0.8393,
                       0.3081, -0.1656, -1.2742, -1.0265,
                       -0.4213, -0.2292, -0.8812, -0.2549,
                       -0.2708, -0.0092,  0.3043,  0.3115,
  };
  conv2_weight = conv2_weight.reshape(0, 4);//4*4
  conv2_weight.convertTo(conv2_weight, CV_32F);

  cv::Mat conv2_bias{ -0.6152, -1.0898, -0.9369, -0.4562 };
  conv2_bias = conv2_bias.reshape(0, 1);//1*4
  conv2_bias.convertTo(conv2_bias, CV_32F);
  cv::resize(conv2_bias, conv2_bias, cv::Size(4, input_v.size()));

  cv::Mat conv3_weight{0.5814,  0.8361, 1.1918,  1.4228};
  conv3_weight = conv3_weight.reshape(0, 4);//4*1
  conv3_weight.convertTo(conv3_weight, CV_32F);


  cv::Mat input{input_v};
  input = input.reshape(0,input_v.size());//n*1

  cv::Mat x = input*conv1_weight;//n*1 1*4 -> n*4

  x = x + conv1_bias;
  sigmoid(x,x);
  x = x*conv2_weight;//n*4 4*4 -> n*4
  x = x+conv2_bias;
  sigmoid(x,x);
  x = x*conv3_weight;//n*4 4*1 -> n*1
  all_values = x.reshape(0,1);
  //value = x.at<float>(input_v.size()-1)*10;

}



//plt面积拟合体积
void PltAreaFitting(const std::vector<float>& vol_v, const float& fusion_rate,
                           float& value, std::vector<float>& all_values){
  if(vol_v.empty()){
    value=0;
    return;
  }
  // std::cout<<"----------plt ori value--------------"<<std::endl;
  // for(auto i :vol_v){
  //   std::cout<<" "<<i<<",";
  // }
  // std::cout<<std::endl;
//plt old
//  double sum = std::accumulate(std::begin(vol_v), std::end(vol_v), 0.0);
//  double mean =  sum / (int)vol_v.size(); //均值
//  float area_mean = (float)mean;
//  std::cout<<"plt area "<<area_mean<<std::endl;
//  //依据像元融合倍率进行区域缩放
//  area_mean = area_mean * (1/fusion_rate)*(1/fusion_rate);
//  //归一化
//  area_mean = area_mean / 1000;
//  PltTinyNetwork(area_mean,  value);



//plt new
  auto vol_v_copy(vol_v);
  float sum = std::accumulate(vol_v.begin(), vol_v.end(), 0.0);
  float mean = sum/vol_v.size();
  vol_v_copy.emplace_back(mean);
  cv::Mat area_data{vol_v_copy};
  area_data = area_data/1000;

  std::vector<float> area_data_v = area_data.reshape(0);

  PltTinyNetwork(area_data_v,  all_values);
  value = all_values[all_values.size()-1]*10;

  // std::cout<<"----------------plt volume value-------------"<<std::endl;
  // for(int i =0; i< all_values.size(); ++i){
  //   std::cout<<", "<<all_values[i];
  // }
  // std::cout<<std::endl;
}


//RBC底面积及侧面积拟合tiny netowrk
void RbcTinyNetwork(const double & input1, const double & input2, float& value){
  std::cout<<"area : incline "<< input1 << " "<<input2<<std::endl;
  cv::Mat conv1_weight{ -0.2080,  1.1787,  0.3284, -0.3995,
                       -0.1844, -2.0602, -0.6971,  1.6079,
                       0.5850, -0.4011, -0.8080,  0.5265};
  conv1_weight = conv1_weight.reshape(0, 3);//3*4
  cv::Mat conv1_bias{0.2962,  0.9444,  0.4666, -1.1170};
  conv1_bias = conv1_bias.reshape(0, 1);//1*4

  cv::Mat conv2_weight{-0.1875,  0.0079, -0.0247,  0.3554,
                       -1.3440, -1.3179,  1.1578, -1.6077,
                       -0.5391, -0.4222,  0.1458, -0.7583,
                       1.2956,  0.8728, -0.9258,  1.0698,
  };
  conv2_weight = conv2_weight.reshape(0, 4);//4*4

  cv::Mat conv2_bias{ -0.5365, -0.5635, -0.1633, -0.4451 };
  conv2_bias = conv2_bias.reshape(0, 1);//1*4


  cv::Mat conv3_weight{1.4867,  1.1952, -1.3927,  1.5872};
  conv3_weight = conv3_weight.reshape(0, 4);//4*1

  cv::Mat input{input1, input2, input1*input2};
  input = input.reshape(0,1);//1*3

  cv::Mat x = input*conv1_weight;//1*3 3*4 -> 1*4
  x = x +conv1_bias;
  sigmoid(x,x);
  x = x*conv2_weight;//1*4 4*4 -> 1*4
  x = x+conv2_bias;
  sigmoid(x,x);
  x = x*conv3_weight;//1*4 4*1 -> 1*1
  value = (float)x.at<double>(0)*100;
}


void CalculateVariance(const std::vector<float>& numbers_v, float& variance){
  if(numbers_v.empty()){
    variance=0;
    return;
  }
  double sum = std::accumulate(std::begin(numbers_v), std::end(numbers_v), 0.0);
  double mean =  sum / (int)numbers_v.size(); //均值
  float area_mean = (float)mean;
  variance = 0;
  std::for_each(numbers_v.begin(), numbers_v.end(), [area_mean, &variance](const std::vector<float>::iterator::value_type& nums){variance += (area_mean-nums)*(area_mean-nums);});
  variance = variance/numbers_v.size();
}

void CalculateStd(const std::vector<float>& numbers_v, float& std_value){
  float variance;
  CalculateVariance(numbers_v, variance);
  std_value=std::sqrt(variance);
}

void RbcAreaInclineFitting(const std::vector<float>& vol_v, const float& fusion_rate,
                        const int& incline_cell_nums, const int&incline_pixels,
                        float& value){
  if(vol_v.empty()){
    value=0;
    return;
  }
  std::cout<<"rbc ori value"<<std::endl;
  for(auto i :vol_v){
    std::cout<<" "<<i<<",";
  }
  std::cout<<std::endl;
  //计算标准差
  float std_value;
  CalculateStd(vol_v, std_value);
  std::cout<<"rbc volume std value "<<std_value<<std::endl;

  double sum = std::accumulate(std::begin(vol_v), std::end(vol_v), 0.0);
  double mean =  sum / (int)vol_v.size(); //均值
  float area_mean = (float)mean;

  float incline_mean = incline_pixels/incline_cell_nums;
  //依据像元融合倍率进行区域缩放
  area_mean = area_mean * (1/fusion_rate)*(1/fusion_rate);
  incline_mean = incline_mean* (1/fusion_rate)*(1/fusion_rate);
  //归一化
  area_mean = area_mean / 10000;
  incline_mean = incline_mean /1000;

  std::cout<<"ara incline "<< area_mean <<" "<<incline_mean<<std::endl;
  RbcTinyNetwork(area_mean, incline_mean, value);
  //避免白细胞流道测试数据过少导致结果为负数
  if(value<0){
    value = 0;
  }
}

//根据不同的类型调用不同拟合算法
void GetRbcVolume(const std::vector<float>& vol_v, const float& fusion_rate,
                  const int& incline_cell_nums, const int&incline_pixels,
                  const VOLUME_FUNC_TYPE& volume_func_type, float& value){


  if(volume_func_type==VOLUME_TYPE_INCLINE){
    RbcAreaInclineFitting(vol_v, fusion_rate, incline_cell_nums, incline_pixels,
                       value);
  }else{
    RbcAreaFitting(vol_v, fusion_rate, value);
  }
  if(value<0){
    value = 0;
  }

}

//plt 体积算法入口
void GetPltVolume(const std::vector<float>& vol_v, const float& fusion_rate, float& value, std::vector<float>& all_values){

    PltAreaFitting(vol_v, fusion_rate, value, all_values);

}



//rdw-cv
void GetRdwCv(const std::vector<float>& vol_v, const float& mcv,  const float& fusion_rate, float& value, std::vector<float>& all_values){
  if(vol_v.empty()){
    value=0;
    return;
  }

  float std_value;
  CalculateStd(vol_v, std_value);

  std::vector<float> vol_v_copy{std_value/mcv*100};
  cv::Mat area_data{vol_v_copy};
  area_data = area_data/1000;
  std::vector<float> area_data_v = area_data.reshape(0);
  RdwCvTinyNetwork(area_data_v,  all_values);
  value = all_values[all_values.size()-1]*10;

}


// template<class ForwardIterator>
// int PseudoArgmin(const ForwardIterator& first_iterator, const ForwardIterator& last_iterator){
//   return std::distance(first_iterator, std::min_element(first_iterator, last_iterator));
// }

// template<class ForwardIterator>
// int PseudoArgMax(const ForwardIterator& first_iterator, const ForwardIterator& last_iterator){
//   return std::distance(first_iterator, std::max_element(first_iterator, last_iterator));
// }



void FindMinDifference(const std::vector<float>& data, const float& target_value, int& min_idx){
  cv::Mat data_mat{data};
  data_mat = data_mat-target_value;
  cv::multiply(data_mat, data_mat, data_mat);
  std::vector<float> data_v = data_mat.reshape(0);
  min_idx = PseudoArgmin(data_v.begin(), data_v.end());
}




void RdwSdTinyNetwork(const std::vector<float>& input_v, std::vector<float>& all_values){

  cv::Mat conv1_weight{ -0.4477,  0.3463, -0.7095, -0.2908};
  conv1_weight.convertTo(conv1_weight, CV_32F);
  conv1_weight = conv1_weight.reshape(0, 1);//1*4

  cv::Mat conv1_bias{1.4306, -1.3713,  3.2006, -0.3561};
  conv1_bias = conv1_bias.reshape(0, 1);//1*4
  conv1_bias.convertTo(conv1_bias, CV_32F);
  cv::resize(conv1_bias, conv1_bias, cv::Size(4,input_v.size()));


  cv::Mat conv2_weight{-0.8803, -0.0291, -0.3414, -1.0174,
                       0.2819,  1.4252, -0.6029,  0.3618,
                       -1.2810,  0.5367, -1.1389, -1.5198,
                       -0.8217,  0.0698, -0.4909, -0.5018,
  };
  conv2_weight = conv2_weight.reshape(0, 4);//4*4
  conv2_weight.convertTo(conv2_weight, CV_32F);

  cv::Mat conv2_bias{ -0.8932,  1.6338, -0.9049, -0.7290};
  conv2_bias = conv2_bias.reshape(0, 1);//1*4
  conv2_bias.convertTo(conv2_bias, CV_32F);
  cv::resize(conv2_bias, conv2_bias, cv::Size(4, input_v.size()));

  cv::Mat conv3_weight{1.4859,  3.7961, 1.4135, 1.6106};
  conv3_weight = conv3_weight.reshape(0, 4);//4*1
  conv3_weight.convertTo(conv3_weight, CV_32F);


  cv::Mat input{input_v};
  input = input.reshape(0,input_v.size());//n*1

  cv::Mat x = input*conv1_weight;//n*1 1*4 -> n*4

  x = x + conv1_bias;
  sigmoid(x,x);
  x = x*conv2_weight;//n*4 4*4 -> n*4
  x = x+conv2_bias;
  sigmoid(x,x);
  x = x*conv3_weight;//n*4 4*1 -> n*1
  all_values = x.reshape(0,1);
  //value = x.at<float>(input_v.size()-1)*10;

}



//rdw-sd
void GetRdwSd(const std::vector<float>& vol_v, const float& fusion_rate, float& value, std::vector<float>& all_values){
  if(vol_v.empty()){
    value=0;
    return;
  }

  int max_idx = PseudoArgMax(vol_v.begin(), vol_v.end());
  std::vector<float> left_data(vol_v.begin(), vol_v.begin()+max_idx+1);
  std::vector<float> right_data(vol_v.begin()+max_idx, vol_v.end());
  float percentage_value = vol_v[max_idx]*RDW_SD_PERCENTAGE;
  int left_min_idx, right_min_idx;
  FindMinDifference(left_data, percentage_value, left_min_idx);
  FindMinDifference(right_data, percentage_value, right_min_idx);
  float raw_rdw_ds = right_min_idx+max_idx-left_min_idx;
  std::vector<float> network_input{raw_rdw_ds/10};
  std::cout<<"11111111111111 "<<raw_rdw_ds<<std::endl;
  RdwSdTinyNetwork(network_input, all_values);
  value = all_values[all_values.size()-1]*10;
	printf("GetRdwSd end \r\n");
}



void HgbTinyNetwork(const std::vector<float>& input_params,
                    std::vector<float>& output_params){
  cv::Mat conv1_weight{ -2.6828,
                        -4.3504};
  conv1_weight = conv1_weight.reshape(0, 2);//2*1
  conv1_weight.convertTo(conv1_weight, CV_32F);

  cv::Mat conv1_bias{3.8624};
  conv1_bias = conv1_bias.reshape(0, 1);//1*1
  conv1_bias.convertTo(conv1_bias, CV_32F);

  cv::Mat conv2_weight{-2.8881};
  conv2_weight = conv2_weight.reshape(0, 1);//1*1
  conv2_weight.convertTo(conv2_weight, CV_32F);

  cv::Mat conv2_bias{2.8752};
  conv2_bias.convertTo(conv2_bias, CV_32F);


  cv::Mat input(input_params);
  input = input.reshape(0,1);//1*2

  cv::Mat x = input*conv1_weight;//1*2 2*1 -> 1*1
  std::cout<<"784"<<std::endl;
  x = x +conv1_bias;
  std::cout<<"786"<<std::endl;
  sigmoid(x,x);
  x = x*conv2_weight;//1*1 1*1 -> 1*1
  std::cout<<"789"<<std::endl;
  x = x+conv2_bias;
  std::cout<<"791"<<std::endl;
  output_params = x.reshape(0,1);
}


void GetHgb(const float& r0, const float& r1,
            const float& m0, const float& m1, float& hgb_value){
  std::cout<<"r0 r1 m0 m1"<<r0<<" "<<r1<<" "<<m0<<" "<<m1<<std::endl;
  float input_param1 = log10(r0/m0);
  float input_param2 = log10(r1/m1);
  std::vector<float> input_params, output_params;
  input_params = {input_param1, input_param2};
  HgbTinyNetwork(input_params, output_params);
  hgb_value = output_params[0]*100;
}

void Delay(const int&  time)//time*1000为秒数
{
    clock_t   now   =   clock();

    while( clock()   -   now   <   time   );

}

// 计算两个框的中心距离平方（用于DIoU）
static float CenterDistanceSquared(float x0, float y0, float x1, float y1) {
	float dx = x1 - x0;
	float dy = y1 - y0;
	return dx * dx + dy * dy;
}

// 计算DIoU
static float CalculateDIoU(float xmin0, float ymin0, float xmax0, float ymax0,
                           float xmin1, float ymin1, float xmax1, float ymax1) {
	float w = std::fmax(0.f, std::fmin(xmax0, xmax1) - std::fmax(xmin0, xmin1));
	float h = std::fmax(0.f, std::fmin(ymax0, ymax1) - std::fmax(ymin0, ymin1));
	float i = w * h;
	float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
	if (u <= 0.f) return 0.f;
	float iou = i / u;

	// 框中心
	float cx0 = (xmin0 + xmax0) * 0.5f;
	float cy0 = (ymin0 + ymax0) * 0.5f;
	float cx1 = (xmin1 + xmax1) * 0.5f;
	float cy1 = (ymin1 + ymax1) * 0.5f;

	// 包含这两个框的最小外接矩形对角线平方
	float enclose_xmin = std::min(xmin0, xmin1);
	float enclose_ymin = std::min(ymin0, ymin1);
	float enclose_xmax = std::max(xmax0, xmax1);
	float enclose_ymax = std::max(ymax0, ymax1);
	float c2 = CenterDistanceSquared(enclose_xmin, enclose_ymin, enclose_xmax, enclose_ymax);
	if (c2 <= 1e-5f) return iou;  // 防止除零

	float center_dist = CenterDistanceSquared(cx0, cy0, cx1, cy1);
	float diou = iou - center_dist / c2;
	return diou;
}

// 替换原始 nms 函数为 DIoU-NMS
static int diou_nms(int validCount, const std::vector<float> &outputLocations,
                    std::vector<int> classIds, std::vector<int> &order,
                    int filterId, float threshold, const int max_det=3500) {
	int valid_obj_nums = 0;
	for (int i = 0; i < validCount; ++i) {
		if (order[i] == -1 || classIds[i] != filterId)
			continue;

		valid_obj_nums++;
		if (valid_obj_nums > max_det) {
			for (; i < validCount; ++i) order[i] = -1;
			return 0;
		}

		int n = order[i];
		for (int j = i + 1; j < validCount; ++j) {
			int m = order[j];
			if (m == -1 || classIds[i] != filterId)
				continue;

			float xmin0 = outputLocations[n * 4 + 0];
			float ymin0 = outputLocations[n * 4 + 1];
			float xmax0 = xmin0 + outputLocations[n * 4 + 2];
			float ymax0 = ymin0 + outputLocations[n * 4 + 3];

			float xmin1 = outputLocations[m * 4 + 0];
			float ymin1 = outputLocations[m * 4 + 1];
			float xmax1 = xmin1 + outputLocations[m * 4 + 2];
			float ymax1 = ymin1 + outputLocations[m * 4 + 3];

			float diou = CalculateDIoU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

			if (diou > threshold) {
				order[j] = -1;  // 抑制冗余框
			}
		}
	}
	return 0;
}

// 修改后的 RemoveDuplicateData 接口不变,增加中心点距离去重
void RemoveDuplicateData(const std::vector<float>& src, const float& iou_thr, float& threshed_cell_nums) {
	int valid_count = src.size() / 4;
	std::vector<int> class_ids(valid_count, 0);  // 所有框默认同类别
	std::vector<int> order(valid_count);
	for (int i = 0; i < valid_count; ++i) {
		order[i] = i;
	}
	int filter_id = 0;

	diou_nms(valid_count, src, class_ids, order, filter_id, iou_thr);

	threshed_cell_nums = 0;
	for (const auto& idx : order) {
		if (idx != -1) {
			threshed_cell_nums++;
		}
	}
}

struct PeakSeekItem {
  int index;
  float clarity;
};
int PeekSeekFirst(std::vector<float> &list, uint32_t *index, float *value, int& peak_nums) {

  if (list.empty()) {
    return -1;
  } else if (list.size() == 1) {
    *index = 0;
    *value = list[0];
    return 0;
  }
  std::vector<int> peakIdList;//峰id
  std::vector<int> minIdList;//左侧波谷id
  std::vector<float> heightPeakList;//峰高
  std::vector<int> widthPeakList;//峰宽
  //筛选峰的阈值
  float heightValue;
  int widthValue;
  int focusPos = -1; // 区分第一个位置
  float focusClarity = 0.0;
  //整合数据
  std::vector<PeakSeekItem> s_clarities;

  PeakSeekItem clarity_item;
  for (int idx = 0; idx < list.size(); ++idx) {
    clarity_item.index = idx;
    clarity_item.clarity = list[idx];
    s_clarities.emplace_back(clarity_item);
  }

  std::vector<PeakSeekItem>::iterator it;
  //初步寻峰
  for (it = s_clarities.begin() + 1; it != s_clarities.end() - 1; it++) {
    if (((it->clarity > (it - 1)->clarity) && (it->clarity >= (it + 1)->clarity)) ||
        ((it->clarity >= (it - 1)->clarity) && (it->clarity > (it + 1)->clarity))) {
      peakIdList.push_back(it->index);
    }
  }
  //如果没有峰，找到区域最大值,同样适用于vector只有2个数
  if (peakIdList.empty()) {
    for (it = s_clarities.begin(); it != s_clarities.end(); it++) {
      if (it->clarity > focusClarity) {
        *index = it->index;
        *value = it->clarity;
        peak_nums = 0;
      }
    }
  } else {//找左侧波谷
    for (unsigned int j = 0; j < peakIdList.size(); j++) {
      if (j == 0) {
        int minimalPos = -1;
        float minimalClarity = s_clarities.at(0).clarity;
        for (unsigned int z = 0; z < peakIdList[j]; z++) {
          if (s_clarities.at(z).clarity < s_clarities.at(0).clarity) {
            minimalClarity = s_clarities.at(z).clarity;
            minimalPos = z;
          } else {
            minimalPos = 0;
          }
        }
        minIdList.push_back(minimalPos);
      } else {
        int minimalPos = -1;
        float minimalClarity = s_clarities.at(peakIdList[j - 1]).clarity;
        for (unsigned int k = s_clarities.at(peakIdList[j - 1]).index;
             k < s_clarities.at(peakIdList[j]).index; k++) {
          if (s_clarities.at(k).clarity < minimalClarity) {
            minimalClarity = s_clarities.at(k).clarity;
            minimalPos = k;
          }
          //          else{
          //            minimalPos = j - 1;
          //          }
        }
        if (minimalPos == -1) {
          minimalPos = j - 1;
        }
        minIdList.push_back(minimalPos);
      }
    }

    //得到最高波谷与波峰的距离
    for (unsigned int i = 0; i < minIdList.size(); i++) {
      float height = s_clarities.at(peakIdList[i]).clarity - s_clarities.at(minIdList[i]).clarity;
      heightPeakList.push_back(height);
      int width = s_clarities.at(peakIdList[i]).index - s_clarities.at(minIdList[i]).index;
      widthPeakList.push_back(width);
    }
    //得到排除其他峰的宽、高阈值
    float height = 0.0;
    int width = 0;
    for (unsigned int i = 0; i < heightPeakList.size(); i++) {
      //综合峰高峰宽
      //			if (heightPeakList[i] > height) {
      //				height = heightPeakList[i];
      //			}
      //			heightValue = height / 2.0;
      //			if (widthPeakList[i] > width) {
      //				width = widthPeakList[i];
      //			}
      //			widthValue = width / 2.0;

      //最宽峰

      if (widthPeakList[i] > width) {
        width = widthPeakList[i];
        height = heightPeakList[i];
        widthValue = width / 2.0;
        heightValue = height / 2.0;
      }

    }
    //删除低峰、窄峰
    //取有意义的峰中的第一个为清晰的聚焦位置
    std::vector<int> exact_peak_idx;
    std::vector<float> exact_peak_clarity;
    for (unsigned int i = 0; i < minIdList.size(); i++) {
      float heightPeak = s_clarities.at(peakIdList[i]).clarity - s_clarities.at(minIdList[i]).clarity;
      int widthPeak = s_clarities.at(peakIdList[i]).index - s_clarities.at(minIdList[i]).index;
      if (heightPeak > heightValue && widthPeak > widthValue) {
        //        *index = s_clarities.at(peakIdList[i]).index ;
        //        *value= s_clarities.at(peakIdList[i]).clarity;
        exact_peak_idx.push_back(s_clarities.at(peakIdList[i]).index);
        exact_peak_clarity.push_back(s_clarities.at(peakIdList[i]).clarity);
      }
    }
    //根据是否有峰选择第一个峰或者第二个峰
    if (exact_peak_idx.empty()) {
      *index = 0;
      *value = 0.f;
    } else if (exact_peak_idx.size() <= 2) {
      *index = exact_peak_idx[0];
      *value = exact_peak_clarity[0];
    } else {
      *index = exact_peak_idx[1];
      *value = exact_peak_clarity[1];
    }
    peak_nums = static_cast<int>(exact_peak_idx.size());
  }
  return 0;
}


using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
int TimeMonitor::Time() {
  auto func_end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  return static_cast<int>(func_end) ;

}

using SVPNG_OUTPUT = std::vector<unsigned char>;
#define SVPNG_PUT(u) (output.push_back(static_cast<unsigned char>(u)))

void svpng(SVPNG_OUTPUT& output, unsigned w, unsigned h, const unsigned char* img, int alpha) {
    static const unsigned t[] = {0,
                                 0x1db71064,
                                 0x3b6e20c8,
                                 0x26d930ac,
                                 0x76dc4190,
                                 0x6b6b51f4,
                                 0x4db26158,
                                 0x5005713c,
                                 /* CRC32 Table */ 0xedb88320,
                                 0xf00f9344,
                                 0xd6d6a3e8,
                                 0xcb61b38c,
                                 0x9b64c2b0,
                                 0x86d3d2d4,
                                 0xa00ae278,
                                 0xbdbdf21c};
    unsigned              a = 1, b = 0, c, p = w * (alpha ? 4 : 3) + 1, x, y, i; /* ADLER-a, ADLER-b, CRC, pitch */
#define SVPNG_U8A(ua, l)    \
    for (i = 0; i < l; i++) \
        SVPNG_PUT((ua)[i]);
#define SVPNG_U32(u)                  \
    do {                              \
        SVPNG_PUT((u) >> 24);         \
        SVPNG_PUT(((u) >> 16) & 255); \
        SVPNG_PUT(((u) >> 8) & 255);  \
        SVPNG_PUT((u) & 255);         \
    } while (0)
#define SVPNG_U8C(u)              \
    do {                          \
        SVPNG_PUT(u);             \
        c ^= (u);                 \
        c = (c >> 4) ^ t[c & 15]; \
        c = (c >> 4) ^ t[c & 15]; \
    } while (0)
#define SVPNG_U8AC(ua, l)   \
    for (i = 0; i < l; i++) \
    SVPNG_U8C((ua)[i])
#define SVPNG_U16LC(u)               \
    do {                             \
        SVPNG_U8C((u) & 255);        \
        SVPNG_U8C(((u) >> 8) & 255); \
    } while (0)
#define SVPNG_U32C(u)                 \
    do {                              \
        SVPNG_U8C((u) >> 24);         \
        SVPNG_U8C(((u) >> 16) & 255); \
        SVPNG_U8C(((u) >> 8) & 255);  \
        SVPNG_U8C((u) & 255);         \
    } while (0)
#define SVPNG_U8ADLER(u)       \
    do {                       \
        SVPNG_U8C(u);          \
        a = (a + (u)) % 65521; \
        b = (b + a) % 65521;   \
    } while (0)
#define SVPNG_BEGIN(s, l) \
    do {                  \
        SVPNG_U32(l);     \
        c = ~0U;          \
        SVPNG_U8AC(s, 4); \
    } while (0)
#define SVPNG_END() SVPNG_U32(~c)
    SVPNG_U8A("\x89PNG\r\n\32\n", 8); /* Magic */
    SVPNG_BEGIN("IHDR", 13);          /* IHDR chunk { */
    SVPNG_U32C(w);
    SVPNG_U32C(h); /*   Width & Height (8 bytes) */
    SVPNG_U8C(8);
    SVPNG_U8C(alpha ? 6 : 2);                 /*   Depth=8, Color=True color with/without alpha (2 bytes) */
    SVPNG_U8AC("\0\0\0", 3);                  /*   Compression=Deflate, Filter=No, Interlace=No (3 bytes) */
    SVPNG_END();                              /* } */
    SVPNG_BEGIN("IDAT", 2 + h * (5 + p) + 4); /* IDAT chunk { */
    SVPNG_U8AC("\x78\1", 2);                  /*   Deflate block begin (2 bytes) */
    for (y = 0; y < h; y++) {                 /*   Each horizontal line makes a block for simplicity */
        SVPNG_U8C(y == h - 1);                /*   1 for the last block, 0 for others (1 byte) */
        SVPNG_U16LC(p);
        SVPNG_U16LC(~p);  /*   Size of block in little endian and its 1's complement (4 bytes) */
        SVPNG_U8ADLER(0); /*   No filter prefix (1 byte) */
        for (x = 0; x < p - 1; x++, img++)
            SVPNG_U8ADLER(*img); /*   Image pixel data */
    }
    SVPNG_U32C((b << 16) | a); /*   Deflate block end with adler (4 bytes) */
    SVPNG_END();               /* } */
    SVPNG_BEGIN("IEND", 0);
    SVPNG_END(); /* IEND chunk {} */
}
