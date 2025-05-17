//
// Created by y on 23-8-4.
//

#ifndef RKNN_ALG_DEMO_DETECTTYPE_H
#define RKNN_ALG_DEMO_DETECTTYPE_H
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "neural_network.h"
namespace ALG_LOCAL{
enum AlgType {
  HGB = 0,
  RBC = 1,
  RET = 2,
  WBC = 3,
  NEU = 4,
  LYM = 5,
  MONO = 6,
  EOS = 7,
  BASO = 8,
  PLT = 9,
  WBC4 = 10,
  SOMATIC = 11,
  BACTERIA = 12,
  BASCLARITY = 13,
  GRADCLARITY = 14,
  WBC4_SINGLE = 15,
  RBC_VOLUME = 16,
  AI_CLARITY = 17,
  PLT_VOLUME = 18,
  AI_CLARITY_FAR_NEAR = 19,
  MILK_GERM = 20,
  RBC_VOL_SPH_BOX = 21,
  RBC_VOL_SPH_SEG = 22,
  MILK_CELL = 23,
  SPHERICAL_FOCAL = 24,
  WBC_SINGLE = 25,
  CLASSIFICATION_CUSTOM = 26,
  CLARITY_MLIK_BOARDLINE = 27,
  PLA = 28, // 疟原虫
};

enum DetectTypeName {
  MILK_TYPE = 0,
  HUMAN_TYPE = 1,
  CAT_TYPE = 2,
  DOG_TYPE = 3,
  BASCLARITY_TYPE = 20,
};

struct AlgParam {
  AlgType alg_type; // 当前检测类型下的算法类型。如人的wbc检测
  bool enable = false;                    // 是否启用该算法
  std::vector<std::string> model_paths_v; // 算法相关路径
  std::vector<float> init_param_float_v; // 初始化所需参数，如con_thr,nms_thr....
    };

    struct AssitFuncParam{
        std::string fun_name;//待检测函数名
        bool enable = false;//是否测试该函数
    };

//初始化参数
    struct InitParam{
        DetectTypeName detect_type;//检测类型
        std::vector<AlgParam> alg_param_v;//细分算法
        std::vector<AssitFuncParam> assit_func_param_v;
    };


//推理参数
    struct ForwardParam{
        DetectTypeName detect_type;//检测类型
        AlgType alg_type;//当前检测类型下的算法类型。如人的wbc检测
        cv::Mat * img_brightness= nullptr;//明场图像
        cv::Mat * img_fluorescence= nullptr;//荧光场图像
        int img_height = 3036;//图像高
        int img_width = 4024;//图像宽

        std::vector<NNetResult> detect_result_v;//推理结果，output
        std::vector<cv::Mat> mat_bright_result_v;//记录算法中间结果， output
        std::vector<cv::Mat> mat_fluo_result_v;//记录算法中间结果， output
        bool processed{false}; //记录该次输入是否被算法处理， output

    };
    //存放血细胞计数相关结果
    struct AlgSampleContex
    {
      float Hgb=0.f;            // hgb 结果
      int Rbc=0;	      // 红细�?
      int Ret=0;              // 网织�?
      int Wbc=0;	      // 白细�?
      int Neu=0;              // �?性粒
      int Lym=0;              // 淋巴
      int Mono=0;              // 单核
      int Eos=0;              // 嗜酸
      int Baso=0;              // 嗜碱
      int Plt=0;              // 血小板
      void* userData;
      std::vector<float> rbc_volume_v;//存放rbc体积
      std::vector<float> plt_volume_v;//存放plt体积

      int incline_cell_nums = 0; //倾斜红细胞个数
      int incline_pixels = 0;   //倾斜红细胞像素个数
    };


    // 项目结果
    struct RstItem
    {
      std::string name;         // 项目名称
      float value;              // 项目�?
      std::string unit;         // 项目单位
    };

    class DetectType {
    public:
        DetectType()=default;
        virtual ~DetectType()=default;
        /*!
         * 单个检测类型（如human）初始化入口
         * @param init_param
         * @return 初始化是否成功
         */
        virtual bool Init(const InitParam& init_param)=0;
        /*!
         * 单个检测类型推理入口
         * @param forward_param input/output
         * @return 推理是否成功
         */
        virtual bool Forward(ForwardParam& forward_param)=0;
        /*!
         * 获取需要处理多张图的统计性结果,如细胞体积
         */
        virtual void GetStatisticResult()=0;
        virtual bool RunAssistFunction()=0;
    };


}



#endif //RKNN_ALG_DEMO_DETECTTYPE_H
