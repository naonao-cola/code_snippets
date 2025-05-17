#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include <list>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#define NNetCtxID_t				void*				/* 神经网络上下文ID */
#define NNetImg_t				cv::Mat				/* 神经网络图像 */
#define NNetGroupMask_t			uint32_t			/* 神经网络分组掩码 */

#define LABEL_NUMS_CUSTOM 9999

//图像resize方式
enum ResizeType{
  NORMAL=0,
  LETTERBOX=1,
  BOTTOMPAD=2,
  LEFT_TOP_CROP=3,
  RGA_NORMAL,
};

/* 神经网络分组枚举 */
typedef enum NNetGroup
{
	NNET_GROUP_MILK = (1 << 0),
	NNET_GROUP_HUMAN = (1 << 1),
	NNET_GROUP_CAT = (1 << 2),
	NNET_GROUP_DOG = (1 << 3),
	NNET_GROUP_CLARITY_AI = (1 << 20),
	NNET_GROUP_ALL = 0xFFFFFFFF,
}NNetGroup_e;

//typedef enum NNetGroup
//{
//  NNET_GROUP_MILK = 0,
//  NNET_GROUP_HUMAN = 1,
//  NNET_GROUP_CAT = 2,
//  NNET_GROUP_DOG = 3,
//  NNET_GROUP_ALL = 0xFFFFFFFF,
//}NNetGroup_e;



/* 神经网络模型ID枚举 */
typedef enum NNetModID {
  NNET_MODID_UNKOWN = 0,
  NNET_MODID_RBC,
  NNET_MODID_WBC,
  NNET_MODID_WBC4,
  NNET_MODID_PLT,
  NNET_MODID_BASO,
  NNET_MODID_RET,
  NNET_MODID_NRBCS,
  NNET_MODID_MILK,
  NNET_MODID_BACTERIA,
  NNET_MODID_BAS_CLARITY,
  NNET_MODID_RBC_INCLINE_DET,
  NNET_MODID_RBC_INCLINE_SEG,
  NNET_MODID_AI_CLARITY,
  NNET_MODID_PLT_VOLUME,
  NNET_MODID_AI_CLARITY_FAR_NEAR,
  NNET_MODID_MILK_GERM,
  NNET_MODID_MILK_CELL,
  NNET_MODID_GERM_RGA,
  NNET_MODID_AI_CLARITY_BASO_FAR_NEAR,
  NNET_MODID_RBC_VOLUME_SHPERICAL,
  NNET_MODID_RBC_QC,
  NNET_MODID_SPHERICAL_FOCAL,
  NNET_MODID_AI_CLARITY_COARSE,
  NNET_MODID_CLASSIFICATION_CUSTOM,
  NNET_MODID_AI_CLARITY_COARSE_FLU_MICRO,
  NNET_MODID_AI_CLARITY_FINE_FLU_MICRO,
  NNET_MODID_CALIB_COUNT,
  NNET_MODID_AI_CLARITY_MILK_BOARDLINE,
  NNET_MODID_PLA,  // 疟原虫模型
  NNET_MODID_PLA4, // 疟原虫模型
} NNetModID_e;

typedef enum NNetTypeID{
  NNET_TYPE_YOLO_RECT = 0,                    //检测模型,结果为矩形
  NNET_TYPE_PP_ROTATED_POLY,                  //检测模型,结果为多边形
  NNET_TYPE_SEG_ALL,                          //分割模型
  NNET_TYPE_CLS_ALL,                          //分类模型
  NNET_TYPE_YOLO_RECT_UNQUAN,                    //检测模型,未量化,结果为矩形
}NNetTypeID_e;

/* 矩形框定义 */
typedef struct BoxRect
{
	int left;
	int right;
	int top;
	int bottom;
        int label;
        std::string name;
        float prop;
}BoxRect_t;
#define NNET_BOX_LEFT(box)				((box)->left)
#define NNET_BOX_RIGHT(box)				((box)->right)
#define NNET_BOX_TOP(box)				((box)->top)
#define NNET_BOX_BOTTOM(box)			((box)->bottom)

/* 神经网络结果定义 */
typedef struct NNetResult
{
	BoxRect_t box;
        std::vector<float> polygon_v; //存储多边形检测结果: {category, conf, point1_x, point1_y, point2_x, point2_y, ...}
        std::vector<cv::Mat> seg_v; //存储分割检测结果:{Mat_0, Mat_1, Mat_k, ...}, vector索引对应类别.
        std::vector<float> category_v; //存储分类结果. {cate_0_prob, cate_1_prob, cate_2_prob,...}, vector索引对应类别
        bool write_rect_box = false; //vector可直接判断空,但其他参数不行,该参数用于记录其他参数是否初始化
}NNetResult_t;
#define NNET_OPT_NAME(opt)              ((opt)->box.name)
#define NNET_OPT_BOX(opt)               (&(opt)->box)
#define NNET_OPT_PROP(opt)              ((opt)->box.prop)

NNetCtxID_t NNet_Init(const std::string& cfg_path);
int NNet_DeInit(NNetCtxID_t ctx_id);
int NNet_AddModel(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id,
                  uint8_t *mod_data, uint32_t mod_size, uint8_t *labels_data, uint32_t labels_size,
                  const ResizeType& resize_type, const float& nms_thr, const float& conf_thr,
                  const NNetTypeID_e& net_type_id, const std::vector<float>& float_param_v);
int NNet_AddModel(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id,
                  uint8_t *mod_data, uint32_t mod_size,const ResizeType &resize_type,
                  float model_type_nums,
                  float nms_nums,
                  float conf_nums,
                  float anchor_nums,
                  float label_nums,
                  float reserved_float_param_nums,
                  float reserved_string_param_nums,
                  const std::vector<float>& model_type,
                  const std::vector<float>& nms_v,
                  const std::vector<float>& conf_v,
                  const std::vector<float>& anchors,
                  const std::vector<std::string>& labels,
                  const std::vector<float>& reserved_float_params,
                  const std::vector<std::string>& reserved_string_params);

int NNet_Inference(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, NNetImg_t *img,
                   std::list<NNetResult_t> &result);

int NNet_GetReservedFloatPrams(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, std::vector<float>& reserved_float_params);




#endif /* _NEURAL_NETWORK_H_ */

