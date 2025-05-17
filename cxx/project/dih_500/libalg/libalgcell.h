#ifndef _LIBALG_H_
#define _LIBALG_H_

#include <string>
#include <list>
#include <vector>
#include <stdint.h>
#include <map>
#define AlgCtxID_t			void*
#define AlgFuncMask_t		uint32_t
//#define AlgCellImg_t		uint8_t*
#define AlgImmData_t		float
#define AlgHgbData_t		float
#define AlgClarityValue_t	float

/*!
 * 清晰度算法返回值含义;血球,荧光微球通用
 * 涉及函数:
 * AlgCell_ClarityGetResultFarNear
 * AlgCell_ClarityGetResultCoarse
 * AlgCell_ClarityGetResultFineFluMicrosphere
 * AlgCell_ClarityGetResultCoarseFluMicrosphere
 */
// 细聚焦返回值
#define AI_CLARITY_FAR_NEAR_FAR             0   //焦平面下方
#define AI_CLARITY_FAR_NEAR_NEAR            1   //焦平面上方
#define AI_CLARITY_FAR_NEAR_COARSE_FOCUS    2   //粗聚焦
#define AI_CLARITY_FAR_NEAR_CLEAR           3   //清晰
#define AI_CLARITY_FAR_NEAR_INDETERMINATION 4   //不确定
#define AI_CLARITY_FAR_NEAR_CHANGE_VIEW     5   //换视野
#define AI_CLARITY_FAR_NEAR_VERY_NEAR       6   //焦平面很上方, 需大步长向下移动,推荐4倍细聚焦步长
#define AI_CLARITY_FAR_NEAR_FORCE_CLEAR     7   //强制将当前焦平面视为清晰.理由:嗜碱细胞可能一个视野无白细胞,算法通过对当前视野已有聚焦过程的逻辑判断,强行将图像视为清晰

// 粗聚焦返回值
#define AI_CLARITY_COARSE_CLEAR                    0   //焦平面上方,清晰或下方
#define AI_CLARITY_COARSE_INDETERMINATION          1   //完全不存在细胞或无法区分焦平面上,下方

// AM300底板划线细聚焦返回值
#define AI_CLARITY_FOCUS_AM               0   //在焦
#define AI_CLARITY_POSITIVE_AM            1   //细正焦
#define AI_CLARITY_NEGATIVE_AM            2   //细负焦
#define AI_CLARITY_FAR_POSITIVE_AM        3   //粗正焦
#define AI_CLARITY_FAR_NEGATIVE_AM        4   //粗负焦
#define AI_CLARITY_FAR_FAR_POSITIVE_AM    5   //远正焦
#define AI_CLARITY_FAR_FAR_NEGATIVE_AM    6   //远负焦
#define AI_CLARITY_OTHER_AM               7   //不确定
/*!
 * AlgCell_XXXOpen std::map key及含义
 */
// key
#define OPEN_PARAM_DEBUG                  "open_param_debug"      //是否开始debug  0 for false, 1 for true
#define OPEN_PARAM_GROUP_IDX              "open_param_group_idx"
#define OPEN_PARAM_QC                     "open_param_qc"         //是否开始质控 0 for false, 1 for true
#define OPEN_PARAM_IMG_H                  "open_param_img_h"         //传入的图像的高
#define OPEN_PARAM_IMG_W                  "open_param_img_w"         //传入的图像的宽
#define OPEN_PARAM_IMG_H_UM               "open_param_img_h_um"      //传入的图像的高对应的微米
#define OPEN_PARAM_ALARM                  "open_param_alarm"         //报警参数
#define OPEN_PARAM_DILUTION               "open_param_dilution"      //各流道稀释倍数
#define OPEN_PARAM_TASK_APPEND_ATT        "open_param_task_append_att" //任务附加属性,{网织红, 有核红},0 for false, 1 for true
#define OPEN_PARAM_CALIB                  "open_param_calib"        //是否为校准模式 0 for false, 1 for true. 该参数与OPEN_PARAM_QC最多只有一个能为1
#define OPEN_PARAM_PLA 					  "open_param_pla" // 是否开启疟原虫
/*!
 * AlgCell_XXXPushImage std::map key及含义,
 */
#define VIEW_PAIR_IDX                     "view_pair_idx"  // {"view_pair_idx:n"}  视图序号,n为序号


/*!
 * AlgCellImageCallback_f std::map key及含义
 */
#define TASK_TYPE                         "task_type"     // {"task_type":n}      //任务类型,n==0,计数;n==1,清晰度
#define TASK_TYPE_HEAMO                    0
#define TASK_TYPE_CLARITY                  1




/* 算法工作模式枚举 */ //值由../人医(动物,牛奶)机型.xml 中MachineIdx给定
typedef enum AlgCellModeID
{
	ALGCELL_MODE_NONE   = 0,
	ALGCELL_MODE_HUMAN  = 1,
	ALGCELL_MODE_ANIMAL = 2,
	ALGCELL_MODE_MILK   = 3,
}AlgCellModeID_e;


/* 算法分组 */ //值由../人医(动物,牛奶)机型.xml 中GroupIdx给定
typedef enum AlgCellGroupID{
  ALG_CELL_GROUP_MILK = 0,
  ALG_CELL_GROUP_HUMAN = 1,
  ALG_CELL_GROUP_CAT = 2,
  ALG_CELL_GROUP_DOG = 3,
}AlgCellGroupID_e;


/* 算法功能掩码 */
typedef enum AlgFunc
{
	ALGCELL_FUNC_HEAMO = (1 << 0),
	ALGCELL_FUNC_HGB = (1 << 1),
	ALGCELL_FUNC_FUSION = (1 << 2),

}AlgFunc_e;

/* 算法视图类型枚举 */
typedef enum AlgCellViewType
{
	ALGCELL_VIEW_TYPE_BRI = 1,
	ALGCELL_VIEW_TYPE_FLU,
}AlgCellViewType_e;

/* 算法图像 */
typedef struct AlgCellImg
{
	uint8_t *data;
	uint32_t size;
	uint32_t width;
	uint32_t height;
}AlgCellImg_t;
#define ALGCELL_IMG_DATA(img)				((img)->data)
#define ALGCELL_IMG_SIZE(img)				((img)->size)
#define ALGCELL_IMG_WIDTH(img)				((img)->width)
#define ALGCELL_IMG_HEIGHT(img)				((img)->height)

/* 算法结果项目 */
typedef struct AlgCellItem
{
	std::string name;
	std::string unit;
        std::string value;
}AlgCellItem_t;
#define ALGCELL_ITEM_NAME(item)             ((item)->name)      // 项目名称
#define ALGCELL_ITEM_UNIT(item)             ((item)->unit)      // 单位
#define ALGCELL_ITEM_VALUE(item)            ((item)->value)     // 数值

/* 算法结果 */
typedef struct AlgCellRst
{
	std::list<AlgCellItem_t> heamo;
	std::vector<float> curve_rbc;
	std::vector<float> curve_plt;
        std::vector<std::string> alarm_results;
}AlgCellRst_t;


/* AI图像处理阶段枚举 */
typedef enum AlgCellStage
{
	ALGCELL_IMG_STATGE_UNDEFINED = 0,
	ALGCELL_IMG_STAGE_RESIZE,
	ALGCELL_IMG_STAGE_INTERENCE,
	ALGCELL_IMG_STAGE_CLARITY,
}AlgCellStage_e;

/**
* 算法图像回调定义
* @param  ctx_id		算法上下文ID
* @param  group_idx		分组索引
* @param  chl_idx		通道索引
* @param  view_order		针对清晰度算法,值为相同视野下的拍摄序号
* @param  view_idx		视图索引
* @param  processed_idx		处理结果图像索引
* @param  image			图像缓存
* @param  stage			图像处理阶段
* @param  userdata		用户数据
* @param  view_pair_idx		视图序号
* @param  call_back_params	call back传入的其他参数
* @return none
*/
typedef void (*AlgCellImageCallback_f)(AlgCtxID_t ctx_id, uint32_t group_idx, uint32_t chl_idx, uint32_t view_order, uint32_t view_idx,
                                       uint32_t processed_idx, AlgCellStage_e stage, AlgCellImg_t *img, void *userdata,
                                       const int& view_pair_idx, std::map<std::string, float> call_back_params);

std::string AlgCell_Version();

AlgCtxID_t AlgCell_Init(void);
int AlgCell_DeInit(AlgCtxID_t ctx_id);

int AlgCell_RunConfigLoad(AlgCtxID_t ctx_id, AlgCellModeID_e mode_id, const char *cfg_path);
int AlgCell_RunConfigUnload(AlgCtxID_t ctx_id);

int AlgCell_HeamoListGroup(AlgCtxID_t ctx_id, std::vector<uint32_t> &list);
int AlgCell_HeamoListChl(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx);
int AlgCell_HeamoListView(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx, uint32_t chl_idx);
int AlgCell_ClarityListGroup(AlgCtxID_t ctx_id, std::vector<uint32_t> &list);
int AlgCell_ClarityListChl(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx);
int AlgCell_ClarityListView(AlgCtxID_t ctx_id, std::vector<uint32_t> &list, uint32_t group_idx, uint32_t chl_idx);

/*!
 * 算法测量开启
 * @param ctx_id                算法上下文ID
 * @param func_mask             功能掩码
 * @param callback              回调函数
 * @param userdata
 * @param open_params           参考 AlgCell_XXXOpen std::map key及含义
 * @return
 */
int AlgCell_HeamoOpen(AlgCtxID_t ctx_id, uint32_t func_mask, AlgCellImageCallback_f callback, void *userdata, const std::map<std::string, std::vector<float>>& open_params);

/*!
 * 算法测量推送血球图片
 * @param ctx_id                    算法上下文ID
 * @param image_list                图像组
 * @param group_idx                 分组索引
 * @param chl_idx                   通道索引
 * @param complementary_params      补充传递参数--  <"view_pair_idx", 0>, 表示该组图像的视野序号为0
 * @return
 */
int AlgCell_HeamoPushImage(AlgCtxID_t ctx_id, std::vector<AlgCellImg_t> &image_list, uint32_t group_idx, uint32_t chl_idx, const std::map<std::string, float>& complementary_params);
int AlgCell_HeamoPushHgb(AlgCtxID_t ctx_id, const std::vector<AlgHgbData_t> &data_list, const std::vector<float> &coef_list);
int AlgCell_HeamoGetResult(AlgCtxID_t ctx_id, AlgCellRst_t &result, uint32_t timeout);
int AlgCell_HeamoClose(AlgCtxID_t ctx_id);

/**
 * @brief  : 获取当前已经处理的图像的数量，按流道进行计算，每个流道 明暗场处理一次，计数 加1
 * @param   zero_flag 是否将全局计数置零的标志，如果为true，将全局计数清0, 表示开启新一轮的测试，重新计数
 * @return 返回已经计数的数量
 * @note   :
**/
int AlgCell_HeamoGetCount(bool zero_flag);

/*!
 *
 * @param
 * @param changed_param_key	改变的字段key,如WBC
 * @param result[out]		下发的前端更改后参数,返回算法修改后参数
 * @return
 */
int AlgCell_ModifyResult(const std::string& changed_param_key, AlgCellRst_t &result);

/*!
 * 细菌计数结果保存
 * @param save_dir 保存文件目录
 * @return
 */
int AlgCell_HeamoPushGermResultDir(const std::string& save_dir);

int AlgCell_ClarityOpen(AlgCtxID_t ctx_id,uint32_t func_mask, AlgCellImageCallback_f callback, void *userdata, const std::map<std::string, std::vector<float>>& open_params);


/*!
 * 聚焦函数接口
 * @param ctx_id 上下文
 * @param group_idx 分组,值目前只能是0
 * @param chl_idx 聚焦算法, 0:梯度最高峰, 1:AI粗聚焦_白细胞, 2:AI聚焦_白细胞, 3:AI聚焦_嗜碱细胞, 4:AI粗聚焦_荧光微球, 5:AI细聚焦_荧光微球, 6:AI粗聚焦_嗜碱, 7:牛奶底板划线聚焦
 * @param img_array 图像数组指针
 * @param array_size 数组大小
 * @param complementary_params      补充传递参数--  <"view_pair_idx", 0>, 表示该组图像的视野序号为0
 * @return
 */
int AlgCell_ClarityPushImage(AlgCtxID_t ctx_id, uint32_t group_idx, uint32_t chl_idx, AlgCellImg_t *img_array, uint32_t array_size, const std::map<std::string, float>& complementary_params);
int AlgCell_ClarityGetResultAll(AlgCtxID_t ctx_id, std::vector<AlgClarityValue_t> &list);
int AlgCell_ClarityGetResultLast(AlgCtxID_t ctx_id, uint32_t *index, AlgClarityValue_t *value);
int AlgCell_ClarityGetResultBest(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value);
/*!
 * 获取远近焦清晰度函数结果
 * @param ctx_id 算法上下文
 * @param index 索引输出指针
 * @param value 输出输出指针;0:焦平面下方, 1:焦平面上方, 2:粗聚焦, 3:清晰, 4:不确定, 5:换视野
 * @return 0 success other fail
 */
int AlgCell_ClarityGetResultFarNear(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value);

/*!
 * 获取粗聚焦结果
 * @param ctx_id 算法上下文
 * @param index 索引输出指针
 * @param value 输出输出指针; 0:焦平面上方,清晰或焦平面下方, 1:完全不存在细胞或无法区分焦平面上,下方
 * @return 0 success other fail
 */
int AlgCell_ClarityGetResultCoarse(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value);

/*!
 * 获取荧光微球细聚焦结果
 * @param ctx_id 算法上下文
 * @param index 索引输出指针
 * @param value 输出输出指针; 0:焦平面上方,清晰或焦平面下方, 1:完全不存在细胞或无法区分焦平面上,下方
 * @return 0 success other fail
 */
int AlgCell_ClarityGetResultFineFluMicrosphere(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value);


/*!
 * 获取荧光微球粗聚焦结果
 * @param ctx_id 算法上下文
 * @param index 索引输出指针
 * @param value 输出输出指针;0:焦平面下方, 1:焦平面上方, 3:清晰, 4:不确定,
 * @return 0 success other fail
 */
int AlgCell_ClarityGetResultCoarseFluMicrosphere(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value);

/*!
 * 获取荧光微球粗聚焦结果
 * @param ctx_id 算法上下文
 * @param index 索引输出指针
 * @param value 输出输出指针;0：在焦，1：细正焦， 2：细负焦， 3：粗正焦， 4：粗负焦， 5：远离正焦， 6：远离负焦， 7：不确定
 * @return 0 success other fail
 */
int AlgCell_ClarityGetResultMilkBoardLine(AlgCtxID_t ctx_id,  uint32_t *index, AlgClarityValue_t *value);

int AlgCell_ClarityClose(AlgCtxID_t ctx_id);

int AlgCell_Stop(AlgCtxID_t ctx_id, uint32_t timeout);

int AlgCell_ParseOpenParams(const std::map<std::string, float>& open_params,
                            bool& debug, uint32_t& group_idx, bool& qc);
#endif /* _LIBALG_H_ */
