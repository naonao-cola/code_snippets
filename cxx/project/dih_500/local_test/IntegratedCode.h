//
// Created by y on 23-8-11.
//

#ifndef RKNN_ALG_DEMO_INTEGRATEDCODE_H
#define RKNN_ALG_DEMO_INTEGRATEDCODE_H


#define MAX_IMG_PAIRS 20000
#define MAX_IMG_NUMS_UNDER_PAIR 5
#include <iostream>
#include <map>
#include "ParseXML.h"
#include "libalgcell.h"
#include "libalgimm.h"
namespace ALG_LOCAL{
namespace INTEGRATE {
//任务类型
enum IntegratedTestType{
  ALG_SAMPLE_FIELD = 0,
  ALG_SAMPLE_HGB=1,
  ALG_SAMPLE_IMMUNE=2,
  ALG_SAMPLE_CLARITY=3,
  ALG_SAMPLE_HYBRID=4,
};



bool ReadImgToBuf(struct AlgCellImg *img_buf, const std::string& img_path, const bool& flip_img= true);
void AlgCellImageCallback_f (AlgCtxID_t ctx_id, uint32_t group_idx, uint32_t chl_idx, uint32_t view_order, uint32_t view_idx, uint32_t processed_idx, AlgCellStage_e stage, AlgCellImg_t *img, void *userdata,
                            const int& view_pair_idx, std::map<std::string, float> call_back_params);


/*!
 * 将字符串中的指定字符串进行替换
 * @param str 需要进行处理的字符串
 * @param old_value 被替换字符串
 * @param new_value 替换为
 * @return
 */
void ReplaceAllDistinct(std::string& str, const std::string& old_value, const std::string& new_value);

class IntegratedCode {
 public:
  IntegratedCode(){};
  ~IntegratedCode();
  /*!
   * 初始化接口
   * @param[in] int_detect_config 整体测试初始化参数
   * @return
   */
  bool Init(const std::vector<XML::IntDetectTypeInitConfig>& int_detect_config, const XML::IntTestDataDir& int_test_data_dir);

  /*!
   * 初始化机型
   * @param detect_type_name 机型名称
   * @return
   */
  bool InitMachine(const std::string& detect_type_name);

  /*!
   * 初始化血球,获取血球测试所需的图像目录
   * @param int_detect_type_init_config 算法参数配置
   * @param int_test_data_dir 整体测试目录配置
   * @return
   */
  bool InitHeamo(const XML::IntDetectTypeInitConfig& int_detect_type_init_config, const XML::IntTestDataDir& int_test_data_dir);

  bool InitClarity(const XML::IntDetectTypeInitConfig& int_detect_type_init_config, const XML::IntTestDataDir& int_test_data_dir);

  bool InitImmune(const XML::IntDetectTypeInitConfig& int_detect_type_init_config, const XML::IntTestDataDir& int_test_data_dir);

  /*!
   * 测试接口
   * @return
   */
  bool TestSample();
 private:
  /*!
   * 计数测试入口
   * @return
   */
  bool TestAlgSampleField(int times=1);

  bool TestAlgSampleHgb();


  /*!
   * 免疫测试入口
   * @return
   */
  bool TestAlgSampleImmune();


  bool TestAlgSampleClarity(int times=1);

  bool ClarityFindClearest(const std::vector<std::string>& img_path_v,
                           std::vector<uint32_t>&index_v,
                           std::vector<float>& clarity_v,
                           const int& view_pair_idx);
  bool ClarityCheckEach(const std::vector<std::string>& img_path_v,
                        std::vector<uint32_t>&index_v,
                        std::vector<float>& clarity_v,
                        const int& view_pair_idx);





  bool HeamoThread();
  bool ClarityThread();
  /*!
   * 多线程测试血球及清晰度算法
   * @return
   */
  bool TestAlgSampleHybrid();
  /*!
   * 打印结果
   */
  void PrintResult(const IntegratedTestType& test_type);

  /*!
   * 将字符串任务类型转换为枚举型
   */
  void MapAlgTypeName();

  /*！
   * 将字符串检测类型转换为group枚举
   */
  void MapSampleTypeName();

  /*!
   * 将字符串检测类型转换为机型枚举
   */
  void MapMachineType();


  //机型
  std::vector<IntegratedTestType> test_type;//存储已初始化的任务类型
  std::map<std::string, IntegratedTestType> alg_type_name_to_type_m;//将字符串算法类型映射为枚举型
  std::map<std::string, int> sample_type_name_to_type_m;//将字符串算法类型映射为枚举型
  std::map<std::string, AlgCellModeID> machine_type_name_to_mode_m;//将机型映射为模式枚举型

  // 血球
  AlgCtxID_t algctx;//血球ctx
  std::vector<float> hgb_coef;
  std::vector<std::vector<std::string>> identified_heamo_channel_config;//筛选后的血球通道信息
  std::vector<std::string> clarity_dir_v;//存储用于检测清晰度算法的目录
  std::string save_dir;//存储中间结果的目录
  int group_idx = 0; //组索引
  int clarity_outer_group_idx = 0; //清晰度算法组id
  int clarity_channel_id = 0; //清晰度算法id
  int img_pair_nums_deemed_same_view = 1; //将两连续读取的x个图像对视为同一视野
  bool debug = false; //是否启用debug
  bool qc = false;    //是否启用质量控制
  bool calib = false;  //是否启用校准计数
  float img_h;
  float img_w;
  float img_h_um;
  std::map<std::string, std::vector<float>> open_params; //open参数
  //免疫
  AlgImmCtxID_t immunectx;//免疫ctx
  std::string card_info_dir;
  std::string data_info_dir;//
  int calibration = 0;
  float coef = 1.f;
};

}
}


#endif //RKNN_ALG_DEMO_INTEGRATEDCODE_H
