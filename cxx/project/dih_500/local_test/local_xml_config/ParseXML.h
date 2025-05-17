//
// Created by y on 23-8-3.
//
#ifndef MY_OWN_TEST_PARSEXML_H
#define MY_OWN_TEST_PARSEXML_H
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "tinyxml2.h"
namespace ALG_LOCAL{
namespace XML{

////////
//单元测试
////////
//图像路径
struct UnitTestDataDir{
  std::string input_bright_dir;//明场图像目录
  std::string input_fluo_dir;//荧光场图像目录
  std::string save_dir;//输出保存目录
};

//单元，血球单个算法的配置，如rbc
struct UnitAlgConfig{
  bool enable{false};//是否启用算法
  std::string alg_name;//算法名,如 "rbc"
  std::vector<std::string> model_paths_v;//算法相关路径
  std::vector<float> float_param_v;//算法配置参数,如nms, conf_thr...
};
//血球单个辅助函数的配置
struct AssistFunctionConfig{
  bool enable{false};
  std::string function_name;//函数名
};

//血球检测类型配置,如human
struct UnitDetectTypeInitConfig{
  bool enable{false};//是否启用当前检测类型
  bool debug{false};
  std::string detect_type_name;//检测类型名,如"human"
  std::vector<UnitAlgConfig> alg_configs_v;//当前检测类型下的各个算法配置
  std::vector<AssistFunctionConfig> assist_function_config_v;
};

//////////
//整体测试类型配置
//////////

//检测项配置
struct IntSampleConfig{
  bool enable{false};//是否启用当前算法大类,如计数或hgb...
  std::string sample_name;//检测类型，
  std::vector<float> float_param_v;//算法参数
};
//检测类型配置
struct IntDetectTypeInitConfig{
  bool enable{false};//是否启用当前检测类型,如human
  bool debug{false};//是否保存中间结果
  std::string detect_type_name;//检测项名称
  std::vector<IntSampleConfig> sample_config_v;//检测项配置
};
//目录配置
struct IntTestDataDir{
  std::string card_info_dir;//试剂卡信息目录
  std::string data_info_dir;//数据目录
  std::vector<std::vector<std::string>> channel_img_dir;//存储通道及相应的图像文件目录
  std::string save_dir;     //输出保存目录
};
///////////////////
//通用类型
//////////////////
//记录单元测试或整体测试
struct TestTypeConfig{
  bool unit_test{false};
  bool integration_test{false};
};
//用于解析xml的类
class ParseXML
{
 public:
  ParseXML();
  ~ParseXML();

  /*!
 * 解析xml的总入口
 * @param xml_path xml文件路径
 * @param[out] test_dirs
 * @param[out] int_test_dirs
 * @param[out] init_params
 * @param[out] int_init_param
 * @param[out] test_type_param
 * @return
   */
  bool ReadParaXml(const std::string& xml_path,
                   UnitTestDataDir& test_dirs,IntTestDataDir& int_test_dirs,
                   std::vector<UnitDetectTypeInitConfig>& init_params,
                   std::vector<IntDetectTypeInitConfig>& int_init_param,
                   TestTypeConfig& test_type_param);



  bool ReadUnitDirConfig(tinyxml2::XMLElement* next_element, UnitTestDataDir& test_dirs);

  bool ReadIntDirChannelConfig(tinyxml2::XMLElement* box_element, std::vector<std::string>& channel_img_dirs);

  bool ReadIntDirConfig(tinyxml2::XMLElement* next_element, IntTestDataDir& int_test_dirs);

  /*!
   * 读取xml文件中的ImageDir node
   * @param next_element
   * @param image_dirs
   * @return 读取是否成功
   */
  bool ReadImageDirConfig(tinyxml2::XMLElement* next_element, UnitTestDataDir& test_dirs, IntTestDataDir& int_test_dirs);

  /*!
             * 读取节点的属性
             * @param next_element 节点
             * @param[out] attribute_map 节点属性map, output
   */
  static void ReadNodeAttribute(tinyxml2::XMLElement* next_element, std::map<std::string, std::string>&attribute_map);

  /*!
             * 将节点属性赋值给按需求赋值给 name enable debug
             * @param attribute_map 节点属性
             * @param[out] name
             * @param[out] enable
             * @param[out] debug
             * @param name_exist
             * @param enable_exist
             * @param debug_exist
             * @return
   */
  bool AttributeConfig(std::map<std::string, std::string> attribute_map, std::string& name, bool& enable, bool& debug,
                       const bool& name_exist, const bool& enable_exist, const bool& debug_exist);

  /*!
             * 读取检测类型下单个alg的配置
             * @param box_element
             * @param[out] one_alg
             * @return
   */
  bool ReadUnitAlgConfig(tinyxml2::XMLElement* box_element, UnitAlgConfig& one_alg);

  /*!
             * 读取辅助函数的配置
             * @param box_element
             * @param[out] one_func
             * @return
   */
  bool ReadAssitFunctionConfig(tinyxml2::XMLElement* box_element, AssistFunctionConfig& one_func);

  /*!
             * 读取单元测试配置的入口
             * @param next_element
             * @param[out] init_params
             * @param[out] test_type_param
             * @return
   */
  /*!
 * 读取xml文件中的DetectType node
 * @param next_element 检测类型xml元素指针，human or animal or ...
 * @param[out] detect_type_init_param 单元测试检测类型初始化参数
 * @return 读取是否成功
   */
  bool ReadUnitDetectTypeConfig(tinyxml2::XMLElement* next_element, UnitDetectTypeInitConfig& detect_type_init_param);

  bool ReadUnitConfig(tinyxml2::XMLElement* next_element, std::vector<UnitDetectTypeInitConfig> &init_params, TestTypeConfig& test_type_param);

  /*!
             * 读取整体测试配置的入口
             * @param next_element
             * @param[out] int_init_param
             * @param[out] test_type_param
             * @return
   */
  bool ReadIntConfig(tinyxml2::XMLElement* next_element,std::vector<IntDetectTypeInitConfig> & int_init_param, TestTypeConfig& test_type_param);

  /*!
             * 读取整体测试单个检测类型的配置, 如human
             * @param next_element
             * @param[out] detect_type_init_param
             * @return
   */
  bool ReadIntDetectTypeConfig(tinyxml2::XMLElement* next_element, IntDetectTypeInitConfig& detect_type_init_param);

  /*!
             * 单个检测类型下的配置,如计数,清晰度,...
             * @param box_element
             * @param[out] sample_config
             * @return
   */
  bool ReadIntSampleConfig(tinyxml2::XMLElement* box_element, IntSampleConfig& sample_config);
};
}
}

#endif //MY_OWN_TEST_PARSEXML_H
