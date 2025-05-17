//
// Created by y on 23-8-4.
//
//本地测试用
#ifndef RKNN_ALG_DEMO_PROJECTMANAGER_H
#define RKNN_ALG_DEMO_PROJECTMANAGER_H
#include <memory>
#include <exception>
#include <sys/stat.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ParseXML.h"
#include "IntegratedCode.h"
#include "UnitTest.h"
#include "replace_std_string.h"
namespace ALG_LOCAL{

class ProjectManager{
 public:
  ProjectManager()=default;
  ~ProjectManager()=default;
  /*!
     * 将xml文件中的配置读取到结构中
     * @param xml_path xml文件路径
     * @return 读取是否成功
   */
  bool GetInitParams(const std::string& xml_path);
  /*!
     * 根据读取的xml中的配置，对算法进行初始化
     * @return 初始化是否成功
   */
  bool Init();

  /*！
     * 根据xml中配置的待测试图像路径推理所有算法
   */
  bool Forward();
  XML::TestTypeConfig test_type_config;//存储选择整体测试或者单元测试
  //unit
  XML::UnitTestDataDir test_data_dir;//存储xml中图像相关路径配置
  std::vector<XML::UnitDetectTypeInitConfig> detect_type_config_v;//存储xml中算法相关配置

  //intregrateion
  XML::IntTestDataDir int_test_data_dir;//存储xml中图像相关路径配置
  std::vector<XML::IntDetectTypeInitConfig> int_detect_type_config_v;//

  //std::shared_ptr<DetectHuman>human {new DetectHuman()};
  std::map<std::string, AlgType> alg_name_to_type_m;//将字符串算法类型映射为枚举型
  std::map<std::string, DetectTypeName> detect_name_to_type_m;//将字符串算法类型映射为枚举型
  UNIT::UnitTest unit_test;
  INTEGRATE::IntegratedCode int_test;

};

}

#endif //RKNN_ALG_DEMO_PROJECTMANAGER_H
