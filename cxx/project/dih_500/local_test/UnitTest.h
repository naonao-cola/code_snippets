//
// Created by y on 23-8-14.
//

#ifndef RKNN_ALG_DEMO_UNITTEST_H
#define RKNN_ALG_DEMO_UNITTEST_H
#include <memory>
#include <exception>
#include <sys/stat.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "DetectType.h"
#include "DetectHuman.h"
#include "ParseXML.h"
#include "replace_std_string.h"
namespace ALG_LOCAL{
namespace UNIT{
class UnitTest {
 public:
  ~UnitTest();
  /*!
     * 根据读取的xml中的配置，对算法进行初始化
     * @return 初始化是否成功
   */
  bool Init(const XML::UnitTestDataDir& img_dir_config,
            const std::vector<XML::UnitDetectTypeInitConfig>& detect_type_config_v);//存储xml中算法相关配置);

  /*！
     * 根据xml中配置的待测试图像路径推理所有算法
   */
  bool ForwardAllDetectTypes();

  bool ForwardOneType(ALG_LOCAL::DetectType * detect_type, const bool& debug,
                      const std::string& detect_type_name,
                      const std::vector<std::string>& alg_names_inside_detect,
                      const std::vector<std::string>& bright_image_path_v,
                      const std::vector<std::string>& fluo_image_path_v);
  /*!
     * 将字符串算法类型映射为枚举型
   */
  void MapAlgName();

  /*!
     * 将字符串检测类型映射为枚举型
   */
  void MapDetectName();

  bool InitOneType(const XML::UnitDetectTypeInitConfig& detect_type_config);

/*  ALG::DetectHuman *human;//人医成员
  bool debug_human{false};//是否debu人医算法
  std::vector<std::string> human_algs;*/

  std::vector<ALG_LOCAL::DetectType *> detect_type_object_v;
  std::vector<bool> detect_type_debug_v;
  std::vector<std::string> detect_type_names_v;
  std::vector<std::vector<std::string>> detect_type_alg_names_v;


  XML::UnitTestDataDir img_dir_config;//存储xml中图像相关路径配置
  std::vector<XML::UnitDetectTypeInitConfig> detect_type_config_v;//存储xml中单元测试相关配置

  XML::TestTypeConfig test_type_config;//存储选择整体测试或者单元测试
  std::vector<XML::IntDetectTypeInitConfig> int_detect_type_config_v;//存储xml中整体测试相关配置


  //std::shared_ptr<DetectHuman>human {new DetectHuman()};
  std::map<std::string, AlgType> alg_name_to_type_m;//将字符串算法类型映射为枚举型
  std::map<std::string, DetectTypeName> detect_name_to_type_m;//将字符串算法类型映射为枚举型

};
}
}



#endif  // RKNN_ALG_DEMO_UNITTEST_H
