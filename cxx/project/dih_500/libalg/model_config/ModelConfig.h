//
// Created by y on 24-8-5.
//

#ifndef MODELCONFIG_H
#define MODELCONFIG_H

#include <vector>
#include <string>

#include "../tinyxml2/tinyxml2.h"

namespace ALG_DEPLOY {
namespace XML {
#define ROOT_NAME               "ModelConfig"
#define MODEL_TYPE              "ModelType"
#define NMS                     "Nms"
#define CONF                    "Conf"
#define ANCHORS                 "Anchors"
#define LABELS                  "Labels"
#define RESERVED_FLOAT_PARAMS   "ReservedFloatParams"
#define RESERVED_STRING_PARAMS  "ReservedStringParam"




/////////////
//部署代码 模型配置
////////////

struct ConfigParams{
  std::vector<float> model_type;
  std::vector<float> nms;
  std::vector<float> conf;
  std::vector<float> anchors;
  std::vector<std::string> labels;
  std::vector<float> reserved_float_params;
  std::vector<std::string> reserved_string_params;
};

class ModelConfig {
 public:
  ModelConfig()=default;
  ~ModelConfig()=default;
  int ReadXmlFile(const std::string& xml_path, ConfigParams& cfg);
 private:
  int ReadModelConfig(tinyxml2::XMLElement* next_element, ConfigParams& cfg);


};

}
}
#endif//MODELCONFIG_H
