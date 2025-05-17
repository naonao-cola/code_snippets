//
// Created by y on 24-8-5.
//
#include <iostream>
#include <sstream>

#include "../tinyxml2/tinyxml2.h"
#include "ModelConfig.h"
#include "DihLogPlog.h"
namespace ALG_DEPLOY {
namespace XML {


/*!
 * ??????????????????
 * @param strs
 * @param delim ???
 * @param[out] splited
 */

void SplitCharacter(const std::string& strs, const char& delim,
                    std::vector<std::string>& splited) {
  if (strs == "") return;
  std::stringstream sstr(strs);
  std::string token;
  while (getline(sstr, token, delim)) {
    splited.push_back(token);
  }
}

/*!
 * ???????????????float vector
 * @param float_param_string
 * @param[out] float_param_v
 * @param delim ???
 */
void SplitFloatParam(const std::string& float_param_string, const char& delim,
                     std::vector<float>& float_param_v) {
  std::vector<std::string> float_param_string_v;
  SplitCharacter(float_param_string, delim, float_param_string_v);
  for (auto iter : float_param_string_v) {
    float_param_v.emplace_back(atof(iter.c_str()));
  }
}

int ModelConfig::ReadXmlFile(const std::string& xml_path, ConfigParams& cfg) {
  ALGLogInfo << "Model config path " << xml_path ;
  // ???xml???????????
  tinyxml2::XMLDocument doc;
  tinyxml2::XMLError ret = doc.LoadFile(xml_path.c_str());
  if (ret != 0) {
    ALGLogError << "Fail to load xml file: " + xml_path ;
    return -1;
  }

  tinyxml2::XMLElement* root_element = doc.RootElement();  // ????

  tinyxml2::XMLElement* next_element = root_element;  // ???????????????

  if (std::string(next_element->Value()) == ROOT_NAME) {
    if (this->ReadModelConfig(next_element, cfg)) {
      ALGLogError << "Error happened on ModelConfig." << std::endl;
      return -2;
    }
  } else {
    ALGLogError << "XML error, root node wrong." << std::endl;
    return -3;
  }
  return 0;
}
int ModelConfig::ReadModelConfig(tinyxml2::XMLElement* next_element,
                                 ConfigParams& cfg) {
  // read all alg config under the detect type
  tinyxml2::XMLElement* line_element = next_element->FirstChildElement();
  for (; line_element; line_element = line_element->NextSiblingElement()) {
    auto element_value = line_element->GetText();
    if (element_value == nullptr) {
//      ALGLogInfo << "Empty params in " << line_element->Value() << std::endl;
      continue;
    }
    if (std::string(line_element->Value()) == MODEL_TYPE) {
      SplitFloatParam(element_value, ',', cfg.model_type);
    } else if (std::string(line_element->Value()) == NMS) {
      SplitFloatParam(element_value, ',', cfg.nms);
    }else if (std::string(line_element->Value()) == CONF) {
      SplitFloatParam(element_value, ',', cfg.conf);
    } else if (std::string(line_element->Value()) == ANCHORS) {
      SplitFloatParam(element_value, ',', cfg.anchors);
    } else if (std::string(line_element->Value()) == LABELS) {
      SplitCharacter(element_value, ',', cfg.labels);
    } else if (std::string(line_element->Value()) == RESERVED_FLOAT_PARAMS) {
      SplitFloatParam(element_value, ',', cfg.reserved_float_params);
    } else if (std::string(line_element->Value()) == RESERVED_STRING_PARAMS) {
      SplitCharacter(element_value, ',', cfg.reserved_string_params);
    } else {
      ALGLogError << "Found unknown name " << line_element->Value() << " in xml";
      return -1;
    }
  }
  return 0;
}
}
}