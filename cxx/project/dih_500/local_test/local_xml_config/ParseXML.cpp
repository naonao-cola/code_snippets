//
// Created by y on 23-8-3.
//
#include "ParseXML.h"
#include <sstream>
// 元素名称定义
#define ROOT_ELT              "Setting"
#define DIR_ELT               "Dir"
#define UNIT_DIR_ELT          "UnitDir"
#define INPUT_BRIGHT_DIR_ELT  "input_bright_dir"
#define INPUT_FLUO_DIR_ELT    "input_fluo_dir"
#define SAVE_DIR_ELT          "save_dir"
#define INTEGRATION_DIR_ELT   "IntegrationDir"
#define CARD_INFO_DIR_ELT     "card_info_dir"
#define DATA_INFO_DIR_ELT     "data_info_dir"
#define CHANNEL_ELT           "channel"
#define INPUT_IMG_DIR_ELT     "input_img_dir"
#define UNIT_TEST_ELT         "UnitTest"
#define DETECT_TYPE_ELT       "DetectType"
#define ALG_ELT               "alg"
#define MODEL_PATH_ELT        "model_path"
#define FLOAT_PARAM_V_ELT     "float_param_v"
#define ASSIST_FUNCTION_ELT   "assist_function"
#define INTEGRATION_TEST_ELT  "IntegrationTest"
#define SAMPLE_TYPE_ELT       "SampleType"
#define DEBUG_ELT             "debug"

//属性名称定义
#define NAME_ATT              "name"
#define ENABLE_ATT            "enable"
#define TRUE_ATT              "true"



namespace ALG_LOCAL{
namespace XML {
ParseXML::ParseXML() = default;
ParseXML::~ParseXML() = default;

void SplitCharacter(const std::string& strs,  const char& delim, std::vector<std::string>& splited) {

  if (strs == "") return;
  std::stringstream sstr(strs);
  std::string token;
  while (getline(sstr, token, delim))
  {
    splited.push_back(token);
  }

}

bool ParseXML::ReadUnitDirConfig(tinyxml2::XMLElement* next_element, UnitTestDataDir& test_dirs){
  tinyxml2::XMLElement *box_element = next_element->FirstChildElement(INPUT_BRIGHT_DIR_ELT);
  if (!box_element) {
    std::cout << "XML error, UnitDir config input_bright_dir dose not exist." << std::endl;
    return false;
  }
  test_dirs.input_bright_dir = std::string(box_element->GetText());

  box_element = next_element->FirstChildElement(INPUT_FLUO_DIR_ELT);
  if (!box_element) {
    std::cout << "XML error, UnitDir config input_fluo_dir dose not exist." << std::endl;
    return false;
  }
  test_dirs.input_fluo_dir = std::string(box_element->GetText());

  box_element = next_element->FirstChildElement(SAVE_DIR_ELT);
  if (!box_element) {
    std::cout << "XML error, UnitDir config save_dir dose not exist." << std::endl;
    return false;
  }
  test_dirs.save_dir = std::string(box_element->GetText());



  return true;
}

bool ParseXML::ReadIntDirChannelConfig(tinyxml2::XMLElement* box_element, std::vector<std::string>& channel_img_dirs){
  //read alg config
  tinyxml2::XMLElement *line_element = box_element->FirstChildElement();
  for (; line_element; line_element = line_element->NextSiblingElement()) {
    if (std::string(line_element->Value()) == INPUT_IMG_DIR_ELT) {
      //预留了更多空白model_path，所以读取时需要判断
      if (line_element->GetText()) {
        channel_img_dirs.emplace_back(line_element->GetText());
      }
    } else {
      std::cout << "XML error, wrong config in input_img_dir node." << std::endl;
      return false;
    }
  }
  return true;
}

bool ParseXML::ReadIntDirConfig(tinyxml2::XMLElement* next_element, IntTestDataDir& int_test_dirs) {
  //immune card
  tinyxml2::XMLElement* box_element =
      next_element->FirstChildElement(CARD_INFO_DIR_ELT);
  if (!box_element) {
    std::cout
        << "XML error, IntegrationDir config card_info_dir dose not exist."
        << std::endl;
    return false;
  }
  int_test_dirs.card_info_dir = std::string(box_element->GetText());
  //immune data
  box_element = next_element->FirstChildElement(DATA_INFO_DIR_ELT);
  if (!box_element) {
    std::cout
        << "XML error, IntegrationDir config data_info_dir dose not exist."
        << std::endl;
    return false;
  }
  int_test_dirs.data_info_dir = std::string(box_element->GetText());

  //heamo channel
  box_element = next_element->FirstChildElement(CHANNEL_ELT);
  for (; box_element; box_element = box_element->NextSiblingElement()) {
    if (std::string(box_element->Value()) == CHANNEL_ELT) {
      std::vector<std::string> channel_img_dirs;
      if (!ReadIntDirChannelConfig(box_element, channel_img_dirs)) {
        return false;
      }
      int_test_dirs.channel_img_dir.emplace_back(channel_img_dirs);
    }
  }

  //heamo save_dir
  box_element = next_element->FirstChildElement(SAVE_DIR_ELT);
  if (!box_element) {
    std::cout
        << "XML error, IntegrationDir config save_dir dose not exist."
        << std::endl;
    return false;
  }
  int_test_dirs.save_dir = std::string(box_element->GetText());
  return true;
}




bool ParseXML::ReadImageDirConfig(tinyxml2::XMLElement *next_element, UnitTestDataDir& test_dirs,IntTestDataDir& int_test_dirs) {
  next_element = next_element->FirstChildElement(DIR_ELT);
  if (!next_element) {
    std::cout << "XML error, Dir node dose not exist." << std::endl;
    return false;
  }
  //unit
  tinyxml2::XMLElement *volume_element = next_element->FirstChildElement(UNIT_DIR_ELT);
  if (!volume_element) {
    std::cout << "XML error, UnitDir node dose not exist." << std::endl;
    return false;
  }
  if(!ReadUnitDirConfig(volume_element, test_dirs)){
    std::cout << "XML error, failed to read UnitDir node." << std::endl;
    return false;
  }

  //integration
  volume_element = next_element->FirstChildElement(INTEGRATION_DIR_ELT);
  if (!volume_element) {
    std::cout << "XML error, IntegrationDir node dose not exist." << std::endl;
    return false;
  }
  if(!ReadIntDirConfig(volume_element, int_test_dirs)){
    std::cout << "XML error, failed to read IntegrationDir node." << std::endl;
    return false;
  }


  return true;
}


bool ParseXML::ReadParaXml(const std::string& xml_path,
                           UnitTestDataDir& test_dirs,IntTestDataDir& int_test_dirs,
                           std::vector<UnitDetectTypeInitConfig>& init_params,
                           std::vector<IntDetectTypeInitConfig>& int_init_param,
                           TestTypeConfig& test_type_param) {

  //读取xml文件中的参数值
  tinyxml2::XMLDocument doc;
  tinyxml2::XMLError ret = doc.LoadFile(xml_path.c_str());
  if (ret != 0) {
    std::cout << "fail to load xml file: " + xml_path<<std::endl;
    return false;
  }


  tinyxml2::XMLElement *root_element = doc.RootElement();        //根目录

  tinyxml2::XMLElement *next_element = root_element;        //根目录下的第一个节点层

  if (std::string(next_element->Value()) == ROOT_ELT)        //读到root节点
  {
    //Dir
    //Use Value() to read tag, Use GetText() to read values
    if (!this->ReadImageDirConfig(next_element, test_dirs, int_test_dirs)) {
      std::cout << "Error happened on ImageDir node." << std::endl;
      return false;
    }
    //UnitTest
    if(!this->ReadUnitConfig(next_element, init_params, test_type_param)){
      std::cout << "Error happened on UnitTest node." << std::endl;
      return false;
    }
    //IntTest
    if(!this->ReadIntConfig(next_element, int_init_param, test_type_param)){
      std::cout << "Error happened on IntTest node." << std::endl;
      return false;
    }

  } else {
    std::cout << "XML error, root node wrong." << std::endl;
    return false;
  }
  return true;

}

void ParseXML::ReadNodeAttribute(tinyxml2::XMLElement *next_element,
                                 std::map<std::string, std::string> &attribute_map) {
  tinyxml2::XMLElement *p_element = next_element->ToElement();
  for (const tinyxml2::XMLAttribute *p_attribute = p_element->FirstAttribute(); p_attribute; p_attribute = p_attribute->Next()) {

    std::string attribute_one_name = p_attribute->Name();
    std::string attribute_one_value = p_attribute->Value();

    if (!attribute_one_value.empty()) {
      attribute_map.insert(
          std::map<std::string, std::string>::value_type(attribute_one_name, attribute_one_value));
    }
  }
}


bool ParseXML::AttributeConfig(std::map<std::string, std::string> attribute_map,
                               std::string& name, bool& enable, bool& debug,
                               const bool& name_exist, const bool& enable_exist,const bool& debug_exist){
  if(name_exist){
    if ( attribute_map.find(NAME_ATT) != attribute_map.end()) {
      name = attribute_map[NAME_ATT];
    }else{
      std::cout << "XML error, attribute name dose not exist name." << std::endl;
    }
  }
  if(enable_exist){
    if (enable_exist&& attribute_map.find(ENABLE_ATT) != attribute_map.end()) {
      enable = (attribute_map[ENABLE_ATT] == TRUE_ATT);
    }else{
      std::cout << "XML error, attribute enable dose not exist enable." << std::endl;
    }
  }
  if(debug_exist){
    if (debug_exist&& attribute_map.find(DEBUG_ELT)!=attribute_map.end()){
      debug = (attribute_map[DEBUG_ELT]==TRUE_ATT);
    } else{
      std::cout << "XML error, attribute debug dose not exist debug." << std::endl;
    }
  }

  return true;
}

bool ParseXML::ReadUnitDetectTypeConfig(tinyxml2::XMLElement *next_element,
                                    UnitDetectTypeInitConfig &detect_type_init_param) {

  //read detect type attribute
  std::map<std::string, std::string> detect_type_attribute_m;
  this->ReadNodeAttribute(next_element, detect_type_attribute_m);

  if(!this->AttributeConfig(detect_type_attribute_m,
                             detect_type_init_param.detect_type_name,
                             detect_type_init_param.enable,
                             detect_type_init_param.debug, true,
                             true, true)){
    std::cout<<"XML error, error happened on detect type."<<std::endl;
    return false;
  }

  //read all alg config under the detect type
  tinyxml2::XMLElement *box_element = next_element->FirstChildElement();
  for (; box_element; box_element = box_element->NextSiblingElement()) {
    if (std::string(box_element->Value()) == ALG_ELT) {
      UnitAlgConfig one_alg;

      if(!ReadUnitAlgConfig(box_element, one_alg)){
        return false;
      }
      detect_type_init_param.alg_configs_v.emplace_back(one_alg);
    }
    //read assist_function
    else if(std::string(box_element->Value()) == ASSIST_FUNCTION_ELT){
      AssistFunctionConfig one_func;
      if(!ReadAssitFunctionConfig(box_element, one_func)){
        return false;
      }

      detect_type_init_param.assist_function_config_v.emplace_back(one_func);
    }

  }
  return true;
}


void SplitFloatParam(const std::string& float_param_string, std::vector<float>& float_param_v, const char& delim){
  std::vector<std::string> float_param_string_v;
  SplitCharacter(float_param_string, delim, float_param_string_v);
  for(auto iter: float_param_string_v){
    float_param_v.emplace_back(atof(iter.c_str()));
  }

}

bool ParseXML::ReadUnitAlgConfig(tinyxml2::XMLElement* box_element, UnitAlgConfig& one_alg){
  //read attribute
  std::map<std::string, std::string> alg_attribute;
  this->ReadNodeAttribute(box_element, alg_attribute);
  bool useless_debug{false};
  if(!this->AttributeConfig(alg_attribute, one_alg.alg_name,
                             one_alg.enable, useless_debug, true,
                             true, false)){
    std::cout<<"XML error， error happened on alg type."<<std::endl;
    return false;
  }

  //read alg config
  tinyxml2::XMLElement *line_element = box_element->FirstChildElement();
  for (; line_element; line_element = line_element->NextSiblingElement()) {
    if (std::string(line_element->Value()) == MODEL_PATH_ELT) {
      //预留了更多空白model_path，所以读取时需要判断
      if (line_element->GetText()) {

        one_alg.model_paths_v.emplace_back(line_element->GetText());
      }

    }else if(std::string(line_element->Value()) == FLOAT_PARAM_V_ELT){
      if (line_element->GetText()) {
        SplitFloatParam(line_element->GetText(), one_alg.float_param_v, ',');
      }
    }


    else {
      std::cout << "XML error, wrong config in alg." << std::endl;
      return false;
    }
  }

  return true;
}

bool ParseXML::ReadAssitFunctionConfig(tinyxml2::XMLElement* box_element, AssistFunctionConfig& one_func){
  std::map<std::string, std::string> func_attribute;
  this->ReadNodeAttribute(box_element, func_attribute);
  try{
    one_func.function_name = func_attribute.at(NAME_ATT);
    if(func_attribute.at(ENABLE_ATT) ==TRUE_ATT){
      one_func.enable= true;
    } else{
      one_func.enable= false;
    }
  }catch(std::exception &e){
    std::cout<<"XML error， assit function type dose not exist name or enable param."<<std::endl;
    return false;
  }
  return true;
}


bool TransformStrToBool(const std::string &str) {
  if (str == TRUE_ATT) {
    return true;
  } else {
    return false;
  }
}

bool ParseXML::ReadUnitConfig(tinyxml2::XMLElement* next_element, std::vector<UnitDetectTypeInitConfig> &init_params,
                              TestTypeConfig& test_type_param){

  next_element = next_element->FirstChildElement(UNIT_TEST_ELT);
  if(!next_element){
    std::cout<<"XML error, UnitTest node dose not exist."<<std::endl;
    return false;
  }
  std::map<std::string, std::string> unit_attribute;
  this->ReadNodeAttribute(next_element, unit_attribute);
  try{
    if(unit_attribute.at(ENABLE_ATT)==TRUE_ATT){
      test_type_param.unit_test = true;
    } else{
      test_type_param.unit_test = false;
    }
  }catch (std::exception &e){
    std::cout<<"XML error, enable attribute of UnitTest  dose not exist."<<std::endl;
  }


  tinyxml2::XMLElement* volume_element = next_element->FirstChildElement(DETECT_TYPE_ELT);
  for (; volume_element && std::string(volume_element->Value()) ==
                               DETECT_TYPE_ELT; volume_element = volume_element->NextSiblingElement()) {
    UnitDetectTypeInitConfig detect_type_init_config;
    if (!ReadUnitDetectTypeConfig(volume_element, detect_type_init_config)) {
      std::cout << "Error happened on DetectType." << std::endl;
      return false;
    }

    init_params.emplace_back(detect_type_init_config);
  }
  return true;
}

bool ParseXML::ReadIntConfig(tinyxml2::XMLElement* next_element, std::vector<IntDetectTypeInitConfig> & int_init_param,
                             TestTypeConfig& test_type_param){
  next_element = next_element->FirstChildElement(INTEGRATION_TEST_ELT);
  if(!next_element){
    std::cout<<"XML error, IntegrationTest  node dose not exist"<<std::endl;
    return false;
  }

  std::map<std::string, std::string> int_attribute;
  this->ReadNodeAttribute(next_element, int_attribute);
  try{
    test_type_param.integration_test=(int_attribute.at(ENABLE_ATT)==TRUE_ATT);
  }catch (std::exception &e){
    std::cout<<"XML error, enable attribute of IntegrationTest   dose not exist."<<std::endl;
  }

  tinyxml2::XMLElement* volume_element = next_element->FirstChildElement(DETECT_TYPE_ELT);
  for (; volume_element && std::string(volume_element->Value()) ==
                               DETECT_TYPE_ELT; volume_element = volume_element->NextSiblingElement()) {
    IntDetectTypeInitConfig detect_type_init_config;
    if (!ReadIntDetectTypeConfig(volume_element, detect_type_init_config)) {
      std::cout << "Error happened on DetectType." << std::endl;
      return false;
    }

    int_init_param.emplace_back(detect_type_init_config);
  }
  return true;
}
bool ParseXML::ReadIntDetectTypeConfig(tinyxml2::XMLElement* next_element, IntDetectTypeInitConfig& detect_type_init_param){
  //read detect type attribute
  std::map<std::string, std::string> detect_type_attribute_m;
  this->ReadNodeAttribute(next_element, detect_type_attribute_m);
//  bool useless{false};
  if(!this->AttributeConfig(detect_type_attribute_m,
                             detect_type_init_param.detect_type_name,
                             detect_type_init_param.enable,
                             detect_type_init_param.debug, true,
                             true, true)){
    std::cout<<"XML error, error happened on DetectType."<<std::endl;
    return false;
  }

  //read all alg config under the detect type
  tinyxml2::XMLElement *box_element = next_element->FirstChildElement();
  for (; box_element; box_element = box_element->NextSiblingElement()) {
    if (std::string(box_element->Value()) == SAMPLE_TYPE_ELT) {
      IntSampleConfig sample_config;

      if(!ReadIntSampleConfig(box_element, sample_config)){
        return false;
      }
      detect_type_init_param.sample_config_v.emplace_back(sample_config);
    } else{
      std::cout<<"XML error, unexpected value " + std::string(box_element->Value())+" is set in SampleType."<<std::endl;
      return false;
    }

  }
  return true;

}


bool ParseXML::ReadIntSampleConfig(tinyxml2::XMLElement* box_element, IntSampleConfig& sample_config){
  //read attribute
  std::map<std::string, std::string> sample_attribute;
  this->ReadNodeAttribute(box_element, sample_attribute);
  bool useless_debug{false};
  if(!this->AttributeConfig(sample_attribute, sample_config.sample_name,
                             sample_config.enable,useless_debug,
                             true, true, false)){
    std::cout<<"XML error， error happened on SampleType."<<std::endl;
    return false;
  }
  //read all alg config under the detect type
  tinyxml2::XMLElement *line_element = box_element->FirstChildElement();
  for (; line_element; line_element = line_element->NextSiblingElement()) {
    if (std::string(line_element->Value()) == FLOAT_PARAM_V_ELT) {
      SplitFloatParam(line_element->GetText(), sample_config.float_param_v, ',');

    } else{
      std::cout<<"XML error, unexpected value " + std::string(box_element->Value())+" is set in float_param_v."<<std::endl;
      return false;
    }

  }
  return true;

}


}
}