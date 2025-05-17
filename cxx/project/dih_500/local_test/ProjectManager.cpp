//
// Created by y on 23-8-4.
//
#include <iostream>
#include "ProjectManager.h"
#include "ParseXML.h"
namespace ALG_LOCAL {
bool ProjectManager::Init() {
  std::cout<<"unit test:"<<this->test_type_config.unit_test<<std::endl;
  std::cout<<"int test:"<<this->test_type_config.integration_test<<std::endl;

  if(this->test_type_config.integration_test&&this->test_type_config.unit_test){
    std::cout<<"Can not initialize UnitTest and IntegrationTest on the simultaneously. "<<std::endl;
    return false;
  }
  if(this->test_type_config.integration_test) {// 初始化整体测试
      if(!this->int_test.Init(this->int_detect_type_config_v, this->int_test_data_dir)){
        std::cout<<"fail to init int test."<<std::endl;
        return false;
      }
      std::cout<<"init int test succeed."<<std::endl;
    } else if(this->test_type_config.unit_test){// 初始化单元测试
      if(!this->unit_test.Init(this->test_data_dir, this->detect_type_config_v)){
        std::cout<<"fail to init unit test."<<std::endl;
        return false;
      }
      std::cout<<"init unit test succeed."<<std::endl;
    }

  return true;

}

bool ProjectManager::Forward(){
  if(this->test_type_config.integration_test){

      if(!this->int_test.TestSample()){// 整体测试推理
        std::cout<<"Fail to forward init test."<<std::endl;
        return false;
      }
      std::cout<<"Forward int test succeed."<<std::endl;
      return true;
  } else if(this->test_type_config.unit_test){// 单元测试推理
      if(! this->unit_test.ForwardAllDetectTypes()){
        std::cout<<"Fail to forward unit test."<<std::endl;
        return false;
      }
      std::cout<<"Forward unit test succeed."<<std::endl;
      return true;
  }
}


bool ProjectManager::GetInitParams(const std::string& xml_path) {
  XML::ParseXML parse_xml{};

  if(!parse_xml.ReadParaXml(xml_path, this->test_data_dir, this->int_test_data_dir,
                             this->detect_type_config_v,this->int_detect_type_config_v,
                             this->test_type_config)){
    return false;
  }
  return true;
}




}

