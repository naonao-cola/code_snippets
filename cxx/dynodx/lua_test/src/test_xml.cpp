/**
 * @FilePath     : /test/src/test_xml.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-05-19 13:03:27
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-05-19 17:06:58
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/

#include "tinyxml2.h"
#include <cstring>
#include <iostream>
#include <string>
int main()
{
    std::string           xml_path = "/home/naonao/demo/cxx/test/src/model_cfg.xml";
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError    ret = doc.LoadFile(xml_path.c_str());

    if (ret != 0) {
        std::cout << "fail to load xml file: " + xml_path << std::endl;
        return -1;
    }
    // 读取
    tinyxml2::XMLElement* root_element = doc.RootElement();   // 根目录

    if (std::string(root_element->Value()) == "ModelSetting") {

        tinyxml2::XMLElement* volume_element = root_element->FirstChildElement("DetectType");
        // 循环读取 DetectType
        for (tinyxml2::XMLElement* iter = root_element->FirstChildElement("DetectType"); iter; iter = iter->NextSiblingElement("DetectType")) {
            const char* name        = iter->Attribute("name");
            const char* elementName = iter->Name();
            if (strcmp(name, "clarity") == 0) {
                // model 节点 的列表
                std::cout << "=================================================================================" << std::endl;
                std::cout << "Element Attribute name : " << name << std::endl;
                for (tinyxml2::XMLElement* model_iter = iter->FirstChildElement("model"); model_iter; model_iter = model_iter->NextSiblingElement("model")) {
                    int         mod_id                     = std::stoi(model_iter->FirstChildElement("mod_id")->GetText());
                    std::string name                       = model_iter->FirstChildElement("name")->GetText();
                    int         multi_label_flag           = std::stoi(model_iter->FirstChildElement("multi_label_flag")->GetText());
                    int         fusion_rate                = std::stoi(model_iter->FirstChildElement("fusion_rate")->GetText());
                    int         group_mask                 = std::stoi(model_iter->FirstChildElement("group_mask")->GetText());
                    int         letterbox                  = std::stoi(model_iter->FirstChildElement("letterbox")->GetText());
                    int         model_type_nums            = std::stoi(model_iter->FirstChildElement("model_type_nums")->GetText());
                    int         nms_nums                   = std::stoi(model_iter->FirstChildElement("nms_nums")->GetText());
                    int         conf_nums                  = std::stoi(model_iter->FirstChildElement("conf_nums")->GetText());
                    int         anchor_nums                = std::stoi(model_iter->FirstChildElement("anchor_nums")->GetText());
                    int         label_nums                 = std::stoi(model_iter->FirstChildElement("label_nums")->GetText());
                    int         reserved_float_param_nums  = std::stoi(model_iter->FirstChildElement("reserved_float_param_nums")->GetText());
                    int         reserved_string_param_nums = std::stoi(model_iter->FirstChildElement("reserved_string_param_nums")->GetText());
                    std::cout << "=========================================" << std::endl;
                    std::cout << "      mod_id : " << mod_id << std::endl;
                    std::cout << "      name : " << name << std::endl;
                    std::cout << "      multi_label_flag : " << multi_label_flag << std::endl;
                    std::cout << "      fusion_rate : " << fusion_rate << std::endl;
                    std::cout << "      group_mask : " << group_mask << std::endl;
                    std::cout << "      letterbox : " << letterbox << std::endl;
                    std::cout << "      model_type_nums : " << model_type_nums << std::endl;
                    std::cout << "      nms_nums : " << nms_nums << std::endl;
                    std::cout << "      conf_nums : " << conf_nums << std::endl;
                    std::cout << "      anchor_nums : " << anchor_nums << std::endl;
                    std::cout << "      label_nums : " << label_nums << std::endl;
                    std::cout << "      reserved_float_param_nums : " << reserved_float_param_nums << std::endl;
                    std::cout << "      reserved_string_param_nums : " << reserved_string_param_nums << std::endl;
                }
            }
        }
    }
}


