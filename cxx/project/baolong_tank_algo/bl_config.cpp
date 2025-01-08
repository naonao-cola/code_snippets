#include "bl_config.h"
#include "utils.h"
#include "logger.h"
// using namespace BL_CONFIG;

// namespace BL_CONFIG {
    void BlParameter::from_bl_json(const json &json_file, bl_config& bl_json) {
        // LOG_INFO("json_file: {}", json_file.dump());
        if ( json_file.contains("type_id")) {
            json_file.at("type_id").get_to(bl_json.type_id);
        }
        json_file.at("device_id").get_to(bl_json.device_id);
        json_file.at("paras").at("detect_classes").get_to(bl_json.params.detect_classes);
        json_file.at("paras").at("model_path").get_to(bl_json.params.model_path);
        json_file.at("template").at("img_path").get_to(bl_json.templates->img_path);
        for(int i = 0; i < 2; i++){
            json_file.at("template").at("shapes")[i].at("points").get_to<std::vector<std::vector<int>>>(bl_json.templates->shapes[i].point);
            json_file.at("template").at("shapes")[i].at("label").get_to(bl_json.templates->shapes->label);
            json_file.at("template").at("shapes")[i].at("shape_type").get_to(bl_json.templates->shapes->shape_type);
        }
        return ;
    }

    bool BlParameter::const_char_to_json(const char* json_str, json& m_config) {
        std::string utf8_config_json_str = AnsiToUtf8(std::string(json_str));
        m_config = json::parse(utf8_config_json_str);
        return true;
    }

    void BlParameter::from_bl_json(const json &json_file, in_param& bl_json){
        if ( json_file.contains("type_id")) {
            json_file.at("type_id").get_to(bl_json.type_id);
        }
        json_file.at("img_name").get_to(bl_json.img_name);
        json_file.at("img_path").get_to(bl_json.img_path);
        json_file.at("img_w").get_to(bl_json.img_w);
        json_file.at("img_h").get_to(bl_json.img_h);
        json_file.at("channel").get_to(bl_json.channel);
    }

    bool BlParameter::result_to_json(json& result_info, bl_config model_config, std::vector<std::string> result){
            json s1, s2;
            result_info["class_list"] = model_config.params.detect_classes;
            s1["image_name"] = model_config.type_id;
            s1["status"] = "OK";
            int i = 0;
            if(!result.empty()){
                for (std::vector<std::string>::iterator it = result.begin(); it < result.end(); it++){
                    s2["label"] = *it;
                    s2["points"] = {{0, 0},{640, 640}};
                    json text;
                    text["confidence"] = 1.00;
                    s2["result"] = text;
                    s2["shapeType"] = "polygon";
                    s1["shapes"][i++] = s2;
                }
            }else{
                s1["shapes"] = json::array();
            }
            result_info["label_set"][0] = s1;
        return true;
    }

    bool BlParameter::result_to_json2(json& result_info, in_param model_config, std::vector<std::string>& result){
            json s1, s2;
            result_info["class_list"] = NULL;
            s1["image_name"] = model_config.img_name;
            s1["status"] = "OK";
            int i = 0;
            if(!result.empty()){
                for (std::vector<std::string>::iterator it = result.begin(); it < result.end(); it++){
                    s2["label"] = *it;
                    result.erase(it);
                    s2["points"] = {{0, 0},{640, 640}};
                    json text;
                    text["confidence"] = acc;
                    s2["result"] = text;
                    s2["shapeType"] = "polygon";
                    s1["shapes"][i++] = s2;
                }
            }else{
                s1["shapes"] = json::array();
            }
            result_info["label_set"][0] = s1;
        return true;
    }

    bool BlParameter::result_to_json3(json& result_info, in_param model_config, std::vector<std::string>& result, img_check imgcheck){
            json s1, s2, s3;
            result_info["class_list"] = NULL;
            s1["image_name"] = model_config.img_name;
            s1["status"] = imgcheck.check_value;
            s3["mean_gv"] = {imgcheck.gv_value[0], imgcheck.gv_value[1], imgcheck.gv_value[2]};
            s1["image_check"] = s3;
            int i = 0;
            if(!result.empty()){
                for (std::vector<std::string>::iterator it = result.begin(); it < result.end(); it++){
                    s2["label"] = *it;
                    result.erase(it);
                    s2["points"] = {{0, 0},{640, 640}};
                    json text;
                    text["confidence"] = acc;
                    s2["result"] = text;
                    s2["shapeType"] = "polygon";
                    s1["shapes"][i++] = s2;
                }
            }else{
                s1["shapes"] = json::array();
            }
            result_info["label_set"][0] = s1;
        return true;
    }

    // bool BlParameter::result_to_json_arry(json& result_arry, json result_info) {
    //     json s1;
    //     s1 = json::arry();
    //     s1.accept(result_info);
    //     s1.contains(result_arry);
    //     s1.get(result_info);
    //     result_info = s1;
    //     return true;
    // }
  
    void BlParameter::from_new_bl_json(const json &json_file, new_bl_config& bl_json) {
        // LOG_INFO("json_file: {}", json_file.dump());
        if ( json_file.contains("type_id")) {
            json_file.at("type_id").get_to(bl_json.type_id);
        }
        if ( json_file.contains("device_id")) {
            json_file.at("device_id").get_to(bl_json.device_id);
        }
        if ( json_file.contains("model_path")) {
            json_file.at("model_path").get_to(bl_json.model_path);
        }
        
		Param cv_param;
		for (auto i = json_file["params"].begin(); i != json_file["params"].end(); i++){
            
            if ( i->contains("detect_item")) {
                i->at("detect_item").get_to(cv_param.detect_Item);
                
            }
            if (i->contains("score")) {
                i->at("score").get_to(cv_param.score);
            }
            // if ( i->at("param").contains("threshold")) {
            //     i->at("param").at("threshold").get_to(cv_param.param.threshold);
            // }
            // if ( i->at("param").contains("radius_min_distance")) {
            //     i->at("param").at("radius_min_distance").get_to(cv_param.param.radius_min_distance);
            // }
            // if ( i->at("param").contains("centerX")) {
            //     i->at("param").at("centerX").get_to(cv_param.param.centerX);
            // }
            // if ( i->at("param").contains("centerY")) {
            //     i->at("param").at("centerY").get_to(cv_param.param.centerY);
            // }
            
			// if (i->contains("box_param")){
			// 	std::cout << "sdasdasdasdasdas" << std::endl;
			// }
			bl_json.params.push_back(cv_param);
		}

        // if (json_file.at("template").contains("img_path")){
        //     json_file.at("template").at("img_path").get_to(bl_json.templates->img_path);
        // }
        
        // for (int i = 0; i < 2; i++) {
        //     if (json_file.at("template").at("shapes")[i].contains("points")){
        //         json_file.at("template").at("shapes")[i].at("points").get_to<std::vector<std::vector<int>>>(bl_json.templates->shapes[i].point);
        //     }
        //     if (json_file.at("template").at("shapes")[i].contains("label")){
        //         json_file.at("template").at("shapes")[i].at("label").get_to(bl_json.templates->shapes->label);
        //     }
        //     if (json_file.at("template").at("shapes")[i].contains("shape_type")){
        //         json_file.at("template").at("shapes")[i].at("shape_type").get_to(bl_json.templates->shapes->shape_type);
        //     }
            
        // }
	
        // for (auto Param: bl_json.params){
        // 	std::cout << "detect_item=" << Param.detect_Item << std::endl;
        // 	std::cout << "centerX=" << Param.param.centerX << std::endl;
        // 	std::cout << "centerY=" << Param.param.centerY << std::endl;
        // 	std::cout << "radius_min_distance=" << Param.param.radius_min_distance << std::endl;
        // 	std::cout << "label=" << Param.param.box_param.label << std::endl;
        // 	std::cout << "threshold=" << Param.param.threshold << std::endl;
        // }
    // return;
    }
// }


