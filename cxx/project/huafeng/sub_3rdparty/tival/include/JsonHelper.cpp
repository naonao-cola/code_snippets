//#include "JsonHelper.h"
//#include "easylogger.h"
//#include "CommonDefine.h"
//#include "StringConvert.h"
//#include <cmath>
//#include <sstream>
//
//using namespace Tival;
//
//std::any JsonHelper::GetParamAny(const json& param, const std::string& key, const std::any& def_val)
//{
//    if (param.contains(key)) {
//        if (param[key].is_string()) {
//            return param[key].get<std::string>();
//        } else if (param[key].is_number_integer()) {
//            return param[key].get<int>();
//        } else if (param[key].is_number_float()) {
//            return param[key].get<double>();
//        } else if (param[key].is_boolean()) {
//            return param[key].get<bool>();
//        } else {
//            return def_val;
//        }
//    } else {
//        return def_val;
//    }
//}
//
//json JsonHelper::ReadJsonFile(std::string filepath)
//{
//    std::ifstream conf_i(filepath);
//    std::stringstream ss_config;
//    ss_config << conf_i.rdbuf();
//    json jsonObj = json::parse(ss_config.str());
//    return std::move(jsonObj);
//}
//
//
//json JsonHelper::ParseJsonText(const char* json_text, bool is_ansi)
//{
//    std::string utf8_text = is_ansi ? StringConvert::AnsiToUtf8(std::string(json_text)) : std::string(json_text);
//    json jsonObj = json::parse(utf8_text);
//    return std::move(jsonObj);
//}
//
//const char* JsonHelper::DumpJson(json jsonObj, bool toAnsi)
//{
//    return toAnsi ? StringConvert::Utf8ToAnsi(jsonObj.dump().c_str()) : jsonObj.dump().c_str();
//}
//
//json JsonHelper::FromTRotateObjList(const std::vector<TRotateRect>& rrects)
//{
//    json rst = json::array();
//    for (auto rrect : rrects) {
//        rst.push_back(FromTRRectObj(rrect));
//    }
//    return rst;
//}
//
//json JsonHelper::FromStringList(const std::vector<std::string>& slist)
//{
//    json rst = json::array();
//    for (auto str : slist) {
//        rst.push_back(str);
//    }
//    return rst;
//}
//
//json JsonHelper::FromTRRectObj(const TRotateRect& rrect)
//{
//    return { rrect.X, rrect.Y, rrect.Angle, rrect.Length1, rrect.Length2 };
//}