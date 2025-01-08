#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <any>
#include "Geometry.h"

using json = nlohmann::json;

namespace Tival
{
    class JsonHelper
    {
    public:
        static std::any GetParamAny(const json& param, const std::string& key, const std::any& def_val);
        static json ReadJsonFile(std::string filepath);
        static json ParseJsonText(const char* json_text, bool is_ansi=true);
        static const char* DumpJson(json jsonObj, bool toAnsi=true);

        template<typename T>
        static T GetParam(const json& param, const std::string& key, const T& def_val)
        {
            if (param.contains(key)) {
                return param[key].get<T>();
            } else {
                return def_val;
            }
        }

        // 数据转换
        static json FromTRRectObj(const TRotateRect& rrect);
        static json FromTRotateObjList(const std::vector<TRotateRect>& rrects);
        static json FromStringList(const std::vector<std::string>& slist);

        template<typename T>
        static json FromValList(const std::vector<T>& valList) {
            json rst = json::array();
            if constexpr (std::is_same<T, cv::Point2i>::value ||
                std::is_same<T, int>::value ||
                std::is_same<T, long>::value ||
                std::is_same<T, double>::value ||
                std::is_same<T, float>::value ||
                std::is_same<T, std::string>::value) {
                for (auto val : valList) {
                    rst.push_back(val);
                }
            } else {
                T tmp;
                LOGW("Convert to json, Unsupported type: {}", type_id(tmp).name());
            }
            return rst;
        }
        // 一维CV对象数组转json
        template<typename T>
        static json FromCvObjList(const std::vector<T>& cvObjList)
        {
            json rst = json::array();
            for (T cvObj : cvObjList) {
                rst.push_back(FromCvObj(cvObj));
            }
            return rst;
        }

        // 二维CV对象数组转json
        template<typename T>
        static json FromCvObjArrList(const std::vector<std::vector<T>>& cvObjArrList)
        {
            json rst = json::array();
            for (auto cvObjArr : cvObjArrList) {
                json subArr = json::array();
                for (T cvObj : cvObjArr) {
                    subArr.push_back(FromCvObj(cvObj));
                }
                rst.push_back(subArr);
            }

            return rst;
        }

        template<typename T>
        static json FromCvObj(const T& cvObj)
        {
            if constexpr (std::is_same<T, cv::Point2i>::value ||
                std::is_same<T, cv::Point2f>::value ||
                std::is_same<T, cv::Point2d>::value ||
                std::is_same<T, cv::Point2l>::value) {
                return { cvObj.x, cvObj.y };
            } else if constexpr (std::is_same<T, cv::Size2i>::value ||
                std::is_same<T, cv::Size2f>::value ||
                std::is_same<T, cv::Size2d>::value ) {
                return { cvObj.width, cvObj.height };
            } else if constexpr (std::is_same<T, cv::Rect2i>::value ||
                std::is_same<T, cv::Rect2f>::value ||
                std::is_same<T, cv::Rect2d>::value ) {
                return { {cvObj.tl().x, cvObj.tl().y}, {cvObj.br().x, cvObj.br().y} };
            } else if constexpr (std::is_same<T, cv::RotatedRect>::value ) {
                cv::Point2f points[4];
                cvObj.points(points);
                return { cvObj.center.x, cvObj.center.y, cvObj.size.width, cvObj.size.height, cvObj.angle };
            } else if constexpr (std::is_same<T, cv::Mat>::value) {
                std::stringstream ss;
                ss<<(void*)cvObj.data;
                return {
                    {"addr", ss.str()},
                    {"img_w", cvObj.cols},
                    {"img_h", cvObj.rows},
                    {"img_c", cvObj.channels()}
                };
            } else {
                throw ("Convert to json, Unsupported type!!");
            }
        }
    };
};