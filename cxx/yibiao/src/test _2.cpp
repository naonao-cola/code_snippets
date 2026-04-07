/**
 * @FilePath     : /yibiao/src/test copy.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-03-31 10:48:17
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-03-31 10:48:17
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#include "httplib.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "../HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo/yolo.hpp"
#include "../HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_pose/yolo_pose.hpp"

// 模型路径
std::string det_modelPath  = "E:\\demo\\cxx\\yibiao\\data\\GAUGE_DET.engine";
std::string pose_modelPath = "E:\\demo\\cxx\\yibiao\\data\\GAUGE_POINTS.engine";

std::shared_ptr<Yolo::Infer>     det_infer;
std::shared_ptr<YoloPose::Infer> pose_infer;

struct GaugeKeypoint
{
    cv::Point2f position;
    float       conf  = 0.f;
    int         index = -1;
};


struct AngleResult
{
    double angleAOB = 0.0;
    double angleAOC = 0.0;
    bool   alarm    = false;
};

double distance(const cv::Point2f& a, const cv::Point2f& b)
{
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return std::sqrt(dx * dx + dy * dy);
}

bool computeABCO(const std::vector<GaugeKeypoint>& redPoints,
                 const std::vector<GaugeKeypoint>& blackPoints,
                 GaugeKeypoint&                    aPoint,
                 GaugeKeypoint&                    bPoint,
                 GaugeKeypoint&                    cPoint,
                 GaugeKeypoint&                    oPoint,
                 cv::Rect                          bbox)
{
    if (blackPoints.size() < 2 || redPoints.size() < 4) {
        return false;
    }

    cv::Point2f bboxCenter(static_cast<float>(bbox.x + bbox.width / 2), static_cast<float>(bbox.y + bbox.height / 2));
    double      b1_dist = distance(blackPoints[0].position, bboxCenter);
    double      b2_dist = distance(blackPoints[1].position, bboxCenter);
    int         oIndex  = -1;
    if (b1_dist < b2_dist) {
        oPoint = blackPoints[0];
        oIndex = 0;
    }
    else {
        oPoint = blackPoints[1];
        oIndex = 1;
    }

    for (size_t i = 0; i < blackPoints.size(); ++i) {
        if (static_cast<int>(i) != oIndex) {
            cPoint = blackPoints[i];
            break;
        }
    }
    if (cPoint.index < 0) {
        return false;
    }
    std::vector<GaugeKeypoint> farRed;
    farRed.reserve(redPoints.size());
    for (const auto& red : redPoints) {
        if (distance(red.position, oPoint.position) > 30) {
            farRed.push_back(red);
        }
    }
    if (farRed.size() < 2) {
        std::vector<std::pair<double, GaugeKeypoint>> redWithDist;
        redWithDist.reserve(redPoints.size());
        for (const auto& red : redPoints) {
            redWithDist.emplace_back(distance(red.position, oPoint.position), red);
        }
        std::sort(redWithDist.begin(), redWithDist.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

        for (const auto& item : redWithDist) {
            if (farRed.size() < 2) {
                farRed.push_back(item.second);
            }
        }
    }
    if (farRed.size() < 2) {
        return false;
    }
    std::sort(farRed.begin(), farRed.end(), [](const GaugeKeypoint& lhs, const GaugeKeypoint& rhs) { return lhs.position.x < rhs.position.x; });
    aPoint = farRed[0];
    bPoint = farRed[1];
    return true;
}

double clockwiseAngle(const cv::Point2f& origin, const cv::Point2f& from, const cv::Point2f& to)
{
    const double v1x = static_cast<double>(from.x) - static_cast<double>(origin.x);
    const double v1y = static_cast<double>(from.y) - static_cast<double>(origin.y);
    const double v2x = static_cast<double>(to.x) - static_cast<double>(origin.x);
    const double v2y = static_cast<double>(to.y) - static_cast<double>(origin.y);

    const double angleFrom = std::atan2(-v1y, v1x);
    const double angleTo   = std::atan2(-v2y, v2x);
    double       deg       = (angleFrom - angleTo) * 180.0 / 3.14159265358979323846;
    if (deg < 0.0) {
        deg += 360.0;
    }
    else if (deg >= 360.0) {
        deg -= 360.0;
    }

    return deg;
}

AngleResult calculateAngles(const GaugeKeypoint& aPoint, const GaugeKeypoint& bPoint, const GaugeKeypoint& cPoint, const GaugeKeypoint& oPoint)
{
    AngleResult result;
    if (aPoint.index < 0 || bPoint.index < 0 || cPoint.index < 0 || oPoint.index < 0) {
        return result;
    }
    result.angleAOB = clockwiseAngle(oPoint.position, aPoint.position, bPoint.position);
    result.angleAOC = clockwiseAngle(oPoint.position, aPoint.position, cPoint.position);
    double BOC;
    // A B 两个点一个在左,一个在右
    if (bPoint.position.x < aPoint.position.x) {
        BOC = clockwiseAngle(oPoint.position, bPoint.position, cPoint.position);
    }
    else {
        BOC = clockwiseAngle(oPoint.position, cPoint.position, bPoint.position);
    }
    std::cout << "angleAOB:" << result.angleAOB << std::endl;
    std::cout << "angleAOC:" << result.angleAOC << std::endl;
    std::cout << "angleBOC:" << BOC << std::endl;
    if (result.angleAOC >= result.angleAOB || BOC >= result.angleAOB) {
        result.alarm = true;
    }
    // result.alarm    = result.angleAOC >= result.angleAOB;
    return result;
}




// 从 URL 中解析 host 和 path
void parseUrl(const std::string& url, std::string& host, std::string& path, int& port)
{
    size_t hostStart = url.find("://");
    if (hostStart != std::string::npos) {
        hostStart += 3;
    }
    else {
        hostStart = 0;
    }

    size_t pathStart = url.find('/', hostStart);
    if (pathStart != std::string::npos) {
        host = url.substr(hostStart, pathStart - hostStart);
        path = url.substr(pathStart);
    }
    else {
        host = url.substr(hostStart);
        path = "/";
    }

    // 检查是否有端口号
    size_t portPos = host.find(':');
    if (portPos != std::string::npos) {
        port = std::stoi(host.substr(portPos + 1));
        host = host.substr(0, portPos);
    }
    else {
        port = (url.find("https://") == 0) ? 443 : 80;
    }
}


void drawABCOLines(cv::Mat& image, const GaugeKeypoint& aPoint, const GaugeKeypoint& bPoint, const GaugeKeypoint& cPoint, const GaugeKeypoint& oPoint)
{
    if (oPoint.index < 0) {
        return;
    }

    const cv::Scalar colorA(255, 0, 0);
    const cv::Scalar colorB(255, 128, 0);
    const cv::Scalar colorC(0, 255, 255);
    const cv::Scalar colorO(0, 255, 0);

    const cv::Point2f oPos = oPoint.position;

    if (aPoint.index >= 0) {
        cv::line(image, oPos, aPoint.position, colorA, 2);
        cv::circle(image, aPoint.position, 8, colorA, 2);
        cv::circle(image, aPoint.position, 3, colorA, -1);
        cv::putText(image, "A", aPoint.position + cv::Point2f(10.f, -10.f), cv::FONT_HERSHEY_SIMPLEX, 0.6, colorA, 2);
    }

    if (bPoint.index >= 0) {
        cv::line(image, oPos, bPoint.position, colorB, 2);
        cv::circle(image, bPoint.position, 8, colorB, 2);
        cv::circle(image, bPoint.position, 3, colorB, -1);
        cv::putText(image, "B", bPoint.position + cv::Point2f(10.f, -10.f), cv::FONT_HERSHEY_SIMPLEX, 0.6, colorB, 2);
    }

    if (cPoint.index >= 0) {
        cv::line(image, oPos, cPoint.position, colorC, 2);
        cv::circle(image, cPoint.position, 8, colorC, 2);
        cv::circle(image, cPoint.position, 3, colorC, -1);
        cv::putText(image, "C", cPoint.position + cv::Point2f(10.f, -10.f), cv::FONT_HERSHEY_SIMPLEX, 0.6, colorC, 2);
    }

    cv::circle(image, oPos, 8, colorO, 2);
    cv::circle(image, oPos, 3, colorO, -1);
    cv::putText(image, "O", oPos + cv::Point2f(10.f, -10.f), cv::FONT_HERSHEY_SIMPLEX, 0.6, colorO, 2);
}


bool MeterDection(
    const std::string& imageUrl, double meterTop, double meterBottom, double meterValue, double& detectionValue, std::string& detectionMessage)
{
    // 输入字段
    std::cout << "meterTop: " << meterTop << std::endl;
    std::cout << "meterBottom: " << meterBottom << std::endl;
    std::cout << "meterValue: " << meterValue << std::endl;
    // 解析 URL
    std::string host, path;
    int         port;
    parseUrl(imageUrl, host, path, port);

    // 创建 HTTP 客户端
    httplib::Client cli(host, port);

    // 发送 GET 请求获取图片
    auto res = cli.Get(path);

    if (res) {
        if (res->status == 200) {
            // 获取图片数据
            std::string imageData = res->body;
            // std::cout << "[MeterDection] Get image. Image size: " << imageData.size() << " byte" << std::endl;
            detectionValue = 0.0;
            std::vector<char> imageDataVec;
            std::copy(imageData.begin(), imageData.end(), std::back_inserter(imageDataVec));
            cv::Mat srcImage = cv::imdecode(cv::Mat(imageDataVec), cv::IMREAD_COLOR);
            detectionValue   = meterValue;

            // 检测
            // if (det_infer.get() != nullptr && pose_infer.get() != nullptr) {
            //    auto boxs = det_infer->commit(srcImage).get();
            //    if (boxs.size() > 0) {
            //        for (auto& box : boxs) {
            //            if (box.confidence < 0.5) {
            //                continue;
            //            }
            //            // 检测到仪表
            //            std::cout << "box: " << box.left << " " << box.top << " " << box.right << " " << box.bottom << " " << box.confidence << " "
            //                      << box.class_label << std::endl;
            //            cv::Rect roi(cv::Point(box.left, box.top), cv::Point(box.right, box.bottom));
            //            if (roi.area() <= 0) {
            //                continue;
            //            }
            //            cv::Mat cropImage = srcImage(roi).clone();
            //            /*cv::imwrite("./cropImage.png", cropImage);*/
            //            // 进行关键点检测
            //            std::vector<GaugeKeypoint> redPoints;
            //            std::vector<GaugeKeypoint> blackPoints;
            //            auto                       objs = pose_infer->commit(cropImage).get();
            //            for (auto& obj : objs) {
            //                std::cout << "keypoints size: " << obj.keypoints.size() << std::endl;
            //                if (obj.keypoints.empty()) {
            //                    continue;
            //                }
            //                for (size_t idx = 0; idx < obj.keypoints.size(); ++idx) {
            //                    const cv::Point3f& kp = obj.keypoints[idx];
            //                    if (kp.z < 0.2f) {
            //                        continue;
            //                    }
            //                    GaugeKeypoint point;
            //                    point.position = cv::Point2f(kp.x + static_cast<float>(roi.x), kp.y + static_cast<float>(roi.y));
            //                    point.conf     = kp.z;
            //                    point.index    = static_cast<int>(idx);
            //                    if (obj.cls == 0) {
            //                        redPoints.push_back(point);
            //                    }
            //                    else if (obj.cls == 1) {
            //                        blackPoints.push_back(point);
            //                    }
            //                }
            //            }
            //            GaugeKeypoint aPoint;
            //            GaugeKeypoint bPoint;
            //            GaugeKeypoint cPoint;
            //            GaugeKeypoint oPoint;
            //            if (computeABCO(redPoints, blackPoints, aPoint, bPoint, cPoint, oPoint, roi)) {
            //                AngleResult angleResult = calculateAngles(aPoint, bPoint, cPoint, oPoint);
            //                /*drawABCOLines(srcImage, aPoint, bPoint, cPoint, oPoint);
            //                cv::imwrite("./baojing.png", srcImage);*/
            //                if (angleResult.alarm) {
            //                    detectionValue = 1.0;
            //                    // detectionMessage = "Detection successful";
            //                    std::cout << "进行报警" << std::endl;
            //                }
            //                else {
            //                    std::cout << "不进行报警" << std::endl;
            //                }
            //            }
            //        }
            //    }
            //}
            // 可选：保存图片到本地
            /*std::ofstream outFile(imageUrl, std::ios::binary);
            if (outFile) {
                outFile.write(imageData.data(), imageData.size());
                outFile.close();
                std::cout << "[MeterDection] Image saved: " << imageUrl << std::endl;
            }*/
            // TODO: 在此处添加仪表检测逻辑
            detectionMessage = "Detection successful";

            return true;
        }
        else {
            std::cerr << "[MeterDection] HTTP failed: " << res->status << std::endl;
            detectionMessage = "Image fetch failed: " + std::to_string(res->status);
            return false;
        }
    }
    else {
        auto err = res.error();
        std::cerr << "[MeterDection] HTTP failed: " << httplib::to_string(err) << std::endl;
        detectionMessage = "Image fetch failed: " + httplib::to_string(err);
        return false;
    }
}

int main()
{
    httplib::Server svr;
    /*if (det_infer.get() == nullptr) {
        det_infer = Yolo::create_infer(det_modelPath, Yolo::Type::V8, 0, 0.25f, 0.5f);
        if (det_infer == nullptr) {
            std::cerr << "Failed to create YOLO inference instance." << std::endl;
            return -1;
        }
    }
    if (pose_infer.get() == nullptr) {
        pose_infer = YoloPose::create_infer(pose_modelPath, 0, 0.25f, 0.8f, YoloPose::NMSMethod::FastGPU, 1024, 2);
        if (pose_infer == nullptr) {
            std::cerr << "Failed to create YOLO Pose inference instance." << std::endl;
            return -1;
        }
    }*/
    // 定义一个简单的 GET 路由
    svr.Get("/TaskAssign", [](const httplib::Request& req, httplib::Response& res) {
        auto imageUrl    = req.get_param_value("imageUrl");
        auto meterTop    = req.get_param_value("meterTop");
        auto meterBottom = req.get_param_value("meterBottom");
        auto meterValue  = req.get_param_value("meterValue");

        if (imageUrl.empty() || ((meterTop.empty() || meterBottom.empty()) && meterValue.empty())) {
            std::ostringstream responseStream;
            responseStream << R"({"status" : 500)"
                           << R"(,"message": "incomplete data"})";
            res.set_content(responseStream.str(), "application/json; charset=utf-8");
            return;
        }

        std::cout << "Task coming, image path: " << imageUrl << std::endl;
        double meterTop_value, meterBottom_value, meterValue_value;
        if (meterTop.empty()) {
            meterTop_value = 0.0;
        }
        else {
            meterTop_value = std::stod(meterTop);
        }
        if (meterBottom.empty()) {
            meterBottom_value = 0.0;
        }
        else {
            meterBottom_value = std::stod(meterBottom);
        }
        if (meterValue.empty()) {
            meterValue_value = 0.0;
        }
        else {
            meterValue_value = std::stod(meterValue);
        }

        double      detectionValue = 0;
        std::string detectionMessage;
        if (MeterDection(imageUrl, meterTop_value, meterBottom_value, meterValue_value, detectionValue, detectionMessage)) {
            std::ostringstream responseStream;
            responseStream << R"({"status" : 200)"
                           << R"(,"value": )" << detectionValue << R"(,"message": ")" << detectionMessage << R"("})";

            res.set_content(responseStream.str(), "application/json; charset=utf-8");
        }
        else {
            std::ostringstream responseStream;
            responseStream << R"({"status" : 500)"
                           << R"(,"message": ")" << detectionMessage << R"("})";

            res.set_content(responseStream.str(), "application/json; charset=utf-8");
        }
    });

    int port = 5245;
    std::cout << "HTTP Server running on http://localhost:" << port << std::endl;
    svr.listen("0.0.0.0", port);

    return 0;
}
