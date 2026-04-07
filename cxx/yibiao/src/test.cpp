
/**
 * @FilePath     : /yibiao/src/test.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-03-31 10:48:17
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-04-03 15:33:07
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#include "gauge_mask.hpp"
#include "httplib.h"
#include <iostream>
#include <opencv2/opencv.hpp>



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
            detectionValue        = 0.0;
            std::vector<char> imageDataVec;
            std::copy(imageData.begin(), imageData.end(), std::back_inserter(imageDataVec));
            cv::Mat srcImage = cv::imdecode(cv::Mat(imageDataVec), cv::IMREAD_COLOR);
            detectionValue   = meterValue;

            GaugePipelineDebug debug;
            GaugeReadResult    result;
            bool               success = ReadGaugePipeline(srcImage, 0, 0, 0, 0.5f, 0.5f, result, &debug);
            if (!success) {
                detectionValue   = meterValue;
                detectionMessage = "Detection fail";
                return false;
            }
            else {
                detectionValue   = result.ratio * (meterTop - meterBottom);
                detectionMessage = "Detection successful";
            }
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
