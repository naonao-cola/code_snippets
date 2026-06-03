/**
 * @FilePath     : /pcl_test/include/lidar_detector.hpp
 * @Description  : 激光雷达碰撞检测 C++ 类（用于 pybind11 / C++ 直接调用）
 * @Author       : weiwei.wang
 * @Date         : 2026-06-06
 * @Version      : 0.0.1
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#ifndef LIDAR_DETECTOR_HPP
#define LIDAR_DETECTOR_HPP

#include <memory>
#include <string>
#include <vector>

struct ObjectInfo
{
    int   object_id;
    int   point_count;
    float center_x;
    float center_y;
    float center_z;
};

struct CollisionPair
{
    int   obj_id_a;
    int   obj_id_b;
    float distance_m;
};

struct DetectResult
{
    std::vector<ObjectInfo>    objects;
    std::vector<CollisionPair> collisions;
};

class LidarDetector
{
public:
    explicit LidarDetector(const std::string& yaml_config_path);
    ~LidarDetector();

    LidarDetector(const LidarDetector&)            = delete;
    LidarDetector& operator=(const LidarDetector&) = delete;

    bool isValid() const;

    bool start();
    void stop();

    void         capture();
    void         encodeImagesToBase64(std::string& color_b64, std::string& depth_b64);
    std::vector<float> getFullCloud();
    DetectResult detectOnce();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif   // LIDAR_DETECTOR_HPP
