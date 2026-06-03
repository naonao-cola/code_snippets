/**
 * @FilePath     : /pcl_test/src/lidar_detector.cpp
 * @Description  : 激光雷达碰撞检测 C++ 类实现（PIMPL 模式，cv::FileStorage 读取配置）
 * @Author       : weiwei.wang
 * @Date         : 2026-06-06
 * @Version      : 0.0.1
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "lidar_detector.hpp"
#include "distance_calc.h"
#include "rsldSDK.h"
#include "yolov8_seg.hpp"

#include <pcl/common/centroid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

#include <cfloat>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

static const char* BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64Encode(const unsigned char* data, size_t len)
{
    std::string result;
    result.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned char a = data[i];
        unsigned char b = (i + 1 < len) ? data[i + 1] : 0;
        unsigned char c = (i + 2 < len) ? data[i + 2] : 0;
        result += BASE64_CHARS[a >> 2];
        result += BASE64_CHARS[((a & 0x03) << 4) | (b >> 4)];
        result += (i + 1 < len) ? BASE64_CHARS[((b & 0x0f) << 2) | (c >> 6)] : '=';
        result += (i + 2 < len) ? BASE64_CHARS[c & 0x3f] : '=';
    }
    return result;
}

struct DetectorConfig
{
    std::string target_ip;
    std::string calib_path;
    std::string model_path;
    int         color_w                = 2448;
    int         color_h                = 1736;
    int         depth_w                = 800;
    int         depth_h                = 600;
    float       collision_distance     = 0.5f;
    float       confidence_threshold   = 0.45f;
    int         random_sample_points   = 2000;
    int         sor_mean_k             = 50;
    float       sor_stddev_mul_thresh  = 1.0f;
    int         infer_width            = 1280;
    int         infer_height           = 1280;
    int         mask_erode_kernel_size = 7;
    int         frame_interval_ms      = 0;
};

static void parseYamlConfig(const std::string& yaml_path, DetectorConfig& cfg)
{
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return;

    fs["target_ip"] >> cfg.target_ip;
    fs["calib_path"] >> cfg.calib_path;
    fs["model_path"] >> cfg.model_path;
    fs["color_w"] >> cfg.color_w;
    fs["color_h"] >> cfg.color_h;
    fs["depth_w"] >> cfg.depth_w;
    fs["depth_h"] >> cfg.depth_h;
    fs["collision_distance"] >> cfg.collision_distance;
    fs["confidence_threshold"] >> cfg.confidence_threshold;
    fs["random_sample_points"] >> cfg.random_sample_points;
    fs["sor_mean_k"] >> cfg.sor_mean_k;
    fs["sor_stddev_mul_thresh"] >> cfg.sor_stddev_mul_thresh;
    fs["infer_width"] >> cfg.infer_width;
    fs["infer_height"] >> cfg.infer_height;
    fs["mask_erode_kernel_size"] >> cfg.mask_erode_kernel_size;
    fs["frame_interval_ms"] >> cfg.frame_interval_ms;

    fs.release();
}

struct InternalObjectCloud
{
    int                                    object_id;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
};

struct LidarDetector::Impl
{
    void*    receiver     = nullptr;
    uint8_t* color_buffer = nullptr;
    uint8_t* depth_buffer = nullptr;
    bool     valid        = false;

    DetectorConfig      config;
    YOLOv8_seg*         yolov8 = nullptr;
    std::vector<Object> detect_objects;

    bool started = false;

    cv::Mat                               stored_color_img;
    ImageData                             stored_imgDepthReg;
    bool                                  has_captured = false;
    std::chrono::steady_clock::time_point last_capture_time;

    std::mutex infer_mutex;
    std::mutex sdk_mutex;
};

static cv::Mat imageDataToMat(const ImageData& img)
{
    int type = 0;
    switch (img.format) {
    case PixelFormat::GRAY8:
        type = CV_8UC1;
        break;
    case PixelFormat::RGB8:
        type = CV_8UC3;
        break;
    case PixelFormat::BGR8:
        type = CV_8UC3;
        break;
    case PixelFormat::RGBA8:
        type = CV_8UC4;
        break;
    case PixelFormat::DEPTH16:
        type = CV_16UC1;
        break;
    case PixelFormat::FLOAT32:
        type = CV_32FC1;
        break;
    default:
        return cv::Mat();
    }
    return cv::Mat(img.height, img.width, type, img.data, img.stride);
}

static void processSingleObject(int idx, const Object& obj, const cv::Mat& color_img, const ImageData& depth_img, void* receiver, const std::string& xmlpath, const DetectorConfig& config, std::vector<InternalObjectCloud>& object_list, std::mutex& list_mutex, std::mutex& sdk_mutex)
{
    if (obj.prob < config.confidence_threshold)
        return;

    cv::Mat  mask         = obj.boxMask;
    cv::Mat  mask_cropped = cv::Mat::zeros(color_img.size(), CV_8UC1);
    cv::Rect obj_rect_int((int)obj.rect.x, (int)obj.rect.y, (int)obj.rect.width, (int)obj.rect.height);
    cv::Rect roi = obj_rect_int & cv::Rect(0, 0, color_img.cols, color_img.rows);

    if (roi.width > 0 && roi.height > 0) {
        cv::Mat mask_to_copy;
        if (mask.size() != roi.size()) {
            cv::resize(mask, mask_to_copy, roi.size(), 0, 0, cv::INTER_NEAREST);
        }
        else {
            mask_to_copy = mask;
        }
        mask_to_copy.copyTo(mask_cropped(roi));
        mask_cropped   = mask_cropped * 255;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(config.mask_erode_kernel_size, config.mask_erode_kernel_size));
        cv::erode(mask_cropped, mask_cropped, kernel);
    }

    std::vector<Point2i> objPixels;
    cv::Mat              locations;
    cv::findNonZero(mask_cropped, locations);
    if (!locations.empty()) {
        objPixels.resize(locations.total());
        std::memcpy(objPixels.data(), locations.data, locations.total() * sizeof(Point2i));
    }

    std::vector<pointxyzrgb> person_points;
    ImageData*               rgbd_placeholder = nullptr;
    std::vector<Point2i>     rgbindexes_placeholder;

    if (!objPixels.empty()) {
        {
            std::lock_guard<std::mutex> lock(sdk_mutex);
            getObjectPoints(receiver, objPixels, depth_img, xmlpath, person_points, rgbd_placeholder, rgbindexes_placeholder);
        }

        if (!person_points.empty()) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            cloud->width    = person_points.size();
            cloud->height   = 1;
            cloud->is_dense = false;
            cloud->points.resize(cloud->width * cloud->height);

            for (size_t k = 0; k < person_points.size(); ++k) {
                cloud->points[k].x = person_points[k].x;
                cloud->points[k].y = person_points[k].y;
                cloud->points[k].z = person_points[k].z;
                cloud->points[k].r = person_points[k].r;
                cloud->points[k].g = person_points[k].g;
                cloud->points[k].b = person_points[k].b;
            }

            if (!cloud->empty()) {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
                if (cloud->size() > (size_t)config.random_sample_points) {
                    pcl::RandomSample<pcl::PointXYZRGB> ran;
                    ran.setInputCloud(cloud);
                    ran.setSample(config.random_sample_points);
                    ran.filter(*downsampled);
                }
                else {
                    downsampled = cloud;
                }

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr           sor_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
                sor.setInputCloud(downsampled);
                sor.setMeanK(config.sor_mean_k);
                sor.setStddevMulThresh(config.sor_stddev_mul_thresh);
                sor.filter(*sor_cloud);

                if (sor_cloud->size() >= 10) {
                    InternalObjectCloud oc;
                    oc.object_id = idx;
                    oc.cloud     = sor_cloud;
                    {
                        std::lock_guard<std::mutex> lock(list_mutex);
                        object_list.push_back(oc);
                    }
                }
            }
        }
    }
}

static std::vector<CollisionPair> detectAllCollisions(const std::vector<InternalObjectCloud>& objects, float warning_distance)
{
    std::vector<CollisionPair> result;
    for (size_t i = 0; i < objects.size(); ++i) {
        for (size_t j = i + 1; j < objects.size(); ++j) {
            int numA = (int)objects[i].cloud->size();
            int numB = (int)objects[j].cloud->size();

            std::vector<float> cloudA(numA * 3);
            std::vector<float> cloudB(numB * 3);

            // 使用 Eigen Map 向量化拷贝 x, y, z (跳过 rgb 字段)
            Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> mapA(cloudA.data(), 3, numA);
            mapA = objects[i].cloud->getMatrixXfMap(3, 4, 0);

            Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> mapB(cloudB.data(), 3, numB);
            mapB = objects[j].cloud->getMatrixXfMap(3, 4, 0);

            float dist = calculateMinDistanceCUDA(cloudA.data(), numA, cloudB.data(), numB);

            if (dist < warning_distance) {
                CollisionPair pair;
                pair.obj_id_a   = objects[i].object_id;
                pair.obj_id_b   = objects[j].object_id;
                pair.distance_m = dist;
                result.push_back(pair);
            }
        }
    }
    return result;
}

// ============================================================
// LidarDetector 实现
// ============================================================

LidarDetector::LidarDetector(const std::string& yaml_config_path)
    : pImpl(std::make_unique<Impl>())
{
    parseYamlConfig(yaml_config_path, pImpl->config);

    pImpl->receiver = CreateInterface(pImpl->config.depth_h, pImpl->config.depth_w);
    if (!pImpl->receiver)
        return;

    pImpl->color_buffer = new uint8_t[pImpl->config.color_w * pImpl->config.color_h * 3];
    pImpl->depth_buffer = new uint8_t[pImpl->config.depth_h * pImpl->config.depth_w * 2];
    pImpl->valid        = true;
}

LidarDetector::~LidarDetector()
{
    if (pImpl->started) {
        StopRecv(pImpl->receiver);
        delete pImpl->yolov8;
        pImpl->yolov8 = nullptr;
    }
    if (pImpl->receiver) {
        DestroyInterface(pImpl->receiver);
    }
    delete[] pImpl->color_buffer;
    delete[] pImpl->depth_buffer;
}

bool LidarDetector::isValid() const
{
    return pImpl->valid;
}

bool LidarDetector::start()
{
    if (!pImpl->valid)
        return false;
    if (pImpl->started)
        return true;
    if (pImpl->config.target_ip.empty())
        return false;

    ::setTargetIp(pImpl->receiver, pImpl->config.target_ip.c_str(), "");
    if (!StartRecv(pImpl->receiver))
        return false;

    if (!pImpl->config.model_path.empty()) {
        pImpl->yolov8 = new YOLOv8_seg(pImpl->config.model_path);
        pImpl->yolov8->make_pipe(true);
    }

    pImpl->started = true;
    return true;
}

void LidarDetector::stop()
{
    if (!pImpl->started)
        return;
    StopRecv(pImpl->receiver);
    delete pImpl->yolov8;
    pImpl->yolov8  = nullptr;
    pImpl->started = false;
}

void LidarDetector::capture()
{
    if (!pImpl->started)
        return;

    // 限速
    if (pImpl->has_captured && pImpl->config.frame_interval_ms > 0) {
        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - pImpl->last_capture_time).count();
        if (elapsed < pImpl->config.frame_interval_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(pImpl->config.frame_interval_ms - elapsed));
        }
    }

    std::lock_guard<std::mutex> lock(pImpl->infer_mutex);

    ImageData imgColorReg, imgDepthReg;
    imgColorReg.data = pImpl->color_buffer;
    imgDepthReg.data = pImpl->depth_buffer;

    memset(pImpl->color_buffer, 0, pImpl->config.color_w * pImpl->config.color_h * 3);
    memset(pImpl->depth_buffer, 0, pImpl->config.depth_h * pImpl->config.depth_w * 2);
    imgColorReg.width = 0;
    imgDepthReg.width = 0;

    getColorVsDepth(pImpl->receiver, &imgColorReg, &imgDepthReg);
    if (imgColorReg.width == 0 || imgDepthReg.width == 0)
        return;

    // 存入成员变量（深拷贝彩色图，浅拷贝深度描述符）
    pImpl->stored_color_img   = imageDataToMat(imgColorReg).clone();
    pImpl->stored_imgDepthReg = imgDepthReg;
    pImpl->has_captured       = true;
    pImpl->last_capture_time  = std::chrono::steady_clock::now();
}

void LidarDetector::encodeImagesToBase64(std::string& color_b64, std::string& depth_b64)
{
    color_b64.clear();
    depth_b64.clear();
    if (!pImpl->has_captured)
        return;

    std::lock_guard<std::mutex> lock(pImpl->infer_mutex);

    // 彩色图 → JPEG → base64
    std::vector<uchar> jpg_buf;
    cv::imencode(".jpg", pImpl->stored_color_img, jpg_buf);
    color_b64 = base64Encode(jpg_buf.data(), jpg_buf.size());

    // 深度图 → 16-bit PNG → base64
    cv::Mat            depth_mat = imageDataToMat(pImpl->stored_imgDepthReg);
    std::vector<uchar> png_buf;
    cv::imencode(".png", depth_mat, png_buf);
    depth_b64 = base64Encode(png_buf.data(), png_buf.size());
}

std::vector<float> LidarDetector::getFullCloud()
{
    std::vector<float> result;
    if (!pImpl->has_captured)
        return result;

    int w = pImpl->stored_color_img.cols;
    int h = pImpl->stored_color_img.rows;

    std::vector<Point2i> all_pixels;
    all_pixels.reserve(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            all_pixels.push_back({x, y});
        }
    }

    std::vector<pointxyzrgb> pts;
    ImageData*               rgbd_placeholder = nullptr;
    std::vector<Point2i>     rgbindexes_placeholder;

    {
        std::lock_guard<std::mutex> lock(pImpl->sdk_mutex);
        getObjectPoints(pImpl->receiver, all_pixels, pImpl->stored_imgDepthReg, pImpl->config.calib_path, pts, rgbd_placeholder, rgbindexes_placeholder);
    }

    if (pts.empty())
        return result;

    result.resize(pts.size() * 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        result[i * 3 + 0] = pts[i].x;
        result[i * 3 + 1] = pts[i].y;
        result[i * 3 + 2] = pts[i].z;
    }
    return result;
}

DetectResult LidarDetector::detectOnce()
{
    DetectResult result;
    if (!pImpl->started || !pImpl->yolov8 || !pImpl->has_captured)
        return result;

    std::lock_guard<std::mutex> lock(pImpl->infer_mutex);

    cv::Mat color_img = pImpl->stored_color_img;

    cv::Size size(pImpl->config.infer_width, pImpl->config.infer_height);
    pImpl->yolov8->copy_from_Mat(color_img, size);
    pImpl->yolov8->infer();
    pImpl->yolov8->postprocess(pImpl->detect_objects, 0.25f, 0.65f, 100, 32, 320, 320);

    std::vector<InternalObjectCloud> object_list;
    std::mutex                       list_mutex;
    std::vector<std::thread>         threads;

    for (int i = 0; i < (int)pImpl->detect_objects.size(); ++i) {
        threads.emplace_back(processSingleObject, i, std::ref(pImpl->detect_objects[i]), std::ref(color_img), std::ref(pImpl->stored_imgDepthReg), pImpl->receiver, pImpl->config.calib_path, std::ref(pImpl->config), std::ref(object_list), std::ref(list_mutex), std::ref(pImpl->sdk_mutex));
    }

    for (auto& t : threads) {
        if (t.joinable())
            t.join();
    }

    // 填充结果
    for (const auto& oc : object_list) {
        ObjectInfo info;
        info.object_id   = oc.object_id;
        info.point_count = (int)oc.cloud->size();

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*oc.cloud, centroid);
        info.center_x = centroid[0];
        info.center_y = centroid[1];
        info.center_z = centroid[2];

        result.objects.push_back(info);
    }

    result.collisions = detectAllCollisions(object_list, pImpl->config.collision_distance);

    return result;
}
