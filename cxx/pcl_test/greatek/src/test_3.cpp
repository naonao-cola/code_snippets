/**
 * @FilePath     : /pcl_test/src/test_3.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-03-18 10:43:08
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-03-18 17:07:37
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "rsldSDK.h"
#include <string>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ox_d.h"
#include "ox_seg.h"
#include <cfloat>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/features/impl/moment_of_inertia_estimation.hpp>


#define TICK(x) auto bench_##x = std::chrono::high_resolution_clock::now();
#define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - bench_##x).count() << "us" << std::endl;

#define SAVE_DATA 0
#define ENABLE_COLLISION_WARNING 1        // 设置为 1 开启预警，0 则只检测实际碰撞
#define COLLISION_WARNING_DISTANCE 0.5f   // 预警距离（米）

cv::Mat imageDataToMat(const ImageData& img)
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
        throw std::runtime_error("Unsupported PixelFormat");
    }
    return cv::Mat(img.height, img.width, type, img.data, img.stride);   // 注意：构造 Mat 时要传入 stride 作为 step 参数
}


const std::vector<std::string> CLASS_NAMES = {"person", "bicycle", "car",     "motorcycle", "airplane",   "bus",       "train",    "truck",    "boat",     "traffic light", "fire hydrant", "stop sign",   "parking meter", "bench",        "bird",           "cat",        "dog",          "horse",         "sheep",        "cow",        "elephant",
                                              "bear",   "zebra",   "giraffe", "backpack",   "umbrella",   "handbag",   "tie",      "suitcase", "frisbee",  "skis",          "snowboard",    "sports ball", "kite",          "baseball bat", "baseball glove", "skateboard", "surfboard",    "tennis racket", "bottle",       "wine glass", "cup",
                                              "fork",   "knife",   "spoon",   "bowl",       "banana",     "apple",     "sandwich", "orange",   "broccoli", "carrot",        "hot dog",      "pizza",       "donut",         "cake",         "chair",          "couch",      "potted plant", "bed",           "dining table", "toilet",     "tv",
                                              "laptop", "mouse",   "remote",  "keyboard",   "cell phone", "microwave", "oven",     "toaster",  "sink",     "refrigerator",  "book",         "clock",       "vase",          "scissors",     "teddy bear",     "hair drier", "toothbrush"};



const std::vector<std::vector<unsigned int>> COLORS = {{0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142}, {119, 172, 48},  {77, 190, 238},  {162, 20, 47},   {76, 76, 76},  {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},   {0, 255, 0},     {0, 0, 255},    {170, 0, 255},  {85, 85, 0},
                                                       {85, 170, 0},    {85, 255, 0},    {170, 85, 0},    {170, 170, 0},  {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128}, {85, 255, 128}, {170, 0, 128},
                                                       {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},  {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},  {0, 170, 255},   {0, 255, 255},   {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255}, {170, 0, 255},  {170, 85, 255},
                                                       {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255}, {255, 170, 255}, {85, 0, 0},      {128, 0, 0},     {170, 0, 0},   {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},      {0, 128, 0},     {0, 170, 0},    {0, 212, 0},    {0, 255, 0},
                                                       {0, 0, 43},      {0, 0, 85},      {0, 0, 128},     {0, 0, 170},    {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},  {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},  {80, 183, 189}, {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {{255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29},  {207, 210, 49}, {72, 249, 10},  {146, 204, 23}, {61, 219, 134}, {26, 147, 52},   {0, 212, 187},
                                                            {44, 153, 168}, {0, 194, 255},   {52, 69, 147},  {100, 115, 255}, {0, 24, 236},   {132, 56, 255}, {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};



struct ObjectOBB
{
    int                                    object_id;
    pcl::PointXYZRGB                       position;     // 物体中心位置
    Eigen::Matrix3f                        rotation;     // 旋转矩阵
    Eigen::Vector3f                        dimensions;   // 长宽高
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    Eigen::Vector3f                        getHalfSizes() const
    {
        return dimensions * 0.5f;
    }

    // 获取8个顶点（用于可视化）
    std::vector<Eigen::Vector3f> getCorners() const
    {
        std::vector<Eigen::Vector3f> corners;
        Eigen::Vector3f              half = getHalfSizes();
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3f local(((i & 1) ? 1 : -1) * half.x(), ((i & 2) ? 1 : -1) * half.y(), ((i & 4) ? 1 : -1) * half.z());
            corners.push_back(Eigen::Vector3f(position.x, position.y, position.z) + rotation * local);
        }
        return corners;
    }
};


// OBB-OBB碰撞检测（SAT算法）
bool intersectOBB(const ObjectOBB& a, const ObjectOBB& b, float warning_distance = 0.0f)
{
    // 获取两OBB的轴
    Eigen::Vector3f axes_a[3] = {a.rotation.col(0), a.rotation.col(1), a.rotation.col(2)};
    Eigen::Vector3f axes_b[3] = {b.rotation.col(0), b.rotation.col(1), b.rotation.col(2)};
    // 中心连线
    Eigen::Vector3f t = Eigen::Vector3f(b.position.x, b.position.y, b.position.z) - Eigen::Vector3f(a.position.x, a.position.y, a.position.z);

    // 测试3个A的轴
    for (int i = 0; i < 3; ++i) {
        float ra = a.getHalfSizes()[i];
        float rb = b.getHalfSizes().dot(Eigen::Vector3f(std::abs(axes_b[0].dot(axes_a[i])), std::abs(axes_b[1].dot(axes_a[i])), std::abs(axes_b[2].dot(axes_a[i]))));
        if (std::abs(t.dot(axes_a[i])) > ra + rb + warning_distance)
            return false;
    }

    // 测试3个B的轴
    for (int i = 0; i < 3; ++i) {
        float rb = b.getHalfSizes()[i];
        float ra = a.getHalfSizes().dot(Eigen::Vector3f(std::abs(axes_a[0].dot(axes_b[i])), std::abs(axes_a[1].dot(axes_b[i])), std::abs(axes_a[2].dot(axes_b[i]))));
        if (std::abs(t.dot(axes_b[i])) > ra + rb + warning_distance)
            return false;
    }

    // 测试9个叉乘轴
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3f L   = axes_a[i].cross(axes_b[j]);
            float           len = L.norm();
            if (len < 1e-6)
                continue;   // 平行轴

            L /= len;
            float ra = 0, rb = 0;
            for (int k = 0; k < 3; ++k) {
                ra += a.getHalfSizes()[k] * std::abs(axes_a[k].dot(L));
                rb += b.getHalfSizes()[k] * std::abs(axes_b[k].dot(L));
            }
            if (std::abs(t.dot(L)) > ra + rb + warning_distance)
                return false;
        }
    }
    return true;   // 所有轴都重叠，发生碰撞
}

// 检测所有OBB之间的碰撞
std::vector<std::pair<int, int>> detectAllCollisions(const std::vector<ObjectOBB>& obbs, float warning_distance = 0.0f)
{
    std::vector<std::pair<int, int>> collisions;
    for (size_t i = 0; i < obbs.size(); ++i) {
        for (size_t j = i + 1; j < obbs.size(); ++j) {
            if (intersectOBB(obbs[i], obbs[j], warning_distance)) {
                collisions.push_back({obbs[i].object_id, obbs[j].object_id});
                std::cout << __LINE__ << "detected Collisions :  object  " << obbs[i].object_id << " and object " << obbs[j].object_id << " Collisioning" << std::endl;
            }
        }
    }
    return collisions;
}

// 可视化OBB（在图像上绘制）
void visualizeOBBs(cv::Mat& image, const std::vector<ObjectOBB>& obbs, const std::vector<std::pair<int, int>>& collisions, const cv::Mat& depth_img, const std::string& xmlpath, void* receiver)
{

    // 建立碰撞查找表
    // std::set<std::pair<int, int>> collision_set;
    // for (auto& col : collisions) {
    //     collision_set.insert(col);
    //     collision_set.insert({col.second, col.first});   // 双向
    // }

    for (size_t i = 0; i < obbs.size(); ++i) {
        const auto& obb = obbs[i];

        // 判断是否有碰撞（用于着色）
        bool has_collision = false;
        for (auto& col : collisions) {
            if (col.first == obb.object_id || col.second == obb.object_id) {
                has_collision = true;
                break;
            }
        }

        // 获取OBB的8个顶点
        // auto corners = obb.getCorners();
        // 将3D点投影到2D图像（简化：使用质心投影）
        // 实际应该投影所有顶点并绘制框，这里简化处理
        // 获取质心对应的图像坐标
        // std::vector<Point2i>     pixel_coords;
        // ImageData*               rgbd_placeholder = nullptr;
        // std::vector<Point2i>     rgbindexes_placeholder;
        // std::vector<pointxyzrgb> center_point;
        // 创建质心点
        // pointxyzrgb center_pt;
        // center_pt.x = obb.position.x;
        // center_pt.y = obb.position.y;
        // center_pt.z = obb.position.z;
        // center_pt.r = 255;
        // center_pt.g = 0;
        // center_pt.b = 0;

        // 这里简化：在图像上绘制文本信息
        std::string info = "ID:" + std::to_string(obb.object_id) + " Size:" + std::to_string((int)(obb.dimensions.x() * 100)) + "x" + std::to_string((int)(obb.dimensions.y() * 100)) + "x" + std::to_string((int)(obb.dimensions.z() * 100));

        if (has_collision) {
            info += " [COLLISION!]";
        }

        // 颜色：碰撞为红色，正常为绿色
        cv::Scalar color = has_collision ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        // 在图像角落显示信息（实际应该在物体位置显示，但需要投影）
        int y_pos = 30 + i * 30;
        cv::putText(image, info, cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    }
}

std::vector<ObjectOBB> computeMultiOBB(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& segmented_clouds, const std::vector<int>& object_ids)
{
    std::vector<ObjectOBB> obb_list;
    for (size_t i = 0; i < segmented_clouds.size(); ++i) {
        auto& cloud = segmented_clouds[i];
        if (cloud->empty() || cloud->points.size() < 10)
            continue;

        // 进一步下采样以加速OBB计算
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(0.07f, 0.07f, 0.07); // 使用较大的体素（5cm）加速计算
        voxel_grid.filter(*downsampled_cloud);

        // 如果下采样后点太少，回退使用原始云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_to_use = downsampled_cloud->size() >= 10 ? downsampled_cloud : cloud;

        pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor;
        feature_extractor.setInputCloud(cloud_to_use);
        feature_extractor.compute();

        pcl::PointXYZRGB min_pt, max_pt, position;
        Eigen::Matrix3f  rotation;
        feature_extractor.getOBB(min_pt, max_pt, position, rotation);

        ObjectOBB obb;
        obb.object_id  = object_ids[i];
        obb.position.x = position.x;
        obb.position.y = position.y;
        obb.position.z = position.z;
        obb.rotation   = rotation;
        // obb.dimensions = max_pt.getVector3fMap() - min_pt.getVector3fMap();
        obb.dimensions = Eigen::Vector3f(max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z);
        obb.cloud      = cloud;
        obb_list.push_back(obb);
        std::cout << __LINE__ << " target " << object_ids[i] << " : position (" << obb.position.x << "," << obb.position.y << "," << obb.position.z << ")"
                  << " size(" << obb.dimensions.x() << "," << obb.dimensions.y() << "," << obb.dimensions.z() << ")" << std::endl;
    }
    return obb_list;
}


int test_realtime_seg_pointcloud()
{
    // 1. 初始化激光雷达
    void* receiver1 = CreateInterface(600, 800);
    // 分配图像缓冲
    uint8_t* colordata = new uint8_t[2448 * 1736 * 3];
    uint8_t* depthdata = new uint8_t[800 * 600 * 2];
    // 设置设备IP
    setTargetIp(receiver1, "192.168.1.64", "");
    if (!StartRecv(receiver1)) {
        printf("Open devices1 failed\n");
        delete[] colordata;
        delete[] depthdata;
        DestroyInterface(receiver1);
        return -1;
    }

    printf("Open devices1 successfully\n");
    // 2. 初始化YOLO分割模型
    auto        detector = std::make_unique<YOLOv8SegDetector>("D:/workspace/weiwei/pcl_test/model/yolov8m-seg.onnx", false, 0.3f, 0.6f);
    std::string xmlpath  = "D:/workspace/weiwei/pcl_test/config/calib_color_readFromCamera 1.yaml";
    ImageData   imgColorReg, imgDepthReg;
    imgColorReg.data = colordata;
    imgDepthReg.data = depthdata;
    int frame_id     = 0;

    while (true) {
        memset(colordata, 0, 2448 * 1736 * 3);
        memset(depthdata, 0, 800 * 600 * 2);
        imgColorReg.width = 0;
        imgDepthReg.width = 0;
        getColorVsDepth(receiver1, &imgColorReg, &imgDepthReg);

        ++frame_id;
        if (imgColorReg.width > 0 && imgDepthReg.width > 0) {
            // 转为OpenCV格式
            cv::Mat color_img = imageDataToMat(imgColorReg);
            // CV_16UC1
            cv::Mat depth_img = imageDataToMat(imgDepthReg);
            // 4. 执行YOLO
            DetectionResult result = detector->Detect(color_img);
            // 存储每个目标的点云和OBB
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> object_clouds;
            std::vector<int>                                    object_ids;
            std::vector<cv::Rect>                               object_rois;
            std::vector<cv::Point3f>                            pt_vec;

            for (size_t i = 0; i < result.boxes.size(); ++i) {
                if (i < result.class_ids.size() && (result.class_ids[i] == 0 || result.class_ids[i] == 39) && i < result.masks.size()) {
                    // 假设 0 是 person 类 或 39 是 杯子类
                    // if (i < result.class_ids.size() && (result.class_ids[i] == 39) && i < result.masks.size()) {
                    cv::Mat mask = result.masks[i];
                    if (mask.size() != color_img.size()) {   // 确保mask尺寸与图像一致
                        cv::resize(mask, mask, color_img.size(), 0, 0, cv::INTER_NEAREST);
                    }
                    // 只保留box区域的mask
                    cv::Mat  mask_cropped = cv::Mat::zeros(color_img.size(), CV_8UC1);
                    cv::Rect roi          = result.boxes[i] & cv::Rect(0, 0, color_img.cols, color_img.rows);
                    if (roi.width > 0 && roi.height > 0) {
                        mask(roi).copyTo(mask_cropped(roi));
                        mask_cropped = mask_cropped * 255;
                    }
                    // 收集mask区域内的像素点坐标
                    std::vector<Point2i> objPixels;
                    cv::Mat              locations;
                    // 使用findNonZero加速
                    cv::findNonZero(mask_cropped, locations);
                    if (!locations.empty()) {
                        objPixels.resize(locations.total());
                        std::memcpy(objPixels.data(), locations.data, locations.total() * sizeof(Point2i));
                    }

                    std::vector<pointxyzrgb> person_points;
                    // 不需要RGBD输出
                    ImageData*           rgbd_placeholder = nullptr;
                    std::vector<Point2i> rgbindexes_placeholder;

                    if (!objPixels.empty()) {
                        // 6. 调用SDK接口获取点云
                        getObjectPoints(receiver1, objPixels, imgDepthReg, xmlpath, person_points, rgbd_placeholder, rgbindexes_placeholder);

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

                            // 去除离群点
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr           sor_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
                            sor.setInputCloud(cloud);
                            sor.setMeanK(50);
                            sor.setStddevMulThresh(1.0);
                            sor.filter(*sor_cloud);

                            // 下采样
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                            pcl::VoxelGrid<pcl::PointXYZRGB>       voxel_grid;
                            voxel_grid.setInputCloud(sor_cloud);
                            voxel_grid.setLeafSize(0.01f, 0.01f, 0.01f);
                            voxel_grid.filter(*downsampled_cloud);

                            if (downsampled_cloud->size() >= 10) {
                                object_clouds.push_back(downsampled_cloud);
                                // 使用检测索引作为ID
                                object_ids.push_back(i);
                                object_rois.push_back(result.boxes[i]);
                            }
                        }
                    }
                    cv::rectangle(color_img, result.boxes[i], cv::Scalar(0, 255, 0), 2);   // 可视化：在图上画框和mask轮廓
                }
            }

            if (object_clouds.size() >= 2) {
                std::cout << __LINE__ << "\n===  detected" << object_clouds.size() << "  targets, compute OBB ===" << std::endl;
                // 计算所有OBB
                auto obb_list = computeMultiOBB(object_clouds, object_ids);
                // 进行碰撞检测
#if ENABLE_COLLISION_WARNING
                auto collisions = detectAllCollisions(obb_list, COLLISION_WARNING_DISTANCE);
#else
                auto collisions = detectAllCollisions(obb_list, 0.0f);
#endif
                // 可视化结果
                visualizeOBBs(color_img, obb_list, collisions, depth_img, xmlpath, receiver1);
            }
            else if (object_clouds.size() == 1) {
                std::cout << "only  one target, skip" << std::endl;
            }
            // 创建自适应大小的窗口
            cv::namedWindow("Realtime Detection", cv::WINDOW_NORMAL);
            // 如果没有检测到人，不暂停，继续采集下一帧
            cv::imshow("Realtime Detection", color_img);
            cv::waitKey(150);
        }
    }
    StopRecv(receiver1);   // 清理资源
    DestroyInterface(receiver1);
    delete[] colordata;
    delete[] depthdata;
    return 0;
}

int main(int argc, char** argv)
{
    test_realtime_seg_pointcloud();
}
