/**
 * @FilePath     : /pcl_test/src/python_bindings.cpp
 * @Description  : pybind11 Python 绑定，将 LidarDetector 暴露为 Python 模块
 * @Author       : weiwei.wang
 * @Date         : 2026-06-06
 * @Version      : 0.0.1
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "lidar_detector.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(lidar_collision, m)
{
    m.doc() = "Lidar collision detection module - 激光雷达碰撞检测模块";

    py::class_<CollisionPair>(m, "CollisionPair").def(py::init<>()).def_readwrite("obj_id_a", &CollisionPair::obj_id_a).def_readwrite("obj_id_b", &CollisionPair::obj_id_b).def_readwrite("distance_m", &CollisionPair::distance_m).def("__repr__", [](const CollisionPair& p) {
        return "<CollisionPair obj_" + std::to_string(p.obj_id_a) + " <-> obj_" + std::to_string(p.obj_id_b) + " dist=" + std::to_string(p.distance_m) + "m>";
    });

    py::class_<ObjectInfo>(m, "ObjectInfo")
        .def(py::init<>())
        .def_readwrite("object_id", &ObjectInfo::object_id)
        .def_readwrite("point_count", &ObjectInfo::point_count)
        .def_readwrite("center_x", &ObjectInfo::center_x)
        .def_readwrite("center_y", &ObjectInfo::center_y)
        .def_readwrite("center_z", &ObjectInfo::center_z)
        .def("__repr__", [](const ObjectInfo& o) { return "<ObjectInfo id=" + std::to_string(o.object_id) + " pts=" + std::to_string(o.point_count) + " center=(" + std::to_string(o.center_x) + "," + std::to_string(o.center_y) + "," + std::to_string(o.center_z) + ")>"; });

    py::class_<DetectResult>(m, "DetectResult").def(py::init<>()).def_readwrite("objects", &DetectResult::objects).def_readwrite("collisions", &DetectResult::collisions).def("__repr__", [](const DetectResult& r) { return "<DetectResult objects=" + std::to_string(r.objects.size()) + " collisions=" + std::to_string(r.collisions.size()) + ">"; });

    py::class_<LidarDetector>(m, "LidarDetector")
        .def(py::init<const std::string&>(), py::arg("yaml_config_path"), "创建雷达检测器实例")

        .def("is_valid", &LidarDetector::isValid)

        .def("start", &LidarDetector::start)
        .def("stop", &LidarDetector::stop)

        .def("capture", &LidarDetector::capture, "仅采集数据并更新内部成员变量（不进行编码）")

        .def(
            "encode_images_to_base64",
            [](LidarDetector& self) {
                std::string color_b64, depth_b64;
                self.encodeImagesToBase64(color_b64, depth_b64);
                return std::make_pair(color_b64, depth_b64);
            },
            "将内部缓存的图像编码为 Base64 字符串并返回 (color_b64, depth_b64)")

        .def(
            "get_full_cloud",
            [](LidarDetector& self) {
                std::vector<float> cloud = self.getFullCloud();
                if (cloud.empty()) {
                    return py::array_t<float>(0);
                }
                size_t             n_points = cloud.size() / 3;
                py::array_t<float> result({(ptrdiff_t)n_points, (ptrdiff_t)3});
                auto               buf = result.request();
                float*             ptr = (float*)buf.ptr;
                std::memcpy(ptr, cloud.data(), cloud.size() * sizeof(float));
                return result;
            },
            "获取全图点云，返回 NumPy 数组 (N, 3)")

        .def("detect_once", &LidarDetector::detectOnce, "执行一帧检测，返回 DetectResult");
}
