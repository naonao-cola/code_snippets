/**
 * @FilePath     : /tray_algo/src/custom/trayTowerDet.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2024-06-20 14:26:35
**/
#pragma once
#include <filesystem> // C++17
#include <time.h>
#include "../framework/BaseAlgo.h"
namespace fs = std::filesystem;

class trayTowerDet : public BaseAlgo {
public:
    trayTowerDet();
    ~trayTowerDet();
    AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    /**
     * @brief 获取参数
     * @param task
     * @return
     */
    bool get_param(InferTaskPtr task);

    /**
     * @brief
     * @param task
     * @param param_map
     * @return
     */
    std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::map<std::string, json> param_map) const;

    /**
     * @brief 二次定位交点的大概位置
     * @param input_img
     * @param input_th_value
     * @param input_pts
     * @return
     */
    std::vector<cv::Point> re_get_rotated_pt(const cv::Mat& input_img, int input_th_value, const std::vector<cv::Point2f>& input_pts);

    /**
     * @brief 获取交点，第三次定位
     * @param img
     * @param pt_vec
     * @param threshold_value
     * @return
     */
    std::vector<cv::Point2f> get_pts(const cv::Mat& img, const std::vector<cv::Point2f>& pt_vec, int threshold_value);

    /**
     * @brief 类型一 3*5类型图片的处理过程
     * @param src
     * @param algo_result
     */
    void img_process1(const cv::Mat& src, AlgoResultPtr algo_result);

    /**
     * @brief 类型二 2*9 类型图片处理过程
     * @param src
     * @param algo_result
     */
    void img_process2(const cv::Mat& src, AlgoResultPtr algo_result);

    /**
     * @brief 写入调试图片
     * @param input_img
     * @param input_pts
     */
    void write_debug_img(const cv::Mat& input_img, const std::vector<std::vector<cv::Point2f>>& input_pts, int unknown_flag = 0);

    // 模板路径
    std::string template_img_path_1_;
    std::string template_img_path_2_;
    // 模板图像
    cv::Mat template_img_1_;
    cv::Mat template_img_2_;
    // 模板一检测的位置点，相对于左上角内交点
    std::vector<cv::Point2f> template_pt_0_;
    // 模板二检测的位置，相对于左上角内交点
    std::vector<cv::Point2f> template_pt_1_;
    // 齿的类型
    int tower_type_ = 0;
    // 判定参数
    int area_th_ = 120;
    int img_th_ = 150;
    int img_th_2_ = 180;

    std::vector<cv::Vec4i> temp_mask_1_{
        cv::Vec4i(227, 74, 21, 21),
        cv::Vec4i(558, 73, 20, 19),
        cv::Vec4i(869, 72, 20, 22),
        cv::Vec4i(1201, 71, 19, 21),
        cv::Vec4i(1512, 71, 19, 19),
        cv::Vec4i(1846, 70, 19, 20),
        cv::Vec4i(2156, 72, 18, 19),
        cv::Vec4i(2489, 70, 20, 23),
        cv::Vec4i(2800, 71, 19, 21),
        cv::Vec4i(3130, 72, 21, 22),
        cv::Vec4i(97, 249, 22, 24),
        cv::Vec4i(688, 210, 21, 23),
        cv::Vec4i(736, 247, 21, 23),
        cv::Vec4i(1332, 208, 22, 24),
        cv::Vec4i(1380, 246, 23, 25),
        cv::Vec4i(1976, 207, 22, 24),
        cv::Vec4i(2024, 245, 23, 26),
        cv::Vec4i(2618, 209, 22, 23),
        cv::Vec4i(2666, 245, 23, 24),
        cv::Vec4i(3257, 211, 23, 22),
        cv::Vec4i(228, 389, 19, 20),
        cv::Vec4i(558, 388, 20, 20),
        cv::Vec4i(868, 387, 20, 20),
        cv::Vec4i(1201, 388, 19, 18),
        cv::Vec4i(1513, 386, 18, 20),
        cv::Vec4i(1846, 385, 19, 20),
        cv::Vec4i(2156, 386, 20, 21),
        cv::Vec4i(2488, 386, 22, 20),
        cv::Vec4i(2800, 387, 19, 19),
        cv::Vec4i(3130, 388, 20, 19),
        cv::Vec4i(228, 530, 18, 19),
        cv::Vec4i(558, 530, 20, 18),
        cv::Vec4i(869, 529, 19, 20),
        cv::Vec4i(1202, 529, 19, 19),
        cv::Vec4i(1514, 528, 18, 20),
        cv::Vec4i(1846, 527, 18, 21),
        cv::Vec4i(2157, 527, 19, 20),
        cv::Vec4i(2489, 526, 20, 22),
        cv::Vec4i(2800, 528, 18, 20),
        cv::Vec4i(3131, 528, 19, 21),
        cv::Vec4i(98, 705, 22, 23),
        cv::Vec4i(689, 666, 21, 23),
        cv::Vec4i(738, 705, 21, 23),
        cv::Vec4i(1332, 666, 20, 22),
        cv::Vec4i(1379, 704, 22, 23),
        cv::Vec4i(1978, 666, 20, 23),
        cv::Vec4i(2024, 702, 23, 26),
        cv::Vec4i(2619, 665, 22, 25),
        cv::Vec4i(2667, 703, 21, 24),
        cv::Vec4i(3259, 666, 21, 23),
        cv::Vec4i(228, 845, 19, 19),
        cv::Vec4i(559, 844, 19, 20),
        cv::Vec4i(868, 845, 19, 19),
        cv::Vec4i(1199, 844, 21, 20),
        cv::Vec4i(1513, 844, 19, 20),
        cv::Vec4i(1846, 844, 18, 20),
        cv::Vec4i(2157, 842, 19, 22),
        cv::Vec4i(2490, 844, 19, 20),
        cv::Vec4i(2799, 842, 20, 21),
        cv::Vec4i(3131, 845, 19, 19),
        cv::Vec4i(229, 984, 19, 21),
        cv::Vec4i(559, 985, 19, 20),
        cv::Vec4i(869, 986, 19, 20),
        cv::Vec4i(1202, 986, 19, 19),
        cv::Vec4i(1513, 986, 18, 21),
        cv::Vec4i(1846, 985, 19, 20),
        cv::Vec4i(2157, 985, 20, 20),
        cv::Vec4i(2489, 984, 21, 21),
        cv::Vec4i(2800, 985, 20, 20),
        cv::Vec4i(3130, 984, 20, 20),
        cv::Vec4i(98, 1159, 21, 24),
        cv::Vec4i(690, 1123, 19, 22),
        cv::Vec4i(736, 1160, 22, 23),
        cv::Vec4i(1332, 1123, 20, 22),
        cv::Vec4i(1380, 1161, 21, 23),
        cv::Vec4i(1976, 1121, 21, 24),
        cv::Vec4i(2024, 1159, 22, 24),
        cv::Vec4i(2620, 1122, 21, 23),
        cv::Vec4i(2667, 1158, 22, 25),
        cv::Vec4i(3260, 1121, 22, 23),
        cv::Vec4i(228, 1300, 19, 20),
        cv::Vec4i(559, 1301, 19, 19),
        cv::Vec4i(869, 1301, 19, 19),
        cv::Vec4i(1202, 1302, 18, 19),
        cv::Vec4i(1512, 1302, 20, 20),
        cv::Vec4i(1846, 1301, 17, 20),
        cv::Vec4i(2156, 1301, 19, 20),
        cv::Vec4i(2489, 1300, 19, 21),
        cv::Vec4i(2801, 1300, 17, 19),
        cv::Vec4i(3131, 1300, 19, 18)};

    std::vector<cv::Vec4i> temp_mask_2_{
        cv::Vec4i(119, 113, 47, 31),
        cv::Vec4i(300, 115, 45, 28),
        cv::Vec4i(483, 115, 46, 27),
        cv::Vec4i(663, 116, 42, 25),
        cv::Vec4i(847, 116, 41, 27),
        cv::Vec4i(1028, 115, 41, 28),
        cv::Vec4i(1212, 115, 41, 28),
        cv::Vec4i(1393, 115, 41, 27),
        cv::Vec4i(1578, 119, 39, 23),
        cv::Vec4i(1759, 115, 41, 25),
        cv::Vec4i(1943, 115, 41, 26),
        cv::Vec4i(2125, 115, 42, 27),
        cv::Vec4i(2311, 116, 39, 27),
        cv::Vec4i(2491, 117, 42, 25),
        cv::Vec4i(2678, 117, 39, 25),
        cv::Vec4i(2859, 117, 45, 26),
        cv::Vec4i(3045, 117, 43, 25),
        cv::Vec4i(3226, 115, 49, 29),
        cv::Vec4i(60, 213, 28, 55),
        cv::Vec4i(369, 213, 42, 55),
        cv::Vec4i(422, 213, 31, 57),
        cv::Vec4i(732, 213, 40, 55),
        cv::Vec4i(779, 213, 35, 55),
        cv::Vec4i(1096, 213, 36, 56),
        cv::Vec4i(1146, 213, 29, 56),
        cv::Vec4i(1466, 213, 29, 54),
        cv::Vec4i(1510, 214, 29, 54),
        cv::Vec4i(1833, 213, 27, 56),
        cv::Vec4i(1879, 214, 28, 55),
        cv::Vec4i(2200, 213, 28, 57),
        cv::Vec4i(2248, 214, 26, 55),
        cv::Vec4i(2567, 214, 29, 57),
        cv::Vec4i(2610, 214, 36, 58),
        cv::Vec4i(2939, 214, 28, 58),
        cv::Vec4i(2978, 216, 39, 56),
        cv::Vec4i(3309, 215, 27, 56),
        cv::Vec4i(62, 536, 29, 54),
        cv::Vec4i(371, 540, 39, 50),
        cv::Vec4i(421, 540, 34, 52),
        cv::Vec4i(735, 541, 38, 50),
        cv::Vec4i(782, 540, 28, 53),
        cv::Vec4i(1100, 542, 28, 49),
        cv::Vec4i(1144, 543, 29, 50),
        cv::Vec4i(1465, 542, 29, 52),
        cv::Vec4i(1511, 543, 26, 52),
        cv::Vec4i(1835, 543, 26, 49),
        cv::Vec4i(1879, 545, 28, 50),
        cv::Vec4i(2203, 544, 25, 49),
        cv::Vec4i(2246, 545, 31, 49),
        cv::Vec4i(2572, 546, 25, 50),
        cv::Vec4i(2615, 544, 32, 52),
        cv::Vec4i(2936, 542, 27, 57),
        cv::Vec4i(2983, 545, 33, 52),
        cv::Vec4i(3311, 546, 21, 54),
        cv::Vec4i(124, 653, 45, 33),
        cv::Vec4i(302, 653, 37, 33),
        cv::Vec4i(486, 654, 39, 33),
        cv::Vec4i(666, 655, 33, 33),
        cv::Vec4i(848, 661, 33, 28),
        cv::Vec4i(1028, 661, 33, 27),
        cv::Vec4i(1213, 661, 34, 29),
        cv::Vec4i(1394, 663, 32, 26),
        cv::Vec4i(1579, 662, 32, 28),
        cv::Vec4i(1758, 660, 36, 32),
        cv::Vec4i(1944, 664, 36, 28),
        cv::Vec4i(2127, 663, 35, 29),
        cv::Vec4i(2313, 663, 35, 28),
        cv::Vec4i(2493, 664, 37, 29),
        cv::Vec4i(2682, 667, 32, 26),
        cv::Vec4i(2864, 666, 35, 28),
        cv::Vec4i(3050, 667, 34, 27),
        cv::Vec4i(3230, 666, 37, 30),
        cv::Vec4i(124, 728, 36, 30),
        cv::Vec4i(301, 729, 36, 29),
        cv::Vec4i(485, 730, 33, 29),
        cv::Vec4i(665, 730, 32, 29),
        cv::Vec4i(848, 731, 33, 30),
        cv::Vec4i(1027, 731, 33, 30),
        cv::Vec4i(1213, 731, 34, 31),
        cv::Vec4i(1393, 731, 34, 30),
        cv::Vec4i(1578, 732, 36, 32),
        cv::Vec4i(1759, 733, 34, 30),
        cv::Vec4i(1945, 733, 34, 30),
        cv::Vec4i(2127, 734, 34, 29),
        cv::Vec4i(2314, 734, 35, 31),
        cv::Vec4i(2495, 734, 35, 33),
        cv::Vec4i(2680, 735, 35, 29),
        cv::Vec4i(2864, 736, 34, 29),
        cv::Vec4i(3049, 737, 37, 29),
        cv::Vec4i(3232, 737, 35, 31),
        cv::Vec4i(62, 822, 30, 55),
        cv::Vec4i(371, 827, 35, 50),
        cv::Vec4i(422, 825, 30, 55),
        cv::Vec4i(731, 825, 36, 56),
        cv::Vec4i(782, 831, 30, 48),
        cv::Vec4i(1097, 829, 30, 51),
        cv::Vec4i(1145, 831, 28, 51),
        cv::Vec4i(1464, 829, 30, 54),
        cv::Vec4i(1510, 832, 30, 51),
        cv::Vec4i(1830, 832, 29, 50),
        cv::Vec4i(1878, 833, 29, 50),
        cv::Vec4i(2201, 833, 26, 53),
        cv::Vec4i(2246, 833, 30, 53),
        cv::Vec4i(2569, 834, 26, 51),
        cv::Vec4i(2613, 835, 32, 50),
        cv::Vec4i(2937, 836, 27, 51),
        cv::Vec4i(2980, 836, 36, 51),
        cv::Vec4i(3306, 838, 24, 48),
        cv::Vec4i(61, 1149, 24, 51),
        cv::Vec4i(370, 1152, 34, 47),
        cv::Vec4i(420, 1152, 25, 49),
        cv::Vec4i(734, 1154, 36, 49),
        cv::Vec4i(780, 1155, 27, 49),
        cv::Vec4i(1099, 1156, 30, 49),
        cv::Vec4i(1147, 1156, 27, 50),
        cv::Vec4i(1464, 1157, 30, 48),
        cv::Vec4i(1512, 1158, 29, 48),
        cv::Vec4i(1832, 1159, 27, 47),
        cv::Vec4i(1878, 1160, 29, 47),
        cv::Vec4i(2199, 1160, 26, 48),
        cv::Vec4i(2244, 1161, 32, 47),
        cv::Vec4i(2569, 1163, 25, 48),
        cv::Vec4i(2613, 1162, 32, 48),
        cv::Vec4i(2937, 1163, 25, 49),
        cv::Vec4i(2974, 1160, 42, 53),
        cv::Vec4i(3308, 1166, 23, 48),
        cv::Vec4i(123, 1273, 36, 26),
        cv::Vec4i(303, 1271, 34, 27),
        cv::Vec4i(481, 1272, 40, 26),
        cv::Vec4i(664, 1271, 37, 28),
        cv::Vec4i(846, 1274, 37, 27),
        cv::Vec4i(1027, 1274, 37, 27),
        cv::Vec4i(1211, 1276, 37, 28),
        cv::Vec4i(1391, 1278, 37, 28),
        cv::Vec4i(1577, 1279, 36, 27),
        cv::Vec4i(1757, 1279, 39, 25),
        cv::Vec4i(1943, 1279, 36, 28),
        cv::Vec4i(2122, 1279, 39, 28),
        cv::Vec4i(2308, 1283, 40, 26),
        cv::Vec4i(2489, 1283, 40, 27),
        cv::Vec4i(2677, 1284, 39, 27),
        cv::Vec4i(2861, 1285, 36, 27),
        cv::Vec4i(3047, 1286, 36, 27),
        cv::Vec4i(3229, 1288, 36, 27)};

    // 模板是否处理的标志
    int img_process_1_flag_ = 0;
    int img_process_2_flag_ = 0;
    // 类型1,3*5类型的图的变量，存储模板的处理结果
    cv::Mat template_img_org_1_;
    cv::Rect temp_rect_1_;
    std::vector<cv::Point2f> temp_pts_1_;
    // 类型2,2*9类型的图的变量。存储模板的处理结果
    cv::Mat template_img_org_2_;
    cv::Rect temp_rect_2_;
    std::vector<cv::Point2f> temp_pts_2_;

    /**
     * @brief 点变换，将点加入到结果中
     * @param input_rect
     * @param wrap_mat
     * @param algo_result
     * @return
     */
    std::vector<cv::Point2f> wrap_point(const cv::Rect& input_rect, const cv::Mat& wrap_mat, AlgoResultPtr algo_result);
    std::string image_file_name_ = "";
    DCLEAR_ALGO_GROUP_REGISTER(trayTowerDet)
};