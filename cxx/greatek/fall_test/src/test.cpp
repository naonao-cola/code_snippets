/**
 * @FilePath     : /test/src/test.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2025-11-21 16:34:14
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2025-11-24 09:50:20
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#include "test.h"

double CalculateDistance(const cv::Point2f& p1, const cv::Point2f& p2)
{
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}



double CalculateAngle(const cv::Point2f& vec1, const cv::Point2f& vec2)
{
    double dot  = vec1.x * vec2.x + vec1.y * vec2.y;
    double len1 = std::sqrt(vec1.x * vec1.x + vec1.y * vec1.y);
    double len2 = std::sqrt(vec2.x * vec2.x + vec2.y * vec2.y);
    if (len1 < 1e-6 || len2 < 1e-6) {
        return 0.0;
    }

    double cos_angle = dot / (len1 * len2);
    cos_angle        = std::max(-1.0, std::min(1.0, cos_angle));
    double angle_rad = std::acos(cos_angle);
    return angle_rad * 180.0 / CV_PI;
}



bool DetectFalling(const std::vector<cv::Point3f>& keypoints)
{
    if (keypoints.size() < 17) {
        return false;
    }

    cv::Point2f left_ear(keypoints[3].x, keypoints[3].y);        // 左耳
    cv::Point2f right_ear(keypoints[4].x, keypoints[4].y);       // 右耳
    cv::Point2f left_hip(keypoints[11].x, keypoints[11].y);      // 左髋
    cv::Point2f right_hip(keypoints[12].x, keypoints[12].y);     // 右髋
    cv::Point2f left_ankle(keypoints[15].x, keypoints[15].y);    // 左脚裹
    cv::Point2f right_ankle(keypoints[16].x, keypoints[16].y);   // 右脚裹



    float confidence_thresh = 0.2f;
    if (keypoints[3].z < confidence_thresh || keypoints[4].z < confidence_thresh || keypoints[11].z < confidence_thresh ||
        keypoints[12].z < confidence_thresh || keypoints[15].z < confidence_thresh || keypoints[16].z < confidence_thresh) {
        return false;
    }

    cv::Point2f C0 = (left_hip + right_hip) * 0.5f;
    cv::Point2f C1 = (left_ear + right_ear) * 0.5f;
    cv::Point2f C2 = (left_ankle + right_ankle) * 0.5f;

    double dist_C0_C1 = CalculateDistance(C0, C1);
    double dist_C0_C2 = CalculateDistance(C0, C2);

    if (dist_C0_C2 < 1e-6) {
        return false;
    }

    double p = dist_C0_C1 / dist_C0_C2;

    cv::Point2f D1 = C1 - C0;
    cv::Point2f D2 = C0 - C2;
    cv::Point2f horizontal_vec(1.0f, 0.0f);

    double theta1 = CalculateAngle(D1, horizontal_vec);
    double theta2 = CalculateAngle(D2, horizontal_vec);

    std::cout << "theta1: " << theta1 << std::endl;
    std::cout << "theta2: " << theta2 << std::endl;
    std::cout << "p: " << p << std::endl;

    if (((theta1 < 45.0 && theta2 < 45.0) || (theta1 > 135.0 && theta2 > 135.0)) && p > 0.7 && p < 1.6) {
        return true;
    }

    return false;
}


bool fall_estimate(const std::vector<cv::Point3f>& kps)
{
    // 设置一个判断是否为摔倒的变量
    bool is_fall = false;

    // 1. 先获取哪些用于判断的点坐标
    cv::Point L_shoulder       = cv::Point((int)kps[5].x, (int)kps[5].y);   // 左肩
    float     L_shoulder_confi = kps[5].z;
    cv::Point R_shoulder       = cv::Point((int)kps[6].x, (int)kps[6].y);   // 右肩
    float     R_shoulder_confi = kps[6].z;
    cv::Point C_shoulder       = cv::Point((int)(L_shoulder.x + R_shoulder.x) / 2, (int)(L_shoulder.y + R_shoulder.y) / 2);   // 肩部中点

    cv::Point L_hip       = cv::Point((int)kps[11].x, (int)kps[11].y);   // 左髋
    float     L_hip_confi = kps[11].z;
    cv::Point R_hip       = cv::Point((int)kps[12].x, (int)kps[12].y);   // 右髋
    float     R_hip_confi = kps[12].z;
    cv::Point C_hip       = cv::Point((int)(L_hip.x + R_hip.x) / 2, (int)(L_hip.y + R_hip.y) / 2);   // 髋部中点

    cv::Point L_knee       = cv::Point((int)kps[13].x, (int)kps[13].y);   // 左膝
    float     L_knee_confi = kps[13].z;
    cv::Point R_knee       = cv::Point((int)kps[14].x, (int)kps[14].y);   // 右膝
    float     R_knee_confi = kps[14].z;
    cv::Point C_knee       = cv::Point((int)(L_knee.x + R_knee.x) / 2, (int)(L_knee.y + R_knee.y) / 2);   // 膝部中点

    cv::Point L_ankle       = cv::Point((int)kps[15].x, (int)kps[15].y);   // 左踝
    float     L_ankle_confi = kps[15].z;
    cv::Point R_ankle       = cv::Point((int)kps[16].x, (int)kps[16].y);   // 右踝
    float     R_ankle_confi = kps[16].z;
    cv::Point C_ankle       = cv::Point((int)(L_ankle.x + R_ankle.x) / 2, (int)(L_ankle.y + R_ankle.y) / 2);   // 计算脚踝中点

    // 2. 第一个判定条件： 若肩的纵坐标最小值min(L_shoulder.y, R_shoulder.y)不低于脚踝的中心点的纵坐标C_ankle.y
    // 且p_shoulders、p_ankle关键点置信度大于预设的阈值，则疑似摔倒。
    if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_ankle_confi > 0.0f && R_ankle_confi > 0.0f) {
        int shoulder_y_min = std::min(L_shoulder.y, R_shoulder.y);
        if (shoulder_y_min >= C_ankle.y) {
            is_fall = true;
            return is_fall;
        }
    }


    // 3. 第二个判断条件：若肩的纵坐标最大值max(L_shoulder.y, R_shoulder.y)大于膝盖纵坐标的最小值min(L_knee.y, R_knee.y)，
    // 且p_shoulders、p_knees关键点置信度大于预设的阈值，则疑似摔倒。
    if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_knee_confi > 0.0f && R_knee_confi > 0.0f) {
        int shoulder_y_max = std::max(L_shoulder.y, R_shoulder.y);
        int knee_y_min     = std::min(L_knee.y, R_knee.y);
        if (shoulder_y_max > knee_y_min) {
            is_fall = true;
            return is_fall;
        }
    }

    // 4, 第三个判断条件：计算关键点最小外接矩形的宽高比。p0～p16在x方向的距离是xmax-xmin，在方向的距离是ymax-ymin，
    // 若(xmax-xmin) / (ymax-ymin)不大于指定的比例阈值，则判定为未摔倒，不再进行后续判定。
    const int num_point = kps.size();   // 17个关键点

    // 初始化xmin, ymin为最大值，xmax, ymax为最小值
    int xmin = std::numeric_limits<int>::max();
    int ymin = std::numeric_limits<int>::max();
    int xmax = std::numeric_limits<int>::min();
    int ymax = std::numeric_limits<int>::min();

    for (int k = 0; k < kps.size(); k++) {
        if (k < num_point) {
            int   kps_x = std::round(kps[k].x);   // 关键点x
            int   kps_y = std::round(kps[k].y);   // 关键点y
            float kps_s = kps[k].z;               // 可见性

            if (kps_s > 0.0f) {
                // 更新xmin, xmax, ymin, ymax
                xmin = std::min(xmin, kps_x);
                xmax = std::max(xmax, kps_x);
                ymin = std::min(ymin, kps_y);
                ymax = std::max(ymax, kps_y);
            }
        }
    }

    // 检查是否存在有效的宽度和高度
    if (xmax > xmin && ymax > ymin) {
        float aspect_ratio = static_cast<float>(xmax - xmin) / (ymax - ymin);

        // 如果宽高比大于指定阈值，则判定为摔倒
        if (aspect_ratio > 0.90f) {
            is_fall = true;
            return is_fall;
        }
    }

    // 5. 第四个判断条件：通过两膝与髋部中心点的连线与地面的夹角判断。首先假定有两点p1＝(x1 ,y1 )，p2＝(x2 ,y2
    // )，那么两点连接线与地面的角度计算公式为： 												θ = arctan((y2-y1) / (x2-x1)) * 180 / pi
    // 此处左膝与髋部的两点是(C_hip, L_knee)，与地面夹角表示为θ1；右膝与髋部的两点 是(C_hip, R_knee)，与地面夹角表示为θ2， 若min(θ1 ,θ2 )＜th1 或
    // max(θ1 ,θ2 )＜th2，且p_knees、 p_hips关键点置信度大于预设的阈值，则疑似摔倒
    if (L_knee_confi > 0.0f && R_knee_confi > 0.0f && L_hip_confi > 0.0f && R_hip_confi > 0.0f) {
        // 左膝与髋部中心的角度
        float theta1 = std::atan2(L_knee.y - C_hip.y, L_knee.x - C_hip.x) * 180.0f / CV_PI;
        // 右膝与髋部中心的角度
        float theta2 = std::atan2(R_knee.y - C_hip.y, R_knee.x - C_hip.x) * 180.0f / CV_PI;

        float min_theta = std::min(std::abs(theta1), std::abs(theta2));
        float max_theta = std::max(std::abs(theta1), std::abs(theta2));

        /*
        根据人体运动规律，阈值th1 和 th2 应设置为代表正常和摔倒之间的界限角度。
        通常情况下，如果人体处于站立或行走状态，膝盖与髋部的连线与地面之间的角度应接近垂直或有一定的倾斜，而当摔倒时，这个角度通常会明显减小。
        th1: 用于判断两膝与髋部的连线与地面的最小角度。可以设定为 20度。如果min(θ1 ,θ2
        )＜th1,即两膝与髋部的连线明显接近平行于地面，则有可能表示摔倒的姿态。 th2: 用于判断两膝与髋部的连线与地面的最大角度。可以设定为
        45度。如果max(θ1 ,θ2 )＜th2,即两膝与髋部的连线即使有倾斜但依然小于正常站立的角度范围，也可能表明摔倒的风险。
        */

        // 设定阈值 th1 和 th2，用于判定是否摔倒
        float th1 = 30.0f;   // 假设的最小角度阈值  // 20, 30 ,25
        float th2 = 70.0f;   // 假设的最大角度阈值  // 35, 40, 45, 50, 60

        // std::cout << "min_theta: " << min_theta  << ", " << "max_theta: " << max_theta << std::endl;

        if ((min_theta) < th1 && (max_theta < th2)) {
            is_fall = true;
            return is_fall;
        }
    }
    // 第五个判断条件：通过肩、髋部、膝盖夹角，髋部、膝盖、脚踝夹角判断。
    // 首先假定有四点p1＝(x1 ,y1 )，p2＝(x2 ,y2 )，p3＝(x3 ,y3 )，p4＝(x4 ,y4 )，其中，p1 p2组 成的向量为v1＝(x2 -x1 ,y2 -y1 )，
    // p3 p4组成的向量为v2＝(x4 -x3 ,y4 -y3 )。v1 v2的夹角计算公式为：
    // θ = arctan((v1 * v2) / (sqrt(v1 * v1) * sqrt(v2 * v2))) * 180 / pi
    // 此处， v1＝(c_shoulder.x - c_hips.x, c_shoulders.y - c_hips.y)
    //	v2＝(c_knees.x -c_hips.x, c_knees .y - c_hips.y)
    //	v3＝(c_hips.x - c_knees.x, c_hips.y - c_knees.y)
    // 	v4＝(c_foot.x - c_knees.x, c_foot.y - c_knees.y)
    // v1 v2两个向量的夹角表示为θ3，v3 v4两个向量的夹角表示为θ4。若θ3＞th3或θ4＜
    // th4，且p_shoulders、p_knees、p_hips、p_foot关键点置信度大于预设的阈值，则疑似摔倒。
    // 第五个判断条件：通过肩、髋部、膝盖夹角，髋部、膝盖、脚踝夹角判断。
    // 如果肩、髋、膝和脚踝关键点的置信度都大于阈值，我们继续进行角度的计算。
    if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_hip_confi > 0.0f && R_hip_confi > 0.0f && L_knee_confi > 0.0f &&
        R_knee_confi > 0.0f && L_ankle_confi > 0.0f && R_ankle_confi > 0.0f) {
        // 计算向量 v1 和 v2
        cv::Point2f v1(C_shoulder.x - C_hip.x, C_shoulder.y - C_hip.y);
        cv::Point2f v2(C_knee.x - C_hip.x, C_knee.y - C_hip.y);

        // 计算向量 v3 和 v4
        cv::Point2f v3(C_hip.x - C_knee.x, C_hip.y - C_knee.y);
        cv::Point2f v4(C_ankle.x - C_knee.x, C_ankle.y - C_knee.y);

        // 计算向量 v1 和 v2 的夹角 θ3
        float dot_product1 = v1.x * v2.x + v1.y * v2.y;
        float magnitude1   = std::sqrt(v1.x * v1.x + v1.y * v1.y) * std::sqrt(v2.x * v2.x + v2.y * v2.y);
        float theta3       = std::acos(dot_product1 / magnitude1) * 180.0f / CV_PI;

        // 计算向量 v3 和 v4 的夹角 θ4
        float dot_product2 = v3.x * v4.x + v3.y * v4.y;
        float magnitude2   = std::sqrt(v3.x * v3.x + v3.y * v3.y) * std::sqrt(v4.x * v4.x + v4.y * v4.y);
        float theta4       = std::acos(dot_product2 / magnitude2) * 180.0f / CV_PI;

        /*
        定义: 𝜃3是肩、髋、膝三点形成的向量夹角。通常情况下，站立时肩、髋和膝盖的夹角应该接近 180度（几乎成一条直线）。
        摔倒判断: 当人摔倒或发生意外时，这个角度可能会急剧减少。一个合理的阈值可以设定为 120度 或 130度。
        定义: 𝜃4是髋、膝、脚踝三点形成的向量夹角。站立或正常行走时，这个角度通常在 160度 到 180度
        之间（接近直线）。在弯曲或下蹲时，这个角度可能会降低。 摔倒判断:
        如果此角度降低到一个较小的值（例如人体接近折叠或蜷缩的状态），可以判断为摔倒。一个合理的阈值可以设定为 60度 或 70度。
        */

        /*
        th3（肩、髋、膝夹角）被设定为70.0f。这个值是基于假设站立时肩、髋和膝盖的夹角应该接近180度（几乎成一条直线），但在摔倒时这个角度可能会急剧减少。
        th4（髋、膝、脚踝夹角）被设定为60.0f。这个值是基于假设站立或正常行走时，这个角度通常在160度到180度之间，而在摔倒或身体接近折叠状态时，这个角度可能会显著降低。
        */

        // 设定角度阈值 th3 和 th4
        float th3 = 70.0f;   // 假设的阈值，肩、髋和膝的角度  // 120.0f, 130.0f
        float th4 = 30.0f;   // 假设的阈值，髋、膝和脚踝的角度  // 60.0f, 70.0f

        // 判断是否符合摔倒条件
        if ((theta3 < th3) && (theta4 < th4)) {
            // std::cout << "theta3: " << theta3  << ", " << "theta4: " << theta4 << std::endl;
            is_fall = true;
        }
        return is_fall;
    }
}