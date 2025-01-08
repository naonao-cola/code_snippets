#pragma once
#include <iostream>
#include <vector>
#include "Geometry.h"

namespace Tival
{
    class ExportAPI TAPI
    {
    public:
        // 角度-弧度 转换
        static double Deg2Rad(double deg);
        static double Rad2Deg(double rad);

        // 直线和X轴方向的角度
        static double AngleLX(double x1, double y1, double x2, double y2);
        static double AngleLX(const TPoint& pt1, const TPoint& pt2);

        // 两条直线的夹角
        static double AngleLL(double a_x1, double a_y1, double a_x2, double a_y2, double b_x1, double b_y1, double b_x2, double b_y2 );
        static double AngleLL(const TPoint& a_pt1, const TPoint& a_pt2, const TPoint& b_pt1, const TPoint& b_pt2);
        static double AngleLL(const TLine& line1, const TLine& line2);

        // 点到直线距离
        static double DistancePL(double x, double y, double lx1, double ly1, double lx2, double ly2);
        static double DistancePL(const TPoint& pt, const TPoint& l_pt1, const TPoint& l_pt2);
        static double DistancePL(const TPoint& pt, const TLine& line);
    
        // 两点距离
        static double DistancePP(double x1, double y1, double x2, double y2);
        static double DistancePP(const TPoint& pt1, const TPoint& pt2);

        // 点到线段（两个端点）的最小最大距离
        static void DistancePS(double x, double y, double sx1, double sy1, double sx2, double sy2, double& dist_min, double& dist_max);
        static void DistancePS(const TPoint& pt, const TLine& line_seg, double& dist_min, double& dist_max);

        // 线段（上的点）到直线的最小最大垂直距离，如果线段和直线不平行，最小距离为0
        static void DistanceSL(double sx1, double sy1, double sx2, double sy2, double lx1, double ly1, double lx2, double ly2, double& dist_min, double& dist_max);
        static void DistanceSL(const TLine& line_seg, const TLine& line, double& dist_min, double& dist_max);

        // 线段1（上的点）到线段2（上的点）最小最大距离
        static void DistanceSS(double s1x1, double s1y1, double s1x2, double s1y2, double s2x1, double s2y1, double s2x2, double s2y2, double& dist_min, double& dist_max);
        static void DistanceSS(const TLine& line_seg1, const TLine& line_seg2, double& dist_min, double& dist_max);

        // 点到直线的投影（垂点）
        static TPoint ProjectionPL(double x, double y, double x1, double y1, double x2, double y2);
        static TPoint ProjectionPL(const TPoint& pt, const TLine& line);

        // 两条直线交点
        static TPoint IntersectLines(double l1x1, double l1y1, double l1x2, double l1y2, double l2x1, double l2y1, double l2x2, double l2y2);
        static TPoint IntersectLines(const TLine& line1, const TLine& line2);

        // 两条线段交点
        static TPoint IntersectSegments(double l1x1, double l1y1, double l1x2, double l1y2, double l2x1, double l2y1, double l2x2, double l2y2);
        static TPoint IntersectSegments(const TLine& line1, const TLine& line2);

        // 两圆（弧）交点
        static std::vector<TPoint> IntersectCircles(double x1, double y1, double r1, double sp1, double ep1, double x2, double y2, double r2, double sp2, double ep2);
        static std::vector<TPoint> IntersectCircles(const TCircle& circle1, const TCircle& circle2);

        // 直线和圆（弧）交点
        static std::vector<TPoint> IntersectLineCircle(double l1x1, double l1y1, double l1x2, double l1y2, double x, double y, double r, double sp, double ep);
        static std::vector<TPoint> IntersectLineCircle(const TLine& line, const TCircle& circle);

        // 线段和圆（弧）交点
        static std::vector<TPoint> IntersectSegmentCircle(double l1x1, double l1y1, double l1x2, double l1y2, double x, double y, double r, double sp, double ep);
        static std::vector<TPoint> IntersectSegmentCircle(const TLine& line, const TCircle& circle);
    };
};