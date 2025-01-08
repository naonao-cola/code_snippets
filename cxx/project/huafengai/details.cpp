#include "details.h"

namespace xx
{
void DrawImg(cv::Mat dispaly, nlohmann::json json_info)
{
}

double computeIoU(cv::Rect box1, cv::Rect box2)
{
    cv::Rect intersection     = box1 & box2;
    cv::Rect unionRect        = box1 | box2;
    int      intersectionArea = intersection.area();
    int      unionArea        = unionRect.area();
    return static_cast<double>(intersectionArea) / unionArea;
}
double FliterBox(std::vector<cv::Rect>& box_vec, cv::Rect box2, double thre)
{
    bool flag = true;
    // #pragma omp parallel for private(flag)
    for (int i = 0; i < box_vec.size(); i++) {
        double score = computeIoU(box_vec[i], box2);
        if (score > thre) {
            flag = false;
        }
    }
    if (flag) {
        box_vec.emplace_back(box2);
        return 1;
    }
    return 0;
}
nlohmann::json pt_json(std::vector<std::vector<cv::Point>> mask)
{
    nlohmann::json pt;
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask[i].size(); j++) {

            pt.push_back({mask[i][j].x, mask[i][j].y});
        }
    }
    return pt;
}
nlohmann::json pt_json(std::vector<cv::Point2f> mask)
{
    nlohmann::json pt;
    for (int j = 0; j < mask.size(); j++) {

        pt.push_back({int(mask[j].x), int(mask[j].y)});
    }
    return pt;
}
cv::Vec4i get_value(std::vector<cv::Point> pts){
    int min_x = INT_MAX, min_y = INT_MAX;
    int max_x = INT_MIN, max_y = INT_MIN;
    for (const auto& point : pts) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);

    }
    return cv::Vec4i(min_x, min_y, max_x, max_y);
}
cv::Point get_center(std::vector<cv::Point> pts){
    int sum_x = 0;
    int sum_y = 0;
    for (int i = 0; i < pts.size();i++) {
        sum_x += pts[i].x;
        sum_y += pts[i].y;
    }
    sum_x = sum_x / pts.size();
    sum_y = sum_y / pts.size();
    return cv::Point(sum_x,sum_y);
}
}   // namespace xx