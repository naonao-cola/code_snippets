


#include <opencv2/opencv.hpp>
#include <vector>


double CalculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);

bool DetectFalling(const std::vector<cv::Point3f>& keypoints);


double CalculateAngle(const cv::Point2f& vec1, const cv::Point2f& vec2);




bool fall_estimate(const std::vector<cv::Point3f>& kps);