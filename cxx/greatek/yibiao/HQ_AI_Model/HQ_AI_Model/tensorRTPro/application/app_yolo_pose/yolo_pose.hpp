#ifndef YOLO_POSE_HPP
#define YOLO_POSE_HPP

#include <common/trt_tensor.hpp>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


namespace YoloPose
{

using namespace std;


struct Box
{
    float               left, top, right, bottom, confidence;
    vector<cv::Point3f> keypoints;
    int                 cls = -1;
    Box()                   = default;

    Box(float left, float top, float right, float bottom, int cls_value, float confidence, int num_keypoints)
        : left(left)
        , top(top)
        , right(right)
        , bottom(bottom)
        , cls(cls_value)
        , confidence(confidence)
    {
        keypoints.reserve(num_keypoints);
    }
};
typedef vector<Box> BoxArray;

enum class NMSMethod : int
{
    CPU     = 0,   // General, for estimate mAP
    FastGPU = 1    // Fast NMS with a small loss of accuracy in corner cases
};

void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

class Infer
{
public:
    virtual void                            setConfidence(float confidence_threshold = 0.25f) = 0;
    virtual void                            setNMS(float nms_threshold = 0.5f)                = 0;
    virtual shared_future<BoxArray>         commit(const cv::Mat& image)                      = 0;
    virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images)            = 0;
};

shared_ptr<Infer> create_infer(const string& engine_file,
                               int           gpuid,
                               float         confidence_threshold        = 0.25f,
                               float         nms_threshold               = 0.5f,
                               NMSMethod     nms_method                  = NMSMethod::FastGPU,
                               int           max_objects                 = 1024,
                               int           key_points                  = 17,
                               int           cls_num                     = 2,
                               bool          use_multi_preprocess_stream = false);

};   // namespace YoloPose

#endif   // YOLO_POSE_HPP