#ifndef DEIM_HPP
#define DEIM_HPP

#include <common/object_detector.hpp>
#include <common/trt_tensor.hpp>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace DEIM
{

using namespace std;
using namespace ObjectDetector;

void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

class Infer
{
public:
    virtual shared_future<BoxArray>         commit(const cv::Mat& image)                      = 0;
    virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images)            = 0;
    virtual void                            setConfidence(float confidence_threshold = 0.25f) = 0;
    virtual void                            setNMS(float nms_threshold = 0.5f)                = 0;
};

shared_ptr<Infer> create_infer(
    const string& engine_file, int gpuid, float confidence_threshold = 0.25f, int max_objects = 1024, bool use_multi_preprocess_stream = false);

};   // namespace DEIM

#endif   // DEIM_HPP
