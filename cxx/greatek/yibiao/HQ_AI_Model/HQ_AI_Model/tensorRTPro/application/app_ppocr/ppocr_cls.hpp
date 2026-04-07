#ifndef PPOCR_CLS_HPP
#define PPOCR_CLS_HPP

#include <common/trt_tensor.hpp>
#include <future>
#include <opencv2/opencv.hpp>
#include <vector>

namespace PaddleOCR
{
namespace Classifier
{

void image_to_tensor(const cv::Mat& image, std::shared_ptr<TRT::Tensor>& tensor, int ibatch);

class TextClassifier
{
public:
    virtual std::shared_future<int>              commit(const cv::Mat& image)                      = 0;
    virtual std::vector<std::shared_future<int>> commits(const std::vector<cv::Mat>& images)       = 0;
    virtual void                                 setConfidence(float confidence_threshold = 0.25f) = 0;
    virtual void                                 setNMS(float nms_threshold = 0.5f)                = 0;
};

std::shared_ptr<TextClassifier> create_classifier(
    const std::string& engine_file, int gpuid, float cls_thresh = 0.9f, int cls_batch_num = 4, bool use_multi_preprocess_stream = false);

};   // namespace Classifier
};   // namespace PaddleOCR


#endif   // PPOCR_CLS_HPP