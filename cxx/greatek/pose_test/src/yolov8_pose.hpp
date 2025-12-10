//
// Created by ubuntu on 4/7/23.
//
#ifndef POSE_NORMAL_YOLOv8_pose_HPP
#define POSE_NORMAL_YOLOv8_pose_HPP

#include "NvInferPlugin.h"
#include "pose_com.hpp"
#include <fstream>

using namespace pose;

class YOLOv8_pose
{
public:
    explicit YOLOv8_pose(const std::string& engine_file_path);

    ~YOLOv8_pose();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    void infer();

    void postprocess(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100);

    static void draw_objects(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs, const std::vector<std::vector<unsigned int>>& SKELETON, const std::vector<std::vector<unsigned int>>& KPS_COLORS, const std::vector<std::vector<unsigned int>>& LIMB_COLORS);

    bool judge(std::vector<float> keypoints);
    int num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

YOLOv8_pose::YOLOv8_pose(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8_pose::~YOLOv8_pose()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_pose::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8_pose::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min((inp_h / height), (inp_w / width));
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8_pose::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto kps_ptr    = row_ptr + 5;

        float score = *scores_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;
            std::vector<float> kps;
            for (int k = 0; k < 17; k++) {
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x       = clamp(kps_x, 0.f, width);
                kps_y       = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }

            bboxes.push_back(bbox);
            labels.push_back(0);
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        obj.kps   = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8_pose::draw_objects(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs, const std::vector<std::vector<unsigned int>>& SKELETON, const std::vector<std::vector<unsigned int>>& KPS_COLORS, const std::vector<std::vector<unsigned int>>& LIMB_COLORS)
{
    res                 = image.clone();
    const int num_point = 17;
    for (auto& obj : objs) {
        cv::rectangle(res, obj.rect, {0, 0, 255}, 2);

        char text[256];
        sprintf(text, "person %.1f%%", obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

        auto& kps = obj.kps;
        for (int k = 0; k < num_point + 2; k++) {
            if (k < num_point) {
                int   kps_x = std::round(kps[k * 3]);
                int   kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto& ske    = SKELETON[k];
            int   pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int   pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }
    }
}



bool YOLOv8_pose::judge(std::vector<float> keypoints)
{
    // 设置一个判断是否为摔倒的变量
    bool                     is_fall = false;
    std::vector<cv::Point3f> kps;
    // 转换为points
    for (int k = 0; k < 17; k++) {
        float x    = keypoints[k * 3];
        float y    = keypoints[k * 3 + 1];
        float conf = keypoints[k * 3 + 2];
        kps.push_back(cv::Point3f(x, y, conf));
    }

    // 1. 先获取哪些用于判断的点坐标
    cv::Point L_shoulder       = cv::Point((int)kps[5].x, (int)kps[5].y);   // 左肩
    float     L_shoulder_confi = kps[5].z;
    cv::Point R_shoulder       = cv::Point((int)kps[6].x, (int)kps[6].y);   // 右肩
    float     R_shoulder_confi = kps[6].z;
    cv::Point C_shoulder       = cv::Point((int)(L_shoulder.x + R_shoulder.x) / 2,
                                     (int)(L_shoulder.y + R_shoulder.y) / 2);   // 肩部中点

    cv::Point L_hip       = cv::Point((int)kps[11].x, (int)kps[11].y);   // 左髋
    float     L_hip_confi = kps[11].z;
    cv::Point R_hip       = cv::Point((int)kps[12].x, (int)kps[12].y);   // 右髋
    float     R_hip_confi = kps[12].z;
    cv::Point C_hip       = cv::Point((int)(L_hip.x + R_hip.x) / 2,
                                (int)(L_hip.y + R_hip.y) / 2);   // 髋部中点

    cv::Point L_knee       = cv::Point((int)kps[13].x, (int)kps[13].y);   // 左膝
    float     L_knee_confi = kps[13].z;
    cv::Point R_knee       = cv::Point((int)kps[14].x, (int)kps[14].y);   // 右膝
    float     R_knee_confi = kps[14].z;
    cv::Point C_knee       = cv::Point((int)(L_knee.x + R_knee.x) / 2,
                                 (int)(L_knee.y + R_knee.y) / 2);   // 膝部中点

    cv::Point L_ankle       = cv::Point((int)kps[15].x, (int)kps[15].y);   // 左踝
    float     L_ankle_confi = kps[15].z;
    cv::Point R_ankle       = cv::Point((int)kps[16].x, (int)kps[16].y);   // 右踝
    float     R_ankle_confi = kps[16].z;
    cv::Point C_ankle       = cv::Point((int)(L_ankle.x + R_ankle.x) / 2,
                                  (int)(L_ankle.y + R_ankle.y) / 2);   // 计算脚踝中点


    // 2. 第一个判定条件： 若肩的纵坐标最小值min(L_shoulder.y, R_shoulder.y)不低于脚踝的中心点的纵坐标C_ankle.y
    // 且p_shoulders、p_ankle关键点置信度大于预设的阈值，则疑似摔倒。
    if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_ankle_confi > 0.0f && R_ankle_confi > 0.0f) {
        int shoulder_y_min = std::min(L_shoulder.y, R_shoulder.y);
        if (shoulder_y_min >= C_ankle.y) {
            is_fall = true;
            std::cout<<"第一个判断条件"<<std::endl;
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
            std::cout << "第二个判断条件" << std::endl;
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
        // if (aspect_ratio > 0.9f) {
        //     is_fall = true;
        //     std::cout << "第三个判断条件, 宽高比: " << aspect_ratio << std::endl;
        //     return is_fall;
        // }
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
            std::cout << "第四个判断条件" << std::endl;
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
    if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_hip_confi > 0.0f && R_hip_confi > 0.0f && L_knee_confi > 0.0f && R_knee_confi > 0.0f && L_ankle_confi > 0.0f && R_ankle_confi > 0.0f) {
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
            std::cout << "第五个判断条件" << std::endl;
        }
        return is_fall;
    }
}

#endif   // POSE_NORMAL_YOLOv8_pose_HPP