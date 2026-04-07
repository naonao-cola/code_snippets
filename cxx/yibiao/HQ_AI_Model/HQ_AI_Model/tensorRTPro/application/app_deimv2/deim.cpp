

#include "deim.hpp"
#include <chrono>
#include <common/cuda_tools.hpp>
#include <common/infer_controller.hpp>
#include <common/monopoly_allocator.hpp>
#include <common/preprocess_kernel.cuh>
#include <ios>

namespace DEIM
{

using namespace cv;
using namespace std;

using ControllerImpl = InferController<Mat, BoxArray, tuple<string, int>>;

class InferImpl : public Infer, public ControllerImpl
{
public:
    /** Require stop to be executed in InferImpl, not in the base class **/
    virtual ~InferImpl()
    {
        stop();
    }

    virtual bool startup(const string& file, int gpuid, float confidence_threshold, int max_objects, bool use_multi_preprocess_stream)
    {
        confidence_threshold_ = confidence_threshold;
        return ControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual void worker(promise<bool>& result) override
    {

        string file  = get<0>(start_param_);
        int    gpuid = get<1>(start_param_);

        TRT::set_device(gpuid);
        auto engine = TRT::load_infer(file);
        if (engine == nullptr) {
            INFOE("Engine %s load failed", file.c_str());
            result.set_value(false);
            return;
        }

        engine->print();

        int  max_batch_size = engine->get_max_batch_size();
        auto input_0        = engine->tensor(engine->get_input_name(0));
        auto input_1        = engine->tensor(engine->get_input_name(1));
        auto output_0       = engine->tensor(engine->get_output_name(0));
        auto output_1       = engine->tensor(engine->get_output_name(1));
        auto output_2       = engine->tensor(engine->get_output_name(2));

        input_width_      = input_0->size(3);
        input_height_     = input_0->size(2);
        tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        stream_           = engine->get_stream();
        gpu_              = gpuid;
        result.set_value(true);

        input_0->resize_single_dim(0, max_batch_size).to_gpu();

        vector<Job> fetch_jobs;
        while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {

            int infer_batch_size = fetch_jobs.size();
            input_0->resize_single_dim(0, infer_batch_size);
            input_1->resize_single_dim(0, infer_batch_size);

            float* input_1_data = input_1->gpu<float>();


            int input_size_0 = input_0->size(1) * input_0->size(2) * input_0->size(3);
            int input_size_1 = input_1->size(1);

            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto& job  = fetch_jobs[ibatch];
                auto& mono = job.mono_tensor->data();

                if (mono->get_stream() != stream_) {
                    checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                }

                input_0->copy_from_gpu(input_0->offset(ibatch), (float*)mono->gpu<float>(), input_size_0);
                input_1->copy_from_gpu(input_1->offset(ibatch), (int32_t*)(mono->gpu<float>() + input_size_0), input_size_1);
                job.mono_tensor->release();
            }

            engine->forward(false);

            int output_size_0 = output_0->size(1);
            int output_size_1 = output_1->size(1);
            int output_size_2 = output_2->size(1) * output_2->size(2);
            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {

                auto& job               = fetch_jobs[ibatch];
                auto& image_based_boxes = job.output;

                std::vector<float>   scores(output_size_0);
                std::vector<int32_t> labels(output_size_1);
                std::vector<float>   boxes(output_size_2);

                checkCudaRuntime(cudaMemcpyAsync(
                    scores.data(), output_0->gpu<float>() + ibatch * output_size_0, output_size_0 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                checkCudaRuntime(cudaMemcpyAsync(labels.data(),
                                                 output_1->gpu<int32_t>() + ibatch * output_size_1,
                                                 output_size_1 * sizeof(int32_t),
                                                 cudaMemcpyDeviceToHost,
                                                 stream_));
                checkCudaRuntime(cudaMemcpyAsync(
                    boxes.data(), output_2->gpu<float>() + ibatch * output_size_2, output_size_2 * sizeof(float), cudaMemcpyDeviceToHost, stream_));

                checkCudaRuntime(cudaStreamSynchronize(stream_));

                for (int m = 0; m < scores.size(); m++) {
                    if (scores[m] >= confidence_threshold_) {
                        Box box;
                        box.left        = boxes[4 * m + 0];
                        box.top         = boxes[4 * m + 1];
                        box.right       = boxes[4 * m + 2];
                        box.bottom      = boxes[4 * m + 3];
                        box.confidence  = scores[m];
                        box.class_label = labels[m];
                        image_based_boxes.emplace_back(box);
                    }
                }

                job.pro->set_value(image_based_boxes);
            }
        }

        stream_ = nullptr;
        tensor_allocator_.reset();
        INFO("Engine destroy.");
    }

    virtual bool preprocess(Job& job, const Mat& image) override
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (tensor_allocator_ == nullptr) {
            INFOE("tensor_allocator_ is nullptr");
            return false;
        }

        job.mono_tensor = tensor_allocator_->query();
        if (job.mono_tensor == nullptr) {
            INFOE("Tensor allocator query failed.");
            return false;
        }

        int origin_img_w = image.cols;
        int origin_img_h = image.rows;

        CUDATools::AutoDevice auto_device(gpu_);
        auto&                 tensor            = job.mono_tensor->data();
        TRT::CUStream         preprocess_stream = nullptr;

        if (tensor == nullptr) {
            tensor = make_shared<TRT::Tensor>();
            tensor->set_workspace(make_shared<TRT::MixMemory>());
            preprocess_stream = stream_;
            tensor->set_stream(preprocess_stream, false);
        }

        preprocess_stream = tensor->get_stream();

        int data_size      = input_width_ * input_height_ * 3;
        int total_elements = data_size + 2;
        tensor->resize(1, total_elements);

        size_t   size_image    = image.cols * image.rows * 3;
        auto     workspace     = tensor->get_workspace();
        uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image);
        uint8_t* image_device  = gpu_workspace;

        uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_image);
        uint8_t* image_host    = cpu_workspace;

        memcpy(image_host, image.data, size_image);
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));

        float            mean[3]   = {0.0f, 0.0f, 0.0f};
        float            std[3]    = {1.0f, 1.0f, 1.0f};
        CUDAKernel::Norm normalize = CUDAKernel::Norm::mean_std(mean, std, 1.0f / 255.0f, CUDAKernel::ChannelType::Invert);

        CUDAKernel::resize_normalize_image(image_device,
                                           image.cols * 3,
                                           image.cols,
                                           image.rows,
                                           tensor->gpu<float>(),
                                           input_width_,
                                           input_height_,
                                           input_width_,
                                           normalize,
                                           preprocess_stream);

        int32_t size_data[2] = {origin_img_w, origin_img_h};
        checkCudaRuntime(
            cudaMemcpyAsync((int32_t*)(tensor->gpu<float>() + data_size), size_data, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, preprocess_stream));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // printf("前处理时间: %.3f ms\n", duration.count() / 1000.0);

        return true;
    }

    virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override
    {
        return ControllerImpl::commits(images);
    }

    virtual std::shared_future<BoxArray> commit(const Mat& image) override
    {
        return ControllerImpl::commit(image);
    }
    virtual void setConfidence(float confidence_threshold)
    {
        return;
    }
    virtual void setNMS(float nms_threshold)
    {
        return;
    }

private:
    int           input_width_          = 0;
    int           input_height_         = 0;
    int           gpu_                  = 0;
    float         confidence_threshold_ = 0;
    TRT::CUStream stream_               = nullptr;
};

shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold, int max_objects, bool use_multi_preprocess_stream)
{
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, gpuid, confidence_threshold, max_objects, use_multi_preprocess_stream)) {
        instance.reset();
    }
    return instance;
}

};   // namespace DEIM