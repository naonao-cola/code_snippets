/*
 * TensorRT 异步推理调度器 (Inference Controller)
 * 提取自 algo_ai/include/private/trt/trt_common/trt_infer_schedule.hpp
 *
 * 核心思想：
 * 利用 std::promise 和 std::future 实现请求-响应模式，结合后台 Worker 线程，
 * 达到前处理、推理、后处理流水线并行的目的，并且支持动态 Batching (commits)。
 */

#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <vector>

template<
    class Input,
    class Output, 
    class StartParam, 
    class JobAdditional
>
class InferController {
public:
    struct Job {
        Input input;
        Output output;
        JobAdditional additional;
        // 绑定独占式 Tensor 内存，避免推理过程中的数据竞争
        // MonopolyAllocator<TRT::Tensor>::MonopolyDataPointer mono_tensor;
        std::shared_ptr<std::promise<Output>> pro;
    };

    virtual ~InferController() { stop(); }

    void stop() {
        run_ = false;
        cond_.notify_all();

        // Cleanup pending jobs
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            while(!jobs_.empty()){
                auto& item = jobs_.front();
                if(item.pro)
                    item.pro->set_value(Output());
                jobs_.pop();
            }
        };

        if(worker_) {
            worker_->join();
            worker_.reset();
        }
    }

    bool startup(const StartParam& param) {
        run_ = true;
        std::promise<bool> pro;
        start_param_ = param;
        worker_ = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));
        return pro.get_future().get();
    }

    virtual std::shared_future<Output> commit(const Input& input) {
        Job job;
        job.pro = std::make_shared<std::promise<Output>>();
        
        // 执行前处理（如 CUDA WarpAffine）
        if(!preprocess(job, input)) {
            job.pro->set_value(Output());
            return job.pro->get_future();
        }
        
        // 压入任务队列，通知 Worker 线程
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            jobs_.push(job);
        }
        cond_.notify_one();
        
        // 返回 Future，主线程可以非阻塞继续执行，直到显式 get()
        return job.pro->get_future();
    }

protected:
    virtual void worker(std::promise<bool>& result) = 0;
    virtual bool preprocess(Job& job, const Input& input) = 0;

    virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size) {
        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_jobs.clear();
        for(int i = 0; i < max_size && !jobs_.empty(); ++i){
            fetch_jobs.emplace_back(std::move(jobs_.front()));
            jobs_.pop();
        }
        return true;
    }

protected:
    StartParam start_param_;
    std::atomic<bool> run_{false};
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;
};
