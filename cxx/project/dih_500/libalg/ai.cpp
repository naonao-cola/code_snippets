#include "ai.h"

#include <string>

#include "replace_std_string.h"
#include "utils.h"
// #include "DihLog.h"
#include "algLog.h"
#include "alg_error_code.h"
#include "event.h"
#include "imgprocess.h"

#define AI_TASK_ITEM_NUM 64
#define AI_TASK_NODE_NUM (AI_TASK_ITEM_NUM * 2)
#define AI_TASK_FLAG (TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC | TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC)
#define AI_DEF_TIMEOUT 0xFFFF
#define BMP_UNIT 4

#ifndef AI_USE_TIMECNT
#define AI_USE_TIMECNT 1
#endif
#if (AI_USE_TIMECNT)

#include "timecnt.h"
// #include "dynoLog.h"

#endif

/**
 * AI推理函数
 * @param  ctx_id	神经网络上下文ID
 * @param  mod_id	模型ID
 * @param  img		神经网络输入图像
 * @param  result	神经网络输出结果
 * @return 0 success other fail
 */
typedef int (*AiInference_f)(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, NNetImg_t* img, std::list<NNetResult_t>& result);

/* AI推理方法注册信息 */
typedef struct AiInferReg
{
    AiInferType_e type;
    AiInference_f func;
} AiInferReg_t;
#define AI_INFER_TYPE(reg) ((reg)->type)   // 推理类型
#define AI_INFER_FUNC(reg) ((reg)->func)   // 推理函数
#define AI_INFER_DEF(type, func) {type, func}

/* AI推理方法注册表 */
std::vector<AiInferReg_t> ai_infer_reglist = {AI_INFER_DEF(AI_INFER_TYPE_NORMAL, NNet_Inference)};

/* AI项目 */
typedef struct AiItem
{
    uint32_t                     group_idx;
    uint32_t                     chl_idx;
    uint32_t                     view_order;
    uint32_t                     view_idx;
    uint32_t                     view_pair_idx;
    uint32_t                     priority;
    AiImgCallback_f              callback;
    void*                        userdata;
    std::list<AiImg_t>           img_list;
    std::map<std::string, float> call_back_params;
} AiItem_t;
#define AI_ITEM_GROUP_IDX(item) ((item)->group_idx)                 // 分组索引
#define AI_ITEM_CHL_IDX(item) ((item)->chl_idx)                     // 通道索引
#define AI_ITEM_VIEW_ORDER(item) ((item)->view_order)               // 视图序号内部
#define AI_ITEM_VIEW_IDX(item) ((item)->view_idx)                   // 视图索引
#define AI_ITEM_VIEW_PAIR_IDX(item) ((item)->view_pair_idx)         // 视图序号外部
#define AI_ITEM_PRIORITY(item) ((item)->priority)                   // 优先级
#define AI_ITEM_CALLBACK(item) ((item)->callback)                   // 图像回调
#define AI_ITEM_USERDATA(item) ((item)->userdata)                   // 用户数据
#define AI_ITEM_IMAGE_LIST(item) ((item)->img_list)                 // 图像列表
#define AI_ITEM_STOP_FLAG(item) ((item)->stop_flag)                 // 停止标志
#define AI_ITEM_CALL_BACK_PARAMS(item) ((item)->call_back_params)   // call back 参数

/* AI上下文 */
typedef struct AiCtx
{
    TaskFlowCtxID_t task_ctxid;
    NNetCtxID_t     nnet_ctxid;
    uint32_t        stop_flag;
} AiCtx_t;
#define AI_CTX_TASK_CTXID(ctx) ((ctx)->task_ctxid)   // 任务流队列上下文ID
#define AI_CTX_NNET_CTXID(ctx) ((ctx)->nnet_ctxid)   // 神经网络上下文ID
#define AI_CTX_STOP_FLAG(item) ((ctx)->stop_flag)    // 停止标志

#define Ai_MoveImageList(dst, src) (dst).splice((dst).end(), (src), (src).begin())

static void* Ai_ItemConstructor(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, void* taskinfo)
{
    AiItem_t* item = new AiItem_t;
    ALGLogInfo << "Construct ai item in task item";
    return item;
}

static AiImg_t* Ai_FindImg(AiItem_t* item, uint32_t view_idx)
{
    if (item == NULL) {
        return NULL;
    }
    uint32_t idx = 0;
    for (std::list<AiImg_t>::iterator it = AI_ITEM_IMAGE_LIST(item).begin(); it != AI_ITEM_IMAGE_LIST(item).end(); it++) {
        AiImg_t* img = &(*it);
        if (idx == view_idx) {
            return img;
        }
        idx++;
    }
    EVWARN(EVID_WARN, "ai: img %d not found", view_idx);
    return NULL;
}

void Ai_ReleaseImg(AiItem_t* item)
{
    item->img_list.clear();
}

static int Ai_ItemCallback(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, TaskFlowNodeID_t node_id, void* taskinfo, void* userdata)
{
    AiCtx_t*  ctx  = (AiCtx_t*)userdata;
    AiItem_t* item = (AiItem_t*)taskinfo;
    if (ctx == NULL || item == NULL) {
        //		//    DLOG(ERROR, "null ctx");
        return -1;
    }
    if (AI_CTX_STOP_FLAG(ctx)) {
        EVINFO(EVID_SAMP_STOP, "ai: stop");
        return -2;
    }
    AiImg_t*                img = Ai_FindImg(item, AI_ITEM_VIEW_IDX(item));
    std::list<NNetResult_t> result;              // 为保持回调函数结构一致,增加该无用值
    int                     processed_idx = 0;   // 为保持回调函数结构一致,增加该无用值
    if (img) {
        if (AI_ITEM_CALLBACK(item)) {
            int ret = (*AI_ITEM_CALLBACK(item))((AiCtxID_t)ctx,
                                                (AiItemID_t)item_id,
                                                img,
                                                AI_ITEM_GROUP_IDX(item),
                                                AI_ITEM_CHL_IDX(item),
                                                AI_ITEM_VIEW_ORDER(item),
                                                AI_ITEM_VIEW_IDX(item),
                                                processed_idx,
                                                AI_IMG_STAGE_INTERENCE,
                                                AI_ITEM_USERDATA(item),
                                                result,
                                                AI_ITEM_VIEW_PAIR_IDX(item),
                                                AI_ITEM_CALL_BACK_PARAMS(item));
            if (ret == 11) {
                ALGLogError << "ai(图像回调): cb = " << ret << " 细胞计数达到最低要求 " << " group " << AI_ITEM_GROUP_IDX(item) << " channel "
                            << AI_ITEM_CHL_IDX(item);
            }
            else if (ret) {
                EVERROR(EVID_ERR, "ai: cb err=%d", ret);
                ALGLogError << "ai(图像回调): cb err= " << ret;
                return ret;
            }

        }
    }
    else {
        //    DLOG(ERROR, "empty img");
        return -3;
    }
    AI_ITEM_VIEW_IDX(item)++;
    return 0;
}

/**
 * AI去初始化
 * @param ctx_id		AI上下文ID
 * @return
 */
int Ai_DeInit(AiCtxID_t ctx_id)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    if (AI_CTX_TASK_CTXID(ctx)) {
        // TaskFlow_DeInit(AI_CTX_TASK_CTXID(ctx));
        AI_CTX_TASK_CTXID(ctx) = NULL;
        delete ctx;
    }
    return 0;
}

/*!
 * 注入task item 的deconstructor
 */
void TaskInfoDestructorCallback_f(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, void* taskinfo)
{
    AiItem_t* item = (AiItem_t*)taskinfo;
    // 回收cv::Mat内存
    Ai_ReleaseImg(item);
}

/**
 * AI初始化
 * @param  none
 * @return AI上下文ID @ref AiCtxID_t
 */
AiCtxID_t Ai_Init(const int& item_nums)
{

    //    if(Log::get_instance()->init("./algWare.log", 5000, 600000)){
    //        printf("Log::get_instance() Ok\r\n");
    //    }
    //    LOG_OUT("开始初始化算法模块");
    AiCtx_t* ctx = new AiCtx_t;
    //        AI_CTX_TASK_CTXID(ctx) = TaskFlow_Init(item_nums, item_nums*2,
    //        Ai_ItemConstructor, TaskInfoDestructorCallback_f, NULL,
    //                                               AI_TASK_FLAG);
    AI_CTX_TASK_CTXID(ctx) = TaskFlow_Init(item_nums, item_nums * 2, Ai_ItemConstructor, NULL, NULL, AI_TASK_FLAG);
    if (AI_CTX_TASK_CTXID(ctx) == NULL) {

        return NULL;
    }
    AI_CTX_STOP_FLAG(ctx) = 0;
#if (AI_USE_TIMECNT)
    TimeCnt_Init("模型推理", 1);
    TimeCnt_Init("像源融合", 1);
    TimeCnt_Init("算法队列推送", 1);
    TimeCnt_Init("梯度计算", 1);
#endif
    return (AiCtxID_t)ctx;
}

int Ai_SetNet(AiCtxID_t ctx_id, NNetCtxID_t nnet_ctxid)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL || nnet_ctxid == NULL) {
        return -1;
    }
    AI_CTX_NNET_CTXID(ctx) = nnet_ctxid;
    return 0;
}

int Ai_ResetNet(AiCtxID_t ctx_id)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    AI_CTX_NNET_CTXID(ctx) = NULL;
    return 0;
}

void bgr_to_rgb(uint8_t* bgr, uint8_t* rgb, int width, int height)
{
    // Ensure BGR and BGR buffers are 16-byte aligned for NEON
    uint8_t* bgr_aligned = (uint8_t*)(((uintptr_t)bgr + 15) & ~15);
    uint8_t* rgb_aligned = (uint8_t*)(((uintptr_t)rgb + 15) & ~15);
#pragma omp parallel for
    for (int q = 0; q < height * width / 16; q++) {
        // Calculate the index for the current pixel
        int index = q * 16 * 3;

        // Load 16 BGR pixels into three vectors.
        uint8x16x3_t bgr_vector = vld3q_u8(bgr + index);
        //    ALGLogError<<"242";
        // Shuffle the bytes to convert from BGR to BGR.
        uint8x16_t b = bgr_vector.val[2];   // Blue
        uint8x16_t g = bgr_vector.val[1];   // Green
        uint8x16_t r = bgr_vector.val[0];   // Red

        // Combine the shuffled bytes into a single vector.
        uint8x16x3_t rgb_vector = {b, g, r};
        //    ALGLogError<<"250 "<<rgb_aligned;
        // Store the result.
        vst3q_u8(rgb + index, rgb_vector);
        //    ALGLogError<<"253";
    }
}

void rgba2rgb_with_neon(const uint8_t* rgba_img, uint8_t* rgb_img, int height, int width)
{
    const int total_pixels  = height * width;
    const int stride_pixels = 16;
#pragma omp parallel for
    for (int i = 0; i < total_pixels; i += stride_pixels) {
        const uint8_t* src = rgba_img + i * 3;
        uint8_t*       dst = rgb_img + i * 3;

        uint8x16x3_t a = vld3q_u8(src);
        uint8x16x3_t b;
        b.val[0] = a.val[0];
        b.val[1] = a.val[1];
        b.val[2] = a.val[2];
        vst3q_u8(dst, b);
    }
}

/**
 * AI转换图像
 * @param list       矩阵列表
 * @param img        图像缓存
 * @param width      图像宽度
 * @param height     图像高度
 * @return
 */

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
int Ai_ConvertImage(std::list<AiImg_t>& list, uint8_t* img, uint32_t width, uint32_t height, const bool& img_fusion, const float& fusion_rate)
{
    if (img == NULL || !(width * height)) {
        return -1;
    }
#if (AI_USE_TIMECNT)
    TimeCnt_Start("像源融合");
#endif
    try {

        //                auto start =
        //                duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        cv::Mat img_ori(int(height), int(width), CV_8UC3, img);

        // cv::Mat
        // img_processed(int(height),int(width),CV_8UC3);//???用中括号初始化,报错.用小括号就不报错

        //                bgr_to_rgb(img, (uint8_t*)img_processed.data, int(width),
        //                int(height)); rgba2rgb_with_neon((uint8_t*)img_ori.data,
        //                (uint8_t*)img_processed.data,
        //                                   int(height), int(width));

        // 内存中获取的图像为bgr格式,需要转换为rgb格式
        cv::Mat img_processed;
        cv::cvtColor(img_ori, img_processed, cv::COLOR_BGR2RGB);
        if (img_fusion) {
            cv::resize(img_processed, img_processed, cv::Size(0, 0), fusion_rate, fusion_rate, cv::INTER_AREA);
            // 计算pad值
            int resized_h  = int(float(height) * fusion_rate);
            int resized_w  = int(float(width) * fusion_rate);
            int pad_bottom = ceil(float(resized_h) / BMP_UNIT) * BMP_UNIT - resized_h;
            int pad_right  = ceil(float(resized_w) / BMP_UNIT) * BMP_UNIT - resized_w;
            cv::copyMakeBorder(img_processed, img_processed, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT, 0);
        }
        list.push_back(img_processed);
        //                auto func_end
        //                =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        //                auto func_cost_time = func_end-start;
        //                std::cout<<"pushing time: "<<func_cost_time<<std::endl;
    }
    catch (std::exception& e) {
        EVERROR(EVID_ERR, "ai: fusion err=%s", e.what())
        return -2;
    }
#if (AI_USE_TIMECNT)
    TimeCnt_End("像源融合");
#endif
    return 0;
}

/*!
 * 清晰度算法调用频繁,且模型对输入图像的细微变化不敏感,此处对清晰度算法的转换进行特殊操作以减少图像copy时间.
 * @param list
 * @param img
 * @param width
 * @param height
 * @param img_fusion
 * @param fusion_rate
 * @return
 */
int Ai_ConvertClarityImage(std::list<AiImg_t>& list,
                           uint8_t*            img,
                           uint32_t            width,
                           uint32_t            height,
                           const bool&         img_fusion,
                           const float&        fusion_rate,
                           const int&          target_width,
                           const int&          target_height,
                           const ResizeType&   resize_type)
{
    if (img == NULL || !(width * height)) {
        return -1;
    }
#if (AI_USE_TIMECNT)
    TimeCnt_Start("像源融合");
#endif

    cv::Mat img_ori(int(height), int(width), CV_8UC3, img);
    std::cout << " 382 ResizeImg 输入 尺寸 宽 " << img_ori.cols << " 高 " << img_ori.rows << std::endl;
    std::cout << " 382 ResizeImg 传入的 缩放类型 " << resize_type << std::endl;
    if (resize_type ==0) {
    //     // 0 类型什么都不做，模型输入原图
        //cv::Mat img_processed;
        //cv::flip(img_ori, img_processed, 0);
        //cv::cvtColor(img_processed, img_processed, cv::COLOR_BGR2RGB);
        //list.push_back(img_processed);
        //cv::resize(img_ori, img_processed, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        //cv::flip(img_ori, img_processed, 0);
        //cv::cvtColor(img_processed, img_processed, cv::COLOR_BGR2RGB);
        //list.push_back(img_ori);
        //return 0;
    }
    // resize -> bgr2rgb -> flip
    cv::Mat img_processed;
    ResizeImg(img_ori, img_processed, cv::Size(target_width, target_height), resize_type);
    // std::cout << " 387 ResizeImg 之后尺寸 宽 " << img_processed.cols << " 高 " << img_processed.rows << std::endl;
    cv::cvtColor(img_processed, img_processed, cv::COLOR_BGR2RGB);
    cv::flip(img_processed, img_processed, 0);

    if (img_fusion) {
        /*      cv::resize(img_processed, img_processed, cv::Size(0, 0),
                            fusion_rate, fusion_rate, cv::INTER_AREA);
                //计算pad值
                int resized_h = int(float(height) * fusion_rate);
                int resized_w = int(float(width) * fusion_rate);
                int pad_bottom =
                    ceil(float(resized_h) / BMP_UNIT) * BMP_UNIT - resized_h;
                int pad_right =
                    ceil(float(resized_w) / BMP_UNIT) * BMP_UNIT - resized_w;
                cv::copyMakeBorder(img_processed, img_processed, 0,
                                    pad_bottom, 0, pad_right,
                                    cv::BORDER_CONSTANT, 0);*/
    }
    list.push_back(img_processed);

#if (AI_USE_TIMECNT)
    TimeCnt_End("像源融合");
#endif
    return 0;
    }

/**
 * AI项目推送
 * @param  ctx_id        AI上下文ID
 * @param  priority      优先级
 * @param  group_idx     分组索引
 * @param  chl_idx       通道索引
 * @param  view_count    视图计数
 * @param  img_list      输入图像列表
 * @param  callback      图像回调
 * @param  userdata      用户数据
 * @return AI项目ID  @ref AiItemID_t
 */
AiItemID_t Ai_ItemPush(AiCtxID_t                           ctx_id,
                       uint32_t                            priority,
                       uint32_t                            group_idx,
                       uint32_t                            chl_idx,
                       uint32_t                            view_order,
                       uint32_t                            view_count,
                       std::list<AiImg_t>&                 img_list,
                       AiImgCallback_f                     callback,
                       void*                               userdata,
                       const std::vector<AiViewReg_t>&     view_list,
                       const int&                          view_pair_idx,
                       const std::map<std::string, float>& call_back_params)
{
    std::cout << "Ai_ItemPush 0" << std::endl;
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
#if (AI_USE_TIMECNT)
    TimeCnt_Start("算法队列推送");
#endif
    std::cout << "Ai_ItemPush 1" << std::endl;
    if (ctx == NULL || !view_count || img_list.size() < view_count) {
        ALGLogError << "Ai_ItemPush 推入图片失败,push param err 推入的视图数目： " << view_count << " 图片数量 " << img_list.size() << "\n";
        return 0;
    }
    std::cout << "Ai_ItemPush 推入的组： " << group_idx << " 通道数 " << chl_idx << " 视图 " << view_pair_idx << " view_count: " << view_count
              << std::endl;

    // for(int i=0;i<5;i++){
    //     int TaskFlow_GetIdleSize_ret = TaskFlow_GetIdleSize(AI_CTX_TASK_CTXID(ctx));
    //     if (TaskFlow_GetIdleSize_ret < img_list.size()) {
    //         std::cout << "TaskFlow_GetIdleSize_ret： " << TaskFlow_GetIdleSize_ret << std::endl;
    //         sleep(1);
    //     }
    //     else{
    //         std::cout << "TaskFlow_GetIdleSize_ret break ： " << TaskFlow_GetIdleSize_ret << " index " << i << std::endl;
    //         break;
    //     }
    // }

    AiItemID_t item_id = TaskFlow_New(AI_CTX_TASK_CTXID(ctx), NULL, NULL, NULL, AI_DEF_TIMEOUT);

    if (!item_id) {
        ALGLogError << "Ai_ItemPush 推入图片失败,task new fail " << "\n";
        return 0;
    }
    std::cout << "Ai_ItemPush 2" << std::endl;
    AiItem_t* item = (AiItem_t*)TaskFlow_GetTaskInfo(AI_CTX_TASK_CTXID(ctx), item_id);
    if (item == NULL) {
        TaskFlow_Del(AI_CTX_TASK_CTXID(ctx), item_id);
        ALGLogError << "Ai_ItemPush 推入图片失败,task info fail " << "\n";
        return 0;
    }
    std::cout << "Ai_ItemPush 3" << std::endl;
#if (AI_USE_TIMECNT)
    TimeCnt_End("算法队列推送");
#endif
    AI_ITEM_GROUP_IDX(item)        = group_idx;
    AI_ITEM_CHL_IDX(item)          = chl_idx;
    AI_ITEM_VIEW_IDX(item)         = 0;
    AI_ITEM_PRIORITY(item)         = priority;
    AI_ITEM_CALLBACK(item)         = callback;
    AI_ITEM_USERDATA(item)         = userdata;
    AI_ITEM_VIEW_ORDER(item)       = view_order;
    AI_ITEM_VIEW_PAIR_IDX(item)    = view_pair_idx;
    AI_ITEM_CALL_BACK_PARAMS(item) = call_back_params;
    if (false == AI_ITEM_IMAGE_LIST(item).empty()) {
        AI_ITEM_IMAGE_LIST(item).clear();
    }
    std::cout << "Ai_ItemPush 4" << std::endl;
    for (uint32_t idx = 0; idx < view_count; idx++) {
        Ai_MoveImageList(AI_ITEM_IMAGE_LIST(item), img_list);
        if (!TaskFlow_Node(AI_CTX_TASK_CTXID(ctx), item_id, AI_ITEM_PRIORITY(item), Ai_ItemCallback, (void*)ctx, AI_DEF_TIMEOUT)) {
            ALGLogError << "Ai_ItemPush 推入图片失败,task node fail " << "\n";
            TaskFlow_Del(AI_CTX_TASK_CTXID(ctx), item_id);
            std::cout << "Ai_ItemPush 推入图片失败,task node fail "  << std::endl;

            return 0;
        }
        if (AI_ITEM_CALLBACK(item)) {
        }
    }
    std::cout << "Ai_ItemPush 5" << std::endl;
    if (TaskFlow_Start(AI_CTX_TASK_CTXID(ctx), item_id)) {
        ALGLogError << "Ai_ItemPush 推入图片失败,task start fail " << "\n";
        return 0;
    }
    return item_id;
}

/**
 * AI项目增加图像
 * @param  ctx_id        AI上下文ID
 * @param  item_id       AI项目ID
 * @param  img           输入图像
 * @return
 */
int Ai_ItemAddImg(AiCtxID_t ctx_id, AiItemID_t item_id, AiImg_t& img)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    AiItem_t* item = (AiItem_t*)TaskFlow_GetTaskInfo(AI_CTX_TASK_CTXID(ctx), item_id);
    if (item) {
        AI_ITEM_IMAGE_LIST(item).push_back(img);
        return TaskFlow_Node(AI_CTX_TASK_CTXID(ctx), item_id, AI_ITEM_PRIORITY(item), Ai_ItemCallback, (void*)ctx, 0xFFFF);
    }
    EVERROR(EVID_ERR, "ai: task info fail")
    return -2;
}

/**
 * AI项目获取图像
 * @param  ctx_id        AI上下文ID
 * @param  item_id       AI项目ID
 * @param  view_idx      视图索引
 * @return
 */
AiImg_t* Ai_ItemGetImg(AiCtxID_t ctx_id, AiItemID_t item_id, uint32_t view_idx)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return NULL;
    }
    AiItem_t* item = (AiItem_t*)TaskFlow_GetTaskInfo(AI_CTX_TASK_CTXID(ctx), item_id);
    if (item) {
        AiImg_t* img = Ai_FindImg(item, view_idx);
        if (img) {
            return img;
        }
    }
    return NULL;
}

/**
 * 删除AI项目
 * @param  ctx_id        AI上下文ID
 * @param  item_id       AI项目ID
 * @return
 */
int Ai_ItemDel(AiCtxID_t ctx_id, AiItemID_t item_id)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    return TaskFlow_Del(AI_CTX_TASK_CTXID(ctx), item_id);
}

/**
 * 等待指定优先级AI项目完成
 * @param  ctx_id        AI上下文ID
 * @param  priority      优先级
 * @param  timeout       超时时间
 * @return
 */
int Ai_WaitPriority(AiCtxID_t ctx_id, uint32_t priority, uint32_t timeout)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == nullptr) {
        ALGLogError << "上下文 ctx 为空";
        return -1;
    }

    return TaskFlow_WaitGroup(AI_CTX_TASK_CTXID(ctx), priority, timeout);
}

/**
 * 等待所有AI项目完成
 * @param  ctx_id        AI上下文ID
 * @param  timeout       超时时间
 * @return
 */
int Ai_WaitAll(AiCtxID_t ctx_id, uint32_t timeout)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    return TaskFlow_WaitAll(AI_CTX_TASK_CTXID(ctx), timeout);
}

int Ai_CleanItemAll(AiCtxID_t ctx_id, uint32_t timeout)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    AI_CTX_STOP_FLAG(ctx) = 1;
    int ret               = TaskFlow_WaitAll(AI_CTX_TASK_CTXID(ctx), timeout);
    if (ret) {
        EVERROR(EVID_ERR, "ai: clean item err=%d", ret)
    }
    AI_CTX_STOP_FLAG(ctx) = 0;
    return ret;
}

static AiMap_t* Ai_FindMapLabels(std::vector<AiMap_t>& map_list, const char* labels_name)
{
    for (uint32_t idx = 0; idx < map_list.size(); idx++) {
        AiMap_t* map = &map_list.at(idx);
        if (!strcmp(AI_MAP_LABELS_NAME(map), labels_name)) {
            return map;
        }
    }
    EVWARN(EVID_WARN, "ai: label not found=%s", labels_name)
    return NULL;
}

static int Ai_CountSingleLabel(void* map_addr, std::vector<AiMap_t>& map_list, std::list<NNetResult_t>& result)
{
    if (map_addr == NULL || true == map_list.empty()) {
        return -1;
    }
    if (false == result.empty()) {
        // 单个类别不需要遍历类别名,直接计数
        NNetResult_t* blk = &result.front();
        AiMap_t*      map = Ai_FindMapLabels(map_list, NNET_OPT_NAME(blk).data());
        if (map) {
            *((AiCntVal_t*)((long)map_addr + (long)AI_MAP_OFFSET(map))) += result.size();
        }
    }
    return 0;
}

static int Ai_CountMultiLabel(void* map_addr, std::vector<AiMap_t>& map_list, std::list<NNetResult_t>& result)
{
    if (map_addr == NULL || true == map_list.empty()) {
        return -1;
    }
    if (false == result.empty()) {
        for (std::list<NNetResult_t>::iterator it = result.begin(); it != result.end(); it++) {
            NNetResult_t* blk = &(*it);
            AiMap_t*      map = Ai_FindMapLabels(map_list, NNET_OPT_NAME(blk).data());
            if (map) {
                if (NNET_OPT_PROP(blk) >= AI_MAP_MIN_PROP(map)) {
                    *((AiCntVal_t*)((long)map_addr + (long)AI_MAP_OFFSET(map))) += 1;
                }
            }
        }
    }
    return 0;
}

/**
 * AI结果计数
 * @param  mod       模型注册信息
 * @param  map_addr  计数器映射地址
 * @param  map_list  计数器映射列表
 * @param  result    神经网络结果
 * @return AI上下文ID @ref AiCtxID_t
 */
int Ai_ResultCount(void* map_addr, std::vector<AiMap_t>& map_list, std::list<NNetResult_t>& result, uint8_t multi_label_flag)
{

    if (map_addr) {
        if (multi_label_flag) {
            return Ai_CountMultiLabel(map_addr, map_list, result);
        }
        else {
            return Ai_CountSingleLabel(map_addr, map_list, result);
        }
    }
    //    DLOG(ERROR, "null map addr");
    return ALG_ERR_COUNT_MODEL_RESULT;
}

/**
 * 添加AI模型
 * @param  ctx_id			AI上下文ID
 * @param  group_id			分组ID
 * @param  mod_id			模型ID
 * @param  mod_data			模型数据
 * @param  mod_size			模型大小
 * @param  labels_data		标签数据
 * @param  labels_size		标签大小
 * @param  multi_label_flag	多标签标记
 * @return 0 success other fail
 */
int Ai_AddModel(AiCtxID_t                       ctx_id,
                NNetGroup_e                     group_id,
                NNetModID_e                     mod_id,
                uint8_t*                        mod_data,
                uint32_t                        mod_size,
                const ResizeType&               resize_type,
                float                           model_type_nums,
                float                           nms_nums,
                float                           conf_nums,
                float                           anchor_nums,
                float                           label_nums,
                float                           reserved_float_param_nums,
                float                           reserved_string_param_nums,
                const std::vector<float>&       model_type,
                const std::vector<float>&       nms,
                const std::vector<float>&       conf,
                const std::vector<float>&       anchors,
                const std::vector<std::string>& labels,
                const std::vector<float>&       reserved_float_params,
                const std::vector<std::string>& reserved_string_params)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    return NNet_AddModel(AI_CTX_NNET_CTXID(ctx),
                         group_id,
                         mod_id,
                         mod_data,
                         mod_size,
                         resize_type,
                         model_type_nums,
                         nms_nums,
                         conf_nums,
                         anchor_nums,
                         label_nums,
                         reserved_float_param_nums,
                         reserved_string_param_nums,
                         model_type,
                         nms,
                         conf,
                         anchors,
                         labels,
                         reserved_float_params,
                         reserved_string_params);
}

static AiInferReg_t* Ai_FindInfer(uint32_t type)
{
    for (uint32_t idx = 0; idx < ai_infer_reglist.size(); idx++) {
        AiInferReg_t* infer = &ai_infer_reglist.at(idx);
        if (AI_INFER_TYPE(infer) == type) {
            return infer;
        }
    }
    return NULL;
}

/**
 * AI推理
 * @param  ctx_id	AI下文ID
 * @param  group_id  分组ID
 * @param  mod_id    模型ID
 * @param  img		输入图像
 * @param  result	神经网络输出结果
 * @param  type      推理类型
 * @return 0 success other fail
 */
int Ai_Inference(AiCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, AiImg_t* img, std::list<NNetResult_t>& result, AiInferType_e type)
{
    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == nullptr || img == nullptr) {

        if (ctx == nullptr) {
            ALGLogError << "ctx: nullptr";
        }

        ALGLogError << "img: nullptr";

        return -1;
    }
    AiInferReg_t* infer = Ai_FindInfer(type);
    if (infer == NULL || AI_INFER_FUNC(infer) == NULL) {
        EVERROR(EVID_ERR, "ai: infer not found=%d", type)
        ALGLogError << "ai: infer not found: " << type;
        return -2;
    }
#if (AI_USE_TIMECNT)
    TimeCnt_Start("模型推理");
#endif
    int ret = (*AI_INFER_FUNC(infer))(AI_CTX_NNET_CTXID(ctx), group_id, mod_id, img, result);

    // std::cout << " 783  Ai_Inference group_id: " << group_id << " mod_id: " << mod_id << std::endl;
    // std::cout << " 783  Ai_Inference 图片大小 : " << img->rows << " " << img->cols << std::endl;
    // for (int i = 0; i < result.begin()->category_v.size(); i++) {
    //     std::cout << " index: " << i << " ret: " << result.begin()->category_v[i];
    // }

    if (ret) {
        EVERROR(EVID_ERR, "ai: infer cb err=%d", ret)
        ALGLogError << "ai: (模型推理)AI_INFER_FUNC NNet_Inference err: " << ret;
    }
#if (AI_USE_TIMECNT)
    else {
        TimeCnt_End("模型推理");
    }
#endif

    return ret;
}

/**
 * 获取AI样本分组注册信息
 * @param  src_list		原始分组列表
 * @param  group_idx		分组索引
 * @return AI样本分组注册信息指针 @ref AiGroupReg_t
 */
AiGroupReg_t* Ai_FindGroup(std::vector<AiGroupReg_t>& src_list, uint32_t group_idx)
{
    if (group_idx < src_list.size()) {
        return &src_list.at(group_idx);
    }
    EVWARN(EVID_WARN, "ai: group not found=g%d", group_idx)
    return NULL;
}

/**
 * 获取AI模型注册信息
 * @param  src_list		原始模型列表
 * @param  group_idx		分组索引
 * @return AI模型注册信息指针 @ref AiModReg_t
 */
AiModReg_t* Ai_FindMod(AiGroupReg_t* group, NNetModID_e mod_id)
{
    if (group) {
        std::vector<AiModReg_t>* src_list = AI_GROUP_MOD_LIST(group);
        if (src_list) {
            for (uint32_t idx = 0; idx < src_list->size(); idx++) {
                AiModReg_t* mod = &src_list->at(idx);
                if ((AI_MOD_ID(mod) == mod_id) && (AI_MOD_GROUP_MASK(mod) & AI_GROUP_ID(group))) {
                    return mod;
                }
            }
        }
        EVWARN(EVID_WARN, "ai: mod not found=gt%dmid%d", AI_GROUP_TYPE(group), mod_id)
    }
    return NULL;
}

/**
 * 获取AI通道注册信息
 * @param  src_list		原始通道列表
 * @param  group		    分组信息
 * @param  chl_idx		通道索引
 * @return AI通道注册信息指针 @ref AiChlReg_t
 */
// 需要判断该流道是否配置了该类型,所需需要遍历
AiChlReg_t* Ai_FindChl(AiGroupReg_t* group, uint32_t chl_idx)
{
    if (group) {
        std::vector<AiChlReg_t>* src_list = AI_GROUP_CHL_LIST(group);
        if (src_list) {
            uint32_t cnt = 0;
            for (uint32_t idx = 0; idx < src_list->size(); idx++) {
                AiChlReg_t* chl = &src_list->at(idx);
                if (AI_CHL_GROUP_MASK(chl) & AI_GROUP_ID(group)) {
                    if (cnt++ == chl_idx) {
                        return chl;
                    }
                }
            }
        }
        EVWARN(EVID_WARN, "ai: mod not found=gt%dc%d", AI_GROUP_TYPE(group), chl_idx)
    }
    return NULL;
}

/**
 * 获取AI视图注册信息
 * @param  group		    分组信息
 * @param  chl		    通道信息
 * @param  view_idx		视图索引
 * @return AI视图注册信息指针 @ref AiViewReg_t
 */
AiViewReg_t* Ai_FindView(AiGroupReg_t* group, AiChlReg_t* chl, uint32_t view_idx)
{
    if (group && chl && AI_CHL_VIEW_LIST(chl)) {
        uint32_t cnt = 0;
        for (uint32_t idx = 0; idx < AI_CHL_VIEW_LIST(chl)->size(); idx++) {
            AiViewReg_t* view = &(AI_CHL_VIEW_LIST(chl)->at(idx));
            if (AI_VIEW_MOD_GROUP_MASK(view) & AI_GROUP_ID(group)) {
                if (cnt++ == view_idx) {
                    return view;
                }
            }
        }
        EVWARN(EVID_WARN, "ai: mod not found=gt%dct%dv%d", AI_GROUP_TYPE(group), AI_CHL_TYPE(chl), view_idx)
    }
    return NULL;
}

/**
 * 获取AI模型注册表
 * @param  dst_list		目标模型列表
 * @param  src_list		原始模型列表
 * @param  group		    分组信息
 * @return 0 success other fail
 */
int Ai_GetModReglist(std::vector<AiModReg_t>& dst_list, AiGroupReg_t* group)
{
    if (group == NULL) {
        return -1;
    }
    std::vector<AiModReg_t>* src_list = AI_GROUP_MOD_LIST(group);
    if (src_list == NULL) {
        EVERROR(EVID_ERR, "ai: mod list not found=gt%d", AI_GROUP_TYPE(group))
        return -2;
    }
    for (uint32_t idx = 0; idx < src_list->size(); idx++) {
        AiModReg_t* mod = &src_list->at(idx);
        if (AI_MOD_GROUP_MASK(mod) & AI_GROUP_ID(group)) {
            dst_list.push_back(*mod);
        }
    }
    return 0;
}

/**
 * 获取AI通道注册表
 * @param  dst_list		目标通道列表
 * @param  src_list		原始通道列表
 * @param  group		    分组信息
 * @return 0 success other fail
 */
int Ai_GetChlReglist(std::vector<AiChlReg_t>& dst_list, AiGroupReg_t* group)
{
    if (group == NULL) {
        return -1;
    }
    std::vector<AiChlReg_t>* src_list = AI_GROUP_CHL_LIST(group);
    if (src_list == NULL) {
        EVERROR(EVID_ERR, "ai: chl list not found=gt%d", AI_GROUP_TYPE(group))
        return -2;
    }
    for (uint32_t idx = 0; idx < src_list->size(); idx++) {
        AiChlReg_t* chl = &src_list->at(idx);
        if (AI_CHL_GROUP_MASK(chl) & AI_GROUP_ID(group)) {
            dst_list.push_back(*chl);
        }
    }
    return 0;
}

/**
 * 获取AI视图注册表
 * @param  dst_list		目标视图列表
 * @param  group		    分组信息
 * @param  chl		    通道信息
 * @return 0 success other fail
 */
int Ai_GetViewReglist(std::vector<AiViewReg_t>& dst_list, AiGroupReg_t* group, AiChlReg_t* chl)
{
    if (group == NULL || chl == NULL) {
        return -1;
    }
    if (AI_CHL_VIEW_LIST(chl) == NULL) {
        EVERROR(EVID_ERR, "ai: view list not found=ct%d", AI_CHL_TYPE(chl))
        return -2;
    }
    for (uint32_t idx = 0; idx < AI_CHL_VIEW_LIST(chl)->size(); idx++) {
        AiViewReg_t* view = &AI_CHL_VIEW_LIST(chl)->at(idx);
        if (AI_VIEW_MOD_GROUP_MASK(view) & AI_GROUP_ID(group)) {
            dst_list.push_back(*view);
        }
    }
    return 0;
}
/*!
 * 遍历所有组查找最大channel数
 * @param group_reglist
 * @param channel_nums
 */
void Ai_FindMaxChannelNums(const std::vector<AiGroupReg_t>& group_reglist, int& channel_nums)
{
    channel_nums = 0;
    for (const auto& group : group_reglist) {
        if (group.chl_reglist->size() > channel_nums) {
            channel_nums = group.chl_reglist->size();
        }
    }
}
/*!
 * 查找指定模型的reserved_params,不同模型内的reserved_params含义不同,由模型部署方指定
 * @param ctx_id
 * @param group_id
 * @param mod_id
 * @param category_conf
 * @return
 */
int Ai_GetNetReservedFloatPrams(AiCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, std::vector<float>& reserved_float_params)
{

    AiCtx_t* ctx = (AiCtx_t*)ctx_id;
    if (ctx == nullptr) {
        ALGLogError << "Null ptr";
        return -1;
    }
    return NNet_GetReservedFloatPrams(ctx->nnet_ctxid, group_id, mod_id, reserved_float_params);
}
