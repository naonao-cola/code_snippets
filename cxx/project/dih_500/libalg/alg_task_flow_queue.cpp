#include "alg_task_flow_queue.h"
// #include "DihLog.h"
#include "algLog.h"

#include <list>
#include <malloc.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <unistd.h>
// #include "AigTaskMutex.h"
#include <chrono>
#include <errno.h>
#include <stdio.h>

/* 任务节点 */
typedef struct TaskFlowNode
{
    TaskFlowNodeID_t   node_id;
    uint32_t           group_idx;
    TaskFlowCallback_f callback;
    void*              userdata;
} TaskFlowNode_t;
#define TASKFLOW_NODE_ID(node) ((node)->node_id)            // 任务节点ID
#define TASKFLOW_NODE_GROUP_IDX(node) ((node)->group_idx)   // 优先级组索引
#define TASKFLOW_NODE_CALLBACK(node) ((node)->callback)     // 用户回调
#define TASKFLOW_NODE_USERDATA(node) ((node)->userdata)     // 用户数据

/* 任务项目 */
typedef struct TaskFlowItem
{
    TaskFlowItemID_t          item_id;
    uint32_t                  start_idx;
    void*                     taskinfo;
    TaskInfoConstructor_f     constructor;
    TaskInfoDestructor_f      destructor;
    std::list<TaskFlowNode_t> node_list;
} TaskFlowItem_t;
#define TASKFLOW_ITEM_ID(item) ((item)->item_id)                // 任务项目ID
#define TASKFLOW_ITEM_START_IDX(item) ((item)->start_idx)       // 启动序号
#define TASKFLOW_ITEM_TASKINFO(item) ((item)->taskinfo)         // 任务信息
#define TASKFLOW_ITEM_CONSTRUCTOR(item) ((item)->constructor)   // 任务信息构建函数
#define TASKFLOW_ITEM_DESTRUCTOR(item) ((item)->destructor)     // 任务信息析构函数
#define TASKFLOW_ITEM_INFOSIZE(item) ((item)->infosize)         // 任务信息大小
#define TASKFLOW_ITEM_NODE_LIST(item) ((item)->node_list)       // 任务节点列表

/* 任务优先级组 */
typedef struct TaskFlowGroup
{
    uint32_t                  wait_empty_cnt;
    sem_t                     sem_empty;
    std::list<TaskFlowItem_t> item_list;
} TaskFlowGroup_t;
#define TASKFLOW_GROUP_WAIT_EMPTY_CNT(group) ((group)->wait_empty_cnt)   // 等待优先级组置空计数
#define TASKFLOW_GROUP_SEM_EMPTY(group) (&(group)->sem_empty)            // 优先级组置空信号
#define TASKFLOW_GROUP_ITEM_LIST(group) ((group)->item_list)             // 任务项目列表

/* 任务队列上下文 */
typedef struct TaskFlowCtx
{
    uint32_t                  cfg_flag;
    uint32_t                  empty_group_cnt;
    uint32_t                  wait_all_empty_cnt;
    uint32_t                  wait_idle_item_cnt;
    uint32_t                  wait_idle_node_cnt;
    sem_t                     sem_all_empty;
    sem_t                     sem_idle_item;
    sem_t                     sem_idle_node;
    sem_t                     sem_start;
    pthread_mutex_t           mut_lock;
    pthread_t                 thr_main;
    uint32_t                  item_id_idx;
    uint32_t                  node_id_idx;
    uint32_t                  start_idx;
    TaskInfoConstructor_f     pre_constructor;
    TaskInfoDestructor_f      pre_destructor;
    void*                     pre_taskinfo;
    TaskFlowGroup_t           priority_group[TASKFLOW_MAX_GROUP_NUM];
    std::list<TaskFlowItem_t> ready_item_list;
    std::list<TaskFlowItem_t> run_item_list;
    std::list<TaskFlowNode_t> run_node_list;
    std::list<TaskFlowItem_t> idle_item_list;
    std::list<TaskFlowNode_t> idle_node_list;
} TaskFlowCtx_t;
#define TASKFLOW_CTX_CFG_FLAG(ctx) ((ctx)->cfg_flag)                       // 配置标志
#define TASKFLOW_CTX_EMPTY_GROUP_CNT(ctx) ((ctx)->empty_group_cnt)         // 优先级组置空计数
#define TASKFLOW_CTX_WAIT_EMPTY_CNT(ctx) ((ctx)->wait_all_empty_cnt)       // 等待全部优先级组置空计数
#define TASKFLOW_CTX_WAIT_IDLE_ITEM_CNT(ctx) ((ctx)->wait_idle_item_cnt)   // 等待空闲任务项目计数
#define TASKFLOW_CTX_WAIT_IDLE_NODE_CNT(ctx) ((ctx)->wait_idle_node_cnt)   // 等待空闲任务节点计数
#define TASKFLOW_CTX_SEM_ALL_EMPTY(ctx) (&(ctx)->sem_all_empty)            // 全部优先级组置空信号
#define TASKFLOW_CTX_SEM_IDLE_ITEM(ctx) (&(ctx)->sem_idle_item)            // 空闲任务项目信号
#define TASKFLOW_CTX_SEM_IDLE_NODE(ctx) (&(ctx)->sem_idle_node)            // 空闲任务节点信号
#define TASKFLOW_CTX_SEM_START(ctx) (&(ctx)->sem_start)                    // 开始任务流信号
#define TASKFLOW_CTX_MUT_LOCK(ctx) (&(ctx)->mut_lock)                      // 队列锁定互斥
#define TASKFLOW_CTX_THR_MAIN(ctx) ((ctx)->thr_main)                       // 主线程
#define TASKFLOW_CTX_ITEM_ID_IDX(ctx) ((ctx)->item_id_idx)                 // 任务项目ID索引
#define TASKFLOW_CTX_NODE_ID_IDX(ctx) ((ctx)->node_id_idx)                 // 任务节点ID索引
#define TASKFLOW_CTX_START_IDX(ctx) ((ctx)->start_idx)                     // 任务启动索引
#define TASKFLOW_CTX_PRE_CONSTRUCTOR(ctx) ((ctx)->pre_constructor)         // 预分配任务信息构造函数
#define TASKFLOW_CTX_PRE_DESTRUCTOR(ctx) ((ctx)->pre_destructor)           // 预分配任务信息析构函数
#define TASKFLOW_CTX_PRE_TASKINFO(ctx) ((ctx)->pre_taskinfo)               // 预分配任务信息
#define TASKFLOW_CTX_PRIORITY_GROUP(ctx) ((ctx)->priority_group)           // 优先级组
#define TASKFLOW_CTX_READY_ITEM_LIST(ctx) ((ctx)->ready_item_list)         // 就绪项目列表
#define TASKFLOW_CTX_RUN_ITEM_LIST(ctx) ((ctx)->run_item_list)             // 运行项目列表
#define TASKFLOW_CTX_RUN_NODE_LIST(ctx) ((ctx)->run_node_list)             // 运行节点列表
#define TASKFLOW_CTX_IDLE_ITEM_LIST(ctx) ((ctx)->idle_item_list)           // 空闲项目列表
#define TASKFLOW_CTX_IDLE_NODE_LIST(ctx) ((ctx)->idle_node_list)           // 空闲节点列表

#define TaskFlow_InsertList(dst, dst_it, src, src_it) (dst).splice((dst_it), (src), (src_it))
#define TaskFlow_MoveListBegin(dst, src) (dst).splice((dst).begin(), (src), (src).begin())
#define TaskFlow_MoveListEnd(dst, src) (dst).splice((dst).end(), (src), --(src).end())

#define TaskFlow_Lock(ctx) pthread_mutex_lock(TASKFLOW_CTX_MUT_LOCK(ctx))
#define TaskFlow_UnLock(ctx) pthread_mutex_unlock(TASKFLOW_CTX_MUT_LOCK(ctx))
#define TaskFlow_SetSem(sem) sem_post(sem)
#define TaskFlow_GetSemForever(sem) sem_wait(sem)

/*static int TaskFlow_GetSem(sem_t *sem_id, uint32_t timeout_ms) {
    struct timespec ts = {0};
    if (clock_gettime(CLOCK_REALTIME, &ts)) {
        DIHLogInfo << "获取时间出错";
        return -1;
    }
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000;
    return sem_timedwait(sem_id, &ts);
}*/


#define MAX_WAIT_INTERVAL_USECOND 5000
// 获取自系统启动的调单递增的时间
inline uint64_t GetTimeConvSeconds(timespec* curTime, uint32_t factor)
{
    // CLOCK_MONOTONIC：从系统启动这一刻起开始计时,不受系统时间被用户改变的影响
    clock_gettime(CLOCK_MONOTONIC, curTime);
    return static_cast<uint64_t>(curTime->tv_sec) * factor;
}

// 获取自系统启动的调单递增的时间 -- 转换单位为微秒
uint64_t GetMonnotonicTime()
{
    timespec curTime;
    uint64_t result = GetTimeConvSeconds(&curTime, 1000000);
    result += static_cast<uint32_t>(curTime.tv_nsec) / 1000;
    return result;
}




/*static int TaskFlow_GetSem(sem_t *sem_id, uint32_t timeout_ms)
{
  const size_t timeoutUs = timeout_ms * 1000; // 延时时间由毫米转换为微秒
  //ALGLogInfo<<"total wait time: "<<timeoutUs;
  size_t timeWait = 1; // 睡眠时间，默认为1微秒
  size_t delayUs = 0; // 剩余需要延时睡眠时间
  const uint64_t startUs = GetMonnotonicTime(); // 循环前的开始时间，单位微秒
  uint64_t elapsedUs = 0; // 过期时间，单位微秒

  int ret = 0;

  do
  {
    // 如果信号量大于0，则减少信号量并立马返回true
    if( sem_trywait( sem_id ) == 0 )
    {
      return 0;
    }

    // 系统信号则立马返回false
    if( errno != EAGAIN )
    {
      ALGLogInfo<<"signal error";
      return -1;
    }


    // 睡眠时间取最小的值
    timeWait = MAX_WAIT_INTERVAL_USECOND;

    // 进行睡眠 单位是微秒
    ret = usleep( timeWait );
    if( ret != 0 )
    {
      ALGLogInfo<<"failed to usleep";
      return -2;
    }


    // 计算开始时间到现在的运行时间 单位是微秒
    elapsedUs = GetMonnotonicTime() - startUs;
    //ALGLogInfo<<"waited time: "<<elapsedUs;
  } while( elapsedUs <= timeoutUs ); // 如果当前循环的时间超过预设延时时间则退出循环
  //ALGLogInfo<<"final wait time: "<<elapsedUs;
  // 超时退出，则返回false
  return -3;
}*/


static int TaskFlow_GetSem(sem_t* sem_id, uint32_t timeout_ms)
{
    int             s;
    struct timespec ts = {0};
    if (clock_gettime(CLOCK_REALTIME, &ts)) {
        ALGLogError << "获取时间出错";
        return -1;
    }
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000;
    // 如果ts.tv_nsec超过1000000000, sem_timedwait 将会返回 22
    // 此处增加值的标准化
    long int sec_quotient   = ts.tv_nsec / 1000000000;
    long int nsec_remainder = ts.tv_nsec % 1000000000;
    ts.tv_sec += sec_quotient;
    ts.tv_nsec = nsec_remainder;

    while ((s = sem_timedwait(sem_id, &ts)) == -1 && errno == EINTR)
        continue;   // Restart if interrupted by handler Check what happened
    if (s == -1 && timeout_ms != 0) {
        ALGLogInfo << "sem_timedwait error code " << errno;
        ALGLogInfo << "nsec " << ts.tv_nsec;
    }

    return s;
}



static void TaskFlow_MoveIdleItem(TaskFlowCtx_t* ctx, std::list<TaskFlowItem_t>& list)
{
    if (ctx) {
        TaskFlow_MoveListBegin(TASKFLOW_CTX_IDLE_ITEM_LIST(ctx), list);
        if (TASKFLOW_CTX_WAIT_IDLE_ITEM_CNT(ctx)) {
            TASKFLOW_CTX_WAIT_IDLE_ITEM_CNT(ctx)--;
            TaskFlow_SetSem(TASKFLOW_CTX_SEM_IDLE_ITEM(ctx));
        }
    }
}

static void TaskFlow_MoveIdleNode(TaskFlowCtx_t* ctx, std::list<TaskFlowNode_t>& list)
{
    if (ctx) {
        TaskFlow_MoveListBegin(TASKFLOW_CTX_IDLE_NODE_LIST(ctx), list);
        if (TASKFLOW_CTX_WAIT_IDLE_NODE_CNT(ctx)) {
            TASKFLOW_CTX_WAIT_IDLE_NODE_CNT(ctx)--;
            TaskFlow_SetSem(TASKFLOW_CTX_SEM_IDLE_NODE(ctx));
        }
    }
}

[[noreturn]] static void* TaskFlowThreadHandler(void* arg)
{
    TaskFlowCtx_t* ctx = (TaskFlowCtx_t*)arg;
    ALGLogInfo << "generate another task thread";
    while (1) {
        //		TaskFlow_GetSem(TASKFLOW_CTX_SEM_START(ctx), 0xFFFE);
        TaskFlow_GetSemForever(TASKFLOW_CTX_SEM_START(ctx));
        //		TaskFlow_w()
        while (!TaskFlow_GetSem(TASKFLOW_CTX_SEM_START(ctx), 0))
            ;
    __PROCESS_START:
        //		DIHLogInfo<<"图像处理 开始1";
        int      ret           = 0;
        uint32_t all_empty_cnt = 0;
        TaskFlow_Lock(ctx);
        while (!TASKFLOW_CTX_RUN_ITEM_LIST(ctx).empty()) {
            TaskFlow_MoveIdleItem(ctx, TASKFLOW_CTX_RUN_ITEM_LIST(ctx));
        }
        while (!TASKFLOW_CTX_RUN_NODE_LIST(ctx).empty()) {
            TaskFlow_MoveIdleNode(ctx, TASKFLOW_CTX_RUN_NODE_LIST(ctx));
        }   // 如果有正在运行的,移动到list front
        TaskFlow_UnLock(ctx);
        for (uint32_t idx = 0; idx < TASKFLOW_MAX_GROUP_NUM; idx++) {
            TaskFlowGroup_t* group = &TASKFLOW_CTX_PRIORITY_GROUP(ctx)[idx];
            if (!TASKFLOW_GROUP_ITEM_LIST(group).empty()) {
                TaskFlow_Lock(ctx);
                TaskFlow_MoveListBegin(TASKFLOW_CTX_RUN_ITEM_LIST(ctx), TASKFLOW_GROUP_ITEM_LIST(group));   // 取当前group下的一个item到正在运行的list
                TaskFlowItem_t* item = &TASKFLOW_CTX_RUN_ITEM_LIST(ctx).front();
                if (!TASKFLOW_ITEM_NODE_LIST(item).empty()) {                                                 // 如果item下节点非空
                    TaskFlow_MoveListBegin(TASKFLOW_CTX_RUN_NODE_LIST(ctx), TASKFLOW_ITEM_NODE_LIST(item));   // 取一个节点放到运行节点队列
                    TaskFlowNode_t* node = &TASKFLOW_CTX_RUN_NODE_LIST(ctx).front();
                    TaskFlow_UnLock(ctx);
                    if (TASKFLOW_NODE_CALLBACK(node)) {   // 运行节点
                        ret = (*TASKFLOW_NODE_CALLBACK(node))((TaskFlowCtxID_t)ctx,
                                                              TASKFLOW_ITEM_ID(item),
                                                              TASKFLOW_NODE_ID(node),
                                                              TASKFLOW_ITEM_TASKINFO(item),
                                                              TASKFLOW_NODE_USERDATA(node));
                    }
                    TaskFlow_Lock(ctx);
                    TaskFlow_MoveIdleNode(ctx, TASKFLOW_CTX_RUN_NODE_LIST(ctx));   // 用完后放到空闲节点
                }
                if (ret) {   // 如果失败,回收全部节点
                    while (!TASKFLOW_ITEM_NODE_LIST(item).empty()) {
                        TaskFlow_MoveIdleNode(ctx, TASKFLOW_ITEM_NODE_LIST(item));
                    }
                }
                if (!TASKFLOW_ITEM_NODE_LIST(item).empty()) {                        // 如果item下节点非空
                    TaskFlowNode_t* node = &TASKFLOW_ITEM_NODE_LIST(item).front();   // 第一个节点
                    if (TASKFLOW_NODE_GROUP_IDX(node) >= TASKFLOW_MAX_GROUP_NUM) {
                        TASKFLOW_NODE_GROUP_IDX(node) = 0;
                    }
                    std::list<TaskFlowItem_t>* group_item_list =
                        &TASKFLOW_GROUP_ITEM_LIST(&TASKFLOW_CTX_PRIORITY_GROUP(ctx)[TASKFLOW_NODE_GROUP_IDX(node)]);
                    for (std::list<TaskFlowItem_t>::iterator it = (*group_item_list).begin();; it++) {
                        TaskFlowItem_t* list_item = &(*it);   // 按序号插入
                        if (TASKFLOW_ITEM_START_IDX(item) < TASKFLOW_ITEM_START_IDX(list_item) || it == (*group_item_list).end()) {
                            TaskFlow_InsertList((*group_item_list), it, TASKFLOW_CTX_RUN_ITEM_LIST(ctx), TASKFLOW_CTX_RUN_ITEM_LIST(ctx).begin());
                            break;
                        }
                    }
                }
                else {
                    if (TASKFLOW_ITEM_DESTRUCTOR(item)) {
                        (*TASKFLOW_ITEM_DESTRUCTOR(item))((TaskFlowCtxID_t)ctx, TASKFLOW_ITEM_ID(item), TASKFLOW_ITEM_TASKINFO(item));
                    }
                    else {
                    }
                    TaskFlow_MoveIdleItem(ctx, TASKFLOW_CTX_RUN_ITEM_LIST(ctx));   // item下节点已运行完毕,移动到空闲队列
                }
                if (TASKFLOW_GROUP_ITEM_LIST(group).empty()) {        // 当前group下item list为空
                    while (TASKFLOW_GROUP_WAIT_EMPTY_CNT(group)) {    // group 下等待空闲的数量
                        if (TASKFLOW_GROUP_WAIT_EMPTY_CNT(group)) {   // 有等待的
                            TASKFLOW_GROUP_WAIT_EMPTY_CNT(group)--;   // 等待的减少一个
                        }
                        TaskFlow_SetSem(TASKFLOW_GROUP_SEM_EMPTY(group));   // 释放一个
                    }
                }
                TaskFlow_UnLock(ctx);
                goto __PROCESS_START;
            }
            if (TASKFLOW_GROUP_ITEM_LIST(group).empty()) {
                while (TASKFLOW_GROUP_WAIT_EMPTY_CNT(group)) {
                    if (TASKFLOW_GROUP_WAIT_EMPTY_CNT(group)) {
                        TASKFLOW_GROUP_WAIT_EMPTY_CNT(group)--;
                    }
                    TaskFlow_SetSem(TASKFLOW_GROUP_SEM_EMPTY(group));
                }
                //				DIHLogInfo<<"图像处理完成";
                //				if (!dyno::aig::task::mutex::AigTaskMutex::setClarityTaskStatus(false)) {
                //					DIHLogInfo << "释放图片清晰度锁失败";
                //				}
            }
            TASKFLOW_CTX_EMPTY_GROUP_CNT(ctx)++;
        }
        if (TASKFLOW_CTX_EMPTY_GROUP_CNT(ctx) < TASKFLOW_MAX_GROUP_NUM) {
            goto __PROCESS_START;
        }
        TaskFlow_Lock(ctx);
        while (TASKFLOW_CTX_WAIT_EMPTY_CNT(ctx)) {   // 有等待全空的
            if (TASKFLOW_CTX_WAIT_EMPTY_CNT(ctx)) {
                TASKFLOW_CTX_WAIT_EMPTY_CNT(ctx)--;
            }
            TaskFlow_SetSem(TASKFLOW_CTX_SEM_ALL_EMPTY(ctx));
        }
        TaskFlow_UnLock(ctx);
    }
}


static int TaskFlow_SetStructor(
    TaskFlowCtx_t* ctx, TaskFlowItem_t* item, TaskInfoConstructor_f constructor, TaskInfoDestructor_f destructor, void* taskinfo)
{
    if (ctx == NULL || item == NULL) {
        return -1;
    }
    TaskInfoConstructor_f temp_constructor = NULL;
    TaskInfoDestructor_f  temp_destructor  = NULL;
    void*                 temp_taskinfo    = NULL;
    if (TASKFLOW_CTX_PRE_CONSTRUCTOR(ctx)) {
        temp_constructor = TASKFLOW_CTX_PRE_CONSTRUCTOR(ctx);
    }
    else if (constructor) {
        temp_constructor = constructor;
    }
    if (TASKFLOW_CTX_PRE_DESTRUCTOR(ctx)) {
        temp_destructor = TASKFLOW_CTX_PRE_DESTRUCTOR(ctx);
    }
    else if (destructor) {
        temp_destructor = destructor;
    }
    if (TASKFLOW_CTX_PRE_TASKINFO(ctx)) {
        temp_taskinfo = TASKFLOW_CTX_PRE_TASKINFO(ctx);
    }
    else {
        temp_taskinfo = taskinfo;
    }
    TASKFLOW_ITEM_CONSTRUCTOR(item) = temp_constructor;
    TASKFLOW_ITEM_DESTRUCTOR(item)  = temp_destructor;
    if (temp_constructor) {
        TASKFLOW_ITEM_TASKINFO(item) = (*TASKFLOW_ITEM_CONSTRUCTOR(item))((TaskFlowCtxID_t)ctx, TASKFLOW_ITEM_ID(item), temp_taskinfo);
    }
    else {
        TASKFLOW_ITEM_TASKINFO(item) = temp_taskinfo;
    }
    return 0;
}

/**
 * 任务流队列初始化
 * @param  pre_item_num		预分配任务项目数量
 * @param  pre_node_num		预分配任务节点数量
 * @param  pre_constructor	预分配任务信息构造函数
 * @param  pre_destructor	预分配任务信息析构函数
 * @param  taskinfo			预分配任务信息
 * @param  cfg_flag			配置标志 @ref TaskFlowFlag_t
 * @return 任务流队列上下文ID @ref TaskFlowCtxID_t
 */
TaskFlowCtxID_t TaskFlow_Init(uint32_t              pre_item_num,
                              uint32_t              pre_node_num,
                              TaskInfoConstructor_f pre_constructor,
                              TaskInfoDestructor_f  pre_destructor,
                              void*                 taskinfo,
                              uint32_t              cfg_flag)
{

    if (!pre_item_num && (cfg_flag & TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC)) {
        return NULL;
    }
    if (!pre_node_num && (cfg_flag & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC)) {
        return NULL;
    }
    TaskFlowCtx_t* ctx                = new TaskFlowCtx_t;
    TASKFLOW_CTX_PRE_CONSTRUCTOR(ctx) = pre_constructor;
    TASKFLOW_CTX_PRE_DESTRUCTOR(ctx)  = pre_destructor;
    TASKFLOW_CTX_PRE_TASKINFO(ctx)    = taskinfo;
    TASKFLOW_CTX_CFG_FLAG(ctx)        = cfg_flag;
    for (uint32_t idx = 0; idx < pre_item_num; idx++) {
        TaskFlowItem_t* item = new TaskFlowItem_t;
        if (TaskFlow_SetStructor(ctx, item, NULL, NULL, taskinfo)) {
            return NULL;
        }
        TASKFLOW_CTX_IDLE_ITEM_LIST(ctx).push_front(*item);
    }
    for (uint32_t idx = 0; idx < pre_node_num; idx++) {
        TaskFlowNode_t* node = new TaskFlowNode_t;
        TASKFLOW_CTX_IDLE_NODE_LIST(ctx).push_front(*node);
    }
    TASKFLOW_CTX_WAIT_EMPTY_CNT(ctx)     = 0;
    TASKFLOW_CTX_WAIT_IDLE_ITEM_CNT(ctx) = 0;
    TASKFLOW_CTX_WAIT_IDLE_NODE_CNT(ctx) = 0;
    for (uint32_t idx = 0; idx < TASKFLOW_MAX_GROUP_NUM; idx++) {
        TaskFlowGroup_t* group               = &TASKFLOW_CTX_PRIORITY_GROUP(ctx)[idx];
        TASKFLOW_GROUP_WAIT_EMPTY_CNT(group) = 0;
        sem_init(TASKFLOW_GROUP_SEM_EMPTY(group), 0, 0);
    }
    sem_init(TASKFLOW_CTX_SEM_ALL_EMPTY(ctx), 0, 0);
    sem_init(TASKFLOW_CTX_SEM_IDLE_ITEM(ctx), 0, 0);
    sem_init(TASKFLOW_CTX_SEM_IDLE_NODE(ctx), 0, 0);
    sem_init(TASKFLOW_CTX_SEM_START(ctx), 0, 0);
    pthread_mutex_init(TASKFLOW_CTX_MUT_LOCK(ctx), NULL);
    if (pthread_create(&TASKFLOW_CTX_THR_MAIN(ctx), NULL, TaskFlowThreadHandler, ctx)) {
        return NULL;
    }
    return (TaskFlowCtxID_t)ctx;
}
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
/**
 * 新建任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  constructor	任务信息构造函数
 * @param  destructor	任务信息析构函数
 * @param  taskinfo		任务信息 This pointer will be used directly when constructor=NULL and destructor=NULL
 * @param  timeout		超时时间 Only valid when (cfg_flag & TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC) is ture
 * @return 任务项目ID 0 fail other success @ref TaskFlowItemID_t
 */
TaskFlowItemID_t TaskFlow_New(
    TaskFlowCtxID_t ctx_id, TaskInfoConstructor_f constructor, TaskInfoDestructor_f destructor, void* taskinfo, uint32_t timeout)
{
    TaskFlowCtx_t* ctx = (TaskFlowCtx_t*)ctx_id;
    if (ctx == NULL) {
        return 0;
    }
    TaskFlow_Lock(ctx);
    if (TASKFLOW_CTX_IDLE_ITEM_LIST(ctx).empty()) {
        if (TASKFLOW_CTX_CFG_FLAG(ctx) & TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC) {
        __WAIT_TDLE_TIME:
            auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            TASKFLOW_CTX_WAIT_IDLE_ITEM_CNT(ctx)++;
            TaskFlow_UnLock(ctx);
            if (TaskFlow_GetSem(TASKFLOW_CTX_SEM_IDLE_ITEM(ctx), timeout)) {
                return 0;
            }
            TaskFlow_Lock(ctx);
            TaskFlow_MoveListEnd(TASKFLOW_CTX_READY_ITEM_LIST(ctx), TASKFLOW_CTX_IDLE_ITEM_LIST(ctx));
            auto func_end       = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            auto func_cost_time = func_end - start;
            ALGLogInfo << "item need more... waiting time " << func_cost_time;
        }
        else {
            printf("item need more...\r\n");
            TaskFlowItem_t* item = new TaskFlowItem_t;
            if (TaskFlow_SetStructor(ctx, item, constructor, destructor, taskinfo)) {
                delete item;
                goto __WAIT_TDLE_TIME;
            }
            TASKFLOW_CTX_READY_ITEM_LIST(ctx).push_back(*item);
        }
    }
    else {
        TaskFlow_MoveListEnd(TASKFLOW_CTX_READY_ITEM_LIST(ctx), TASKFLOW_CTX_IDLE_ITEM_LIST(ctx));
    }
    TaskFlowItem_t* item = &TASKFLOW_CTX_READY_ITEM_LIST(ctx).back();
    if (!TASKFLOW_CTX_ITEM_ID_IDX(ctx)) {
        TASKFLOW_CTX_ITEM_ID_IDX(ctx)++;
    }
    TASKFLOW_ITEM_ID(item) = TASKFLOW_CTX_ITEM_ID_IDX(ctx)++;
    while (!TASKFLOW_ITEM_NODE_LIST(item).empty()) {
        TaskFlow_MoveListBegin(TASKFLOW_CTX_IDLE_NODE_LIST(ctx), TASKFLOW_ITEM_NODE_LIST(item));
    }
    if (TASKFLOW_ITEM_TASKINFO(item) == NULL) {
        ALGLogInfo << "null taskinfo";
        if (TaskFlow_SetStructor(ctx, item, constructor, destructor, taskinfo)) {
            TaskFlow_UnLock(ctx);
            return 0;
        }
        if (TASKFLOW_ITEM_TASKINFO(item) == NULL) {
            ALGLogInfo << "still null taskinfo";
        }
    }
    TaskFlow_UnLock(ctx);
    return TASKFLOW_ITEM_ID(item);
}

static TaskFlowItem_t* TaskFlow_FindListItem(std::list<TaskFlowItem_t>& list, TaskFlowItemID_t item_id)
{
    for (std::list<TaskFlowItem_t>::iterator it = list.begin(); it != list.end(); it++) {
        TaskFlowItem_t* item = &(*it);
        if (TASKFLOW_ITEM_ID(item) == item_id) {
            return item;
        }
    }
    return NULL;
}

static TaskFlowItem_t* TaskFlow_FindItem(TaskFlowCtx_t* ctx, TaskFlowItemID_t item_id)
{
    if (ctx == NULL || item_id == 0) {
        return NULL;
    }
    TaskFlowItem_t* item = TaskFlow_FindListItem(TASKFLOW_CTX_READY_ITEM_LIST(ctx), item_id);
    if (item == NULL) {
        item = TaskFlow_FindListItem(TASKFLOW_CTX_RUN_ITEM_LIST(ctx), item_id);
        if (item == NULL) {
            for (uint32_t idx = 0; idx < TASKFLOW_MAX_GROUP_NUM; idx++) {
                std::list<TaskFlowItem_t>* group_item_list = &TASKFLOW_GROUP_ITEM_LIST(&TASKFLOW_CTX_PRIORITY_GROUP(ctx)[idx]);
                item                                       = TaskFlow_FindListItem((*group_item_list), item_id);
                if (item) {
                    break;
                }
            }
        }
    }
    return item;
}

static TaskFlowNodeID_t TaskFlow_AddNode(
    TaskFlowCtx_t* ctx, TaskFlowItem_t* item, uint32_t group_idx, TaskFlowCallback_f callback, void* userdata, uint32_t timeout)
{
    if (item == NULL) {
        return 0;
    }
    if (TASKFLOW_CTX_IDLE_NODE_LIST(ctx).empty()) {
        if (TASKFLOW_CTX_CFG_FLAG(ctx) & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC) {
            TASKFLOW_CTX_WAIT_IDLE_NODE_CNT(ctx)++;
            TaskFlow_UnLock(ctx);
            if (TaskFlow_GetSem(TASKFLOW_CTX_SEM_IDLE_NODE(ctx), timeout)) {
                return 0;
            }
            TaskFlow_Lock(ctx);
            TaskFlow_MoveListEnd(TASKFLOW_ITEM_NODE_LIST(item), TASKFLOW_CTX_IDLE_NODE_LIST(ctx));
        }
        else {
            TaskFlowNode_t* node = new TaskFlowNode_t;
            TASKFLOW_ITEM_NODE_LIST(item).push_back(*node);
        }
    }
    else {
        TaskFlow_MoveListEnd(TASKFLOW_ITEM_NODE_LIST(item), TASKFLOW_CTX_IDLE_NODE_LIST(ctx));
    }
    TaskFlowNode_t* node = &TASKFLOW_ITEM_NODE_LIST(item).back();
    if (!TASKFLOW_CTX_NODE_ID_IDX(ctx)) {
        TASKFLOW_CTX_NODE_ID_IDX(ctx)++;
    }
    TASKFLOW_NODE_ID(node)        = (TaskFlowNodeID_t)TASKFLOW_CTX_NODE_ID_IDX(ctx)++;
    TASKFLOW_NODE_GROUP_IDX(node) = group_idx;
    TASKFLOW_NODE_CALLBACK(node)  = callback;
    TASKFLOW_NODE_USERDATA(node)  = userdata;
    return TASKFLOW_NODE_ID(node);
}

/**
 * 添加任务节点到任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @param  group_idx		优先级组序号
 * @param  callback		用户回调
 * @param  userdata		用户数据
 * @param  timeout		超时时间 Only valid when (cfg_flag & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC) is ture
 * @return 任务节点ID 0 fail other success @ref TaskFlowNodeID_t
 */
TaskFlowNodeID_t TaskFlow_Node(
    TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, uint32_t group_idx, TaskFlowCallback_f callback, void* userdata, uint32_t timeout)
{
    TaskFlowNodeID_t node_id = 0;
    TaskFlowCtx_t*   ctx     = (TaskFlowCtx_t*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return 0;
    }
    TaskFlow_Lock(ctx);
    TaskFlowItem_t* item = TaskFlow_FindItem(ctx, item_id);
    if (item) {
        node_id = TaskFlow_AddNode(ctx, item, group_idx, callback, userdata, timeout);
        TaskFlow_UnLock(ctx);
        return node_id;
    }
    TaskFlow_UnLock(ctx);
    return 0;
}

/**
 * 开始任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return 0 success other fail
 */
int TaskFlow_Start(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id)
{
    TaskFlowCtx_t* ctx = (TaskFlowCtx_t*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return -1;
    }
    TaskFlow_Lock(ctx);
    for (std::list<TaskFlowItem_t>::iterator it = TASKFLOW_CTX_READY_ITEM_LIST(ctx).begin(); it != TASKFLOW_CTX_READY_ITEM_LIST(ctx).end(); it++) {
        TaskFlowItem_t* item = &(*it);
        if (TASKFLOW_ITEM_ID(item) == item_id) {
            if (!TASKFLOW_CTX_START_IDX(ctx)) {
                TASKFLOW_CTX_START_IDX(ctx)++;
            }
            TASKFLOW_ITEM_START_IDX(item) = TASKFLOW_CTX_START_IDX(ctx)++;
            uint32_t group_idx            = 0;
            if (!TASKFLOW_ITEM_NODE_LIST(item).empty()) {
                TaskFlowNode_t* node = &TASKFLOW_ITEM_NODE_LIST(item).front();
                if (TASKFLOW_NODE_GROUP_IDX(node) < TASKFLOW_MAX_GROUP_NUM) {
                    group_idx = TASKFLOW_NODE_GROUP_IDX(node);
                }
            }
            std::list<TaskFlowItem_t>* group_item_list = &TASKFLOW_GROUP_ITEM_LIST(&TASKFLOW_CTX_PRIORITY_GROUP(ctx)[group_idx]);
            TaskFlow_InsertList((*group_item_list), (*group_item_list).end(), TASKFLOW_CTX_READY_ITEM_LIST(ctx), it);
            TaskFlow_UnLock(ctx);
            TaskFlow_SetSem(TASKFLOW_CTX_SEM_START(ctx));
            return 0;
        }
    }
    TaskFlow_UnLock(ctx);
    return -2;
}

/**
 * 获取任务信息
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return 任务信息
 */
void* TaskFlow_GetTaskInfo(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id)
{
    TaskFlowCtx_t* ctx = (TaskFlowCtx_t*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return NULL;
    }
    TaskFlowItem_t* item = TaskFlow_FindItem(ctx, item_id);
    if (item) {
        return TASKFLOW_ITEM_TASKINFO(item);
    }
    return NULL;
}

/**
 * 删除任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return
 */
int TaskFlow_Del(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id)
{
    TaskFlowCtx_t* ctx = (TaskFlowCtx_t*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return -1;
    }
    TaskFlow_Lock(ctx);
    TaskFlowItem_t* item = TaskFlow_FindItem(ctx, item_id);
    if (item) {
        while (!TASKFLOW_ITEM_NODE_LIST(item).empty()) {
            TaskFlow_MoveIdleNode(ctx, TASKFLOW_ITEM_NODE_LIST(item));
        }
        return 0;
    }
    return -2;
}

/**
 * 等待分组任务流完成
 * @param  ctx_id		任务流队列上下文ID
 * @param  group_idx		优先级组索引
 * @param  timeout		超时时间
 * @return
 */
int TaskFlow_WaitGroup(TaskFlowCtxID_t ctx_id, uint32_t group_idx, uint32_t timeout)
{
    TaskFlowCtx_t* ctx = (TaskFlowCtx_t*)ctx_id;
    if (ctx == NULL || group_idx >= TASKFLOW_MAX_GROUP_NUM) {
        ALGLogInfo << "Clarity_WaitCplt  超时 最终错误 TaskFlow_WaitGroup";
        return -1;
    }
    TaskFlowGroup_t* group = &TASKFLOW_CTX_PRIORITY_GROUP(ctx)[group_idx];
    TaskFlow_Lock(ctx);
    TASKFLOW_GROUP_WAIT_EMPTY_CNT(group)++;
    TaskFlow_UnLock(ctx);
    TaskFlow_SetSem(TASKFLOW_CTX_SEM_START(ctx));
    //	for (int i = 0; i < timeout * 10; i++) {
    //		usleep(100000);
    //                ALGLogInfo<<"wait time "<<i;
    //		bool clarityTaskStatus;
    //		dyno::aig::task::mutex::AigTaskMutex::getClarityTaskStatus(clarityTaskStatus);
    //		if (!clarityTaskStatus) {
    //			return 0;
    //		}
    //	}
    //	DIHLogInfo << "等待图片处理完成，超时：5秒";
    //	return -1;
    return TaskFlow_GetSem(TASKFLOW_GROUP_SEM_EMPTY(group), timeout);
}

/**
 * 等待所有任务流完成
 * @param  ctx_id		任务流队列上下文ID
 * @param  timeout		超时时间
 * @return
 */
int TaskFlow_WaitAll(TaskFlowCtxID_t ctx_id, uint32_t timeout)
{
    TaskFlowCtx_t* ctx = (TaskFlowCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    TaskFlow_Lock(ctx);
    TASKFLOW_CTX_WAIT_EMPTY_CNT(ctx)++;
    TaskFlow_UnLock(ctx);
    TaskFlow_SetSem(TASKFLOW_CTX_SEM_START(ctx));
    return TaskFlow_GetSem(TASKFLOW_CTX_SEM_ALL_EMPTY(ctx), timeout);
}

#include <vector>

#define TIMEOUT_FOREVER 0xFFFFFFFF

typedef struct TaskInfoTest
{
    const char*      info;
    std::vector<int> val;
} TaskInfoTest_t;

void* info_init(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, void* taskinfo)
{
    TaskInfoTest_t* data = new TaskInfoTest_t;
    data->info           = "TASKINFO:";
    for (uint32_t idx = 0; idx < item_id; idx++) {
        data->val.push_back(idx);
    }
    return (void*)data;
}

void info_deinit(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, void* taskinfo) {}

int CB2(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, TaskFlowNodeID_t node_id, void* taskinfo, void* userdata)
{
    TaskInfoTest_t* data      = (TaskInfoTest_t*)taskinfo;
    char            buf[2048] = "";
    int             len       = 0;
    for (auto it : data->val) {
        len += snprintf(buf + len, 2048 - len, "%d ", it);
    }
    printf("CB2>> %s (%s)\r\n", data->info, buf);
    sleep(1);
    return 0;
}

int CB1(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, TaskFlowNodeID_t node_id, void* taskinfo, void* userdata)
{
    TaskFlow_Node(ctx_id, item_id, 0, CB2, NULL, TIMEOUT_FOREVER);
    printf("CB1>> item(%d) node(%d) task(%s) userdata(%s)\r\n", item_id, node_id, NULL, userdata);
    return 0;
}

void TaskFlow_AutoNewTask(TaskFlowCtxID_t ctx_id, uint32_t task_id, uint32_t node_num)
{
    // char *taskinfo = (char*)malloc(64);
    char taskinfo[64] = "";
    snprintf(taskinfo, 64, "Task %d", task_id);
    uint32_t item_id = TaskFlow_New(ctx_id, info_init, info_deinit, NULL, TIMEOUT_FOREVER);
    for (uint32_t idx = 0; idx < 1; idx++) {
        char* userdata = (char*)malloc(64);
        snprintf(userdata, 64, "Step %d", idx);
        TaskFlow_Node(ctx_id, item_id, 0, CB1, (void*)userdata, TIMEOUT_FOREVER);
    }
    int ret = TaskFlow_Start(ctx_id, item_id);
    printf("TaskFlow_AutoNewTask(%d) = %d <\r\n", task_id, ret);
    return;
}

void TaskFlow_TestExample(void)
{
    int ret = 0;
    printf("TaskFlow Test Example >>>\r\n");
    void* ctxid = TaskFlow_Init(5, 10, NULL, NULL, NULL, TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC | TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC);
    printf("TaskFlow_Init ret=0x%x\r\n", ctxid);

    for (uint32_t idx = 0; idx < 50; idx++) {
        TaskFlow_AutoNewTask(ctxid, idx, 5);
    }
}
