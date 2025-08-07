#include "task.h"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <list>
#include <malloc.h>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <thread>

class Semaphore
{
public:
    Semaphore() {};
    ~Semaphore() {};
    void signal()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.notify_one();
    }
    // 超时唤醒 返回std::cv_status::timeout    0，
    // 通知唤醒 返回std::cv_status::no_timeout 1
    int wait_for(int64_t timeout_ms)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms)) == std::cv_status::timeout) {
            return -1;
        }
        return 0;
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
};


/* 任务节点 */
struct TaskFlowNode
{
    int64_t          node_id;     // 任务节点ID
    int64_t          group_idx;   // 优先级组索引
    TaskFlowCallback callback;    // 用户回调
    void*            userdata;    // 用户数据
};

/* 任务项目 */
struct TaskFlowItem
{
    int64_t                 item_id;       // 任务项目ID
    int64_t                 start_idx;     // 启动序号
    void*                   taskinfo;      // 任务信息
    TaskInfoConstructor     constructor;   // 任务信息构建函数
    TaskInfoDestructor      destructor;    // 任务信息析构函数
    std::list<TaskFlowNode> node_list;     // 任务节点列表
};

/* 任务优先级组 */
struct TaskFlowGroup
{
    int64_t                 wait_empty_cnt;   // 等待优先级组置空计数
    Semaphore               sem_empty;        // 优先级组置空信号
    std::list<TaskFlowItem> item_list;        // 任务项目列表
};


struct TaskFlowCtx
{
    int64_t                      cfg_flag;             // 配置标志
    int64_t                      empty_group_cnt;      // 优先级组置空计数
    int64_t                      wait_all_empty_cnt;   // 等待全部优先级组置空计数
    int64_t                      wait_idle_item_cnt;   // 等待空闲任务项目计数
    int64_t                      wait_idle_node_cnt;   // 等待空闲任务节点计数
    Semaphore                    sem_all_empty;        // 全部优先级组置空信号
    Semaphore                    sem_idle_item;        // 空闲任务项目信号
    Semaphore                    sem_idle_node;        // 空闲任务节点信号
    Semaphore                    sem_start;            // 开始任务流信号
    std::mutex                   mut_lock;             // 队列锁定互斥
    std::shared_ptr<std::thread> thr_main;             // 主线程
    int64_t                      item_id_idx;          // 任务项目ID索引
    int64_t                      node_id_idx;          // 任务节点ID索引
    int64_t                      start_idx;            // 任务启动索引
    TaskInfoConstructor          pre_constructor;      // 预分配任务信息构造函数
    TaskInfoDestructor           pre_destructor;       // 预分配任务信息析构函数
    void*                        pre_taskinfo;         // 预分配任务信息
    TaskFlowGroup                priority_group[2];    // 优先级组
    std::list<TaskFlowItem>      ready_item_list;      // 就绪项目列表
    std::list<TaskFlowItem>      run_item_list;        // 运行项目列表
    std::list<TaskFlowNode>      run_node_list;        // 运行节点列表
    std::list<TaskFlowItem>      idle_item_list;       // 空闲项目列表
    std::list<TaskFlowNode>      idle_node_list;       // 空闲节点列表
};


/**
 将list 里面开头的一个元素 转到空闲项目
*/
static void TaskFlow_MoveIdleItem(TaskFlowCtx* ctx, std::list<TaskFlowItem>& list)
{
    if (ctx) {
        ctx->idle_item_list.splice(ctx->idle_item_list.begin(), list, list.begin());
        if (ctx->wait_idle_item_cnt) {
            ctx->wait_idle_item_cnt--;
            ctx->sem_idle_item.signal();
        }
        std::cout << " \n 正在移动到空闲的 item \n";
    }
}

/**
将list 的node的 开头的一个元素 移动到 空闲node
*/
static void TaskFlow_MoveIdleNode(TaskFlowCtx* ctx, std::list<TaskFlowNode>& list)
{
    if (ctx) {
        ctx->idle_node_list.splice(ctx->idle_node_list.begin(), list, list.begin());
        if (ctx->wait_idle_node_cnt) {
            ctx->wait_idle_node_cnt--;
            ctx->sem_idle_node.signal();
        }
    }
}

[[noreturn]] static void* TaskFlowThreadHandler(void* arg)
{
    TaskFlowCtx* ctx = (TaskFlowCtx*)arg;

    while (1) {

        ctx->sem_start.wait_for(0);
        while (!ctx->sem_start.wait_for(0))
            ;

    __PROCESS_START:

        int     ret           = 0;
        int64_t all_empty_cnt = 0;
        ctx->mut_lock.lock();
        while (!ctx->run_item_list.empty()) {
            // std::cout << "ctx->run_item_list  " << __FUNCTION__ << " " << __LINE__ << " " << ctx->run_item_list.size() << "\n";
            // std::cout << "将正在运行的 item 移动到空闲的 item " << __FUNCTION__ << " " << __LINE__ << " "<< "\n";
            TaskFlow_MoveIdleItem(ctx, ctx->run_item_list);
        }
        while (!ctx->run_node_list.empty()) {
            TaskFlow_MoveIdleNode(ctx, ctx->run_node_list);
        }
        ctx->mut_lock.unlock();

        for (int64_t idx = 0; idx < 2; idx++) {
            TaskFlowGroup* group = &ctx->priority_group[idx];
            if (!group->item_list.empty()) {
                ctx->mut_lock.lock();
                // 取当前group下的一个item到正在运行的list
                ctx->run_item_list.splice(ctx->run_item_list.begin(), group->item_list, group->item_list.begin());

                TaskFlowItem* item = &ctx->run_item_list.front();
                if (!item->node_list.empty()) {
                    printf("当前 item的node非空 ，它的的 taskinfo %s\r\n", static_cast<char*>(item->taskinfo));

                    // 如果item下节点非空// 取一个节点放到运行节点队列
                    ctx->run_node_list.splice(ctx->run_node_list.begin(), item->node_list, item->node_list.begin());
                    TaskFlowNode* node = &ctx->run_node_list.front();
                    ctx->mut_lock.unlock();
                    if (node->callback) {   // 运行节点
                        std::cout << "执行函数 " << __FUNCTION__ << __LINE__ << "\n";
                        ret = (*(node->callback))((void*)ctx, item->item_id, node->node_id, item->taskinfo, node->userdata);
                    }
                    ctx->mut_lock.lock();
                    TaskFlow_MoveIdleNode(ctx, ctx->run_node_list);   // 用完后放到空闲节点
                }
                if (ret) {   // 如果失败,回收全部节点
                    while (!item->node_list.empty()) {
                        TaskFlow_MoveIdleNode(ctx, item->node_list);
                    }
                }
                if (!item->node_list.empty()) {                      // 如果item下节点非空
                    TaskFlowNode* node = &item->node_list.front();   // 第一个节点
                    if (node->group_idx >= 2) {
                        node->group_idx = 0;
                    }
                    std::list<TaskFlowItem>* group_item_list = &(&(ctx->priority_group)[node->group_idx])->item_list;
                    for (std::list<TaskFlowItem>::iterator it = (*group_item_list).begin();; it++) {
                        TaskFlowItem* list_item = &(*it);   // 按序号插入
                        if (item->start_idx < list_item->start_idx || it == (*group_item_list).end()) {
                            (*group_item_list).splice((it), ctx->run_item_list, ctx->run_item_list.begin());
                            break;
                        }
                    }
                }
                else {
                    if (item->destructor) {
                        (*item->destructor)((void*)ctx, item->item_id, item->taskinfo);
                    }
                    else {
                    }
                    std::cout << "将正在运行的 item 移动到空闲的 item " << __FUNCTION__ << " " << __LINE__ << " " << "\n";
                    TaskFlow_MoveIdleItem(ctx, ctx->run_item_list);   // item下节点已运行完毕,移动到空闲队列
                }
                if (group->item_list.empty()) {        // 当前group下item list为空
                    while (group->wait_empty_cnt) {    // group 下等待空闲的数量
                        if (group->wait_empty_cnt) {   // 有等待的
                            group->wait_empty_cnt--;   // 等待的减少一个
                        }
                        // sem_post(&group->sem_empty);
                        group->sem_empty.signal();   // 释放一个
                    }
                }
                ctx->mut_lock.unlock();
                goto __PROCESS_START;
            }
            if (group->item_list.empty()) {
                while (group->wait_empty_cnt) {
                    if (group->wait_empty_cnt) {
                        group->wait_empty_cnt--;
                    }
                    // sem_post(&group->sem_empty);
                    group->sem_empty.signal();
                }
            }
            ctx->empty_group_cnt++;
        }
        if (ctx->empty_group_cnt < 2) {
            goto __PROCESS_START;
        }
        ctx->mut_lock.lock();
        while (ctx->wait_all_empty_cnt) {   // 有等待全空的
            if (ctx->wait_all_empty_cnt) {
                ctx->wait_all_empty_cnt--;
            }
            ctx->sem_all_empty.signal();
        }
        ctx->mut_lock.unlock();
    }
}


static int TaskFlow_SetStructor(TaskFlowCtx* ctx, TaskFlowItem* item, TaskInfoConstructor constructor, TaskInfoDestructor destructor, void* taskinfo)
{
    if (ctx == NULL || item == NULL) {
        return -1;
    }

    TaskInfoConstructor temp_constructor = NULL;
    TaskInfoDestructor  temp_destructor  = NULL;
    void*               temp_taskinfo    = NULL;
    if (ctx->pre_constructor) {
        temp_constructor = ctx->pre_constructor;
    }
    else if (constructor) {
        temp_constructor = constructor;
    }
    if (ctx->pre_destructor) {
        temp_destructor = ctx->pre_destructor;
    }
    else if (destructor) {
        temp_destructor = destructor;
    }
    if (ctx->pre_taskinfo) {
        temp_taskinfo = ctx->pre_taskinfo;
    }
    else {
        temp_taskinfo = taskinfo;
    }
    item->constructor = temp_constructor;
    item->destructor  = temp_destructor;
    if (temp_constructor) {
        item->taskinfo = (*item->constructor)((void*)ctx, item->item_id, temp_taskinfo);
    }
    else {
        item->taskinfo = temp_taskinfo;
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
 * @return 任务流队列上下文ID @ref void*
 */
void* TaskFlow_Init(int64_t pre_item_num, int64_t pre_node_num, TaskInfoConstructor pre_constructor, TaskInfoDestructor pre_destructor, void* taskinfo, int64_t cfg_flag)
{

    if (!pre_item_num && (cfg_flag & TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC)) {
        return NULL;
    }
    if (!pre_node_num && (cfg_flag & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC)) {
        return NULL;
    }
    TaskFlowCtx* ctx     = new TaskFlowCtx;
    ctx->pre_constructor = pre_constructor;
    ctx->pre_destructor  = pre_destructor;
    ctx->pre_taskinfo    = taskinfo;
    ctx->cfg_flag        = cfg_flag;
    for (int64_t idx = 0; idx < pre_item_num; idx++) {
        TaskFlowItem* item = new TaskFlowItem;
        if (TaskFlow_SetStructor(ctx, item, NULL, NULL, taskinfo)) {
            return NULL;
        }
        ctx->idle_item_list.push_front(*item);
    }
    for (int64_t idx = 0; idx < pre_node_num; idx++) {
        TaskFlowNode* node = new TaskFlowNode;
        ctx->idle_node_list.push_front(*node);
    }
    ctx->wait_all_empty_cnt = 0;
    ctx->wait_idle_item_cnt = 0;
    ctx->wait_idle_node_cnt = 0;
    for (int64_t idx = 0; idx < 2; idx++) {
        TaskFlowGroup* group  = &ctx->priority_group[idx];
        group->wait_empty_cnt = 0;
    }
    ctx->thr_main = std::make_shared<std::thread>(TaskFlowThreadHandler, ctx);
    return (void*)ctx;
}

/**

将空闲的 item 移动到 准备状态的item ， 并且将item的对应的node 移动到 准备的node
*/
int64_t TaskFlow_New(void* ctx_id, TaskInfoConstructor constructor, TaskInfoDestructor destructor, void* taskinfo, int64_t timeout)
{
    if (ctx_id == NULL)
        return 0;

    TaskFlowCtx* ctx = reinterpret_cast<TaskFlowCtx*>(ctx_id);

    ctx->mut_lock.lock();
    if (ctx->idle_item_list.empty() && (ctx->cfg_flag & TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC)) {
        ctx->wait_idle_item_cnt++;
        ctx->mut_lock.unlock();

        if (ctx->sem_idle_item.wait_for(timeout))
            return 0;
        ctx->mut_lock.lock();
        ctx->ready_item_list.splice(ctx->ready_item_list.end(), ctx->idle_item_list, --ctx->idle_item_list.end());
    }
    else {
        ctx->ready_item_list.splice(ctx->ready_item_list.end(), ctx->idle_item_list, --ctx->idle_item_list.end());
    }

    TaskFlowItem* item = &ctx->ready_item_list.back();
    if (!ctx->item_id_idx) {
        ctx->item_id_idx++;
    }
    item->item_id = ctx->item_id_idx++;
    std::cout << "\n item 的index 是：  " << item->item_id << "\n";
    if (!item->node_list.empty()) {
        ctx->idle_node_list.splice(ctx->idle_node_list.begin(), item->node_list);
    }
    if (item->taskinfo == NULL) {
        std::cout << "null taskinfo";
        if (TaskFlow_SetStructor(ctx, item, constructor, destructor, taskinfo)) {
            ctx->mut_lock.unlock();
            return 0;
        }
        if (item->taskinfo == NULL) {
            std::cout << "still null taskinfo \n";
        }
    }
    ctx->mut_lock.unlock();
    return item->item_id;
}

static TaskFlowItem* TaskFlow_FindListItem(std::list<TaskFlowItem>& list, int64_t item_id)
{
    for (std::list<TaskFlowItem>::iterator it = list.begin(); it != list.end(); it++) {
        TaskFlowItem* item = &(*it);
        if (item->item_id == item_id) {
            return item;
        }
    }
    return NULL;
}

static TaskFlowItem* TaskFlow_FindItem(TaskFlowCtx* ctx, int64_t item_id)
{
    if (ctx == NULL || item_id == 0) {
        return NULL;
    }
    TaskFlowItem* item = TaskFlow_FindListItem(ctx->ready_item_list, item_id);
    if (item == NULL) {
        item = TaskFlow_FindListItem(ctx->run_item_list, item_id);
        if (item == NULL) {
            for (int64_t idx = 0; idx < 2; idx++) {
                std::list<TaskFlowItem>* group_item_list = &(&ctx->priority_group[idx])->item_list;
                item                                     = TaskFlow_FindListItem((*group_item_list), item_id);
                if (item) {
                    break;
                }
            }
        }
    }
    return item;
}

//将空闲的node 移动到 list 里面
static int64_t TaskFlow_AddNode(TaskFlowCtx* ctx, TaskFlowItem* item, int64_t group_idx, TaskFlowCallback callback, void* userdata, int64_t timeout)
{
    if (item == NULL)
        return 0;

    if (ctx->idle_node_list.empty() && (ctx->cfg_flag & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC)) {
        ctx->wait_idle_node_cnt++;
        ctx->mut_lock.unlock();
        if (ctx->sem_idle_node.wait_for(timeout))
            return 0;
        ctx->mut_lock.lock();
        item->node_list.splice(item->node_list.end(), ctx->idle_node_list, --ctx->idle_node_list.end());
    }
    else {
        item->node_list.splice(item->node_list.end(), ctx->idle_node_list, --ctx->idle_node_list.end());
    }
    TaskFlowNode* node = &item->node_list.back();
    if (!ctx->node_id_idx) {
        ctx->node_id_idx++;
    }
    node->node_id   = (int64_t)ctx->node_id_idx++;
    node->group_idx = group_idx;
    node->callback  = callback;
    node->userdata  = userdata;
    return ((node)->node_id);
}

/**
 * 添加任务节点到任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @param  group_idx		优先级组序号
 * @param  callback		用户回调
 * @param  userdata		用户数据
 * @param  timeout		超时时间 Only valid when (cfg_flag & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC) is ture
 * @return 任务节点ID 0 fail other success @ref int64_t
 */
int64_t TaskFlow_Node(void* ctx_id, int64_t item_id, int64_t group_idx, TaskFlowCallback callback, void* userdata, int64_t timeout)
{
    int64_t      node_id = 0;
    TaskFlowCtx* ctx     = (TaskFlowCtx*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return 0;
    }
    ctx->mut_lock.lock();
    TaskFlowItem* item = TaskFlow_FindItem(ctx, item_id);
    if (item) {
        node_id = TaskFlow_AddNode(ctx, item, group_idx, callback, userdata, timeout);
        ctx->mut_lock.unlock();
        return node_id;
    }
    ctx->mut_lock.unlock();
    return 0;
}

/**
 * 开始任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return 0 success other fail
 */
int TaskFlow_Start(void* ctx_id, int64_t item_id)
{
    TaskFlowCtx* ctx = (TaskFlowCtx*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return -1;
    }
    ctx->mut_lock.lock();
    for (std::list<TaskFlowItem>::iterator it = ctx->ready_item_list.begin(); it != ctx->ready_item_list.end(); it++) {
        TaskFlowItem* item = &(*it);
        if (item->item_id == item_id) {
            if (!ctx->start_idx) {
                ctx->start_idx++;
            }
            item->start_idx   = ctx->start_idx++;
            int64_t group_idx = 0;
            if (!item->node_list.empty()) {
                TaskFlowNode* node = &item->node_list.front();
                if (node->group_idx < 2) {
                    group_idx = ((node)->group_idx);
                }
            }
            std::list<TaskFlowItem>* group_item_list = &(&ctx->priority_group[group_idx])->item_list;
            group_item_list->splice(group_item_list->end(), ctx->ready_item_list, it);
            ctx->mut_lock.unlock();
            ctx->sem_start.signal();
            return 0;
        }
    }
    ctx->mut_lock.unlock();
    return -2;
}

/**
 * 获取任务信息
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return 任务信息
 */
void* TaskFlow_GetTaskInfo(void* ctx_id, int64_t item_id)
{
    TaskFlowCtx* ctx = (TaskFlowCtx*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return NULL;
    }
    TaskFlowItem* item = TaskFlow_FindItem(ctx, item_id);
    if (item) {
        return (item->taskinfo);
    }
    return NULL;
}

/**
 * 删除任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return
 */
int TaskFlow_Del(void* ctx_id, int64_t item_id)
{
    TaskFlowCtx* ctx = (TaskFlowCtx*)ctx_id;
    if (ctx == NULL || item_id == 0) {
        return -1;
    }
    ctx->mut_lock.lock();
    TaskFlowItem* item = TaskFlow_FindItem(ctx, item_id);
    if (item) {
        while (!item->node_list.empty()) {
            TaskFlow_MoveIdleNode(ctx, item->node_list);
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
int TaskFlow_WaitGroup(void* ctx_id, int64_t group_idx, int64_t timeout)
{
    TaskFlowCtx* ctx = (TaskFlowCtx*)ctx_id;
    if (ctx == NULL || group_idx >= 2) {
        std::cout << "Clarity_WaitCplt  超时 最终错误 TaskFlow_WaitGroup \n";
        return -1;
    }
    TaskFlowGroup* group = &ctx->priority_group[group_idx];
    ctx->mut_lock.lock();
    group->wait_empty_cnt++;
    ctx->mut_lock.unlock();
    ctx->sem_start.signal();
    return group->sem_empty.wait_for(timeout);
}

/**
 * 等待所有任务流完成
 * @param  ctx_id		任务流队列上下文ID
 * @param  timeout		超时时间
 * @return
 */
int TaskFlow_WaitAll(void* ctx_id, int64_t timeout)
{
    TaskFlowCtx* ctx = (TaskFlowCtx*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    ctx->mut_lock.lock();
    ctx->wait_all_empty_cnt++;
    ctx->mut_lock.unlock();
    ctx->sem_start.signal();
    return ctx->sem_all_empty.wait_for(timeout);
}



#define TIMEOUT_FOREVER 0xFFFFFFFF




/*************************测试********************************** */

// node 的回调函数
int CB1(void* ctx_id, int64_t item_id, int64_t node_id, void* taskinfo, void* userdata)
{
    printf(" 回调函数打印--------------item_id  %ld node_id %ld  %s    \r\n", item_id, node_id, (char*)userdata);
    return 0;
}

void TaskFlow_AutoNewTask(void* ctx_id, int64_t task_id, int64_t node_num)
{
    char taskinfo[64] = "";
    snprintf(taskinfo, 64, "task_id: %ld", task_id);
    char* userdata = (char*)malloc(64);
    snprintf(userdata, 64, "Step %ld", task_id);
    int64_t item_id = TaskFlow_New(ctx_id, NULL, NULL, taskinfo, TIMEOUT_FOREVER);
    int64_t node_id = TaskFlow_Node(ctx_id, item_id, 0, CB1, (void*)userdata, TIMEOUT_FOREVER);
    int ret = TaskFlow_Start(ctx_id, item_id);
    printf("\n TaskFlow_AutoNewTask(%ld) = %d <\r\n", task_id, ret);
    return;
}




void TaskFlow_TestExample(void)
{
    int ret = 0;
    printf("TaskFlow Test Example >>>\n");
    void* ctxid = TaskFlow_Init(2, 5, NULL, NULL, NULL, TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC | TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC);
    std::cout << "TaskFlow_Init ret = " << ctxid;
    for (int64_t idx = 0; idx < 10; idx++) {
        std::cout << "正在运行 " << idx << "\n";
        TaskFlow_AutoNewTask(ctxid, idx, 5);
    }
    TaskFlow_WaitAll(ctxid, TIMEOUT_FOREVER);
}
