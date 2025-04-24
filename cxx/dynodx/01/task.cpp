/**
 * @FilePath     : /code_snippets/cxx/dynodx/01/task.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-04-24 13:14:57
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-04-24 13:14:57
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#include "task.h"
#include <list>
#include <malloc.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <errno.h>
#include <stdio.h>

/* 任务节点 */
struct TaskFlowNode {
  uint32_t node_id;            // 任务节点ID
  uint32_t group_idx;          // 优先级组索引
  TaskFlowCallback_f callback; // 用户回调
  void *userdata;              // 用户数据
};

/* 任务项目 */
struct TaskFlowItem {
  uint32_t item_id;                  // 任务项目ID
  uint32_t start_idx;                // 启动序号
  void *taskinfo;                    // 任务信息
  TaskInfoConstructor_f constructor; // 任务信息构建函数
  TaskInfoDestructor_f destructor;   // 任务信息析构函数
  std::list<TaskFlowNode> node_list; // 任务节点列表
};

/* 任务优先级组 */
struct TaskFlowGroup {
  uint32_t wait_empty_cnt;           // 等待优先级组置空计数
  sem_t sem_empty;                   // 优先级组置空信号
  std::list<TaskFlowItem> item_list; // 任务项目列表
};

struct TaskFlowCtx {
  uint32_t cfg_flag;                       // 配置标志
  uint32_t empty_group_cnt;                // 优先级组置空计数
  uint32_t wait_all_empty_cnt;             // 等待全部优先级组置空计数
  uint32_t wait_idle_item_cnt;             // 等待空闲任务项目计数
  uint32_t wait_idle_node_cnt;             // 等待空闲任务节点计数
  sem_t sem_all_empty;                     // 全部优先级组置空信号
  sem_t sem_idle_item;                     // 空闲任务项目信号
  sem_t sem_idle_node;                     // 空闲任务节点信号
  sem_t sem_start;                         // 开始任务流信号
  pthread_mutex_t mut_lock;                // 队列锁定互斥
  pthread_t thr_main;                      // 主线程
  uint32_t item_id_idx;                    // 任务项目ID索引
  uint32_t node_id_idx;                    // 任务节点ID索引
  uint32_t start_idx;                      // 任务启动索引
  TaskInfoConstructor_f pre_constructor;   // 预分配任务信息构造函数
  TaskInfoDestructor_f pre_destructor;     // 预分配任务信息析构函数
  void *pre_taskinfo;                      // 预分配任务信息
  TaskFlowGroup priority_group[2];         // 优先级组
  std::list<TaskFlowItem> ready_item_list; // 就绪项目列表
  std::list<TaskFlowItem> run_item_list;   // 运行项目列表
  std::list<TaskFlowNode> run_node_list;   // 运行节点列表
  std::list<TaskFlowItem> idle_item_list;  // 空闲项目列表
  std::list<TaskFlowNode> idle_node_list;  // 空闲节点列表
};

#define MAX_WAIT_INTERVAL_USECOND 5000

static int TaskFlow_GetSem(sem_t *sem_id, uint32_t timeout_ms) {
  int s;
  struct timespec ts = {0};
  if (clock_gettime(CLOCK_REALTIME, &ts)) {
    std::cout << "获取时间出错";
    return -1;
  }
  ts.tv_sec += timeout_ms / 1000;
  ts.tv_nsec += (timeout_ms % 1000) * 1000;
  // 如果ts.tv_nsec超过1000000000, sem_timedwait 将会返回 22
  // 此处增加值的标准化
  long int sec_quotient = ts.tv_nsec / 1000000000;
  long int nsec_remainder = ts.tv_nsec % 1000000000;
  ts.tv_sec += sec_quotient;
  ts.tv_nsec = nsec_remainder;

  while ((s = sem_timedwait(sem_id, &ts)) == -1 && errno == EINTR)
    continue; // Restart if interrupted by handler Check what happened
  if (s == -1 && timeout_ms != 0) {
    std::cout << "sem_timedwait error code " << errno;
    std::cout << "nsec " << ts.tv_nsec;
  }

  return s;
}

/**
 将list 里面开头的一个元素 转到空闲项目
*/
static void TaskFlow_MoveIdleItem(TaskFlowCtx *ctx,
                                  std::list<TaskFlowItem> &list) {
  if (ctx) {
    ctx->idle_item_list.splice(ctx->idle_item_list.begin(), list, list.begin());
    if (ctx->wait_idle_item_cnt) {
      ctx->wait_idle_item_cnt--;
      sem_post((&(ctx)->sem_idle_item));
    }
  }
}

/**
将list 的node的 开头的一个元素 移动到 空闲node
*/
static void TaskFlow_MoveIdleNode(TaskFlowCtx *ctx,
                                  std::list<TaskFlowNode> &list) {
  if (ctx) {
    ctx->idle_node_list.splice(ctx->idle_node_list.begin(), list, list.begin());
    if (ctx->wait_idle_node_cnt) {
      ctx->wait_idle_node_cnt--;
      sem_post(&ctx->sem_idle_node);
    }
  }
}

[[noreturn]] static void *TaskFlowThreadHandler(void *arg) {
  TaskFlowCtx *ctx = (TaskFlowCtx *)arg;

  std::cout << "\n generate another task thread";
  while (1) {

    sem_wait(&ctx->sem_start);

    while (!TaskFlow_GetSem(&ctx->sem_start, 0))
      ;

  __PROCESS_START:

    int ret = 0;
    uint32_t all_empty_cnt = 0;
    pthread_mutex_lock(&ctx->mut_lock);
    while (!ctx->run_item_list.empty()) {
      TaskFlow_MoveIdleItem(ctx, ctx->run_item_list);
    }
    while (!ctx->run_node_list.empty()) {
      TaskFlow_MoveIdleNode(ctx, ctx->run_node_list);
    }
    pthread_mutex_unlock(&ctx->mut_lock);

    for (uint32_t idx = 0; idx < 2; idx++) {
      TaskFlowGroup *group = &ctx->priority_group[idx];
      if (!group->item_list.empty()) {
        pthread_mutex_lock(&ctx->mut_lock);
        // 取当前group下的一个item到正在运行的list
        ctx->run_item_list.splice(ctx->run_item_list.begin(), group->item_list,
                                  group->item_list.begin());

        TaskFlowItem *item = &ctx->run_item_list.front();
        if (!item->node_list.empty()) {
          // 如果item下节点非空// 取一个节点放到运行节点队列
          ctx->run_node_list.splice(ctx->run_node_list.begin(), item->node_list,
                                    item->node_list.begin());
          TaskFlowNode *node = &ctx->run_node_list.front();
          pthread_mutex_unlock(&ctx->mut_lock);
          if (node->callback) { // 运行节点
            std::cout << "执行函数 " << __FUNCTION__ << __LINE__ << "\n";
            ret = (*(node->callback))((void *)ctx, item->item_id, node->node_id,
                                      item->taskinfo, node->userdata);
          }
          pthread_mutex_lock(&ctx->mut_lock);
          TaskFlow_MoveIdleNode(ctx, ctx->run_node_list); // 用完后放到空闲节点
        }
        if (ret) { // 如果失败,回收全部节点
          while (!((item)->node_list).empty()) {
            TaskFlow_MoveIdleNode(ctx, item->node_list);
          }
        }
        if (!item->node_list.empty()) {                  // 如果item下节点非空
          TaskFlowNode *node = &item->node_list.front(); // 第一个节点
          if (node->group_idx >= 2) {
            node->group_idx = 0;
          }
          std::list<TaskFlowItem> *group_item_list =
              &(&(ctx->priority_group)[node->group_idx])->item_list;
          for (std::list<TaskFlowItem>::iterator it =
                   (*group_item_list).begin();
               ; it++) {
            TaskFlowItem *list_item = &(*it); // 按序号插入
            if (item->start_idx < list_item->start_idx ||
                it == (*group_item_list).end()) {
              (*group_item_list)
                  .splice((it), ctx->run_item_list, ctx->run_item_list.begin());
              break;
            }
          }
        } else {
          if (item->destructor) {
            (*item->destructor)((void *)ctx, item->item_id, item->taskinfo);
          } else {
          }
          TaskFlow_MoveIdleItem(
              ctx, ctx->run_item_list); // item下节点已运行完毕,移动到空闲队列
        }
        if (group->item_list.empty()) {   // 当前group下item list为空
          while (group->wait_empty_cnt) { // group 下等待空闲的数量
            if (group->wait_empty_cnt) {  // 有等待的
              group->wait_empty_cnt--;    // 等待的减少一个
            }
            sem_post(&group->sem_empty); // 释放一个
          }
        }
        pthread_mutex_unlock(&ctx->mut_lock);
        goto __PROCESS_START;
      }
      if (group->item_list.empty()) {
        while (group->wait_empty_cnt) {
          if (group->wait_empty_cnt) {
            group->wait_empty_cnt--;
          }
          sem_post(&group->sem_empty);
        }
      }
      ctx->empty_group_cnt++;
    }
    if (ctx->empty_group_cnt < 2) {
      goto __PROCESS_START;
    }
    pthread_mutex_lock(&ctx->mut_lock);
    while (ctx->wait_all_empty_cnt) { // 有等待全空的
      if (ctx->wait_all_empty_cnt) {
        ctx->wait_all_empty_cnt--;
      }
      sem_post(&ctx->sem_all_empty);
    }
    pthread_mutex_unlock((&(ctx)->mut_lock));
  }
}

static int TaskFlow_SetStructor(TaskFlowCtx *ctx, TaskFlowItem *item,
                                TaskInfoConstructor_f constructor,
                                TaskInfoDestructor_f destructor,
                                void *taskinfo) {
  if (ctx == NULL || item == NULL) {
    return -1;
  }
  TaskInfoConstructor_f temp_constructor = NULL;
  TaskInfoDestructor_f temp_destructor = NULL;
  void *temp_taskinfo = NULL;
  if (ctx->pre_constructor) {
    temp_constructor = ctx->pre_constructor;
  } else if (constructor) {
    temp_constructor = constructor;
  }
  if (ctx->pre_destructor) {
    temp_destructor = ctx->pre_destructor;
  } else if (destructor) {
    temp_destructor = destructor;
  }
  if (ctx->pre_taskinfo) {
    temp_taskinfo = ctx->pre_taskinfo;
  } else {
    temp_taskinfo = taskinfo;
  }
  item->constructor = temp_constructor;
  item->destructor = temp_destructor;
  if (temp_constructor) {
    item->taskinfo =
        (*item->constructor)((void *)ctx, item->item_id, temp_taskinfo);
  } else {
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
void *TaskFlow_Init(uint32_t pre_item_num, uint32_t pre_node_num,
                    TaskInfoConstructor_f pre_constructor,
                    TaskInfoDestructor_f pre_destructor, void *taskinfo,
                    uint32_t cfg_flag) {

  if (!pre_item_num && (cfg_flag & TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC)) {
    return NULL;
  }
  if (!pre_node_num && (cfg_flag & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC)) {
    return NULL;
  }
  TaskFlowCtx *ctx = new TaskFlowCtx;
  ctx->pre_constructor = pre_constructor;
  ctx->pre_destructor = pre_destructor;
  ctx->pre_taskinfo = taskinfo;
  ctx->cfg_flag = cfg_flag;
  for (uint32_t idx = 0; idx < pre_item_num; idx++) {
    TaskFlowItem *item = new TaskFlowItem;
    if (TaskFlow_SetStructor(ctx, item, NULL, NULL, taskinfo)) {
      return NULL;
    }
    ctx->idle_item_list.push_front(*item);
  }
  for (uint32_t idx = 0; idx < pre_node_num; idx++) {
    TaskFlowNode *node = new TaskFlowNode;
    ctx->idle_node_list.push_front(*node);
  }
  ctx->wait_all_empty_cnt = 0;
  ctx->wait_idle_item_cnt = 0;
  ctx->wait_idle_node_cnt = 0;
  for (uint32_t idx = 0; idx < 2; idx++) {
    TaskFlowGroup *group = &ctx->priority_group[idx];
    group->wait_empty_cnt = 0;
    sem_init(&group->sem_empty, 0, 0);
  }
  sem_init(&ctx->sem_all_empty, 0, 0);
  sem_init(&ctx->sem_idle_item, 0, 0);
  sem_init(&ctx->sem_idle_node, 0, 0);
  sem_init(&ctx->sem_start, 0, 0);
  pthread_mutex_init(&ctx->mut_lock, NULL);
  if (pthread_create(&ctx->thr_main, NULL, TaskFlowThreadHandler, ctx)) {
    return NULL;
  }
  return (void *)ctx;
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
 * @param  taskinfo		任务信息 This pointer will be used directly when
 * constructor=NULL and destructor=NULL
 * @param  timeout		超时时间 Only valid when (cfg_flag &
 * TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC) is ture
 * @return 任务项目ID 0 fail other success @ref uint32_t
 */
uint32_t TaskFlow_New(void *ctx_id, TaskInfoConstructor_f constructor,
                      TaskInfoDestructor_f destructor, void *taskinfo,
                      uint32_t timeout) {
  TaskFlowCtx *ctx = (TaskFlowCtx *)ctx_id;
  if (ctx == NULL) {
    return 0;
  }
  pthread_mutex_lock(&ctx->mut_lock);
  if (ctx->idle_item_list.empty()) {
    if (ctx->cfg_flag & TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC) {
    __WAIT_TDLE_TIME:
      auto start =
          duration_cast<milliseconds>(system_clock::now().time_since_epoch())
              .count();
      ctx->wait_idle_item_cnt++;
      pthread_mutex_unlock(&ctx->mut_lock);
      if (TaskFlow_GetSem(&ctx->sem_idle_item, timeout)) {
        return 0;
      }
      pthread_mutex_lock((&ctx->mut_lock));
      ctx->ready_item_list.splice(ctx->ready_item_list.end(),
                                  ctx->idle_item_list,
                                  --ctx->idle_item_list.end());
      auto func_end =
          duration_cast<milliseconds>(system_clock::now().time_since_epoch())
              .count();
      auto func_cost_time = func_end - start;
      std::cout << "item need more... waiting time " << func_cost_time;
    } else {
      printf("item need more...\r\n");
      TaskFlowItem *item = new TaskFlowItem;
      if (TaskFlow_SetStructor(ctx, item, constructor, destructor, taskinfo)) {
        delete item;
        goto __WAIT_TDLE_TIME;
      }
      ctx->ready_item_list.push_back(*item);
    }
  } else {
    ctx->ready_item_list.splice(ctx->ready_item_list.end(), ctx->idle_item_list,
                                --ctx->idle_item_list.end());
  }
  TaskFlowItem *item = &ctx->ready_item_list.back();
  if (!ctx->item_id_idx) {
    ctx->item_id_idx++;
  }
  item->item_id = ctx->item_id_idx++;
  while (!item->node_list.empty()) {
    ctx->idle_node_list.splice(ctx->idle_node_list.begin(), item->node_list,
                               item->node_list.begin());
  }
  if (item->taskinfo == NULL) {
    std::cout << "null taskinfo";
    if (TaskFlow_SetStructor(ctx, item, constructor, destructor, taskinfo)) {
      pthread_mutex_unlock(&ctx->mut_lock);
      return 0;
    }
    if (item->taskinfo == NULL) {
      std::cout << "still null taskinfo \n";
    }
  }
  pthread_mutex_unlock(&ctx->mut_lock);
  return item->item_id;
}

static TaskFlowItem *TaskFlow_FindListItem(std::list<TaskFlowItem> &list,
                                           uint32_t item_id) {
  for (std::list<TaskFlowItem>::iterator it = list.begin(); it != list.end();
       it++) {
    TaskFlowItem *item = &(*it);
    if (item->item_id == item_id) {
      return item;
    }
  }
  return NULL;
}

static TaskFlowItem *TaskFlow_FindItem(TaskFlowCtx *ctx, uint32_t item_id) {
  if (ctx == NULL || item_id == 0) {
    return NULL;
  }
  TaskFlowItem *item = TaskFlow_FindListItem(ctx->ready_item_list, item_id);
  if (item == NULL) {
    item = TaskFlow_FindListItem(ctx->run_item_list, item_id);
    if (item == NULL) {
      for (uint32_t idx = 0; idx < 2; idx++) {
        std::list<TaskFlowItem> *group_item_list =
            &(&ctx->priority_group[idx])->item_list;
        item = TaskFlow_FindListItem((*group_item_list), item_id);
        if (item) {
          break;
        }
      }
    }
  }
  return item;
}

static uint32_t TaskFlow_AddNode(TaskFlowCtx *ctx, TaskFlowItem *item,
                                 uint32_t group_idx,
                                 TaskFlowCallback_f callback, void *userdata,
                                 uint32_t timeout) {
  if (item == NULL) {
    return 0;
  }
  if (ctx->idle_node_list.empty()) {
    if (ctx->cfg_flag & TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC) {
      ctx->wait_idle_node_cnt++;
      pthread_mutex_unlock(&ctx->mut_lock);
      if (TaskFlow_GetSem(&ctx->sem_idle_node, timeout)) {
        return 0;
      }
      pthread_mutex_lock(&ctx->mut_lock);

      item->node_list.splice(item->node_list.end(), ctx->idle_node_list,
                             --ctx->idle_node_list.end());
    } else {
      TaskFlowNode *node = new TaskFlowNode;
      item->node_list.push_back(*node);
    }
  } else {
    item->node_list.splice(item->node_list.end(), ctx->idle_node_list,
                           --ctx->idle_node_list.end());
  }
  TaskFlowNode *node = &item->node_list.back();
  if (!ctx->node_id_idx) {
    ctx->node_id_idx++;
  }
  node->node_id = (uint32_t)ctx->node_id_idx++;
  node->group_idx = group_idx;
  node->callback = callback;
  node->userdata = userdata;
  return ((node)->node_id);
}

/**
 * 添加任务节点到任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @param  group_idx		优先级组序号
 * @param  callback		用户回调
 * @param  userdata		用户数据
 * @param  timeout		超时时间 Only valid when (cfg_flag &
 * TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC) is ture
 * @return 任务节点ID 0 fail other success @ref uint32_t
 */
uint32_t TaskFlow_Node(void *ctx_id, uint32_t item_id, uint32_t group_idx,
                       TaskFlowCallback_f callback, void *userdata,
                       uint32_t timeout) {
  uint32_t node_id = 0;
  TaskFlowCtx *ctx = (TaskFlowCtx *)ctx_id;
  if (ctx == NULL || item_id == 0) {
    return 0;
  }
  pthread_mutex_lock(&ctx->mut_lock);
  TaskFlowItem *item = TaskFlow_FindItem(ctx, item_id);
  if (item) {
    node_id =
        TaskFlow_AddNode(ctx, item, group_idx, callback, userdata, timeout);
    pthread_mutex_unlock(&ctx->mut_lock);
    return node_id;
  }
  pthread_mutex_unlock(&ctx->mut_lock);
  return 0;
}

/**
 * 开始任务项目
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return 0 success other fail
 */
int TaskFlow_Start(void *ctx_id, uint32_t item_id) {
  TaskFlowCtx *ctx = (TaskFlowCtx *)ctx_id;
  if (ctx == NULL || item_id == 0) {
    return -1;
  }
  pthread_mutex_lock(&ctx->mut_lock);
  for (std::list<TaskFlowItem>::iterator it = ctx->ready_item_list.begin();
       it != ctx->ready_item_list.end(); it++) {
    TaskFlowItem *item = &(*it);
    if (item->item_id == item_id) {
      if (!ctx->start_idx) {
        ctx->start_idx++;
      }
      item->start_idx = ctx->start_idx++;
      uint32_t group_idx = 0;
      if (!item->node_list.empty()) {
        TaskFlowNode *node = &item->node_list.front();
        if (node->group_idx < 2) {
          group_idx = ((node)->group_idx);
        }
      }
      std::list<TaskFlowItem> *group_item_list =
          &(&ctx->priority_group[group_idx])->item_list;

      group_item_list->splice(group_item_list->end(), ctx->ready_item_list, it);

      pthread_mutex_unlock(&ctx->mut_lock);
      sem_post(&ctx->sem_start);
      return 0;
    }
  }
  pthread_mutex_unlock((&(ctx)->mut_lock));
  return -2;
}

/**
 * 获取任务信息
 * @param  ctx_id		任务流上下文ID
 * @param  item_id		任务项目ID
 * @return 任务信息
 */
void *TaskFlow_GetTaskInfo(void *ctx_id, uint32_t item_id) {
  TaskFlowCtx *ctx = (TaskFlowCtx *)ctx_id;
  if (ctx == NULL || item_id == 0) {
    return NULL;
  }
  TaskFlowItem *item = TaskFlow_FindItem(ctx, item_id);
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
int TaskFlow_Del(void *ctx_id, uint32_t item_id) {
  TaskFlowCtx *ctx = (TaskFlowCtx *)ctx_id;
  if (ctx == NULL || item_id == 0) {
    return -1;
  }
  pthread_mutex_lock(&ctx->mut_lock);
  TaskFlowItem *item = TaskFlow_FindItem(ctx, item_id);
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
int TaskFlow_WaitGroup(void *ctx_id, uint32_t group_idx, uint32_t timeout) {
  TaskFlowCtx *ctx = (TaskFlowCtx *)ctx_id;
  if (ctx == NULL || group_idx >= 2) {
    std::cout << "Clarity_WaitCplt  超时 最终错误 TaskFlow_WaitGroup \n";
    return -1;
  }
  TaskFlowGroup *group = &ctx->priority_group[group_idx];
  pthread_mutex_lock(&ctx->mut_lock);
  group->wait_empty_cnt++;
  pthread_mutex_unlock(&ctx->mut_lock);
  sem_post(&ctx->sem_start);
  return TaskFlow_GetSem(&group->sem_empty, timeout);
}

/**
 * 等待所有任务流完成
 * @param  ctx_id		任务流队列上下文ID
 * @param  timeout		超时时间
 * @return
 */
int TaskFlow_WaitAll(void *ctx_id, uint32_t timeout) {
  TaskFlowCtx *ctx = (TaskFlowCtx *)ctx_id;
  if (ctx == NULL) {
    return -1;
  }
  pthread_mutex_lock(&ctx->mut_lock);
  ctx->wait_all_empty_cnt++;
  pthread_mutex_unlock(&ctx->mut_lock);
  sem_post(&ctx->sem_start);
  return TaskFlow_GetSem(&ctx->sem_all_empty, timeout);
}

#include <vector>

#define TIMEOUT_FOREVER 0xFFFFFFFF

struct TaskInfoTest {
  const char *info;
  std::vector<int> val;
};

void *info_init(void *ctx_id, uint32_t item_id, void *taskinfo) {
  TaskInfoTest *data = new TaskInfoTest;
  data->info = "TASKINFO:";
  for (uint32_t idx = 0; idx < item_id; idx++) {
    data->val.push_back(idx);
  }
  return (void *)data;
}

/*************************测试********************************** */
void info_deinit(void *ctx_id, uint32_t item_id, void *taskinfo) {}

int CB2(void *ctx_id, uint32_t item_id, uint32_t node_id, void *taskinfo,
        void *userdata) {
  TaskInfoTest *data = (TaskInfoTest *)taskinfo;
  char buf[2048] = "";
  int len = 0;
  for (auto it : data->val) {
    len += snprintf(buf + len, 2048 - len, "%d ", it);
  }
  printf("CB2>> %s (%s)\r\n", data->info, buf);
  sleep(1);
  return 0;
}

int CB1(void *ctx_id, uint32_t item_id, uint32_t node_id, void *taskinfo,
        void *userdata) {
  TaskFlow_Node(ctx_id, item_id, 0, CB2, NULL, TIMEOUT_FOREVER);
  printf("CB1>> item(%d) node(%d) task(%s) userdata(%s)\r\n", item_id, node_id,
         NULL, userdata);
  return 0;
}

void TaskFlow_AutoNewTask(void *ctx_id, uint32_t task_id, uint32_t node_num) {
  char taskinfo[64] = "";
  snprintf(taskinfo, 64, "Task %d", task_id);
  uint32_t item_id =
      TaskFlow_New(ctx_id, info_init, info_deinit, NULL, TIMEOUT_FOREVER);

  char *userdata = (char *)malloc(64);
  snprintf(userdata, 64, "Step %d", 1);
  TaskFlow_Node(ctx_id, item_id, 0, CB1, (void *)userdata, TIMEOUT_FOREVER);
  int ret = TaskFlow_Start(ctx_id, item_id);

  printf("TaskFlow_AutoNewTask(%d) = %d <\r\n", task_id, ret);
  return;
}

void TaskFlow_TestExample(void) {
  int ret = 0;
  printf("TaskFlow Test Example >>>\n");
  void *ctxid = TaskFlow_Init(2, 4, NULL, NULL, NULL,
                              TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC |
                                  TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC);

  std::cout << "TaskFlow_Init ret = " << ctxid;
  for (uint32_t idx = 0; idx < 8; idx++) {
    TaskFlow_AutoNewTask(ctxid, idx, 5);
  }
}
