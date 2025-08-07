
#ifndef _IMAGE_FLOW_QUEUE_H_
#define _IMAGE_FLOW_QUEUE_H_

#include <stdint.h>

#define TASKFLOW_MAX_GROUP_NUM 2

#define TaskFlowCtxID_t void*
#define TaskFlowItemID_t uint32_t
#define TaskFlowNodeID_t uint32_t

typedef enum TaskFlowFlag
{
    TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC = (1 << 0),   // 关闭任务项目自动分配
    TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC = (1 << 1),   // 关闭任务节点自动分配
} TaskFlowFlag_t;

/**
 * 任务信息构造函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  taskinfo      任务信息
 * @return 任务信息指针
 */
typedef void* (*TaskInfoConstructor_f)(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, void* taskinfo);

/**
 * 任务信息析构函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  taskinfo      任务信息
 * @return none
 */
typedef void (*TaskInfoDestructor_f)(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, void* taskinfo);

/**
 * 任务流回调函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  node_id       任务节点ID
 * @param  taskinfo      任务信息
 * @param  userdata      用户数据
 * @return 0 continue the process other abort
 */
typedef int (*TaskFlowCallback_f)(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, TaskFlowNodeID_t node_id, void* taskinfo, void* userdata);

TaskFlowCtxID_t  TaskFlow_Init(uint32_t              pre_item_num,
                               uint32_t              pre_node_num,
                               TaskInfoConstructor_f pre_constructor,
                               TaskInfoDestructor_f  pre_destructor,
                               void*                 taskinfo,
                               uint32_t              cfg_flag);
TaskFlowItemID_t TaskFlow_New(
    TaskFlowCtxID_t ctx_id, TaskInfoConstructor_f constructor, TaskInfoDestructor_f destructor, void* taskinfo, uint32_t timeout);
TaskFlowNodeID_t TaskFlow_Node(
    TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id, uint32_t group_idx, TaskFlowCallback_f callback, void* userdata, uint32_t timeout);
int   TaskFlow_Start(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id);
void* TaskFlow_GetTaskInfo(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id);
int   TaskFlow_Del(TaskFlowCtxID_t ctx_id, TaskFlowItemID_t item_id);
int   TaskFlow_WaitGroup(TaskFlowCtxID_t ctx_id, uint32_t group_idx, uint32_t timeout);
int   TaskFlow_WaitAll(TaskFlowCtxID_t ctx_id, uint32_t timeout);

void TaskFlow_TestExample(void);


// int TaskFlow_GetIdleSize(TaskFlowCtxID_t ctx_id);

#endif /* _IMAGE_FLOW_QUEUE_H_ */
