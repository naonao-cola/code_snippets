/**
 * @FilePath     : /test/src/task.h
 * @Description  :
 * @Author       : error: git config user.name & please set dead value or install git
 * @Date         : 2025-04-23 16:02:27
 * @Version      : 0.0.1
 * @LastEditors  : error: git config user.name & please set dead value or install git
 * @LastEditTime : 2025-04-23 16:07:59
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#ifndef _IMAGE_FLOW_QUEUE_H_
#define _IMAGE_FLOW_QUEUE_H_

#include <iostream>
#include <stdint.h>


#define TASKFLOW_MAX_GROUP_NUM 2

enum  TaskFlowFlag
{
    TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC = 1,   // 关闭任务项目自动分配
    TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC = 2,   // 关闭任务节点自动分配
} ;

/**
 * 任务信息构造函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  taskinfo      任务信息
 * @return 任务信息指针
 */
typedef void* (*TaskInfoConstructor_f)(void* ctx_id, uint32_t item_id, void* taskinfo);

/**
 * 任务信息析构函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  taskinfo      任务信息
 * @return none
 */
typedef void (*TaskInfoDestructor_f)(void* ctx_id, uint32_t item_id, void* taskinfo);

/**
 * 任务流回调函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  node_id       任务节点ID
 * @param  taskinfo      任务信息
 * @param  userdata      用户数据
 * @return 0 continue the process other abort
 */
typedef int (*TaskFlowCallback_f)(void* ctx_id, uint32_t item_id, uint32_t node_id, void* taskinfo, void* userdata);

void* TaskFlow_Init(uint32_t pre_item_num, uint32_t pre_node_num, TaskInfoConstructor_f pre_constructor, TaskInfoDestructor_f pre_destructor, void* taskinfo, uint32_t cfg_flag);

uint32_t TaskFlow_New(void* ctx_id, TaskInfoConstructor_f constructor, TaskInfoDestructor_f destructor, void* taskinfo, uint32_t timeout);

uint32_t TaskFlow_Node(void* ctx_id, uint32_t item_id, uint32_t group_idx, TaskFlowCallback_f callback, void* userdata, uint32_t timeout);

int TaskFlow_Start(void* ctx_id, uint32_t item_id);

void* TaskFlow_GetTaskInfo(void* ctx_id, uint32_t item_id);

int TaskFlow_Del(void* ctx_id, uint32_t item_id);

int TaskFlow_WaitGroup(void* ctx_id, uint32_t group_idx, uint32_t timeout);

int TaskFlow_WaitAll(void* ctx_id, uint32_t timeout);

void TaskFlow_TestExample(void);

#endif /* _IMAGE_FLOW_QUEUE_H_ */
