
/**
 * @FilePath     : /test/src/task.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-04-23 16:02:27
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-04-30 14:21:23
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/

#ifndef _IMAGE_FLOW_QUEUE_H_
#define _IMAGE_FLOW_QUEUE_H_

#include <stdint.h>


enum TaskFlowFlag
{
    TASKFLOW_FLAG_CLOSE_ITEM_AUTO_ALLOC = 1,   // 关闭任务项目自动分配
    TASKFLOW_FLAG_CLOSE_NODE_AUTO_ALLOC = 2,   // 关闭任务节点自动分配
};

/**
 * 任务信息构造函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  taskinfo      任务信息
 * @return 任务信息指针
 */
typedef void* (*TaskInfoConstructor)(void* ctx_id, int64_t item_id, void* taskinfo);

/**
 * 任务信息析构函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  taskinfo      任务信息
 * @return none
 */
typedef void (*TaskInfoDestructor)(void* ctx_id, int64_t item_id, void* taskinfo);

/**
 * 任务流回调函数定义
 * @param  ctx_id        任务流上下文ID
 * @param  item_id       任务项目ID
 * @param  node_id       任务节点ID
 * @param  taskinfo      任务信息
 * @param  userdata      用户数据
 * @return 0 continue the process other abort
 */
typedef int (*TaskFlowCallback)(void* ctx_id, int64_t item_id, int64_t node_id, void* taskinfo, void* userdata);



/**
 * @brief  :
 * @param   pre_item_num
 * @param   pre_node_num
 * @param   pre_constructor
 * @param   pre_destructor
 * @param   taskinfo
 * @param   cfg_flag
 * @return
 * @note   :
**/
void* TaskFlow_Init(int64_t pre_item_num, int64_t pre_node_num, TaskInfoConstructor pre_constructor, TaskInfoDestructor pre_destructor, void* taskinfo, int64_t cfg_flag);

/**
 * @brief  :
 * @param   ctx_id
 * @param   constructor
 * @param   destructor
 * @param   taskinfo
 * @param   timeout
 * @return
 * @note   :
**/
int64_t TaskFlow_New(void* ctx_id, TaskInfoConstructor constructor, TaskInfoDestructor destructor, void* taskinfo, int64_t timeout);


/**
 * @brief  :
 * @param   ctx_id
 * @param   item_id
 * @param   group_idx
 * @param   callback
 * @param   userdata
 * @param   timeout
 * @return
 * @note   :
**/
int64_t TaskFlow_Node(void* ctx_id, int64_t item_id, int64_t group_idx, TaskFlowCallback callback, void* userdata, int64_t timeout);

/**
 * @brief  :
 * @param   ctx_id
 * @param   item_id
 * @return
 * @note   :
**/
int TaskFlow_Start(void* ctx_id, int64_t item_id);

/**
 * @brief  :
 * @param   ctx_id
 * @param   item_id
 * @return
 * @note   :
**/
void* TaskFlow_GetTaskInfo(void* ctx_id, int64_t item_id);


/**
 * @brief  :
 * @param   ctx_id
 * @param   item_id
 * @return
 * @note   :
**/
int TaskFlow_Del(void* ctx_id, int64_t item_id);


/**
 * @brief  :
 * @param   ctx_id
 * @param   group_idx
 * @param   timeout
 * @return
 * @note   :
**/
int TaskFlow_WaitGroup(void* ctx_id, int64_t group_idx, int64_t timeout);


/**
 * @brief  :
 * @param   ctx_id
 * @param   timeout
 * @return
 * @note   :
**/
int TaskFlow_WaitAll(void* ctx_id, int64_t timeout);


/**
 * @brief  :
 * @return
 * @note   :
**/
void TaskFlow_TestExample(void);

#endif /* _IMAGE_FLOW_QUEUE_H_ */
