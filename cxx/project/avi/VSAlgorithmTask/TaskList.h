
#pragma once

#include <ppl.h>

using namespace Concurrency;
//用于管理异步任务的任务列表的类。
//处理Task的动态创建和销毁。
class CTaskList
{
private:
	//CList<task_group*, task_group*>	m_ListRunningTask;
	CList<task_group*, task_group*>	m_ListFinishedTask;
	CRITICAL_SECTION	m_cs;

		ULONG	m_lCntRunningTask;		//从内存中删除CtaskList对象时,如果当前有正在运行的Task,则等待的计数

public:
	CTaskList()
	{
		m_lCntRunningTask = 0;
		InitializeCriticalSectionAndSpinCount(&m_cs,4000);	
	}
	~CTaskList()
	{
		task_group* pTaskGroup = NULL;

		while(m_lCntRunningTask)
			Sleep(0);

		EnterCriticalSection(&m_cs);
		DeleteAllFinishedTask();
		LeaveCriticalSection(&m_cs);
		DeleteCriticalSection(&m_cs);
	}

		//如果任务结束列表中有已结束的Task,则销毁内存并创建新的Task组并返回。
	task_group* AllocNewTaskGroup()
	{
		EnterCriticalSection(&m_cs);
		DeleteAllFinishedTask();
		task_group* pTask = new task_group;
		m_lCntRunningTask++;
		LeaveCriticalSection(&m_cs);
		return pTask;
	}

		//Task的任务结束后,在任务结束列表中注册。
	int EndTaskGroup(task_group* pEndTask)
	{
		EnterCriticalSection(&m_cs);
		AddTailFinishedTask(pEndTask);
		m_lCntRunningTask--;
		LeaveCriticalSection(&m_cs);
		return 0;
	}

private:
	int AddTailFinishedTask(task_group* pEndTask)
	{
		POSITION ps = m_ListFinishedTask.AddTail(pEndTask);

		return 0;
	}

	int DeleteAllFinishedTask()
	{
		task_group* pTaskGroup = NULL;

		for(POSITION pos = m_ListFinishedTask.GetHeadPosition(); pos != NULL;)
		{
			pTaskGroup = m_ListFinishedTask.GetNext(pos);

			if (pTaskGroup != NULL)
			{
				pTaskGroup->wait();
				delete pTaskGroup;
				pTaskGroup = NULL;
			}
		}
		m_ListFinishedTask.RemoveAll();
		return 0;
	}
};
