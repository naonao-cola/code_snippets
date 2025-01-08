#pragma once
#include "../stdafx.h"
#include <Windows.h>
#include <iostream>
#include <vector>
#include "./AIRuntime/AIRuntimeDataStruct.h"
using namespace std;

#define DEF_PRE_THRD_CNT 4
#define DEF_POST_THRD_CNT 2
#define DEF_PRE_THRD_PRIORITY   THREAD_PRIORITY_NORMAL
#define DEF_POST_THRD_PRIORITY  THREAD_PRIORITY_ABOVE_NORMAL
#define PREP_QUEUE_MAX_SIZE 128
#define POST_QUEUE_MAX_SIZE 128
#define WAIT_DEQUEUE_TIMEOUT 1000

#ifndef DECLARE_SINGLETON

#define DECLARE_SINGLETON(class_name)	\
    static class_name* get_instance() { \
        static class_name instance;		\
        return &instance;				\
    }

#endif // !DECLARE_SINGLETON

#ifndef HIDE_CREATE_METHODS
#define HIDE_CREATE_METHODS(class_name) \
    class_name(); \
    ~class_name(); \
    class_name(const class_name&) = delete; \
    class_name& operator=(const class_name&) = delete; 
#endif // !HIDE_CREATE_METHODS


using TimePoint = std::chrono::system_clock::time_point;

struct tTaktTime
{
	TimePoint startTime;
	TimePoint endTime;
	long long costTimeMs;
};


struct tAlgoResults
{
	tTaktTime tt;
	std::vector<std::shared_ptr<stModelResult>> vecResults;
};

//using AlgoResultPtr = std::shared_ptr<tAlgoResults>;

struct tAlgoInspParam
{
	int nImageNum;
	int nROINum;
	int nAlgNum;
	bool bpInspectEnd;
	HANDLE hInspectEnd;
	cv::Rect2i roiRect;
	tAlgoResults algoResults;
	// test_
	cv::Mat imgdata;
	std::vector<cv::Mat> imgs;
};

