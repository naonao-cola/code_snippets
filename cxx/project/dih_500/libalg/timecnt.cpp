#include "timecnt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <list>
#include <string>
#include <time.h>
#include <iostream>

#include "event.h"

#include "DihLogPlog.h"

typedef struct TimeCnt
{
	std::string name;
	int64_t start_time;
	int64_t min_single_time;
	int64_t max_single_time;
	int64_t full_time;
	int64_t count;
	uint8_t open_print;
}TimeCnt_t;

std::list<TimeCnt_t> cnt_list;

static int64_t TimeCnt_GetTimeMs(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (int64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

void TimeCnt_Init(const char *name, uint8_t open_print)
{
	int exit_flag=0;
    for (auto& cnt : cnt_list) {
        if (!cnt.name.compare(name)) {
            //如果有已经存在的名字 则返回
			exit_flag++;
        }
    }
    if (exit_flag>0) {
		return ;
	}
    TimeCnt_t *cnt = new TimeCnt_t;
    cnt->start_time=0;
    cnt->min_single_time=0;
    cnt->max_single_time=0;
    cnt->full_time=0;
    cnt->count=0;
    cnt->name   = name;
    cnt->open_print = open_print;
	cnt_list.push_back(*cnt);
}

void TimeCnt_Start(const char *name)
{
	for(auto &cnt:cnt_list)
	{
		if(!cnt.name.compare(name))
		{
			cnt.start_time = TimeCnt_GetTimeMs();
		}
	}
}

int64_t TimeCnt_End(const char* name)
{
	for(auto &cnt:cnt_list)
	{
		if(!cnt.name.compare(name))
		{
			int64_t now_time = TimeCnt_GetTimeMs() - cnt.start_time;
			if(now_time < 0)
			{
				printf("timecnt err!\r\n");
                ALGLogError << "timecnt err!" << "\r\n";
            }
			if(now_time > cnt.max_single_time)
			{
				cnt.max_single_time = now_time;
				if(!cnt.count)
				{
					cnt.min_single_time = now_time;
				}
			}
			if(now_time < cnt.min_single_time)
			{
				cnt.min_single_time = now_time;
			}
			cnt.full_time += now_time;
			cnt.count++;
			if(cnt.open_print)
			{
				//EVINFO(EVID_INFO, "tc: %d %s  %d ms", cnt.count, cnt.name.c_str(), now_time)
                ALGLogInfo << "tc: " << cnt.count << "  " << cnt.name.c_str() << "  "<< now_time;
            }
            if (!cnt.name.compare("nms")) {
                return now_time;
            }
        }

	}
}


void TimeCnt_PrintResult(void)
{
	printf("时间统计 >>>\r\n--------------------\r\n");
    ALGLogInfo << "时间统计 >>>\r\n--------------------\r\n";
    for (auto& cnt : cnt_list)
    {
		printf(" 名称    : %s\r\n 最小用时: %d ms (%.3f s)\r\n 最大用时: %d ms (%.3f s)\r\n 总计用时: %d ms (%.3f s)\r\n 总计次数: %d\r\n 平均用时: %.3f ms (%.3f s)\r\n--------------------\r\n",
			cnt.name.c_str(),
			cnt.min_single_time,
			((double)cnt.min_single_time)/1000.0,
			cnt.max_single_time,
			((double)cnt.max_single_time)/1000.0,
			cnt.full_time,
			((double)cnt.full_time)/1000.0,
			cnt.count,
			((double)cnt.full_time)/((double)cnt.count),
			(((double)cnt.full_time)/((double)cnt.count))/1000.0
		);

        ALGLogInfo << "名称    :  " << cnt.name.c_str() << " \r\n"
                   << "最小用时: " << cnt.min_single_time << "  ms " << "\r\n"
                   << "最大用时: " << cnt.max_single_time << "  ms " << "\r\n"
                   << "总计用时: " << cnt.full_time << "  ms " << "\r\n"
                   << "总计次数: " << cnt.count << "\r\n"
                   << "平均用时: " << ((double)cnt.full_time) / ((double)cnt.count) << "  ms " << "\r\n"
                   << " --------------------" << "\r\n";
    }
}
