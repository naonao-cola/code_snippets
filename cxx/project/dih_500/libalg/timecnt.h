/**
 * @FilePath     : /DIH-ALG/libalg/timecnt.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-05-07 10:09:24
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-07-21 13:50:22
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#ifndef _TIME_CNT_H_
#define _TIME_CNT_H_

#include <stdio.h>
#include <time.h>

void    TimeCnt_Init(const char* name, uint8_t open_print);
void    TimeCnt_Start(const char* name);
int64_t TimeCnt_End(const char* name);
void    TimeCnt_PrintResult(void);

#endif /* _TIME_CNT_H_ */
