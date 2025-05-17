#ifndef _TIME_CNT_H_
#define _TIME_CNT_H_

#include <stdio.h>
#include <time.h>

void TimeCnt_Init(const char *name, uint8_t open_print);
void TimeCnt_Start(const char *name);
void TimeCnt_End(const char *name);
void TimeCnt_PrintResult(void);

#endif /* _TIME_CNT_H_ */