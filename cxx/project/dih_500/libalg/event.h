#ifndef _EVENT_H_
#define _EVENT_H_

#define EV_PRINT_FILE_INFO          1
#define EV_BUILD_FOR_ANDROID        0

#include <stdio.h>
#include <stdint.h>
#if(EV_BUILD_FOR_ANDROID)
#include <android/log.h>
#endif /* EV_BUILD_FOR_ANDROID */

typedef enum EvID
{
    EVID_INFO = 0,
    EVID_WARN,
    EVID_ERR,
    
    EVID_SAMP_START,
    EVID_SAMP_CHL_START,
    EVID_SAMP_FIELD_START,
    EVID_SAMP_END,
    EVID_SAMP_STOP,
    EVID_SAMP_RESULT,

    EVID_FILE_READ,

    EVID_FOCUS_START,
    EVID_FOCUS_END,

    EVID_IMM_TEST_START,
    EVID_IMM_TEST_END,

    EVID_HGB_TEST_START,
    EVID_HGB_TEST_END,

    EVID_MTR_MOVE_FAIL,

    EVID_COM_INIT_FAIL,

}EvID_e;

#define EvFuncMaks_t        uint32_t
#define EvID_t            int

typedef enum EvFunc
{
    EVFUNC_CALLBACK = (1 << 0),
    EVFUNC_PRINT = (1 << 1),
    EVFUNC_PRINT_ANDROID = (1 << 2),
}EvFunc_e;


typedef enum EVType
{
    EVTYPE_INFO = 0,
    EVTYPE_WARNING,
    EVTYPE_ERROR,
    EVTYPE_FATAL
}EVType_e;


typedef void (*EventCallback_f)(EVType_e type, EvID_t id, const char *info, uint32_t timestamp, void *userdata);

void EV_Init(EvFuncMaks_t func_mask, EventCallback_f callback, void *userdata);
uint32_t Ev_GetTimestamp(void);
void EV_EventCall(uint32_t timestamp, EVType_e type, EvID_t id, const char *fmt, ...);

extern uint32_t global_func_mask;

#if(EV_PRINT_FILE_INFO)
#define EV_Printf(tsp, type, fmt, ...)                  printf("%s %d @ %s:%d %s > " fmt "\r\n", type, tsp, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define EV_Call(tsp, type, id, fmt, ...)                EV_EventCall(tsp, type, id, "%d @ %s:%d %s > " fmt "\r\n", tsp, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else /* EV_PRINT_FILE_INFO */
#define EV_Printf(tsp, fmt, ...)                        printf(fmt, ##__VA_ARGS__)
#define EV_Call(tsp, type, id, fmt, ...)                EV_EventCall(type, id, fmt, ##__VA_ARGS__)
#endif /* EV_PRINT_FILE_INFO */

#if(EV_BUILD_FOR_ANDROID)
#if(EV_PRINT_FILE_INFO)
#define EV_PrintAndroid(tsp, type, tag, fmt, ...)         __android_log_print(type, tag, "%d @ %s:%d %s > " fmt "\r\n", tsp,  __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else /* EV_PRINT_FILE_INFO */
#define EV_PrintAndroid(tsp, type, tag, fmt, ...)        __android_log_print(type, tag, fmt, ##__VA_ARGS__)
#endif /* EV_PRINT_FILE_INFO */
#else /* EV_BUILD_FOR_ANDROID */
//#define EV_PrintAndroid(tsp, type, tag, fmt, ...)        NOP()
#endif /* EV_BUILD_FOR_ANDROID */

#define EV_TAG              "dihlog"

#define EVTYPE_TXT_FATAL    "fatal"
#define EVTYPE_TXT_ERR      "err  "
#define EVTYPE_TXT_WARN     "warn "
#define EVTYPE_TXT_INFO     "info "
#define EVTYPE_TXT_DBG      "dbg  "
#define EVDEBUG(type, id, fmt, ...) \
if(global_func_mask){\
    uint32_t tsp = Ev_GetTimestamp();\
    if(global_func_mask & EVFUNC_PRINT){\
        switch(type){\
            case EVTYPE_FATAL:EV_Printf(tsp, EVTYPE_TXT_FATAL, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_ERROR:EV_Printf(tsp, EVTYPE_TXT_ERR, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_WARNING:EV_Printf(tsp, EVTYPE_TXT_WARN, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_INFO:EV_Printf(tsp, EVTYPE_TXT_INFO, fmt, ##__VA_ARGS__);break;\
            default:EV_Printf(tsp, EVTYPE_TXT_DBG, fmt, ##__VA_ARGS__);break;\
        }\
    }\
}
/*if(global_func_mask){\
    uint32_t tsp = Ev_GetTimestamp();\
    if(global_func_mask & EVFUNC_PRINT){\
        switch(type){\
            case EVTYPE_FATAL:EV_Printf(tsp, EVTYPE_TXT_FATAL, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_ERROR:EV_Printf(tsp, EVTYPE_TXT_ERR, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_WARNING:EV_Printf(tsp, EVTYPE_TXT_WARN, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_INFO:EV_Printf(tsp, EVTYPE_TXT_INFO, fmt, ##__VA_ARGS__);break;\
            default:EV_Printf(tsp, EVTYPE_TXT_DBG, fmt, ##__VA_ARGS__);break;\
        }\
    }\
    if(global_func_mask & EVFUNC_PRINT_ANDROID){\
        switch(type){\
            case EVTYPE_FATAL:\
            case EVTYPE_ERROR:EV_PrintAndroid(tsp, ANDROID_LOG_ERROR, EV_TAG, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_WARNING:EV_PrintAndroid(tsp, ANDROID_LOG_WARN, EV_TAG, fmt, ##__VA_ARGS__);break;\
            case EVTYPE_INFO:EV_PrintAndroid(tsp, ANDROID_LOG_INFO, EV_TAG, fmt, ##__VA_ARGS__);break;\
            default:EV_PrintAndroid(tsp, ANDROID_LOG_DEBUG, EV_TAG, fmt, ##__VA_ARGS__);break;\
        }\
    }\
    if(global_func_mask & EVFUNC_CALLBACK){\
        EV_Call(tsp, type, id, fmt, ##__VA_ARGS__);\
    }\
}*/
#define EVFATAL(id, fmt, ...)             EVDEBUG(EVTYPE_FATAL, id, fmt, ##__VA_ARGS__)
#define EVERROR(id, fmt, ...)             EVDEBUG(EVTYPE_ERROR, id, fmt, ##__VA_ARGS__)
#define EVWARN(id, fmt, ...)              EVDEBUG(EVTYPE_WARNING, id, fmt, ##__VA_ARGS__)
#define EVINFO(id, fmt, ...)              EVDEBUG(EVTYPE_INFO, id, fmt, ##__VA_ARGS__)




#endif /* _EVENT_MONITOR_H_ */