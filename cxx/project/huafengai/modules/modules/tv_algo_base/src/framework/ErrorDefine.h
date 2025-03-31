#pragma once

enum class ErrorCode : int {
   OK = 0,                     // 正常运行
    ABNORMAL_IMAGE,             // 图片异常
    ABNORMAL_IMG_BRIGHTNESS,  // 图片亮度异常
    ABNORMAL_ANGLE,             // 角度异常
    NOT_FOUND_TARGET,           // 目标查找失败
    OUT_OF_MEMORY,              // 内存不足
    WRONG_PARAM,                // 参数异常
    NOT_READY,                  // 未准备好（未初始化）
    TIME_OUT,                   // 检测超时
    DUPLICATE_ALGO_NAME,        // 算法名重复
    INVALID_IMG_DATA,           // 图片buffer异常，转换cvMat失败
    QUEUE_OVERFLOW,             // 队列溢出
    WRONG_STATE,                // 状态错误
    LICENSE_ERROR,              // 授权检查失败
    AI_INIT_FAIL,               // AI 初始化Runtime失败
    AI_LOAD_MODEL_FAIL,         // AI 加载模型失败
    AI_INFER_FAIL,              // AI 推理失败
    INVALID_HANDLE,             // 非法句柄
    UNKNOWN_ERROR = 999         // 未知错误
};

#define CHECK_ERRORCODE_RETURN(err_code) \
    if (err_code != ErrorCode::OK)       \
        return err_code;