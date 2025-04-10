#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <time.h>
#include <chrono>
#include <assert.h>
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"

#ifndef LOG_LEVEL
#define LOG_LEVEL "info"
#endif

#ifndef LOG_DIR
#define LOG_CONSOLE false  //控制台日志
#define LOG_DIR "./log"
#else
#define LOG_CONSOLE false
#endif

static inline int NowDateToInt()
{
    time_t now;
    time(&now);

    tm p;
#ifdef _WIN32
    localtime_s(&p, &now);
#else
    localtime_r(&now, &p);
#endif
    int now_date = (1900 + p.tm_year) * 10000 + (p.tm_mon + 1) * 100 + p.tm_mday;
    return now_date;
}

static inline int NowTimeToInt()
{
    time_t now;
    time(&now);
    tm p;
#ifdef _WIN32
    localtime_s(&p, &now);
#else
    localtime_r(&now, &p);
#endif

    int now_int = p.tm_hour * 10000 + p.tm_min * 100 + p.tm_sec;
    return now_int;
}

class XLogger
{
    public:
        static XLogger* getInstance()
        {
            static XLogger xlogger;
            return &xlogger;
        }

        std::shared_ptr<spdlog::logger> getLogger()
        {
            return m_logger;
        }

    private:
        // make constructor private to avoid outside instance
        XLogger()
        {
            // hardcode log path
            const std::string log_dir = LOG_DIR;
            const std::string logger_name_prefix = "baolong_";

            // decide print to console or log file
            bool console = LOG_CONSOLE;

            // decide the log level
            std::string level = LOG_LEVEL;

            try
            {
                int date = NowDateToInt();
                int time = NowTimeToInt();
                const std::string logger_name = logger_name_prefix + std::to_string(date) + "_" + std::to_string(time);

                if (console)
                    m_logger = spdlog::stdout_color_st(logger_name); // single thread console output faster
                else
                    m_logger = spdlog::create_async<spdlog::sinks::rotating_file_sink_mt>(logger_name,
                            log_dir + "/" + logger_name + ".log", 100 * 1024 * 1024, 10); // 10 * 100M

                // custom format
                m_logger->set_pattern("%Y-%m-%d %H:%M:%S.%f <thread %t> [%l] [%@] %v");

                if (level == "debug")
                {
                    m_logger->set_level(spdlog::level::debug);
                    m_logger->flush_on(spdlog::level::debug);
                }
                else if (level == "info")
                {
                    m_logger->set_level(spdlog::level::info);
                    m_logger->flush_on(spdlog::level::info);
                }
                else if (level == "warn")
                {
                    m_logger->set_level(spdlog::level::warn);
                    m_logger->flush_on(spdlog::level::warn);
                }
                else if (level == "error")
                {
                    m_logger->set_level(spdlog::level::err);
                    m_logger->flush_on(spdlog::level::err);
                }
                else if (level == "off")
                {
                    m_logger->set_level(spdlog::level::off);
                    m_logger->flush_on(spdlog::level::off);
                }
            }
            catch (const spdlog::spdlog_ex& ex)
            {
                std::cout << "Log initialization failed: " << ex.what() << std::endl;
            }
        }

        ~XLogger()
        {
            spdlog::drop_all(); // must do this
        }

        XLogger(const XLogger&) = delete;
        XLogger& operator=(const XLogger&) = delete;

    private:
        std::shared_ptr<spdlog::logger> m_logger;
};

// use embedded macro to support file and line number
#define LOG_DEBUG(...) SPDLOG_LOGGER_CALL(XLogger::getInstance()->getLogger().get(), spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...) SPDLOG_LOGGER_CALL(XLogger::getInstance()->getLogger().get(), spdlog::level::info, __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_CALL(XLogger::getInstance()->getLogger().get(), spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_CALL(XLogger::getInstance()->getLogger().get(), spdlog::level::err, __VA_ARGS__)

#define DBG_ASSERT(x) do {\
        if (!(x)) {\
            LOG_ERROR("Assert Error!");\
        }\
    } while(0)
