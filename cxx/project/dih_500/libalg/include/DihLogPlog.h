//
// Created by dyno on 2023/11/9.
//

#ifndef DIH_500WARE_DIHLOG_H
#define DIH_500WARE_DIHLOG_H
#include <iostream>
#include "plog/Log.h"
#define PLOG_VERBOSE_(instanceId)        PLOG_(instanceId, plog::verbose)
#define PLOG_DEBUG_(instanceId)          PLOG_(instanceId, plog::debug)
#define PLOG_INFO_(instanceId)           PLOG_(instanceId, plog::info)
#define PLOG_WARNING_(instanceId)        PLOG_(instanceId, plog::warning)
#define PLOG_ERROR_(instanceId)          PLOG_(instanceId, plog::error)
#define PLOG_FATAL_(instanceId)          PLOG_(instanceId, plog::fatal)
#define PLOG_NONE_(instanceId)           PLOG_(instanceId, plog::none)

#define  DIHLogInfo  PLOG_INFO_(DIH)
#define DIHLogWarning  PLOG_WARNING_(DIH)
#define DIHLogError  PLOG_ERROR_(DIH)
#define DIHLogFatal  PLOG_FATAL_(DIH)
#define DIHLogDebug PLOG_DEBUG_(DIH)
#define DIHLogVerbose  PLOG_VERBOSE_(DIH)
#define DIHLogNone  PLOG_NONE_(DIH)

#define ALGLogInfo  PLOG_INFO_(ALG)
#define ALGLogWarning  PLOG_WARNING_(ALG)
#define ALGLogError  PLOG_ERROR_(ALG)
#define ALGLogFatal  PLOG_FATAL_(ALG)
#define ALGLogDebug PLOG_DEBUG_(ALG)
#define ALGLogVerbose  PLOG_VERBOSE_(ALG)
#define ALGLogNone  PLOG_NONE_(ALG)

enum // Define log instances. Default is 0 and is omitted from this enum.
{
    DIH = 1,
    ALG
};
void dihPLogConfig(const std::string &logPath);
void algPLogConfig(const std::string &logPath);


#endif //DIH_500WARE_DIHLOG_H
