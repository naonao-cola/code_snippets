#pragma once
#include "CommonDefine.h"

namespace Tival
{
    const int RS_OK = 0;
    const int RS_NOT_FOUND = 1;
    const int RS_WRONG_PARAM = 2;
    const int RS_NOT_READY = 3;
    const int RS_TIMEOUT = 4;
    const int RS_NO_RESULT = 5;
    const int RS_OP_FAIL = 6;
    const int RS_OP_NO_LICENSE = 7;

    const int RS_UNKOWN_ERR = 99;

    class ResutBase
    {
    public:
        int status = RS_OK;
        int num_instances = 0;
        virtual json ToJson() const = 0;
    };
};