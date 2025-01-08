#pragma once

/**
 * @FilePath     : /connector_algo/src/custom/sub_3rdparty/tival/include/TivalCore.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-05-22 10:03:49
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
**/
#include "CommonDefine.h"
#include "FindLine.h"
#include "FindCircle.h"
#include "DataCode.h"
#include "ShapeBasedMatching.h"
#include "TAPI.h"


namespace Tival
{
    class ExportAPI TivalCore
    {
    public:
        static int Init(int solutionId);
    };
}