/**
 * @FilePath     : /test/src/main.cpp
 * @Description  :
 * @Author       : error: git config user.name & please set dead value or install git
 * @Date         : 2025-04-23 16:02:58
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-05-23 09:56:11
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/

#include "task.h"




#include <iostream>
#include <stdio.h>
#include <string>

#ifdef WIN32
#include <objbase.h>
#else
#include <uuid/uuid.h>
#endif

#define MAX_LEN 128


/*
**@brief: get windows guid or linux uuid
**@return: string type windows guid or linux uuid
*/
std::string GetGuid()
{
    using namespace std;
    char szuuid[MAX_LEN] = {0};
#ifdef WIN32
    GUID guid;
    CoCreateGuid(&guid);
    _snprintf_s(szuuid,
                sizeof(szuuid),
                "{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
                guid.Data1,
                guid.Data2,
                guid.Data3,
                guid.Data4[0],
                guid.Data4[1],
                guid.Data4[2],
                guid.Data4[3],
                guid.Data4[4],
                guid.Data4[5],
                guid.Data4[6],
                guid.Data4[7]);
#else
    uuid_t uuid;
    uuid_generate(uuid);
    uuid_unparse(uuid, szuuid);
#endif

    return std::string(szuuid);
}



int main()
{


    //TaskFlow_TestExample();


    std::string strGuid = GetGuid();
    std::cout << strGuid.c_str() << std::endl;
   
    return 0;
}
