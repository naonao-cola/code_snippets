#pragma once
/*

*/
#ifndef _LUA_PORT_H_
#define _LUA_PORT_H_


// #include "system_layer_link.h"
// #include "rtthread.h"
// #include "dfs_posix.h"
// #include "db_port.h"


/*#include "..\immunofluorescence-dih200\lua-5.3.4\lua.h"
#include "..\immunofluorescence-dih200\lua-5.3.4\lauxlib.h"
#include "..\immunofluorescence-dih200\lua-5.3.4\lualib.h"*/
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include <stdint.h>

#define LUA_PORT_MAX_PARAM_NUM          10

/*
typedef struct
{
    uint32_t project_id;
    uint32_t sample_id;
    uint32_t algo_id;
    uint32_t max_param_num;
    uint32_t *raw_data;
    uint32_t max_data_num;
    float param[LUA_PORT_MAX_PARAM_NUM];
    float t_c_ratio;
    float result;
}LuaPort_TestDataTypeDef;*/

#ifdef __cplusplus
    extern "C" {
#endif

  int lua_port_call_main(char* output, const char* lua_str, const char* input);

  // int lua_port_call_algo(DB_AlgoTypeDef *algo, LuaPort_TestDataTypeDef *data);
  int LuaPort_CallMain(char* output, char* lua_str, char* input);
  int LuaPort_CallMainDoFile(char* output, const char* file_name, char* input);

  void test_luaport(void);

#ifdef __cplusplus
};
#endif



#endif