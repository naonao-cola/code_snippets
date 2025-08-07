
#include "lua_port.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define LUAPORT_VERSIONS        "LUA_PORT_V1.0.0"

/* portlib function */

/* portlib function */
static void stackDump(lua_State* L)
{
  int i = 0;
  int top = lua_gettop(L); //获取栈顶下标
  for (i = top; i <= top; --i)
  {
    if (i == 0)
    {
      break;
    }
    int t = lua_type(L, i); //获取栈上数据类型
    switch (t)
    {
    case LUA_TSTRING:
    {
      printf("|INDEX='%d','LUA_TSTRING=%s'|\n", i,lua_tostring(L, i));
    }
    break;
    case LUA_TBOOLEAN:
    {
      printf("|INDEX='%d','LUA_TBOOLEAN=%s'|\n", i,lua_toboolean(L, i) ? "true" : "false");
    }
    case LUA_TNUMBER:
    {
      printf("|INDEX='%d','LUA_TNUMBER=%g'|\n", i,lua_tonumber(L, i));
    }
    break;
    default:
    {
      printf("|INDEX='%d','DEFAULT=%s'|\n", i,lua_typename(L, t));
    }
    break;
    }
  }
}

static int get_versions(lua_State* L)
{
  lua_pushstring(L, LUAPORT_VERSIONS);
  return 1;
}


/*static int print_result(lua_State* L)
{
    char* addr = (char*)luaL_checkinteger(L, 1);
    char* result = luaL_checkstring(L, 2);  //检查函数的第2个参数是否是一个字符串并返回这个字符串。
    strcpy(addr, result);
    return 1;
}*/

static int print_result(lua_State* L)
{
  char* addr = (char*)luaL_checkstring(L, 1);
  char* result = (char*)luaL_checkstring(L, 2);  //检查函数的第2个参数是否是一个字符串并返回这个字符串。
  strcpy(addr, result);
  lua_pushstring(L, result);
  stackDump(L);
  printf("lua call  in print_result\n");
  return 1;
}

static const struct luaL_Reg portlib[] = {
    {"get_versions", get_versions},
    {"print_result", print_result},
    {NULL, NULL}
};


LUALIB_API int luaopen_portlib(lua_State* L)
{
  //    luaL_newmetatable(L, "portlib_matatable");
  luaL_newlib(L, portlib);
  return 1;
}

extern int luaopen_cjson(lua_State* l);


void LuaPort_RequireLib(lua_State* L)
{
  luaL_requiref(L, "portlib", luaopen_portlib, 1);
  luaL_requiref(L, "cjson", luaopen_cjson, 1);
}



int LuaPort_DoString(const char* lua_str)
{
  lua_State* L = luaL_newstate();
  int ret = 0;
  luaL_openlibs(L);
  LuaPort_RequireLib(L);
  ret = luaL_dostring(L, lua_str);
  if (ret)
  {
    printf("[FIA] err: lua executed failed!\n");
  }
  lua_close(L);
  return ret;
}




int LuaPort_DoFile(const char* file_name)
{
  lua_State* L = luaL_newstate();
  int ret = 0;
  luaL_openlibs(L);
  LuaPort_RequireLib(L);
  ret = luaL_dofile(L, file_name);
  if (ret)
  {
    printf("[FIA] err: lua executed failed!\n");
  }
  lua_close(L);
  return ret;
}

int LuaPort_CallMainDoFile(char* output, const char*file_name, char* input)
{
  lua_State* L = luaL_newstate();  //初始化lua虚拟机
  int ret = 0;
  // 打开所有的标准库
  luaL_openlibs(L);
  LuaPort_RequireLib(L);
  //ret = luaL_dofile(L, "F:\\Dynodx_Algorithm\\immunofluorescence\\immunofluorescence\\alg.lua");
  ret = luaL_dofile(L, file_name); //加载lua脚本并运行,如果没有错误，函数返回0；有错则返回1。
  printf("%d\n", ret);
  if (ret)
  {
    printf("[lua port] err: lua string executed failed!\n");
    lua_close(L);
    return ret;
  }
  lua_getglobal(L, "main");   //获取lua中全局方法main并把它放到lua的栈顶（lua_setglobal为设置全局变量）
  lua_pushinteger(L, (int)output);  //往lua栈里面压入两个参数（作为main的参数）
  lua_pushstring(L, input);
  lua_call(L, 2, 2);  //调用lua函数，2表示参数的个数，1为返回值的个数

  stackDump(L);
  printf("lua call over\n");
  strcpy(output, lua_tostring(L,-1));
  ret = (int)lua_tonumber(L, -2);  //从栈顶读取返回值
  lua_pop(L, 4);  //把返回值从栈顶删除

  stackDump(L);

  lua_close(L); //关闭lua虚拟机
  printf("lua close over\n");
  return ret;
}

int LuaPort_CallMain(char* output, char* lua_str, char* input)
{
  lua_State* L = luaL_newstate();
  int ret = 0;
  luaL_openlibs(L);
  LuaPort_RequireLib(L);
  ret = luaL_dostring(L, lua_str);
  if (ret)
  {
    printf("[lua port] err: lua string executed failed!\n");
    lua_close(L);
    return ret;
  }
  lua_getglobal(L, "main");
  lua_pushinteger(L, (int)output);
  lua_pushstring(L, input);
  lua_call(L, 2, 0);
  ret = (int)lua_tonumber(L, -1);
  lua_pop(L, 1);
  lua_close(L);
  return ret;
}

/*
sysStatus_t LuaPort_CallAlgo(DB_AlgoTypeDef *algo, LuaPort_TestDataTypeDef *data)
{
    char output[1024];
    char input[4196];

    LuaPort_CallMain(output, algo->formula, input);
}*/


void lua_main_newstate_callback(lua_State* L)
{
  LuaPort_RequireLib(L);
}



/*
// #include "cjson_port.h"
char lp_cjson_text[12000];
void test_luaport(void)
{

    uint32_t index;
    cJSONPort_AlgoInputTypeDef ipt;
    uint32_t raw_data[700];
    uint32_t tick;

    ipt.algo_id = 123;
    ipt.project_id = 456;
    ipt.sample_id = 789;

    tick = sysKernelGetTickCount();
    ipt.max_param_num = 7;
    for(index = 0; index < ipt.max_param_num; index++)
    {
        ipt.param[index] = rand();
    }

    ipt.raw_data = raw_data;
    ipt.max_data_num = 700;
    for(index = 0; index < ipt.max_data_num; index++)
    {
        ipt.raw_data[index] = rand()&0xFFFF;
    }

    cJSONPort_EncodeAlgoInput(lp_cjson_text, 12000, &ipt);
    rt_kprintf("\nLUA + CJSON - TEST -\n");
    printf("cjson text:\n%s\n", lp_cjson_text);

    char opt_text[64];
//    const char *lua = "lib = require('portlib')\n function main(opt,ipt)\n str = 'output success!' print(lib.get_versions())\n lib.print_result(opt, str)\n  print(ipt)\n return 1\nend\n ";
     char *lua = "lib = require('portlib')\n print(lib.get_versions())\n function main(opt,ipt)\n data = cjson.decode(ipt)\n   result = {algo_id = data[\"algo_id\"], sample_id = data[\"sample_id\"], d700 = data[\"raw_data\"][693]}\n  optext = cjson.encode(result)\n  lib.print_result(opt,optext) print(optext)  return 1\nend\n ";

    LuaPort_CallMain(opt_text, lua, lp_cjson_text);
    printf("opt: %s\n", opt_text);

}*/
//MSH_CMD_EXPORT(test_luaport, lua port test func);
