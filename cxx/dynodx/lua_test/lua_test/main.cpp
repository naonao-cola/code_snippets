/**
 * @FilePath     : /test/lua_test/main.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-05-21 13:57:05
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-05-21 15:14:10
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/

#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

extern "C" {
#include "lauxlib.h"
#include "lua.h"
#include "lualib.h"
}

static void printLuaStack(lua_State* L)
{
    int nIndex;
    int nType;
    printf("================栈顶================\n");
    printf("   索引  类型          值\n");
    std::string value(250, '\0');
    for (nIndex = lua_gettop(L); nIndex > 0; --nIndex) {
        if (nIndex == 0) {
            break;
        }
        nType = lua_type(L, nIndex);
        switch (lua_type(L, nIndex)) {
        case LUA_TSTRING:
        {
            printf("   (%d)  %s         %s\n", nIndex, lua_typename(L, nType), lua_tostring(L, nIndex));
        } break;
        case LUA_TNUMBER:
        {
            printf("   (%d)  %s         %g\n", nIndex, lua_typename(L, nType), lua_tonumber(L, nIndex));
        } break;
        case LUA_TBOOLEAN:
        {
            snprintf(value.data(), value.size(), "%d", lua_tonumber(L, nIndex));
            printf("   (%d)  %s         %s\n", nIndex, lua_typename(L, nType), lua_toboolean(L, nIndex) ? "true" : "false");
        } break;
        default:
            value = "?";
            printf("   (%d)  %s         %s\n", nIndex, lua_typename(L, nType), lua_typename(L, nType));
            break;
        }
    }
    printf("================栈底================\n");
}

void traverse_table(lua_State* L, int index)
{
    printf("================ 进入迭代 ================\n");
    // 为了使被操作的表在栈顶，我们需要作一些操作来确保
    bool ontop = index == -1;

    // 如果不在栈顶上，就压入此表的一份引用到栈顶
    if (!ontop)
        lua_pushvalue(L, index);

    // 现在的栈是这样的：-1 => table

    // 好了，现在表已经在栈顶上了，像前面操作 next 的方式一样操作
    // 1. 先压入 nil 调用 next 开始迭代
    lua_pushnil(L);
    // 现在的栈是这样的：-1 => nil, -2 => table

    // 2. 现在循环调用 next 来获取下一组键值
    // 第1个参数待查询的键，第2个参数是表的索引
    // 如果没有可迭代的元素了，lua_next 会返回 0


    while (lua_next(L, -2)) {
        // 现在的栈是这样的：-1 => value, -2 => key, -3 => table
        // 3. 值已经拿到了，可以使用了。但还有一点需要注意：
        //    如果 key 不是字符串的话，不能使用 lua_tostring，原因请看
        //    官方文档：http://www.lua.org/manual/5.3/manual.html#lua_next
        //    如果真的想用 lua_tostring 的话，可以先压入一份 key 的拷贝。
        // 我这里为了简单起见，使用了字符串，就不需要再考虑了。
        const char* key = lua_tostring(L, -2);
        const char* val = lua_tostring(L, -1);
        printf("%s => %s\n", key, val);

        // 现在完后，干掉 value 保留 key，为下一次迭代作准备
        lua_pop(L, 1);
        // 现在的栈是这样的：-1 => key, -2 => table
        // 已经回到开始遍历时的栈帧了，可以下一次遍历了
    }

    // lua_next 不止会返回0，而且还会帮我们把最后一个 key 给弹出去，只保留最初那个表
    // 现在的栈是这样的：-1 => table

    // 好了，让一切回到最初
    if (!ontop)
        lua_pop(L, 1);

    printf("================ 退出迭代 ================\n");
}

/*
https://blog.csdn.net/qq826364410/article/details/88624824

*/

int main()
{
    // 1.创建Lua状态
    lua_State* L = luaL_newstate();
    if (L == NULL) {
        return -1;
    }
    luaL_openlibs(L);
    // 2.加载Lua文件
    int bRet = luaL_loadfile(L, "/home/naonao/demo/cxx/test/lua_test/hello.lua");
    if (bRet) {
        cout << "load file error" << endl;
        return -1;
    }

    // 3.运行Lua文件
    bRet = lua_pcall(L, 0, 0, 0);
    if (bRet) {
        cout << "pcall error" << endl;
        return -1;
    }

    // 4.读取变量
    // 1.把 str 压栈 2.由lua去寻找全局变量str的值，并将str的值返回栈顶（替换str）
    // 如果存在相同命名的其他变量、table或函数，就会报错（读取位置发生访问冲突）
    lua_getglobal(L, "str");
    string str = lua_tostring(L, -1);
    cout << "str = " << str.c_str() << endl;   // str = I am so cool~

    // 5.读取table，把table压栈
    lua_getglobal(L, "tbl");
    //遍历
    traverse_table(L, -1);


    // 1.把name压入栈中，2.由lua去寻找table中name键的值，并将键值返回栈顶（替换name）
    // 相当于lua_pushstring(L, "name") + lua_gettable(L, -2)执行结果是一样的


    lua_getfield(L, -1, "name");
    //  把name压入栈中
    //  弹出栈上的name，并从表中找到name的键值，把结果放在栈上相同的位置
    // lua_gettable 从栈中获取一个表（位于 index 位置），然后从栈顶获取一个键（key），并使用这个键从表中获取值。

    // lua_pushstring(L, "name");
    //lua_gettable(L, -2);
    //  printf("================ 第二种方式 ================\n");
    //  printLuaStack(L);

     str = lua_tostring(L, -1);
     cout << "tbl:name = " << str.c_str() << endl;   // tbl:name = shun

    // lua_getfield 从栈中获取一个表（位于 index 位置），然后使用指定的字符串键 k 从表中获取值。
     lua_getfield(L, -2, "id");
     int id = lua_tonumber(L, -1);
     cout << "tbl:name = " << str.c_str() << endl;
     cout << "tbl:id = " << id << endl;

     // 6.读取函数
     lua_getglobal(L, "add");   // 获取函数，压入栈中
     lua_pushnumber(L, 10);     // 压入第一个参数
     lua_pushnumber(L, 20);     // 压入第二个参数
     // 栈过程：参数出栈->保存参数->参数出栈->保存参数->函数出栈->调用函数->返回结果入栈
     // 调用函数，调用完成以后，会将返回值压入栈中，2表示参数个数，1表示返回结果个数。
     int iRet = lua_pcall(L, 2, 1, 0);   // 调用函数，调用完成以后，会将返回值压入栈中，2表示参数个数，1表示返回结果个数。
     if (iRet)                           // 调用出错
     {
         const char* pErrorMsg = lua_tostring(L, -1);
         cout << pErrorMsg << endl;
         lua_close(L);
         return -1;
    }
    if (lua_isnumber(L, -1))   // 取值输出
    {
        double fValue = lua_tonumber(L, -1);
        cout << "Result is " << fValue << endl;
    }

    printLuaStack(L);
    // 至此，栈中的情况是：
    //=================== 栈顶 ===================
    // 索引    类型      值
    // 5或-1   int       30
    // 4或-2   int       20114442
    // 3或-3   string    shun
    // 2或-4   table	 tbl
    // 1或-5   string	 I am so cool~
    //=================== 栈底 ===================


    lua_pushstring(L, "Master");
    printLuaStack(L);
    // 会将"Master"值出栈，保存值，找到到table的name键，如果键存在，存储到name键中
    lua_setfield(L, 2, "name");
    // 读取
    lua_getfield(L, 2, "name");
    str = lua_tostring(L, -1);

    printf("================ 查看栈================\n");
    printLuaStack(L);
    cout << "tbl:name = " << str.c_str() << endl;

    // 创建新的table
    lua_newtable(L);
    lua_pushstring(L, "A New Girlfriend");
    lua_setfield(L, -2, "name");
    // 读取
    lua_getfield(L, -1, "name");
    str = lua_tostring(L, -1);
    cout << "newtbl:name = " << str.c_str() << endl;

    printf("================ 查看栈2 ================\n");
    printLuaStack(L);

    // 7.关闭state
    //  销毁指定 Lua 状态机中的所有对象， 并且释放状态机中使用的所有动态内存。
    //  （如果有垃圾收集相关的元方法的话，会调用它们）


    //  7.关闭state
    lua_close(L);
    return 0;
}
