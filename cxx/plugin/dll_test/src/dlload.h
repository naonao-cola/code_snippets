#ifndef DLLOAD_H
#define DLLOAD_H

#if defined(WIN32)
#include <windows.h>
typedef HMODULE  MODULE_HANDLE;
#endif

#if defined(__linux__)
typedef void *  MODULE_HANDLE;
#endif

MODULE_HANDLE gdl_Open(const char *plname);
void gdl_Close(MODULE_HANDLE h);
void *gdl_GetProc(MODULE_HANDLE h, const char *pfname);
char* gdl_GetLastError();

#endif