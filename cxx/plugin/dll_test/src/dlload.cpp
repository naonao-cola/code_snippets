#include "dlload.h"

#if defined(WIN32)
#include <windows.h>
#endif

#if defined(__linux__)
#include <dlfcn.h>
#endif

MODULE_HANDLE gdl_Open(const char *plname)
{
#if defined(WIN32)
    return LoadLibraryA (plname);
#endif

#if defined(__linux__)
   return dlopen( plname, RTLD_NOW|RTLD_GLOBAL);
#endif
};

void gdl_Close(MODULE_HANDLE h)
{
    if(h)
    {
#if defined(WIN32)
        FreeLibrary(h);
#endif
#if defined(__linux__)
        dlclose (h);
#endif
    }
};

void *gdl_GetProc(MODULE_HANDLE h, const char *pfname)
{
   if(h)
   {
#if defined(WIN32)
        return (void *)GetProcAddress(h, pfname);
#endif

#if defined(__linux__)
        return dlsym(h,pfname);
#endif
    }
    return 0;
};

char* gdl_GetLastError()
{
#if defined(WIN32)
    return (char*)::GetLastError();
#endif

#if defined(__linux__)
    return dlerror();
#endif
}