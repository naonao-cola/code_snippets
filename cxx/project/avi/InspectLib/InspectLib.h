#pragma once

#ifdef X64
#pragma comment(lib,"ClientSockDll_X64.lib")
#else
#pragma comment(lib,"ClientSockDll.lib")
#endif

#ifdef INSPECTLIB_EXPORTS
#define INSPECTLIB_API __declspec(dllexport)
#else
#define INSPECTLIB_API __declspec(dllimport)
#endif

class INSPECTLIB_API CInspectLib {
public:
	CInspectLib(void);
	// TODO: Add methods.
};

extern INSPECTLIB_API int nInspectLib;

INSPECTLIB_API int fnInspectLib(void);


