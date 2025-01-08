#pragma once
#ifdef INSPECTLIB_EXPORTS
#define INSPECTLIB_API __declspec(dllexport)
#else
#define INSPECTLIB_API __declspec(dllimport)
#endif

class INSPECTLIB_API CAIInspectLib {
public:
	CAIInspectLib(void);
};

extern INSPECTLIB_API int nAIInspectLib;

INSPECTLIB_API int fnAIInspectLib(void);

