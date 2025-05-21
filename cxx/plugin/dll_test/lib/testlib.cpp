/**
 * @FilePath     : /dll_test/lib/testlib.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2022-09-22 11:23:20
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-05-17 14:33:24
 * @Copyright (c) 2025 by G, All Rights Reserved.
**/
#include "metaObject.h"
//g++ -Wall -shared -fPIC -o test.so testlib.cpp

#include <iostream>
#include <string>

#ifdef WIN32
#ifdef __cplusplus
#define EXPORT_DLL extern "C" __declspec(dllexport)
#else
#define EXPORT_DLL __declspec(dllexport)
#endif
#else
#define EXPORT_DLL extern "C"
#endif


class MetaObject_child : public MetaObject {
public:
	virtual int add(int a, int b) const
	{
		return a+b;
	};
	virtual void setVal(int _val){
		val = _val;
	};
	virtual int getVal() const{
		return val;
	};
private:
	int val;
};

// the class factories
EXPORT_DLL MetaObject* create() {
    return new MetaObject_child();
}

EXPORT_DLL void destroy(MetaObject* p) {
    delete p;
}

//the funs factories
EXPORT_DLL void testfunc01(int a)
{
	std::cout << "a="<<a<<std::endl;
}

EXPORT_DLL int testfunc02(int b)
{
	return b*b;
}