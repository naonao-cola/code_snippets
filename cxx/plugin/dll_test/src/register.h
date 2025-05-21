#ifndef REGISTER_H
#define REGISTER_H

#include <map>

#include "dlload.h"
#include "metaObject.h"

/*
class Argument
{
public:
	Argument(const char* name = 0, const void* data = 0);
	~Argument();
	void*	data() const;
	const char*	name() const;
};
*/

class RegisterM
{
public:
	enum MethodType { Method, Constructor };//函数类型
	enum Access { Private, Protected, Public };//访问方式
	enum CallType {Asynchronous,Synchronous};//函数调用方式
	enum SetType {SetVal,getVal};//属性值设置
public:
	RegisterM(){};
	~RegisterM(){};
	//注册类库
	int registerObject(const char* objectName, const char* conf);
	//注销类库
	bool unregisterObject(const char* objectName);
	//创建实例类
	create_t* getInstance(const char* objectName);
	//析构实例类
	destroy_t* rmInstance(const char* objectName);

	void* getFunc(const char* objectName,char* funcName);
	//属性
	// bool invokeP(MetaObject* object,const char* name,Argument returnVal,SetType stype);
	//函数调用
	// bool invokeF(MetaObject* object,const char* name,Argument returnVal,Argument argVal[]);
private:
	MODULE_HANDLE index ( const char * Name );
	// const char* Name ( MODULE_HANDLE index );
private:
	std::map<char *, MODULE_HANDLE> libmap;
};

#endif