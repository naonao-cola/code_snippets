#ifndef METAOBJECT_H
#define METAOBJECT_H

class MetaObject
{
public:
	MetaObject(){};
	// MetaObject(const MetaObject& rhs);
	virtual ~MetaObject(){};

	//构成元素
	//元素使用

	virtual int add(int a, int b) const = 0;
	virtual void setVal(int _val) =0;
	virtual int getVal() const = 0;
};

//
typedef MetaObject* create_t();
typedef void destroy_t(MetaObject*);
//typedef _declspec(dllimport) void destroy_t(MetaObject*); //window

#endif
