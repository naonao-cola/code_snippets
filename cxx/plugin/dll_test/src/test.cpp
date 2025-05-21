#include "register.h"

#include <stdint.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {

	RegisterM *rm = new RegisterM();
	char libname[128] = "test";
	#ifdef WIN32
	#ifdef _DEBUG
	char libpath[128] = ".\\Debug\\testlibd.dll";
	#else
	char libpath[128] = ".\\Release\\testlib.dll";
	#endif
	#endif
	#ifdef __linux__
	char libpath[128] = "./linux/libtestlib.so";
	#endif
	int ret = rm->registerObject(libname,libpath);
	if (ret>0)
	{
		std::cout<<"registerObject success"<<std::endl;
		create_t* create_instance = rm->getInstance(libname);
		if (NULL!=create_instance)
		{
			std::cout<<"getInstance success"<<std::endl;
			MetaObject* _instance = create_instance();
			int _sum = _instance->add(7,8);
			std::cout<<"sum="<<_sum<<std::endl;
			_instance->setVal(15);
			std::cout<<"_instance->val="<<_instance->getVal()<<std::endl;

			void (*pa)(int a);
			*(void**)(&pa) = rm->getFunc((char*)libname,(char*)"testfunc01");
			pa(6);
			int (*fa)(int a);
			*(void**)(&fa) = rm->getFunc((char*)libname,(char*)"testfunc02");
			std::cout<<"fa(5)="<<fa(5)<<std::endl;

			destroy_t* destroy_instance = rm->rmInstance(libname);
			if (NULL!=destroy_instance)
			{
				std::cout<<"rmInstance success"<<std::endl;
				destroy_instance(_instance);
			}
		}else{
			std::cout<<"getInstance failed"<<std::endl;
		}
		bool re = rm->unregisterObject(libname);
		if (re)
		{
			std::cout<<"unregisterObject success"<<std::endl;
		}
	}
	return 0;
};