#include "register.h"

#include <stdio.h>
//#include <unistd.h>
#include <iostream>

int RegisterM::registerObject(const char* objectName, const char* conf)
{
	MODULE_HANDLE load_handle= gdl_Open(conf);
	if (!load_handle) {
        std::cerr << "Cannot load library: " << conf 
			<< " Error:" << gdl_GetLastError() << '\n';
        return -1;
    }
	libmap[const_cast<char*>(objectName)]=load_handle;
	return 1;
}

bool RegisterM::unregisterObject(const char* objectName)
{
	std::map<char *, MODULE_HANDLE>::iterator it=libmap.find(const_cast<char*>(objectName));
	if (it!=libmap.end())
	{
		gdl_Close(it->second);
		libmap.erase(it);
		return true;
	}
	return false;
}

create_t* RegisterM::getInstance(const char* objectName)
{
	MODULE_HANDLE _handle = index(objectName);
	if (NULL!=_handle)
	{
		create_t* create_instance = (create_t*) gdl_GetProc(_handle, "create");
	    const char* dlsym_error = gdl_GetLastError();
	    if (dlsym_error) {
	        std::cerr << "Cannot load symbol create error: " << dlsym_error << '\n';
	        return NULL;
	    }else{
	    	return create_instance;
	    }
	}
	return NULL;
}

destroy_t* RegisterM::rmInstance(const char* objectName)
{
	MODULE_HANDLE _handle = index(objectName);
	if (NULL!=_handle)
	{
		destroy_t* destroy_instance = (destroy_t*) gdl_GetProc(_handle, "destroy");
	    const char* dlsym_error = gdl_GetLastError();
	    if (dlsym_error) {
	        std::cerr << "Cannot load symbol create: " << dlsym_error << '\n';
	        return NULL;
	    }else{
	    	return destroy_instance;
	    }
	}
	return NULL;
}

void* RegisterM::getFunc(const char* objectName,char* funcName)
{
	MODULE_HANDLE _handle = index(objectName);
	if (NULL!=_handle)
	{
	    void* ret = gdl_GetProc(_handle,funcName);
	    const char* dlsym_error = gdl_GetLastError();
	    if (dlsym_error) {
	        std::cerr << "Cannot load symbol create: " << dlsym_error << '\n';
	        return NULL;
	    }else{
	    	return ret;
	    }
	}
	return NULL;
}

MODULE_HANDLE RegisterM::index ( const char * Name )
{
	std::map<char *, MODULE_HANDLE>::iterator it=libmap.find(const_cast<char*>(Name));
	if (it!=libmap.end())
	{
		return it->second;
	}else{
		std::cerr << "Cannot find library: " << Name << '\n';	
	}
	return NULL;
}