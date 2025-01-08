#include "stdafx.h"
#include "AIAlgoExample.h"
#include "AIInspectDefine.h"

#include "AIInpectAlgoInterface.h"
#include "AIRuntime/logger.h"

#include <mutex>

std::mutex lockInspect;
AIInspectAlgoInterface* GetAIInspectAlgo(AIInspectAlgo algo) {
	if (algo == AIInspectAlgo::AIInspectAlgoDemo) {
		lockInspect.lock();
		auto obj = AIAlgoExample::get_instance();
		if (!obj->is_init) {
			obj->is_init = true;
			obj->Initialize(1, 0, 1, 0);
		}
		lockInspect.unlock();
		return AIAlgoExample::get_instance();
	} 
	else
	{
		LOG_INFOE("Unknown algorithm type: {}", algo);
		return nullptr;
	}
}


