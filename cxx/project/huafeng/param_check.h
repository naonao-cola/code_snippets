
#include <sstream>
#include "../framework/Defines.h"
#include "../utils/Logger.h"

#if (defined(__GNUC__) && __GNUC__ >= 3) || defined(__clang__)
static inline bool (likely)(bool x) { return __builtin_expect((x), true); }
static inline bool (unlikely)(bool x) { return __builtin_expect((x), false); }
#else
static inline bool (likely)(bool x) { return x; }
static inline bool (unlikely)(bool x) { return x; }
#endif


#define TVALGO_FUNCTION_BEGIN \
		AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();\
		algo_result->status = RunStatus::OK;\


#define TVALGO_FUNCTION_END \
		return algo_result;\

#define TVALGO_FUNCTION_RETURN_ERROR_PARAM(info) \
		algo_result->status = RunStatus::WRONG_PARAM;\
		return algo_result;\


#define TVALGO_FUNCTION_LOG(info) \
		LOGD("algo log run file {}, line {} info {}", __FILE__, __LINE__,info);\



bool InIntSet(const std::string& name, int val, std::set<int> valSet, bool canAny);
bool InStringSet(const std::string& name, std::string val, std::set<std::string> valSet, bool canEmpty);
bool InDoubleRange(const std::string& name, double  val, double minVal, double maxVal, bool canAny);
bool InIntRange(const std::string& name, int val, int minVal, int maxVal, bool canAny);
bool PositiveInt(const std::string& name, int val, bool canAny);