#include "param_check.h"
bool InIntSet(const std::string& name, int val, std::set<int> valSet, bool canAny)
{
    if (canAny && val == -1) return true;
    if (valSet.count(val) == 0) {
        std::ostringstream validVals;
        for (auto it = valSet.begin(); it != valSet.end(); ++it) {
            validVals << *it;
            if (it != valSet.end()) {
                validVals << ", ";
            }
        }
        LOGE("Invalid param[{}]={}, valid values are:{}", name, val, validVals.str());
        return false;
    }
    return true;
}

bool InStringSet(const std::string& name, std::string val, std::set<std::string> valSet, bool canEmpty)
{
    if (canEmpty && val == "") return true;
    if (valSet.count(std::string(val)) == 0) {
        std::ostringstream validVals;
        for (auto it = valSet.begin(); it != valSet.end(); ++it) {
            validVals << *it;
            if (it != valSet.end()) {
                validVals << ", ";
            }
        }
        LOGE("Invalid param[{}]={}, valid values are:{}", name, val, validVals.str());
        return false;
    }
    return true;
}

bool InDoubleRange(const std::string& name, double val, double minVal, double maxVal, bool canAny)
{
    if (canAny && val == -1) return true;
    if (val < minVal || val > maxVal) {
        LOGE("Invalid param[{}]={}, min-max:{}-{}", name, val, minVal, maxVal);
        return false;
    }
    return true;
}

bool InIntRange(const std::string& name, int val, int minVal, int maxVal, bool canAny)
{
    if (canAny && val == -1) return true;
    if (val < minVal || val > maxVal) {
        LOGE("Invalid param[{}]={}, min-max:{}-{}", name, val, minVal, maxVal);
        return false;
    }
    return true;
}

bool PositiveInt(const std::string& name, int val, bool canAny)
{
    if (canAny && val == -1) return true;
    if (val <= 0) {
        LOGE("Invalid param[{}]={}", name, val);
        return false;
    }
    return true;
}
