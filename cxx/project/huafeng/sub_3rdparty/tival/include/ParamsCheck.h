#include "CommonDefine.h"
#include "JsonHelper.h"


#define USE_TDSET 1

#if USE_TDSET
    #include "Tival.h"
    #define TYPE_INT 1
    #define TYPE_DOUBLE 2
    #define TYPE_STRING 4
#endif

namespace Tival {
    class ParamsCheck
    {
    public:
        // 判断int是否包含在指定集合范围内
        static bool ParamsCheck::InIntSet(const std::string& name, int val,  const std::set<int>& valSet, bool canAny=false) {
            if (canAny && val == -1) return true;
            if (valSet.count(val) == 0) {
                LOGE("Invalid param[{}]={}, valid values are:{}", name, val, Value2Str(valSet));
                return false;
            }
            return true;
        }

        // 判断字符串是否包含在指定集合范围内
        static bool ParamsCheck::InStringSet(const std::string& name, const std::string& val, const std::set<std::string>& valSet, bool canAny=false)
        {
            if (canAny && val == "") return true;
            if (valSet.count(val) == 0) {
                LOGE("Invalid param[{}]={}, valid values are:{}", name, val, Value2Str(valSet));
                return false;
            }
            return true;
        }

#if USE_TDSET
    static bool ParamsCheck::InIntSet(const std::string& name, const TDSet& val,  const std::set<int>& valSet, bool canAny=false)
    {
        if (!IsNumericValue(val)) {
            LOGE("Invalid param[{}]={}", name, Value2Str(val));
            return false;
        }
        return InIntSet(name, val.I(), valSet, canAny);
    }

    static bool ParamsCheck::InStringSet(const std::string& name, const TDSet& val, const std::set<std::string>& valSet, bool canAny=false)
    {
        if (!IsStringValue(val)) {
            LOGE("Invalid param[{}]={}", name, Value2Str(val));
            return false;
        }
        return InStringSet(name, std::string(val.S().Text()), valSet, canAny);
    }
#endif

        // 判断在（min~max）范围内
        // canAny: 目前用-1或者空字符“”串表示该参数不指定
        // mastEven: 是否必须是偶数
        template<typename T>
        static bool InRange(const std::string& name, const T& val, double minVal, double maxVal, bool canAny=false, bool mastEven=false)
        {
            if (canAny && IsAny(val)) {
                return true;
            }
            if (mastEven && !IsEvenValue(val)) {
                LOGE("Invalid param[{}]={}, min-max:{}-{} mastEven:{}", name, Value2Str(val), Value2Str(minVal), Value2Str(maxVal), mastEven);
                return false;
            }

            if (IsNumericValue(val)) {
                if (val < minVal || val > maxVal) {
                    LOGE("Invalid param[{}]={}, min-max:{}-{} mastEven:{}", name, Value2Str(val), Value2Str(minVal), Value2Str(maxVal), mastEven);
                    return false;
                } else {
                    return true;
                }
            }

            LOGE("Invalid param[{}]={}, min-max:{}-{} mastEven:{}", name, Value2Str(val), Value2Str(minVal), Value2Str(maxVal), mastEven);
            return false;
        }

        // 判断正数
        template<typename T>
        static bool Positive(const std::string& name, const T& val, bool canAny=false, bool mastEven=false)
        {
            if (canAny && IsAny(val)) {
                return true;
            }

            if (mastEven && !IsEvenValue(val)) {
                return false;
            }

            if (IsNumericValue(val)) {
                if (val > 0) return true;
                LOGE("Invalid param[{}], Unsupported value type:", name, typeid(val).name());
                return false;
            }

            LOGE("Invalid param[{}], Unsupported value type:", name, typeid(val).name());
            return false;
        }

        // 判断非负数
        template<typename T>
        static bool NonNegative(const std::string& name, const T& val, bool canAny=false, bool mastEven=false)
        {
            if (canAny && IsAny(val)) {
                return true;
            }

            if (mastEven && !IsEvenValue(val)) {
                return false;
            }

            if (IsNumericValue(val)) {
                return val >= 0;
            }
            LOGE("Invalid param[{}], Unsupported value type:", name, typeid(val).name());
            return false;
        }

        // 判断是否是偶数
        template<typename T>
        static bool IsEvenValue(const T& val)
        {
            if constexpr (std::is_same<T, int>::value || std::is_same<T, long>::value) {
                if (val % 2 == 0) {
                    return true;
                } else {
                    LOGW("Check even value fail, Not even value:{}", val.I());
                    return false;
                }
            }
#if USE_TDSET
            else if constexpr(std::is_same<T, TDSet>::value) {
                if (val.Type() == TYPE_INT) {
                    if (val.I() % 2 == 0) {
                        return true;
                    } else {
                        LOGW("Check even value fail, Not even value:{}", val.I());
                        return false;
                    }
                } else {
                    LOGW("Check even value fail, invalid TDset type:{}", val.Type());
                    return false;
                }
            }
#endif
           LOGW("Check even value fail, unkown value type:{}", typeid(val).name());
           return false;
        }

        // 判断是否是Any, 目前用-1或者空字符“”串表示该参数不指定
        template<typename T>
        static bool IsAny(const T& val) {
            if (IsNumericValue(val)) {
                return val == -1;
            }
            if (IsStringValue(val)) {
                if constexpr (std::is_same<T, TDSet>::value
                    || std::is_same<T, std::string>::value) {
                    return val == "";
                }
            }
            return false;
        }

        // 检测是否是数值类型
        template<typename T>
        static bool IsNumericValue(const T& val)
        {
            if constexpr (std::is_same<T, int>::value || std::is_same<T, long>::value ||
                std::is_same<T, float>::value || std::is_same<T, double>::value)
            {
                return true;
            }
#if USE_TDSET
            else if constexpr (std::is_same<T, TDSet>::value) {
                return val.Type() == TYPE_INT || val.Type() == TYPE_DOUBLE;
            }
#endif
            return false;
        }

        // 判断是否是字符串类型
        template<typename T>
        static bool IsStringValue(const T& val)
        {
            if constexpr (std::is_same<T, std::string>::value) {
                return true;
            }
#if USE_TDSET
            if constexpr(std::is_same<T, TDSet>::value) {
                return val.Type() == TYPE_STRING;
            }
#endif
            return false;
        }

        // 任意数据类型转字符串
        template<typename T>
        static std::string Value2Str(const T& val) {
            if constexpr (std::is_same<T, int>::value || std::is_same<T, long>::value ||
                std::is_same<T, float>::value || std::is_same<T, double>::value) {
                return std::to_string(val);
            } else if constexpr (std::is_same<T, std::string>::value) {
                return val;
            } else if constexpr (std::is_same<T, std::set<int>>::value ||
                std::is_same<T, std::set<std::string>>::value ||
                std::is_same<T, std::vector<int>>::value ||
                std::is_same<T, std::vector<std::string>>::value) {
                std::ostringstream oss;
                for (auto it = val.begin(); it != val.end(); ++it) {
                    oss<<*it;
                    if (it != val.end()) {
                        oss<<", ";
                    }
                }
                return oss.str();
            }
#if USE_TDSET
            else if (std::is_same<T, TDSet>::value) {
                if (IsNumericValue(val)) {
                    return std::to_string(val.D());
                } else if (IsStringValue(val)) {
                    return std::string(val.S().Text());
                }
            }
#endif
            return std::string("unkown value.");
        }
    };
}
