#pragma once


#include <string>
#include <sstream>
#include <locale>
#include <wtypes.h>
#include <codecvt>

namespace AIInspectUtils {

std::string TCHARToString(const TCHAR* tcharArray)
{
    std::wstring_convert<std::codecvt_utf8_utf16<TCHAR>> converter;

    std::wstring wideString(tcharArray);

    return converter.to_bytes(wideString);
}

} // namespace AIInspectUtils