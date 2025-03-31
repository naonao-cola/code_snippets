#pragma once

#include <windows.h>
#include <locale.h>
#include <iostream>

class StringConvert
{
public:
    static wchar_t *AnsiToUnicode(const char *lpszStr)
    {
        wchar_t *lpUnicode;
        int nLen;

        if (NULL == lpszStr)
            return NULL;

        nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, NULL, 0);
        if (0 == nLen)
            return NULL;

        lpUnicode = new wchar_t[nLen + 1];
        if (NULL == lpUnicode)
            return NULL;

        memset(lpUnicode, 0, sizeof(wchar_t) * (nLen + 1));
        nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, lpUnicode, nLen);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            return NULL;
        }

        return lpUnicode;
    }

    static char *UnicodeToAnsi(const wchar_t *lpszStr)
    {
        char *lpAnsi;
        int nLen;

        if (NULL == lpszStr)
            return NULL;

        nLen = WideCharToMultiByte(CP_ACP, 0, lpszStr, -1, NULL, 0, NULL, NULL);
        if (0 == nLen)
            return NULL;

        lpAnsi = new char[nLen + 1];
        if (NULL == lpAnsi)
            return NULL;

        memset(lpAnsi, 0, nLen + 1);
        nLen = WideCharToMultiByte(CP_ACP, 0, lpszStr, -1, lpAnsi, nLen, NULL, NULL);
        if (0 == nLen)
        {
            delete[] lpAnsi;
            return NULL;
        }

        return lpAnsi;
    }

    static char *AnsiToUtf8(const char *lpszStr)
    {
        wchar_t *lpUnicode;
        char *lpUtf8;
        int nLen;

        if (NULL == lpszStr)
            return NULL;

        nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, NULL, NULL);
        if (0 == nLen)
            return NULL;

        lpUnicode = new wchar_t[nLen + 1];
        if (NULL == lpUnicode)
            return NULL;

        memset(lpUnicode, 0, sizeof(wchar_t) * (nLen + 1));
        nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, lpUnicode, nLen);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            return NULL;
        }

        nLen = WideCharToMultiByte(CP_UTF8, 0, lpUnicode, -1, NULL, 0, NULL, NULL);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            return NULL;
        }

        lpUtf8 = new char[nLen + 1];
        if (NULL == lpUtf8)
        {
            delete[] lpUnicode;
            return NULL;
        }

        memset(lpUtf8, 0, nLen + 1);
        nLen = WideCharToMultiByte(CP_UTF8, 0, lpUnicode, -1, lpUtf8, nLen, NULL, NULL);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            delete[] lpUtf8;
            return NULL;
        }

        delete[] lpUnicode;

        return lpUtf8;
    }

    static char *Utf8ToAnsi(const char *lpszStr)
    {
        wchar_t *lpUnicode;
        char *lpAnsi;
        int nLen;

        if (NULL == lpszStr)
            return NULL;

        nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, NULL, NULL);
        if (0 == nLen)
            return NULL;

        lpUnicode = new wchar_t[nLen + 1];
        if (NULL == lpUnicode)
            return NULL;

        memset(lpUnicode, 0, sizeof(wchar_t) * (nLen + 1));
        nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, lpUnicode, nLen);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            return NULL;
        }

        nLen = WideCharToMultiByte(CP_ACP, 0, lpUnicode, -1, NULL, 0, NULL, NULL);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            return NULL;
        }

        lpAnsi = new char[nLen + 1];
        if (NULL == lpAnsi)
        {
            delete[] lpUnicode;
            return NULL;
        }

        memset(lpAnsi, 0, nLen + 1);
        nLen = WideCharToMultiByte(CP_ACP, 0, lpUnicode, -1, lpAnsi, nLen, NULL, NULL);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            delete[] lpAnsi;
            return NULL;
        }

        delete[] lpUnicode;

        return lpAnsi;
    }

    static char *UnicodeToUtf8(const wchar_t *lpszStr)
    {
        char *lpUtf8;
        int nLen;

        if (NULL == lpszStr)
            return NULL;

        nLen = WideCharToMultiByte(CP_UTF8, 0, lpszStr, -1, NULL, 0, NULL, NULL);
        if (0 == nLen)
            return NULL;

        lpUtf8 = new char[nLen + 1];
        if (NULL == lpUtf8)
            return NULL;

        memset(lpUtf8, 0, nLen + 1);
        nLen = WideCharToMultiByte(CP_UTF8, 0, lpszStr, -1, lpUtf8, nLen, NULL, NULL);
        if (0 == nLen)
        {
            delete[] lpUtf8;
            return NULL;
        }

        return lpUtf8;
    }

    static wchar_t *Utf8ToUnicode(const char *lpszStr)
    {
        wchar_t *lpUnicode;
        int nLen;

        if (NULL == lpszStr)
            return NULL;

        nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, NULL, 0);
        if (0 == nLen)
            return NULL;

        lpUnicode = new wchar_t[nLen + 1];
        if (NULL == lpUnicode)
            return NULL;

        memset(lpUnicode, 0, sizeof(wchar_t) * (nLen + 1));
        nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, lpUnicode, nLen);
        if (0 == nLen)
        {
            delete[] lpUnicode;
            return NULL;
        }

        return lpUnicode;
    }

    static bool AnsiToUnicode(const char *lpszAnsi, wchar_t *lpszUnicode, int nLen)
    {
        int nRet = MultiByteToWideChar(CP_ACP, 0, lpszAnsi, -1, lpszUnicode, nLen);
        return (0 == nRet) ? FALSE : TRUE;
    }

    static bool UnicodeToAnsi(const wchar_t *lpszUnicode, char *lpszAnsi, int nLen)
    {
        int nRet = WideCharToMultiByte(CP_ACP, 0, lpszUnicode, -1, lpszAnsi, nLen, NULL, NULL);
        return (0 == nRet) ? FALSE : TRUE;
    }

    static bool AnsiToUtf8(const char *lpszAnsi, char *lpszUtf8, int nLen)
    {
        wchar_t *lpszUnicode = AnsiToUnicode(lpszAnsi);
        if (NULL == lpszUnicode)
            return FALSE;

        int nRet = UnicodeToUtf8(lpszUnicode, lpszUtf8, nLen);

        delete[] lpszUnicode;

        return (0 == nRet) ? FALSE : TRUE;
    }

    static bool Utf8ToAnsi(const char *lpszUtf8, char *lpszAnsi, int nLen)
    {
        wchar_t *lpszUnicode = Utf8ToUnicode(lpszUtf8);
        if (NULL == lpszUnicode)
            return FALSE;

        int nRet = UnicodeToAnsi(lpszUnicode, lpszAnsi, nLen);

        delete[] lpszUnicode;

        return (0 == nRet) ? FALSE : TRUE;
    }

    static bool UnicodeToUtf8(const wchar_t *lpszUnicode, char *lpszUtf8, int nLen)
    {
        int nRet = WideCharToMultiByte(CP_UTF8, 0, lpszUnicode, -1, lpszUtf8, nLen, NULL, NULL);
        return (0 == nRet) ? FALSE : TRUE;
    }

    static bool Utf8ToUnicode(const char *lpszUtf8, wchar_t *lpszUnicode, int nLen)
    {
        int nRet = MultiByteToWideChar(CP_UTF8, 0, lpszUtf8, -1, lpszUnicode, nLen);
        return (0 == nRet) ? FALSE : TRUE;
    }

    static std::wstring AnsiToUnicode(const std::string &strAnsi)
    {
        std::wstring strUnicode;

        wchar_t *lpszUnicode = AnsiToUnicode(strAnsi.c_str());
        if (lpszUnicode != NULL)
        {
            strUnicode = lpszUnicode;
            delete[] lpszUnicode;
        }

        return strUnicode;
    }
    static std::string UnicodeToAnsi(const std::wstring &strUnicode)
    {
        std::string strAnsi;

        char *lpszAnsi = UnicodeToAnsi(strUnicode.c_str());
        if (lpszAnsi != NULL)
        {
            strAnsi = lpszAnsi;
            delete[] lpszAnsi;
        }

        return strAnsi;
    }

    static std::string AnsiToUtf8(const std::string &strAnsi)
    {
        std::string strUtf8;

        char *lpszUtf8 = AnsiToUtf8(strAnsi.c_str());
        if (lpszUtf8 != NULL)
        {
            strUtf8 = lpszUtf8;
            delete[] lpszUtf8;
        }

        return strUtf8;
    }

    static std::string Utf8ToAnsi(const std::string &strUtf8)
    {
        std::string strAnsi;

        char *lpszAnsi = Utf8ToAnsi(strUtf8.c_str());
        if (lpszAnsi != NULL)
        {
            strAnsi = lpszAnsi;
            delete[] lpszAnsi;
        }

        return strAnsi;
    }

    static std::string UnicodeToUtf8(const std::wstring &strUnicode)
    {
        std::string strUtf8;

        char *lpszUtf8 = UnicodeToUtf8(strUnicode.c_str());
        if (lpszUtf8 != NULL)
        {
            strUtf8 = lpszUtf8;
            delete[] lpszUtf8;
        }

        return strUtf8;
    }

    static std::wstring Utf8ToUnicode(const std::string &strUtf8)
    {
        std::wstring strUnicode;

        wchar_t *lpszUnicode = Utf8ToUnicode(strUtf8.c_str());
        if (lpszUnicode != NULL)
        {
            strUnicode = lpszUnicode;
            delete[] lpszUnicode;
        }

        return strUnicode;
    }
};
