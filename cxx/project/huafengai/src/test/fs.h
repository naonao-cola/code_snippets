/**
* @author  闹闹
* @brief   文件系统库
* @details 为天地立心，为生民立命，为往圣继绝学，为万世开太平。
* @version 1.0.0.1
* @date    2020/08/13 14:04:59
* @copyright Copyright (c) 2050
**********************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-8-13 15:50:18   <td>1.0.0.3    <td>naonao     <td>总体格式修改，重载了userlog
* </table>
*
***********************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-8-14 10:13:31   <td>1.0.0.4    <td>naonao     <td>修改部分注释,修改了#if defined的格式,去除成员变量，将读取文件的函数改为静态函数
* </table>
*
************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-8-14 11:51:17   <td>1.0.0.5    <td>naonao     <td>增加了函数调用失败的错误信息
* </table>
*
*************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-8-17 16:11:10   <td>1.0.0.6    <td>naonao     <td>增加了获取文件与字符串的md5，增加了根据md5和文件名删除相同文件，
* </table>
*
**************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-8-19 17:44:07   <td>1.0.0.7    <td>naonao     <td>修改了函数名称，作为统一标识
* </table>
*
***************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-8-28 11:59:01   <td>1.0.0.8    <td>naonao     <td>修改了函数FT_GetAllFormatFiles，FT_GetAllFiles,增加了过滤的文件夹。其他增加过滤文件夹的函数可以参考
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-9-1 10:24:08    <td>1.0.0.9    <td>naonao     <td>去除了using namespace std;防止污染空间名，用的话，放在命名空间里面，去除了类，改为命名空间，修改了格式，命名空间的短写不要放在头文件，将函数的静态属性去除
* </table>
*
***************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-9-8 14:53:51   <td>1.0.0.10    <td>naonao     <td>修改了FT_GetCopyFileA函数，使用第二种方法，去除了时间慢，占用内存大的问题，方法一在x86下分配大内存会失败，果断采用方法二
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-9-14 16:16:09   <td>1.0.0.11   <td>naonao     <td>修改了函数FT_GetAllFormatFiles，FT_GetAllFiles，在传入单斜线格式的路径时出现无效的BUG。
* </table>
*
**************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-9-30 09:07:59   <td>1.0.0.13   <td>naonao     <td>重载FT_GetAllFormatFiles
* </table>
*
***************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2020-10-14 16:53:21  <td>1.0.0.14   <td>naonao     <td>增加了过滤字符串的函数filterfield
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2021年1月4日14:00:11 <td>1.0.0.15   <td>naonao     <td>删除线程错误信息，改为全局信息，将.h与.cpp 文件合并为.hpp。
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2021年1月27日10:55:32<td>1.0.0.16   <td>naonao     <td>修改代码格式，去除冗余部分，在考虑是否需要加入文件读写的内容。希望代码之间耦合性不要太高。
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2021-4-16 11:31:08   <td>1.0.0.17   <td>naonao     <td>修改Windows的代码，去除平台依赖，FT_AlterFileName 重命名修改，FT_FolderIsEmpty 判断文件夹非空，FT_GetCopyFileA  文件复制函数修改，禁用了一部分函数，此部分函数是WINDOWS平台下的函数，有需要时可以打开
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2021/6/3 20:39:41    <td>1.0.0.18   <td>naonao     <td>删除掉不必要的部分，补全std,获取时间函数进行修改
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2021年8月8日15:08:38 <td>1.0.0.18   <td>naonao     <td>去掉不常用的部分，改变命名为fo
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2021/9/18 15:44:27   <td>1.0.0.19   <td>naonao     <td>版本修改，增加函数
* </table>
*
****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2021/10/20 20:42:41  <td>1.0.0.20   <td>naonao     <td>增加c++文件的读写功能，便于日常应用。以二进制读写为主，未考虑文本文件。
* </table>
*
*****************************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2023/2/1 15:38:07    <td>1.0.0.21   <td>naonao     <td>考虑文本文件。
* </table>
*
***/
#pragma once
#ifndef _OPERATINGTOOLS_H_
#define _OPERATINGTOOLS_H_

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <filesystem>
namespace fs = std::filesystem;

#if defined(_WIN32) || defined(_WIN64) || defined(_WINDOWS) || defined(_MSC_VER)
#ifndef WINDOWS
#define WINDOWS
#include <windows.h>
#endif
#else
#include <sys/time.h>
#endif
//标准库
#include <time.h>
#include <tchar.h>
#include <stdlib.h>
#include <thread>
#include <exception>
#include <regex>
#include <io.h>
#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <map>
#include<typeinfo>
#include <sstream>
namespace nao {
	namespace  fl {
		/**
		 * @brief 目录是否存在
		 * @param path
		 * @return 返回结果
		*/
		bool existsDir(const std::string& path);

		/**
		 * @brief 判断是否是目录
		 * @param path
		 * @return 返回结果
		*/
		bool isDir(const std::string& path);

		/**
		 * @brief 创建目录
		 * @param filename
		 * @return 返回结果
		*/
		bool creatDir(const std::string& filename);

		/**
		 * @brief 文件夹是否为空
		 * @param foldername
		 * @return 返回结果
		*/
		bool folderIsEmpty(const std::string& foldername);

		/**
		 * @brief 获取文件名
		 * @param filename
		 * @return 返回结果
		*/
		std::string base_name(const std::string& filename);

		std::string parent_parh(const std::string& filename);

		/**
		 * @brief 获取扩展名
		 * @param filename
		 * @return 返回结果
		*/
		std::string extension_name(const std::string& filename);

		/**
		 * @brief 获取文件大小
		 * @param filename
		 * @return 返回结果
		*/
		uintmax_t getFileSize(const std::string& filename);

		/**
		 * @brief 移除目录
		 * @param filename
		 * @return 返回结果
		*/
		uintmax_t removeDir(const std::string& filename);

		/**
		 * @brief 获取所有文件
		 * @param path    路径
		 * @param files   结果数组
		 * @param filter_directory  需要过滤的文件夹
		 * @return 返回结果
		*/
		bool getAllFiles(const std::string& path, std::vector<std::string>& files, const std::vector<std::string>& filter_directory = std::vector<std::string>());

		/**
		 * @brief 获取特定格式的文件名，递归目录，不区分后缀大小写
		 * @param path 路径
		 * @param files 结果数组
		 * @param format 格式
		 * @param filter_directory 需要过滤的文件夹
		 * @return 返回结果
		*/
		bool getAllFormatFiles(const std::string& path, std::vector<std::string>& files,
			const std::string& format = "(.*)(.png|jpg|bmp)",
			const std::vector<std::string>& filter_directory = std::vector<std::string>());

		/**
		 * @brief 获取特定格式的文件 overload
		 * @param path
		 * @param files
		 * @param format
		 * @param filter_directory
		 * @return 返回结果
		*/
		bool getAllFormatFiles(const std::vector<std::string>& path, std::vector<std::string>& files,
			const std::string& format = "(.*)(.png|jpg|bmp)",
			const std::vector<std::string>& filter_directory = std::vector<std::string>());

		/**
		 * @brief 获取文件标志位，格式为：文件路径+空格+标志符
		 * @param strPath
		 * @return 返回结果
		*/
		int getClassFlag(const std::string& strPath);

		/**
		 * @brief 获取文件列表到当前目录的txt，以追加的方式写入文件,format是格式,lable是标志符
		 * @param filePath 文件路径
		 * @param lable    文件标识符
		 * @param distAll  生成文件的目录，，默认当前目录。
		 * @param format   文件格式
		*/
		void getFileList(const std::string& filePath, const int& lable = 255, const std::string& distAll = "", const std::string& format = "");

		/**
		 * @brief 根据文件名删除，将dstfile中与orgfile相同的文件删除，要删除的源文件在dst中,flag是删除标识符，默认不删除
		 * @param orgfile
		 * @param dstfile
		 * @param samefilelist
		 * @param flag
		 * @return 返回结果
		*/
		int removeDuplicateFile(const std::string& orgfile, const std::string& dstfile, std::vector<std::string>& samefilelist,
			const int flag = 0);

		/**
		 * @brief 更改文件名称
		 * @param filename
		 * @param newname
		 * @return 返回结果
		*/
		int alterFileName(const std::string& filename, const std::string& newname);

		/**
		 * @brief 复制文件
		 * @param orgPath
		 * @param dstPath
		 * @return 返回结果
		*/
		bool  getCopyFile(const std::string& orgPath, const std::string& dstPath);

		/**
		 * @brief 获取时间
		 * @return 返回结果
		*/
		double getDoubleTime();

		/**
		 * @brief 读取文件列表文件与标志位,读到lablefilelist中
		 * @param filename
		 * @param lablefilelist
		*/
		void readFileList(const std::string& filename, std::multimap<std::string, int>& lablefilelist);

		/**
		 * @brief 读取不含标志位的文件列表文件(一行是一个路径),读到nolablefilelist
		 * @param filename
		 * @param nolablefilelist
		 * @return 返回结果
		*/
		bool readNoLableFileList(const std::string& filename, std::vector<std::string>& nolablefilelist);

		/**
		 * @brief 从字符串列表中过滤掉包含特殊字符的字符串
		 * @param srcStringList
		 * @param dstStringList
		 * @param format
		 * @return 返回结果
		*/
		bool filterField(const std::vector<std::string>& srcStringList, std::vector<std::string>& dstStringList,
			const std::vector<std::string>& format = std::vector<std::string>{ ".*_20200926.*" });

		/**
		 * @brief 从字符串列表中过滤掉包含特殊字符的字符串
		 * @param srcStringList
		 * @param dstStringList
		 * @param format
		 * @return 返回结果
		*/
		bool filterField(const std::vector<std::string>& srcStringList, std::vector<std::string>& dstStringList, const std::string& format);

		/**
		 * @brief 读写文件
		 * open mode:
		 * 'r': read         open if exists
		 * 'a': append       created if not exists
		 * 'w': write        created if not exists, truncated if exists
		 * 'r_t'文本模式
		 * 'a_t'
		 * 'w_t'
		*/
		class file {
		public:
			static const int seek_beg = 0;
			static const int seek_cur = 1;
			static const int seek_end = 2;
			/**
			 * @brief 第 1 个版本是默认构造函数，创建一个空的 file 对象，不会打开任何文件。
			*/
			file() {}

			/**
			 * @brief 第 2 个版本是 move 构造函数，支持将 file 对象放到 STL 容器中。
			 * @param f
			*/
			file(file&& f) { _path = f._path; _fin = std::move(f._fin); _fout = std::move(f._fout); }

			/**
			 * @brief 第 3-4 个版本，打开指定的文件，参数 path 是文件路径，参数 mode 是打开模式。
			 * @param path
			 * @param mode  r  a  w
			*/
			file(const char* path, std::string mode) { this->open(path, mode); }
			file(const std::string& path, std::string mode) : file(path.c_str(), mode) {}

			/*析构函数，关闭之前打开的文件，释放相关资源*/
			~file();

			file(const file& x) = delete;
			void operator=(const file& x) = delete;
			void operator=(file&& x) = delete;

			/**
			 * @brief 将 fs::file 转换为 bool 类型，文件成功打开时返回 true，否则返回 false。
			*/
			explicit operator bool() const;

			/**
			 * @brief 文件未打开或打开失败时返回 true，否则返回 false。
			 * @return 返回结果
			*/
			bool operator!() const { return !(bool)(*this); }

			/**
			 * @brief 此方法返回 file 内部文件路径的引用。
			 * 若 file 对象并未关联任何文件，则返回值会引用一个空字符串。
			 * @return 返回结果
			*/
			const std::string& path() const;

			/**
			 * @brief 此方法返回文件的大小，文件未打开或打开失败时，调用此方法会返回 -1。
			 * @return 返回结果
			*/
			int64_t size()  const { return getFileSize(this->path()); }

			/**
			 * @brief 判断文件是否存在。
			 * @return 返回结果
			*/
			bool exists() const { return existsDir(this->path()); }

			/**
			 * @brief 此方法打开指定的文件，path 是文件路径，mode 是打开模式，见构造函数中的说明。此方法在打开文件前，会先关闭之前打开的文件。
			 * @param path 路径
			 * @param mode 模式
			 * @param type 默认以二进制形式读写
			 * @return
			*/
			bool open(const char* path, std::string mode);
			bool open(const std::string& path, std::string mode) { return this->open(path.c_str(), mode); }

			/**
			 * @brief 关闭文件，析构函数中会自动调用此方法。
			 * 多次调用此方法是安全的。
			*/
			void close();

			/**
			 * @brief 设置文件指针的当前位置，参数 off 是偏移位置，参数 whence 是起始位置，可以是file::seek_beg, file::seek_cur, file::seek_end 中的一种。此方法对以 'a' (append) 模式打开的文件无效。
			 * @param off
			 * @param whence
			*/
			void seek(int64_t off, int whence = seek_beg);

			/**
			 * @brief 第 1 个版本读取数据到指定的 buffer 中，n 是要读取的字节数，返回实际读取的字节数。第 2 个版本与第 1 个版本类似，但以 fastring 的形式返回读取的数据，n 是要读取的字节数。此方法在遇到文件尾或发生错误时，实际读取的字节数可能小于 n。
			 * @param buf
			 * @param n
			 * @return
			*/
			size_t read(void* buf, size_t n);
			std::string read(size_t n);

			/**
			 * @brief 写入数据
			 * @param s
			 * @param n
			 * @return
			 * 字符串写入还是字符串
			*/
			size_t write(const void* s, size_t n);
			size_t write(const char* s) { return this->write(s, strlen(s)); }
			size_t write(const std::string& s) { return this->write(s.data(), s.size()); }
			size_t write(char c) { return this->write(&c, 1); }

		private:
			std::ifstream _fin;
			std::ofstream _fout;
			std::string _path;
		};//class file

		/**
		 * @brief fl::file 不支持缓存，写小文件性能较差，为此，另外实现了支持缓存的 fl::fstream 类，fl::fstream 只用于写文件，不支持读操作。
		*/
		class fstream {
		public:
			/**
			 * @brief 第 1 个版本是默认构造函数，内部缓存大小为 8k。
			*/
			fstream() {
				_cap = 8192;
				_s.resize(0);
			}

			/**
			 * @brief 第 2 个版本是 move 构造函数，可以将 fstream 对象放到 STL 容器中。
			 * @param fs
			*/
			fstream(fstream&& fs) : _s(std::move(fs._s)), _f(std::move(fs._f)) {}

			/**
			 * @brief 第 3 个版本用参数 cap 指定缓存的大小。
			 * @param cap
			*/
			explicit fstream(size_t cap) {
				_cap = cap;
				_s.resize(0);
			}

			/**
			 * @brief 第 4-5 个版本打开指定的文件，path 是文件路径，mode 是模式，cap 是缓存大小，默认为 8k。
			 * @param path
			 * @param mode
			 * @param cap
			*/
			fstream(const char* path, std::string mode, size_t cap = 8192) :_f(path, mode == "w" ? "w" : "a"), _cap(cap) { _s.resize(0); }
			fstream(const std::string& path, std::string mode, size_t cap = 8192) : fstream(path.c_str(), mode, cap) {}

			~fstream() { this->close(); }

			/**
			 * @brief 将 fs::fstream 转换为 bool 类型，文件成功打开时返回 true，否则返回 false。
			*/
			explicit operator bool() const { return (bool)_f; }
			bool operator!() const { return !(bool)_f; }

			/**
			 * @brief 打开指定的文件，参数 path 是文件路径，参数 mode 是打开模式，见构造函数中的说明
			 * @param path
			 * @param mode
			 * @return
			*/
			bool open(const char* path, std::string mode) {
				this->close();
				return _f.open(path, mode == "w" ? "w" : "a");
			}
			bool open(const std::string& path, char mode) {
				return this->open(path.c_str(), mode);
			}

			/**
			 * @brief 调整缓存容量，参数 n 是容量大小。若 n 小于之前的容量，则缓存容量保持不变。
			 * @param n
			*/
			void reserve(size_t n) {
				if (n <= _cap) {
					return;
				}
				_cap = n;
				_s.resize(_cap);
			}

			void flush() {
				if (!_s.empty()) {
					_f.write(_s.data(), _s.size());
					_s.clear();
				}
			}

			void close() {
				this->flush();
				_f.close();
			}

			/**
			 * @brief 追加数据，参数 n 是数据的长度。
			 * @param s
			 * @param n
			 * @return
			*/
			fstream& append(const void* s, size_t n) {
				if (_cap < _s.size() + n) this->flush();
				n <= _cap ? ((void)_s.append((char*)s, n)) : ((void)_f.write(s, n));
				return *this;
			}

			fstream& operator<<(const char* s) {
				return this->append(s, strlen(s));
			}

			fstream& operator<<(const std::string& s) {
				return this->append(s.data(), s.size());
			}

			//其他类型的直接写
			template<typename T>
			fstream& operator<<(T v) {
				if (_cap < _s.size() + 24) this->flush();
				this->append((const char*)&v, sizeof(T));
				return *this;
			}

		private:
			file _f;
			std::string _s;
			size_t _cap;
		};// class fstream
	}//namespace fl
}//namespace operatingtools

#endif // !_OPERATINGTOOLS_H_
