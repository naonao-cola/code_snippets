#pragma once
#ifndef __TOOLS_H__
#define __TOOLS_H__
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <type_traits>

namespace nao {
	namespace xx {
		class Cout {
		public:
			Cout() { this->mutex().lock(); }
			Cout(const char* file, unsigned int line) {
				this->mutex().lock();
				this->stream() << '[' << file << ':' << line << ']' << ' ';
			}
			~Cout() {
				this->stream() << std::endl;
				::fwrite(this->stream().str().c_str(), 1, this->stream().str().size(), stderr);
				//清空数据
				this->stream().str("");
				this->mutex().unlock();
			}
			std::mutex& mutex() {
				static std::mutex kMtx;
				return kMtx;
			}
			std::ostringstream& stream() {
				static std::ostringstream  kStream;
				return kStream;
			}
		};//class Cout

		//defer的实现
		template <typename F>
		struct Defer {
			Defer(F&& f) : _f(std::forward<F>(f)) {}
			~Defer() { _f(); }
			typename std::remove_reference<F>::type _f;
		};

		template <typename F>
		inline Defer<F> create_defer(F&& f)
		{
			return Defer<F>(std::forward<F>(f));
		}
#define _nao_defer_name_cat(x, n) x##n
#define _nao_defer_make_name(x, n) _nao_defer_name_cat(x, n)
#define _nao_defer_name _nao_defer_make_name(_nao_defer_, __LINE__)
	}//namaspace xx

	namespace now {
		typedef int64_t int64;
		typedef uint32_t uint32;
		//时间戳
		int64 now_ms();
		int64 now_us();
		// "%Y-%m-%d %H:%M:%S" ==> 2018-08-08 08:08:08
		std::string str_time(const char* fm = "%Y-%m-%d %H:%M:%S");
		//自1970-01-01 00:00:00以来的时间,现在时刻的时间戳。使用这个
		int64 epoch_ms();
		int64 epoch_us();
		//休眠
		void sleep_sec(uint32 n);
		void sleep_ms(uint32 n);
		//计时器
		class Timer {
		public:
			Timer() {
				_start = now_us();
			}
			void restart() {
				_start = now_us();
			}
			int64 us() const {
				return now_us() - _start;
			}
			int64 ms()const {
				return this->us() / 1000;
			}
		private:
			int64 _start;
		};//class Timer
	}//namespace now

	//类型转换函数
	template <typename in_type, typename out_type>
	void typeConvert(const in_type& in_value, out_type& out_value) {
		std::stringstream stream;
		stream << in_value;
		stream >> out_value;
	}
}//namespace nao

//控制台输出
#define XOUT   nao::xx::Cout().stream()
#define XLOG   nao::xx::Cout(__FILE__, __LINE__).stream()
//Defer 功能
#define DEFER(e) auto _nao_defer_name = nao::xx::create_defer([&](){ e; })
//禁止拷贝与赋值
#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
    ClassName(const ClassName&) = delete; \
    void operator=(const ClassName&) = delete
//unlikely 功能
#if (defined(__GNUC__) && __GNUC__ >= 3) || defined(__clang__)
static inline bool (likely)(bool x) { return __builtin_expect((x), true); }
static inline bool (unlikely)(bool x) { return __builtin_expect((x), false); }
#else
static inline bool (likely)(bool x) { return x; }
static inline bool (unlikely)(bool x) { return x; }
#endif

#define CLIP_RANGE(value, min, max)  ( (value) > (max) ? (max) : (((value) < (min)) ? (min) : (value)) )
#define SWAP(a, b, t)  do { t = a; a = b; b = t; } while(0)
#include <chrono>
#define TICK(x) auto bench_##x = std::chrono::high_resolution_clock::now();
#define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - bench_##x).count() << "us" << std::endl;

#endif  //__TOOLS_H__