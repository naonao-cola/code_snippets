#include "tools.h"
#ifdef _WIN32
//windows 平台
#include <time.h>
// for struct timeval
#include <winsock2.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
namespace nao {
	namespace _Mono {
		typedef int64_t int64;
		inline int64 _QueryFrequency() {
			LARGE_INTEGER freq;
			QueryPerformanceFrequency(&freq);
			return freq.QuadPart;
		}

		inline int64 _QueryCounter() {
			LARGE_INTEGER counter;
			QueryPerformanceCounter(&counter);
			return counter.QuadPart;
		}

		inline const int64& _Frequency() {
			static int64 freq = _QueryFrequency();
			return freq;
		}

		inline int64 ms() {
			int64 count = _QueryCounter();
			const int64& freq = _Frequency();
			return (count / freq) * 1000 + (count % freq * 1000 / freq);
		}

		inline int64 us() {
			int64 count = _QueryCounter();
			const int64& freq = _Frequency();
			return (count / freq) * 1000000 + (count % freq * 1000000 / freq);
		}
	}// namespace _Mono
	namespace now {
		int64 now_ms() {
			return _Mono::ms();
		}

		int64 now_us() {
			return _Mono::us();
		}

		std::string str_time(const char* fm) {
			int64 x = time(0);
			struct tm t;
			_localtime64_s(&t, &x);
			char buf[256];
			const std::size_t r = strftime(buf, sizeof(buf), fm, &t);
			return std::string(buf, r);
		}

		inline int64 filetime() {
			FILETIME ft;
			LARGE_INTEGER x;
			GetSystemTimeAsFileTime(&ft);
			x.LowPart = ft.dwLowDateTime;
			x.HighPart = ft.dwHighDateTime;
			return x.QuadPart - 116444736000000000ULL;
		}

		int64 epoch_ms() {
			return filetime() / 10000;
		}

		int64 epoch_us() {
			return filetime() / 10;
		}

		void sleep_sec(uint32 n) {
			::Sleep(n * 1000);
		}

		void sleep_ms(uint32 n) {
			::Sleep(n);
		}
	}//namespace now
}//namespace nao

#else
//非windows平台
#include <time.h>
#include <sys/time.h>
namespace nao {
	namespace _Mono {
#ifdef CLOCK_MONOTONIC
		inline int64 ms() {
			struct timespec t;
			clock_gettime(CLOCK_MONOTONIC, &t);
			return static_cast<int64>(t.tv_sec) * 1000 + t.tv_nsec / 1000000;
		}

		inline int64 us() {
			struct timespec t;
			clock_gettime(CLOCK_MONOTONIC, &t);
			return static_cast<int64>(t.tv_sec) * 1000000 + t.tv_nsec / 1000;
		}
#else
		inline int64 ms() {
			return epoch::ms();
		}

		inline int64 us() {
			return epoch::us();
		}
#endif
	} //namespace  _Mono
	namespace now {
		int64 now_ms() {
			return _Mono::ms();
		}

		int64 now_us() {
			return _Mono::us();
		}

		std::string str_time(const char* fm) {
			time_t x = time(0);
			struct tm t;
			localtime_r(&x, &t);
			char buf[256];
			const size_t r = strftime(buf, sizeof(buf), fm, &t);
			return std::string(buf, r);
		}

		int64 epoch_ms() {
			struct timeval t;
			gettimeofday(&t, 0);
			return static_cast<int64>(t.tv_sec) * 1000 + t.tv_usec / 1000;
		}

		int64 epoch_us() {
			struct timeval t;
			gettimeofday(&t, 0);
			return static_cast<int64>(t.tv_sec) * 1000000 + t.tv_usec;
		}

		void sleep_sec(uint32 n) {
			struct timespec ts;
			ts.tv_sec = n;
			ts.tv_nsec = 0;
			while (nanosleep(&ts, &ts) == -1 && errno == EINTR);
		}

		void sleep_ms(uint32 n) {
			struct timespec ts;
			ts.tv_sec = n / 1000;
			ts.tv_nsec = n % 1000 * 1000000;
			while (nanosleep(&ts, &ts) == -1 && errno == EINTR);
		}
	}//namespace now
}//namespace nao
#endif //_WIN32