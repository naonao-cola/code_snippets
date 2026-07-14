#pragma once
#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__
#include <thread>
#include <mutex>
#include <iostream>
#include <future>
#include <chrono>
#include <queue>
#include <vector>
#include <assert.h>
#include <stddef.h>
#include <condition_variable>

namespace nao {
	namespace threadpool {
		class Event {
		private:
			Event(const Event&) = delete;
			Event& operator=(const Event&) = delete;
		public:
			bool _flag = false;
			bool _all = false;
			std::mutex _mu;
			std::condition_variable _con;
		public:
			Event() = default;
			void wait();
			template<typename _Rep, typename _Period> bool waitFor(const std::chrono::duration<_Rep, _Period>& duration);
			template<typename _Clock, typename _Duration>bool waitUntil(const std::chrono::time_point<_Clock, _Duration>& point);
			void notifyOne();
			void notifyAll();
			void reset();
		};

		template<typename T>
		class MutexObject {
		private:
			T          _data;
			std::mutex _mu;
		public:
			MutexObject() {}
			template<typename... Args> MutexObject(Args... args) :_data(args...) {}
			MutexObject<T>& operator = (const T& data) {
				this->_data = data;
				return *this;
			}
			MutexObject<T>& operator = (const MutexObject<T>& other) {
				this->_data = other._data;
				return *this;
			}
			operator T& () { return _data; }
			operator T && () { return std::move(_data); }
			T* operator -> () { return &_data; }
			T* operator & () { return operator->(); }
			std::mutex& mutex() { return _mu; }
			T& data() { return _data; }
		};

#define MUTEXOBJECT_LOCK_GUARD(obj) std::lock_guard<std::mutex> lock(obj.mutex())
#define MUTEXOBJECT_UNIQUE_LOCK(obj) std::unique_lock<std::mutex> lock(obj.mutex())

		class ThreadPool {
			typedef std::function<void()> Task;
			typedef std::queue<Task>  TaskQueue;
			typedef std::shared_ptr<std::thread> ThreadPtr;
			typedef std::vector<ThreadPtr> Pool;
		private:
			MutexObject<Pool>      _pool;
			MutexObject<TaskQueue> _taskQueue;
			Event                  _event_;
			std::size_t            _coreCnt;
			bool                   _expand;
			std::size_t            _maxCnt;
			std::atomic<bool>      _run;

		public:
			ThreadPool(std::size_t coreCnt = 1, bool expand = false, std::size_t maxCnt = std::thread::hardware_concurrency()) :_coreCnt(coreCnt), _expand(coreCnt ? expand : true), _maxCnt(maxCnt), _run(true) {}
			~ThreadPool();
			void start();
			void close();
			template<typename Fun, typename...Args> std::future<typename std::result_of<Fun(Args...)>::type> submit(Fun&& fun, Args&&... args);
		private:
			bool _needNewThread();
			void _newThread();
			void _dispath(bool core);
			void _killSelf();
			Task _pickOneTask();
		};
	}//namespace threadpool
}//namespace nao

#include "thread_pool-inl.h"
#endif  //__THREAD_POOL_H__