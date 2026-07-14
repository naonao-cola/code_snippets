#include "thread_pool.h"

namespace nao {
	namespace threadpool {
		template<typename _Rep, typename _Period>
		bool Event::waitFor(const std::chrono::duration<_Rep, _Period>& duration) {
			std::unique_lock<std::mutex> lock(_mu);
			bool ret = true;
			ret = _con.wait_for(lock, duration, [this]() {return this->_flag || this->_all; });
			if (ret && !_all) _flag = false;
			return ret;
		}

		template<typename _Clock, typename _Duration>
		bool Event::waitUntil(const std::chrono::time_point<_Clock, _Duration>& point) {
			std::unique_lock<std::mutex> lock(_mu);
			bool ret = true;
			ret = _con.wait_until(lock, point, [this]() {return this->_flag || this->_all; });
			if (ret && !_all) _flag = false;
			return ret;
		}

		template<typename Fun, typename...Args>
		std::future<typename std::result_of<Fun(Args...)>::type> ThreadPool::submit(Fun&& fun, Args&&... args) {
			if (!_run.load())
				throw std::runtime_error("ThreadPool has closed");
			typedef typename std::result_of<Fun(Args...)>::type ReturnType;
			auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::bind(std::forward<Fun>(fun), std::forward<Args>(args)...));
			do
			{
				MUTEXOBJECT_LOCK_GUARD(_taskQueue);
				_taskQueue->emplace([task]() {
					(*task)();
					});
			} while (false);
			_event_.notifyOne();
			if (_needNewThread())
			{
				_newThread();
			}
			return task->get_future();
		}
	}//namespace threadpool
}//namespace nao