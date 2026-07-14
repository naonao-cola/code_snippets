#include "thread_pool.h"

namespace nao {
	namespace threadpool {
		void Event::wait() {
			std::unique_lock<std::mutex> lock(_mu);
			_con.wait(lock, [this]() {return this->_flag || this->_all; });
			if (!_all)_flag = false;
		}

		void Event::notifyOne() {
			std::lock_guard<std::mutex> lock(_mu);
			_flag = true;
			_con.notify_one();
		}

		void Event::notifyAll() {
			std::lock_guard<std::mutex> lock(_mu);
			_all = true;
			_con.notify_all();
		}

		void Event::reset() {
			std::lock_guard<std::mutex> lock(_mu);
			_flag = _all = false;
		}

		ThreadPool::~ThreadPool() {
			close();
		}

		void ThreadPool::start() {
			_run = true;
			_event_.reset();
		}

		void ThreadPool::close() {
			_run = false;
			_event_.notifyAll();
			Pool vec;
			do {
				MUTEXOBJECT_LOCK_GUARD(_pool);
				vec = _pool.data();
			} while (false);
			std::for_each(std::begin(vec), std::end(vec), [](const ThreadPtr& it) {
				if (it->joinable()) {
					if (std::this_thread::get_id() == it->get_id()) {
						it->detach();
					}
					else {
						it->join();
					}
				}
				});
		}

		bool ThreadPool::_needNewThread() {
			do {
				MUTEXOBJECT_LOCK_GUARD(_pool);
				if (_pool->empty())
					return true;
				if (_pool->size() == _maxCnt)
					return false;
			} while (false);
			do {
				MUTEXOBJECT_LOCK_GUARD(_taskQueue);
				return _taskQueue->size() > 0;
			} while (false);
			assert(false);
		}

		void ThreadPool::_newThread() {
			MUTEXOBJECT_LOCK_GUARD(_pool);
			if (_pool->size() < _coreCnt) {
				_pool->emplace_back(new std::thread(std::bind(&ThreadPool::_dispath, this, true)));
			}
			else if (_expand) {
				_pool->emplace_back(new std::thread(std::bind(&ThreadPool::_dispath, this, false)));
			}
		}

		void ThreadPool::_dispath(bool core) {
			while (_run.load()) {
				if (Task task = _pickOneTask()) {
					task();
				}
				else if (!_event_.waitFor(std::chrono::minutes(1)) && !core) {
					_killSelf();
					break;
				}
			}
		}

		void ThreadPool::_killSelf() {
			MUTEXOBJECT_LOCK_GUARD(_pool);
			auto it = std::find_if(std::begin(_pool.data()), std::end(_pool.data()), [](const ThreadPtr& it) {
				return std::this_thread::get_id() == it->get_id();
				});

			(*it)->detach();
			_pool->erase(it);
		}

		ThreadPool::Task ThreadPool::_pickOneTask() {
			MUTEXOBJECT_LOCK_GUARD(_taskQueue);
			Task ret = nullptr;
			if (!_taskQueue->empty()) {
				ret = std::move(_taskQueue->front());
				_taskQueue->pop();
			}
			return ret;
		}
	}//namespace threadpool
}//namespace nao