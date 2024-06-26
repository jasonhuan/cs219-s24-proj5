#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());

  ~ThreadPool();

  // queue task for execution by the thread pool 
  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
      using return_type = typename std::result_of<F(Args...)>::type;

      auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
      );

      std::future<return_type> res = task->get_future();
      {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop) {
          throw std::runtime_error("attemping to enqueue on stopped ThreadPool");
      }
      tasks.emplace([task]() { (*task)(); });
      }
      cv.notify_one();
      return res;
  }

private:
  std::vector<std::thread> threads;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable cv;

  bool stop = false;
};

#endif // THREADPOOL_H
