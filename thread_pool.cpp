#include "thread_pool.h"

ThreadPool::ThreadPool(size_t num_threads) 
{ 
  // create worker threads 
  for (size_t i = 0; i < num_threads; ++i) { 
    threads.emplace_back([this] { 
      while (true) { 
        std::function<void()> task; 
        { 
          // lock the queue so that data can be shared safely 
          std::unique_lock<std::mutex> lock( queue_mutex); 
  
          // wait for a task to execute or if the pool is stopped 
          cv.wait(lock, [this] { 
              return !tasks.empty() || stop; 
          }); 
  
          // exit the thread if the pool is stopped and there are no tasks 
          if (stop && tasks.empty()) { 
              return; 
          } 
  
          // get the next task from the queue 
          task = move(tasks.front()); 
          tasks.pop(); 
        } 
  
        task(); 
      } 
    }); 
  } 
} 
  
// destructor stops the thread pool 
ThreadPool::~ThreadPool() { 
  { 
    // lock the queue to update the stop flag safely 
    std::unique_lock<std::mutex> lock(queue_mutex); 
    stop = true; 
  } 

  // notify all threads to stop
  cv.notify_all(); 
  
  // join threads to ensure they are finished
  for (auto& thread : threads) { 
    thread.join(); 
  } 
} 
