#pragma once
#include <coroutine>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuda2omp {
template<class T> decltype(auto) kernel_arg(void** args, unsigned index) {
  return *static_cast<std::remove_reference_t<T>*>(args[index]);
}
template<class T> unsigned extent_x(const T& extent) {
  if constexpr (requires { extent.x; }) return extent.x;
  else return static_cast<unsigned>(extent);
}
template<class T=void> class Task;
template<class T> struct Promise {
  T value{}; std::coroutine_handle<> continuation{};
  Task<T> get_return_object(); std::suspend_always initial_suspend() noexcept{return {};}
  struct Final { bool await_ready() noexcept{return false;} void await_resume() noexcept{}
    std::coroutine_handle<> await_suspend(std::coroutine_handle<Promise> h) noexcept { return h.promise().continuation ? h.promise().continuation : std::noop_coroutine(); }};
  Final final_suspend() noexcept{return {};} void unhandled_exception(){std::terminate();}
  template<class U> void return_value(U&& v){value=std::forward<U>(v);}
};
template<> struct Promise<void> { std::coroutine_handle<> continuation{}; Task<void> get_return_object();
  std::suspend_always initial_suspend() noexcept{return {};}
  struct Final { bool await_ready() noexcept{return false;} void await_resume() noexcept{}
    std::coroutine_handle<> await_suspend(std::coroutine_handle<Promise> h) noexcept{return h.promise().continuation?h.promise().continuation:std::noop_coroutine();}};
  Final final_suspend() noexcept{return {};} void return_void(){} void unhandled_exception(){std::terminate();}
};
template<class T> class Task { public: using promise_type=Promise<T>; using H=std::coroutine_handle<Promise<T>>; H h{};
  explicit Task(H x):h(x){} Task(Task&&o):h(std::exchange(o.h,{})){} Task(const Task&)=delete;
  ~Task(){if(h)h.destroy();} H release(){return std::exchange(h,{});}
  struct Awaiter { H h; bool await_ready(){return false;} H await_suspend(std::coroutine_handle<> p){h.promise().continuation=p;return h;}
    T await_resume(){if constexpr(std::is_void_v<T>){h.destroy();return;}else{T v=std::move(h.promise().value);h.destroy();return v;}}};
  Awaiter operator co_await() && {return {release()};}
};
template<class T> Task<T> Promise<T>::get_return_object(){return Task<T>{std::coroutine_handle<Promise>::from_promise(*this)};}
inline Task<void> Promise<void>::get_return_object(){return Task<void>{std::coroutine_handle<Promise>::from_promise(*this)};}

struct BlockScheduler;
struct ThreadContext { unsigned threadIdx_x,blockIdx_x,blockDim_x,gridDim_x; BlockScheduler* scheduler;
  struct Barrier { BlockScheduler* s; bool await_ready()const noexcept{return false;} void await_resume()const noexcept{}
    void await_suspend(std::coroutine_handle<> h)const; };
  Barrier syncthreads(){return {scheduler};}
};
struct BlockScheduler { unsigned count; std::vector<std::coroutine_handle<>> waiting;
  explicit BlockScheduler(unsigned n):count(n){} void arrive(std::coroutine_handle<> h){waiting.push_back(h);}
  void run(std::vector<std::coroutine_handle<>>& roots){for(auto h:roots)h.resume();
    while(true){bool done=true;for(auto h:roots)done&=h.done();if(done)break;
      if(waiting.size()!=count){std::cerr<<"cuda2omp: divergent barrier ("<<waiting.size()<<"/"<<count<<")\n";std::abort();}
      auto wave=std::move(waiting);waiting.clear();for(auto h:wave)h.resume();}
    for(auto h:roots)h.destroy(); }
};
inline void ThreadContext::Barrier::await_suspend(std::coroutine_handle<> h)const{s->arrive(h);}

template<class Shared,class Factory> void launch(unsigned grid,unsigned block,Factory factory){
  #pragma omp parallel for
  for(int b=0;b<(int)grid;++b){Shared shared{};BlockScheduler sched(block);std::vector<std::coroutine_handle<>> roots;roots.reserve(block);
    std::vector<ThreadContext> contexts; contexts.reserve(block);
    for(unsigned t=0;t<block;++t){contexts.push_back({t,(unsigned)b,block,grid,&sched});auto task=factory(contexts.back(),shared);roots.push_back(task.release());}
    sched.run(roots);}
}
}
