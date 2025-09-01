#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <ratio>

#include <boost/noncopyable.hpp>

struct App : private boost::noncopyable {
  using Timestamp = std::chrono::duration<uint64_t, std::micro>;
  using DeltaTime = std::chrono::duration<double, std::milli>;
  using Clock = std::chrono::high_resolution_clock;
  struct FrameData {
    Timestamp time_now;
    DeltaTime delta_time;
  };
  std::function<void(FrameData)> on_frame_;
  struct GLFWwindow *window_;
  Clock::time_point last_frame_;

  App(struct AppBuilder &);
  ~App();
  void Run();
};

struct AppBuilder {
  std::function<std::function<void(App::FrameData)>(App &)> on_frame;
  int win_width = 1024;
  int win_height = 1024;
  const char *win_title = "null";
  bool vsync = false;

  AppBuilder();
  std::unique_ptr<App> Build();
};
