#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <ratio>

#include <boost/noncopyable.hpp>

struct App : private boost::noncopyable {
  using Timestamp = std::chrono::duration<uint64_t, std::milli>;
  struct FrameData {
    Timestamp time_now;
  };
  std::function<void(FrameData)> on_frame_;
  struct GLFWwindow *window_;

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
