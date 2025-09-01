#include <format>
#include <iostream>
#include <stdexcept>

#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include "app.hpp"

App::App(AppBuilder &b) : on_frame_([](auto) {}) {
  glfwSetErrorCallback([](int code, const char *msg) {
    std::cerr << std::format("glfw error: {} ({})\n", msg, code);
  });
  glfwInit();
  if (!glfwInit()) {
    throw std::runtime_error("glfw init error");
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
  glfwWindowHint(GLFW_RESIZABLE, false);
  window_ = glfwCreateWindow(b.win_width, b.win_height, b.win_title, nullptr,
                             nullptr);
  glfwMakeContextCurrent(window_);
  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    throw std::runtime_error("GL loader failure");
  }
  if (GLAD_GL_ARB_debug_output) {
    glDebugMessageCallbackARB(
        [](GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum severity,
           GLsizei len, const char *msg, const void * /*user*/) {
          std::cerr << "OpenGL: ";
          std::cerr.write(msg, len) << "\n";
          if (severity == GL_DEBUG_SEVERITY_HIGH_ARB) {
            throw std::runtime_error("OpenGL fatal error");
          }
        },
        nullptr);
  }
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);

  if (b.vsync) {
    glfwSwapInterval(1);
  } else {
    glfwSwapInterval(0);
  }
}

App::~App() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

std::unique_ptr<App> AppBuilder::Build() {
  auto app = std::make_unique<App>(*this);
  app->on_frame_ = on_frame(*app);
  return app;
}

AppBuilder::AppBuilder() : on_frame([](auto &) { return [](auto) {}; }) {}

void App::Run() {
  const auto t0 = Clock::now();
  last_frame_ = t0;
  while (!glfwWindowShouldClose(window_)) {
    const auto now = Clock::now();
    auto time_passed = now - t0;
    auto delta = now - last_frame_;
    on_frame_(FrameData{
        .time_now =
            std::chrono::duration_cast<std::chrono::milliseconds>(time_passed),
        .delta_time = delta});
    last_frame_ = now;
    glfwPollEvents();
    glfwSwapBuffers(window_);
  }
}
