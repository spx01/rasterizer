#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <hip/hip_runtime.h>

#include <hip/hip_gl_interop.h>

#include <boost/core/noncopyable.hpp>
#include <boost/format.hpp>
#include <boost/scope_exit.hpp>

#include <chrono>
#include <iostream>

using boost::format;

static inline void SoftExit(int code) { throw code; }

#define HIP_CHECK(condition)                                             \
  {                                                                      \
    const hipError_t error = condition;                                  \
    if (error != hipSuccess) {                                           \
      std::cerr << "hip error encountered: \"" << hipGetErrorName(error) \
                << "\" at " __FILE__ << ":" << __LINE__ << "\n";         \
      SoftExit(-1);                                                      \
    }                                                                    \
  }

template <typename T>
__host__ __device__ constexpr auto ceiling_div(const T &dividend,
                                               const T &divisor) {
  return (dividend + divisor - 1) / divisor;
}

constexpr int kTexWidth = 1024;
constexpr int kTexHeight = 1024;
constexpr int kTileSize = 16;

GLFWwindow *g_wnd;
GLuint g_tex;
hipGraphicsResource_t g_hip_tex;
int g_hip_dev;

static __global__ void SineKern(uchar4 *buf, float time) {
  constexpr float kFreq = 10.f;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = y * kTexWidth + x;
  float u = 2.f * x / kTexWidth - 1.f;
  float v = 2.f * y / kTexHeight - 1.f;

  if (x < kTexWidth && y < kTexHeight) {
    float val = sinf(u * kFreq + time) * cosf(v * kFreq + time);
    auto scaled = uchar3(255 * ((val + 1) / 2));
    buf[i] = uchar4(scaled.x, scaled.y, scaled.z, 255);
  }
}

static GLFWwindow *CreateWindow() {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
  GLFWwindow *wnd = glfwCreateWindow(800, 800, "crast", nullptr, nullptr);
  if (!wnd) {
    std::cerr << "glfw window creation failure\n";
    SoftExit(-1);
  }
  glfwSetFramebufferSizeCallback(
      wnd, [](auto, int w, int h) { glViewport(0, 0, w, h); });
  return wnd;
}

static void InitGl() {
  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    std::cerr << "gl loader failure\n";
    SoftExit(-1);
  }
  if (GLAD_GL_ARB_debug_output) {
    // set up automatic error logging
    glDebugMessageCallbackARB(
        [](auto, auto, auto, GLenum severity, GLsizei len, const char *msg,
           auto) {
          std::cerr << "OpenGL: ";
          std::cerr.write(msg, len) << "\n";
          if (severity == GL_DEBUG_SEVERITY_HIGH_ARB) {
            SoftExit(-1);
          }
        },
        nullptr);
  }
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
}

static int PickHipDev() {
  unsigned int dev_count;
  int hip_dev;
  HIP_CHECK(hipGLGetDevices(&dev_count, &hip_dev, 1, hipGLDeviceListAll));
  if (dev_count == 0) {
    std::cerr << "no opengl intercompatible devices found\n";
    SoftExit(-1);
  }
  return hip_dev;
}

static void CreateDestTex(GLuint &tex, hipGraphicsResource_t &hip_tex, int w,
                          int h) {
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  HIP_CHECK(hipGraphicsGLRegisterImage(&hip_tex, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsWriteDiscard));
}

struct Renderer : private boost::noncopyable {
  std::chrono::high_resolution_clock::time_point start_time_;
  hipStream_t gl_stream_;
  hipGraphicsResource_t tex_res_;
  uchar4 *hip_buf_;
  size_t pitch_;

  Renderer(int hip_dev, hipGraphicsResource_t tex_res) {
    start_time_ = std::chrono::high_resolution_clock::now();
    tex_res_ = tex_res;

    HIP_CHECK(hipStreamCreate(&gl_stream_));

    // use the 2D allocation API to get a better pitch value
    // this should help with the device-to-device copying we perform
    // this buffer will be used by the compute kernel directly
    HIP_CHECK(hipMallocPitch(&hip_buf_, &pitch_, 4 * kTexWidth, kTexHeight));
  }

  void Unmap() {
    HIP_CHECK(hipGraphicsUnmapResources(1, &tex_res_, gl_stream_));
  }

  void Update() {
    auto now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     now - this->start_time_)
                     .count();
    SineKern<<<dim3(ceiling_div(kTexWidth, kTileSize),
                    ceiling_div(kTexHeight, kTileSize)),
               dim3(kTileSize, kTileSize), 0, gl_stream_>>>(hip_buf_, time);

    // temporarily map the OpenGL texture as a HIP resource so that we may copy
    // the data from the kernel's output to it
    hipArray_t tex_buf;
    HIP_CHECK(hipGraphicsMapResources(1, &tex_res_, gl_stream_));
    HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&tex_buf, tex_res_, 0, 0));

    // unmap it after the operation is done (or in case of an error)
    BOOST_SCOPE_EXIT(this_) { this_->Unmap(); }
    BOOST_SCOPE_EXIT_END

    HIP_CHECK(hipMemcpy2DToArrayAsync(tex_buf, 0, 0, hip_buf_, pitch_,
                                      4 * kTexWidth, kTexHeight,
                                      hipMemcpyDefault, gl_stream_));
    HIP_CHECK(hipStreamSynchronize(gl_stream_));
  }

  ~Renderer() {
    HIP_CHECK(hipStreamSynchronize(gl_stream_));
    Unmap();
    HIP_CHECK(hipStreamDestroy(gl_stream_));
  }
};

static int Run() {
  glfwSetErrorCallback([](int code, const char *msg) {
    std::cerr << format{"glfw error encountered: %1% (%2%)\n"} % msg % code;
  });
  if (!glfwInit()) {
    std::cerr << "glfw init error\n";
    SoftExit(-1);
  }
  BOOST_SCOPE_EXIT(void) {
    std::cout << "graceful termination\n";
    glfwTerminate();
  }
  BOOST_SCOPE_EXIT_END

  g_wnd = CreateWindow();
  glfwMakeContextCurrent(g_wnd);
  InitGl();
  glfwSwapInterval(1);

  g_hip_dev = PickHipDev();
  HIP_CHECK(hipSetDevice(g_hip_dev));

  CreateDestTex(g_tex, g_hip_tex, kTexWidth, kTexHeight);
  Renderer ren(g_hip_dev, g_hip_tex);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  while (!glfwWindowShouldClose(g_wnd)) {
    ren.Update();

    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, g_tex);
    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);

    // be lazy, get it done *immediately*
    glVertex3f(-0.5f, -0.5f, 0.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(0.5f, -0.5f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(0.5f, 0.5f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(-0.5f, 0.5f, 0.0f);
    glTexCoord2f(0.0f, 1.0f);

    glEnd();

    glDisable(GL_TEXTURE_2D);

    glfwSwapBuffers(g_wnd);
    glfwPollEvents();
  }

  return 0;
}

int main() {
  try {
    return Run();
  } catch (int e) {
    return e;
  }
}
