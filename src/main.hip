#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <hip/hip_runtime.h>

#include <hip/hip_gl_interop.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

#include <boost/core/noncopyable.hpp>
#include <boost/format.hpp>
#include <boost/scope_exit.hpp>

#include <chrono>
#include <iostream>
#include <vector>

#include "pipeline.hpp"

using boost::format;

static inline void SoftExit(int code) { throw code; }

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
    unsigned char c = (unsigned char)(255 * ((val + 1) / 2));
    buf[i] = make_uchar4(c, 0, 0, 255);
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
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void **>(&hip_buf_), &pitch_,
                             4 * kTexWidth, kTexHeight));
  }

  void Unmap() {
    HIP_CHECK(hipGraphicsUnmapResources(1, &tex_res_, gl_stream_));
  }

  void Update() {
    auto now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     now - this->start_time_)
                     .count();
    SineKern<<<dim3(hutil::CeilDiv(kTexWidth, kTileSize),
                    hutil::CeilDiv(kTexHeight, kTileSize)),
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
    HIP_CHECK(hipStreamDestroy(gl_stream_));
  }
};

struct TestingMesh {
  size_t tris_count;
  device_ptr<float4> verts;
  device_ptr<uint4> tri_idxs;
};

template <typename T>
static device_ptr<T> CreateDeviceBuffer(const std::vector<T> &data) {
  auto d_buf = hutil::DevMalloc<T>(data.size());
  HIP_CHECK(hipMemcpy(d_buf.get(), data.data(), data.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return d_buf;
}

static TestingMesh CreateTestMesh(int w, int h) {
  constexpr int kGridDim = 32;

  std::vector<float4> verts;
  verts.reserve(kGridDim * kGridDim);

  float shift_x = static_cast<float>(w) / (2 * kGridDim);
  float shift_y = static_cast<float>(h) / (2 * kGridDim);
  for (int y = 0; y < kGridDim; ++y) {
    for (int x = 0; x < kGridDim; ++x) {
      float fx = static_cast<float>(x) / kGridDim * w + shift_x;
      float fy = static_cast<float>(y) / kGridDim * h + shift_y;
      verts.emplace_back(make_float4(fx, fy, 0.f, 0.f));
    }
  }

  std::vector<uint4> tris;
  auto grid2idx = [](int x, int y) { return y * kGridDim + x; };
  auto add_tri = [&](int x0, int y0, int x1, int y1, int x2, int y2) {
    tris.emplace_back(
        make_uint4(grid2idx(x0, y0), grid2idx(x1, y1), grid2idx(x2, y2), 0));
  };

  // clang-format off
  constexpr int kTrigs[][3][2] = {
    // {{ 1,  1}, {11,  8}, { 5, 23}},
    // {{ 7, 12}, {25,  0}, {20, 19}},
    // {{12, 22}, {15, 20}, {14, 23}},
    // {{ 1, 27}, { 8, 23}, { 4, 29}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  1}},
    {{ 0,  0}, { 1,  0}, { 0,  2}},
  };
  // clang-format on
  for (const auto &tri : kTrigs) {
    add_tri(tri[0][0], tri[0][1], tri[1][0], tri[1][1], tri[2][0], tri[2][1]);
  }

  std::cout << format{"created %1% vertices and %2% triangles\n"} %
                   verts.size() % tris.size();

  return {
      .tris_count = tris.size(),
      .verts = CreateDeviceBuffer(verts),
      .tri_idxs = CreateDeviceBuffer(tris),
  };
}

static void TestPipeline() {
  using namespace pipeline;
  PipelineParams params{};
  Pipeline p(params);
  auto mesh = CreateTestMesh(kTexWidth, kTexHeight);

  BOOST_SCOPE_EXIT(mesh) {
    HIP_CHECK(hipFree(mesh.verts.get()));
    HIP_CHECK(hipFree(mesh.tri_idxs.get()));
  }
  BOOST_SCOPE_EXIT_END

  BinningBatch batch{
      .tris_count = mesh.tris_count,
      .verts = mesh.verts,
      .tri_idxs = mesh.tri_idxs,
  };
  p.FeedBinningAsync(batch);

  auto stream = p.GetStream();
  HIP_CHECK(hipStreamSynchronize(p.GetStream()));

  std::vector<uint> bins(p.MaxItemsBins());
  HIP_CHECK(hipMemcpy(bins.data(), p.hip_bins_.get(),
                      sizeof(uint) * bins.size(), hipMemcpyDeviceToHost));
  std::vector<uint> bin_counts(p.tile_count_x_ * p.tile_count_y_);
  HIP_CHECK(hipMemcpy(bin_counts.data(), p.hip_bin_counts_.get(),
                      bin_counts.size() * sizeof(uint), hipMemcpyDeviceToHost));

  int cur_bin = 0;
  int i = 0;
  while (i < bins.size() && cur_bin < bin_counts.size()) {
    int bin_count = bin_counts[cur_bin];
    if (bin_count == 0) {
      ++cur_bin;
      continue;
    }
    std::cout << "bin " << cur_bin << ": ";
    for (int j = 0; j < bin_count; ++j) {
      std::cout << bins[i++] << " ";
    }
    ++cur_bin;
    std::cout << '\n';
  }
}

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

  TestPipeline();

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
