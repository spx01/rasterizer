#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <hip/hip_runtime.h>

#include <hip/hip_gl_interop.h>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <glm/ext/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <vector>

#include "app.hpp"
#include "mesh.hpp"
#include "obj_mesh.hpp"
#include "pipeline.hpp"

using boost::format;
namespace po = boost::program_options;
using namespace std::chrono_literals;
namespace chrono = std::chrono;

static int PickHipDev() {
  unsigned int dev_count;
  int hip_dev;
  HIP_CHECK(hipGLGetDevices(&dev_count, &hip_dev, 1, hipGLDeviceListAll));
  if (dev_count == 0) {
    std::cerr << "no opengl intercompatible devices found\n";
    exit(-1);
  }
  return hip_dev;
}

struct TestMesh : mesh::Mesh {
  mesh::VertexData GetVertexData() const override {
    constexpr int n_verts = 3;
    constexpr std::array<std::array<float, 2>, 3> positions{{
        {-0.5F, -0.5F},
        {0.5F, -0.5F},
        {0.0F, 0.5F},
    }};
    constexpr std::array<std::array<float, 4>, 3> colors{{
        {1.F, 0.F, 0.F, 1.F},
        {0.F, 1.F, 0.F, 1.F},
        {0.F, 0.F, 1.F, 1.F},
    }};

    auto pos = std::vector<float3>{};
    auto norm = std::vector<float3>{};
    auto tex = std::vector<float2>{};
    auto col = std::vector<float3>{};

    for (int i = 0; i < n_verts; ++i) {
      auto p = positions[i];
      auto c = colors[i];
      pos.push_back(make_float3(p[0], p[1], 0.F));
      norm.push_back(make_float3(0.F, 0.F, 1.F));
      tex.push_back(make_float2(0.0F, 0.0F));
      col.push_back(make_float3(c[0], c[1], c[2]));
    }
    return mesh::VertexData{
        .positions = std::move(pos),
        .normals = std::move(norm),
        .texcoords = std::move(tex),
        .colors = std::move(col),
    };
  }

  buf::Buf<uint> GetTrianglesPadded() const override {
    return std::vector<uint>{0, 1, 2, 0};
  }
};

static glm::mat3x3 InverseTranspose(const glm::mat3x3 &m) {
  return glm::transpose(glm::inverse(m));
}

static glm::mat3x3 NormalTransformMat(const glm::mat4x4 &mv_mat) {
  return InverseTranspose(glm::mat3x3(mv_mat));
}

struct FrameStats {
  using Dt = chrono::duration<double, std::milli>;
  std::vector<Dt> frame_times_;
  bool first_ = true;

  void Update(Dt dt) {
    if (!first_) {
      frame_times_.push_back(dt);
    }
    first_ = false;
  }

  struct Stats {
    Dt avg, min, max;
  };
  Stats GetStats() {
    if (frame_times_.size() <= 1) {
      return Stats{};
    }

    frame_times_.pop_back();
    double sum = 0.0;
    double min = 1e9;
    double max = 0.0;
    for (const auto &dt : frame_times_) {
      sum += dt.count();
      min = std::min(min, dt.count());
      max = std::max(max, dt.count());
    }
    return Stats{.avg = Dt(sum / static_cast<double>(frame_times_.size())),
                 .min = Dt(min),
                 .max = Dt(max)};
  }
};

struct Renderer {
  int hip_dev_;
  GLuint gl_tex_;
  std::unique_ptr<pipeline::Pipeline> pipeline_;
  pipeline::Handle<pipeline::kTargetTexture> target_tex_;
  std::unique_ptr<mesh::Mesh> mesh_;
  pipeline::Handle<pipeline::kVertexBuffer> vb_;
  pipeline::Handle<pipeline::kIndexBuffer> ib_;
  int width_, height_;
  glm::mat4 mv_mat_;
  glm::mat4 proj_mat_;
  FrameStats stats_;
  bool collect_stats_;

  explicit Renderer(bool collect_stats)
      : hip_dev_(0),
        gl_tex_(0),
        target_tex_(nullptr),
        mesh_(std::make_unique<ObjMesh>("assets/suzanne.obj", "assets")),
        vb_(nullptr),
        ib_(nullptr),
        width_(0),
        height_(0),
        mv_mat_(),
        proj_mat_(),
        collect_stats_(collect_stats) {}

  ~Renderer() {}

  void Init(int width, int height) {
    width_ = width;
    height_ = height;
    hip_dev_ = PickHipDev();
    HIP_CHECK(hipSetDevice(hip_dev_));
    pipeline_ = std::make_unique<pipeline::Pipeline>(
        pipeline::Params{.tex_width = width, .tex_height = height});
    target_tex_ = pipeline_->CreateTargetTex();
    gl_tex_ = pipeline_->GetTargetTexGlId(target_tex_);
    auto h = mesh_->UploadTo(mesh::kTrianglesPadded, *pipeline_);
    vb_ = h.vb;
    ib_ = h.ib;

    const auto eye = glm::identity<glm::mat4>();
    const glm::mat4 trans = glm::translate(eye, glm::vec3(0.F, 0.F, -0.7F));
    const glm::mat4 scale = glm::scale(eye, glm::vec3(0.13F));
    mv_mat_ = trans * scale;
  }

  void SetGlState() {
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glClearColor(0.F, 0.F, 0.F, 1.F);
    glEnable(GL_TEXTURE_2D);
  }

  void Render(uint64_t time_ms) {
    constexpr uint64_t ms_period = 3000;

    const auto timer_start = std::chrono::high_resolution_clock::now();

    const float t =
        static_cast<float>(time_ms % ms_period) / static_cast<float>(ms_period);
    const float ang0 = t * glm::pi<float>() * 2.F;
    const int cnt = 5;

    pipeline_->Begin(pipeline::DrawContext{.target = target_tex_});
    const glm::mat4 trans_y =
        glm::translate(glm::identity<glm::mat4>(), glm::vec3(0.F, 0.5F, 0.F));
    for (int i = 0; i < cnt; ++i) {
      const auto ang = glm::pi<float>() / (static_cast<float>(cnt) / 2) *
                       static_cast<float>(i);
      const glm::mat4 rot_z = glm::rotate(glm::identity<glm::mat4>(), ang,
                                          glm::vec3(0.F, 0.F, 1.F));
      const glm::mat4 rot_y = glm::rotate(glm::identity<glm::mat4>(),
                                          ang0 + ang, glm::vec3(0.F, 1.F, 0.F));
      auto m = rot_z * trans_y * rot_y * mv_mat_;
      pipeline_->SetMvpMat(m);
      pipeline_->SetNormalMat(NormalTransformMat(m));
      pipeline_->DrawTrianglesPadded(vb_, ib_);
    }
    pipeline_->End();

    const auto timer_end = std::chrono::high_resolution_clock::now();
    if (collect_stats_) {
      stats_.Update(chrono::duration_cast<chrono::duration<double, std::milli>>(
          timer_end - timer_start));
    }

    SetGlState();
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0F, 0.0F);
    glVertex3f(-1.0F, -1.0F, 0.0F);
    glTexCoord2f(1.0F, 0.0F);
    glVertex3f(1.0F, -1.0F, 0.0F);
    glTexCoord2f(1.0F, 1.0F);
    glVertex3f(1.0F, 1.0F, 0.0F);
    glTexCoord2f(0.0F, 1.0F);
    glVertex3f(-1.0F, 1.0F, 0.0F);
    glEnd();
  }
};

static bool SecondPassed(App::Timestamp t1, App::Timestamp t2) {
  return t2 - t1 >= chrono::seconds(1);
}

int main(int argc, char **argv) {  // NOLINT
  std::optional<int> quit_after_seconds{};
  po::options_description desc;
  desc.add_options()("profile", "run in profiling mode (automatically exits)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  bool profiling = false;
  if (vm.contains("profile")) {
    profiling = true;
    quit_after_seconds = 5;
    std::cout << "profiling mode\n";
  }

  struct State {
    std::optional<App::Timestamp> last_report;
    int frame_count = 0;
    GLFWwindow *win = nullptr;
    App::Timestamp auto_close_time = App::Timestamp::max();
    FrameStats total_stats;
  };

  auto r_ = std::make_shared<Renderer>(profiling);
  auto s_ = std::make_shared<State>();
  AppBuilder b;
  b.vsync = false;
  b.on_frame = [=](App &app) {
    r_->Init(b.win_width, b.win_height);
    s_->win = app.window_;
    if (quit_after_seconds) {
      s_->auto_close_time = *quit_after_seconds * 1s;
    }
    return [r_, s_](App::FrameData d) {
      auto &r = *r_;
      auto &s = *s_;
      s.total_stats.Update(d.delta_time);
      r.Render(chrono::duration_cast<chrono::milliseconds>(d.time_now).count());
      ++s.frame_count;
      if (!s.last_report || SecondPassed(*s.last_report, d.time_now)) {
        auto fps = s.frame_count;
        s.frame_count = 0;
        s.last_report = d.time_now;
        std::cout << format{"fps: %d\n"} % fps;
      }
      if (d.time_now > s_->auto_close_time) {
        glfwSetWindowShouldClose(s_->win, true);
      }
    };
  };
  auto app = b.Build();
  app->Run();
  if (profiling) {
    {
      const auto [avg, min, max] = r_->stats_.GetStats();
      std::cout
          << format{"pipeline frame times - avg %.3f ms, min %.3f ms, max %.3f ms\n"} %
                 avg.count() % min.count() % max.count();
    }
    {
      const auto [avg, min, max] = s_->total_stats.GetStats();
      std::cout
                  << format{"total frame times - avg %.3f ms, min %.3f ms, max %.3f ms\n"} %
                     avg.count() % min.count() % max.count();
    }
  }
}
