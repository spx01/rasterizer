#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <glm/ext/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <vector>

#include "app.hpp"
#include "mesh.hpp"
#include "pipeline.hpp"

using boost::format;
namespace po = boost::program_options;
using namespace std::chrono_literals;

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
    const int n_verts = 3;
    std::array<std::array<float, 2>, 3> positions{{
        {-0.5F, -0.5F},
        {0.5F, -0.5F},
        {0.0F, 0.5F},
    }};
    std::array<std::array<float, 4>, 3> colors{{
        {1.F, 0.F, 0.F, 1.F},
        {0.F, 1.F, 0.F, 1.F},
        {0.F, 0.F, 1.F, 1.F},
    }};

    auto pos = std::vector<float3>{};
    auto norm = std::vector<float3>{};
    auto tex = std::vector<float2>{};
    auto col = std::vector<float4>{};

    for (int i = 0; i < n_verts; ++i) {
      auto p = positions[i];
      auto c = colors[i];
      pos.push_back(make_float3(p[0], p[1], 0.F));
      norm.push_back(make_float3(0.F, 0.F, 1.F));
      tex.push_back(make_float2(0.0F, 0.0F));
      col.push_back(make_float4(c[0], c[1], c[2], c[3]));
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

struct Renderer {
  int hip_dev_;
  GLuint gl_tex_;
  std::unique_ptr<pipeline::Pipeline> pipeline_;
  pipeline::Handle<pipeline::kTargetTexture> target_tex_;
  TestMesh mesh_;
  pipeline::Handle<pipeline::kVertexBuffer> vb_;
  pipeline::Handle<pipeline::kIndexBuffer> ib_;
  int width_, height_;
  glm::mat4 mvp_;

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
    auto h = mesh_.UploadTo(mesh::kTrianglesPadded, *pipeline_);
    vb_ = h.vb;
    ib_ = h.ib;

    const auto id = glm::identity<glm::mat4>();
    const auto rot =
        glm::rotate(id, glm::pi<float>() / 4.F, glm::vec3(0.F, 0.F, 1.F));
    const auto scale = glm::scale(id, glm::vec3(1 / 4.F));
    const auto trans = glm::translate(id, glm::vec3(0.0F, 0.5F, 0.F));
    mvp_ = trans * rot * scale;
  }

  void SetGlState() {
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glClearColor(0.F, 0.F, 0.F, 1.F);
    glEnable(GL_TEXTURE_2D);
  }

  void Render(uint64_t milliseconds = 0) {
    const uint64_t ms_period = 1000;
    const float t = (milliseconds % ms_period) / static_cast<float>(ms_period);
    const float ang0 = t * glm::pi<float>() * 2.F;

    pipeline_->Begin(pipeline::DrawContext{.target = target_tex_});
    for (int i = 0; i < 32; ++i) {
      const auto ang = glm::pi<float>() / 16.F * i;
      glm::mat4 rot_z = glm::rotate(glm::identity<glm::mat4>(), ang,
                                    glm::vec3(0.F, 0.F, 1.F));
      glm::mat4 rot_y = glm::rotate(glm::identity<glm::mat4>(), ang0 + ang,
                                    glm::vec3(0.F, 1.F, 0.F));
      pipeline_->SetMvpMat(rot_z * rot_y * mvp_);
      pipeline_->DrawTrianglesPadded(vb_, ib_);
    }
    pipeline_->End();

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
  return t2 - t1 >= std::chrono::seconds(1);
}

int main(int argc, char **argv) {
  std::optional<int> quit_after_seconds{};
  po::options_description desc;
  desc.add_options()("profile", "run in profiling mode (automatically exits)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("profile")) {
    quit_after_seconds = 5;
    std::cout << "profiling mode\n";
  }

  struct State {
    std::optional<App::Timestamp> last_report;
    int frame_count = 0;
    GLFWwindow *win = nullptr;
    App::Timestamp auto_close_time = App::Timestamp::max();
  };

  AppBuilder b;
  b.vsync = false;
  b.on_frame = [=](App &app) {
    auto r_ = std::make_shared<Renderer>();
    auto s_ = std::make_shared<State>();
    r_->Init(b.win_width, b.win_height);
    s_->win = app.window_;
    if (quit_after_seconds) {
      s_->auto_close_time = *quit_after_seconds * 1s;
    }
    return [r_, s_](App::FrameData d) {
      auto &r = *r_;
      auto &s = *s_;
      r.Render(d.time_now.count());
      ++s.frame_count;
      if (!s.last_report || SecondPassed(*s.last_report, d.time_now)) {
        auto fps = s.frame_count;
        s.frame_count = 0;
        s.last_report = d.time_now;
        std::cout << format("fps: %d\n") % fps;
      }
      if (d.time_now > s_->auto_close_time) {
        glfwSetWindowShouldClose(s_->win, true);
      }
    };
  };
  auto app = b.Build();
  app->Run();
}
