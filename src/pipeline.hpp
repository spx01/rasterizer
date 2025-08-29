#pragma once

#include <list>
#include <optional>
#include <variant>

#include <boost/noncopyable.hpp>

#include <hip/hip_runtime.h>

#include <glm/mat4x4.hpp>

#include "glad/glad.h"  // IWYU pragma: keep

#include "device_buffer.hpp"
#include "hip_util.hpp"
#include "util.hpp"

namespace pipeline {

struct Params {
  int tex_width = 1024;
  int tex_height = 1024;

  int max_trig_batch = 0x200;
  int raster_tile_dim = 32;

  // The texture type is fixed as RGBA8

  bool order_independent = false;
};

/// A batch to pass to the binning stage (bypassing any vertex processing in the
/// pipeline).
struct BinningBatch {
  size_t tris_count;

  // Only the first 3 elements of each vertex are used
  device_ptr<float4> verts;

  // Only the first 3 elements of each triangle are used
  // these represent indices into the vertex buffer
  device_ptr<uint4> tri_idxs;

  // Heuristic for what proportion of screen space the average triangle AABB
  // covers
  double avg_aabb_coverage = 0.05;
};

/// Column-major order
struct Mat4 {
  float data[16];
};

/// Column-major order
struct Mat3 {
  float data[9];
};

struct UniformState {
  Mat4 mvp_mat;
  Mat3 normal_mat;
  UniformState();
  bool operator==(const UniformState &other) const;
};

enum ResourceType {
  kInvalidResource = 0,
  kTargetTexture,
  kVertexBuffer,
  kIndexBuffer,
};

template <template <typename> typename T>
struct VertexData {
  /// Only the first 3 elements are used
  T<float4> positions;
  /// Only the first 3 elements are used
  T<float4> normals;

  T<float2> texcoords;
  T<float4> colors;
};

struct Resource {
  using Invalid = std::monostate;

  struct TargetTex {
    GLuint id;
    hipGraphicsResource_t hip_handle;
    TargetTex() = default;
    TargetTex(const TargetTex &) = delete;
    ~TargetTex();
  };

  struct VertexBuffer {
    VertexData<DeviceBuffer> bufs;
  };

  struct IndexBuffer {
    DeviceBuffer<uint> buf;
  };

  std::variant<Invalid, TargetTex, VertexBuffer, IndexBuffer> data =
      std::monostate();
};

template <ResourceType T>
using Handle = Resource *;

struct ResourceMgr {
  std::list<Resource> resources;
  Resource *Create();
};

struct DrawContext {
  Handle<kTargetTexture> target;
};

struct Pipeline : private boost::noncopyable {
  static constexpr int kBinningBlockSize = 256;

  Pipeline(const Params &p);
  ~Pipeline();

  auto GetStream() { return stream_; }
  size_t MaxItemsBins() const {
    return static_cast<size_t>(param_.max_trig_batch) * tile_count_x_ *
           tile_count_y_;
  }

  void Begin(DrawContext ctx);
  void End();
  void Flush();
  void SetMvpMat(const glm::mat4 &mvp);
  void SetNormalMat(const glm::mat3 &normal);
  /// `ib` must contain consecutive triangles with 3 indices and one padding
  /// index each
  void DrawTrianglesPadded(Handle<kVertexBuffer> vb, Handle<kIndexBuffer> ib);
  GLuint GetTargetTexGlId(Handle<kTargetTexture> h);

  [[nodiscard]] Handle<kTargetTexture> CreateTargetTex();
  [[nodiscard]] Handle<kVertexBuffer> UploadVertexBuffer(
      VertexData<CSpan> verts);
  [[nodiscard]] Handle<kIndexBuffer> UploadIndexBuffer(CSpan<uint> idxs);

  struct Buffers;
  std::unique_ptr<Buffers> bufs_;

  ResourceMgr res_;
  Params param_;
  hipStream_t stream_;
  int tile_count_x_;
  int tile_count_y_;
  std::vector<UniformState> uniform_buf_;
  std::optional<DrawContext> draw_context_;
};

}  // namespace pipeline
