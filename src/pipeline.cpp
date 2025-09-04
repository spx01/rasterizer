#include "pipeline.hpp"

#include "glad/glad.h"

#include <hip/hip_runtime.h>

#include <hip/hip_gl_interop.h>

#include <thrust/reduce.h>

#include <hipcub/hipcub.hpp>

#include <glm/ext/matrix_transform.hpp>

#include <boost/format.hpp>
#include <boost/scope_exit.hpp>

#include "device_buffer.hpp"

using namespace pipeline;

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

struct Pipeline::Buffers {
  DeviceBuffer<float4> vert_screen_pos;
  DeviceBuffer<float4> vert_screen_normal;

  DeviceBuffer<uint> bins;
  DeviceBuffer<uint> bins_sorted;
  DeviceBuffer<uint> bin_counts;
  DeviceBuffer<uint> bin_offsets;
  DeviceBuffer<uchar> temp_storage;

  DeviceBuffer<uchar4> target;
  size_t target_pitch;
  DeviceBuffer<float> depth;
  size_t depth_pitch;
};

static void Sync(Pipeline &p) { HIP_CHECK(hipStreamSynchronize(p.stream_)); }

static void StateBufferInit(Pipeline &p) {
  UniformState state0{};
  state0.mvp_mat = std::bit_cast<Mat4>(glm::identity<glm::mat4>());
  state0.normal_mat = std::bit_cast<Mat3>(glm::identity<glm::mat3>());
  p.uniform_buf_.push_back(state0);
  p.uniform_buf_.push_back(state0);
}

static const UniformState &StateBufferCommit(Pipeline &p) {
  auto &buf = p.uniform_buf_;
  auto size = buf.size();
  assert(size >= 2);
  const auto &last_committed = buf[size - 2];
  if (buf.back() != last_committed) {
    buf.push_back(buf.back());
  }
  return buf.back();
}

Resource::TargetTex::~TargetTex() {
  glDeleteTextures(1, &id);
  HIP_CHECK(hipGraphicsUnregisterResource(hip_handle));
}

Resource *ResourceMgr::Create() {
  resources.emplace_back();
  return &resources.back();
}

Pipeline::Pipeline(const Params &p)
    : bufs_(std::make_unique<Buffers>()), param_(p) {
  tile_count_x_ = hutil::CeilDiv(param_.tex_width, param_.raster_tile_dim);
  tile_count_y_ = hutil::CeilDiv(param_.tex_height, param_.raster_tile_dim);
  size_t bins_count = static_cast<size_t>(tile_count_x_) * tile_count_y_;

  HIP_CHECK(hipStreamCreate(&stream_));

  bufs_->vert_screen_pos =
      DeviceBuffer<float4>(p.max_vertex_buffer_size * sizeof(float4), stream_);
  bufs_->vert_screen_normal =
      DeviceBuffer<float4>(p.max_vertex_buffer_size * sizeof(float4), stream_);

  bufs_->bins = DeviceBuffer<uint>(MaxItemsBins(), stream_);
  bufs_->bins_sorted = DeviceBuffer<uint>(MaxItemsBins(), stream_);
  bufs_->bin_counts = DeviceBuffer<uint>(bins_count, stream_);
  bufs_->bin_offsets = DeviceBuffer<uint>(bins_count + 2, stream_);
  bufs_->temp_storage = DeviceBuffer<uchar>(nullptr, 0, stream_);

  bufs_->bins.FillBytes(0);
  bufs_->bin_counts.FillBytes(0);
  bufs_->bin_offsets.FillBytes(0);

  bufs_->target = DeviceBuffer<uchar4>(
      [&](device_ptr<uchar4> &p) {
        HIP_CHECK(hipMallocPitch(
            reinterpret_cast<void **>(&p), &bufs_->target_pitch,
            param_.tex_width * sizeof(uchar4), param_.tex_height));
        assert(bufs_->target_pitch % sizeof(uchar4) == 0);
        return bufs_->target_pitch / sizeof(uchar4) * param_.tex_height;
      },
      stream_);
  bufs_->depth = DeviceBuffer<float>(
      [&](device_ptr<float> &p) {
        HIP_CHECK(hipMallocPitch(
            reinterpret_cast<void **>(&p), &bufs_->depth_pitch,
            param_.tex_width * sizeof(float), param_.tex_height));
        assert(bufs_->depth_pitch % sizeof(float) == 0);
        return bufs_->depth_pitch / sizeof(float) * param_.tex_height;
      },
      stream_);
}

Pipeline::~Pipeline() {
  Sync(*this);
  HIP_CHECK(hipStreamDestroy(stream_));
}

UniformState::UniformState() { memset(this, 0, sizeof(*this)); }

bool UniformState::operator==(const UniformState &other) const {
  return memcmp(this, &other, sizeof(*this)) == 0;  // NOLINT
}

struct BinningProcessing {
  struct Inputs {
    size_t tri_count;
    const float4 *__restrict__ verts;
    const uint4 *__restrict__ tris;
    int2 dim_in_tiles;
    int tile_dim;
    int num_bins;
  };
  struct Outputs {
    uint *__restrict bins;
    uint *__restrict bin_counts;
    uint *__restrict__ bin_offsets;
  };
};

static __host__ __device__ void TriangleBoundingBox(const float2 &v0,
                                                    const float2 &v1,
                                                    const float2 &v2,
                                                    int &min_x, int &min_y,
                                                    int &max_x, int &max_y) {
  min_x = (int)floorf(fminf(fminf(v0.x, v1.x), v2.x));
  min_y = (int)floorf(fminf(fminf(v0.y, v1.y), v2.y));
  max_x = (int)ceilf(fmaxf(fmaxf(v0.x, v1.x), v2.x));
  max_y = (int)ceilf(fmaxf(fmaxf(v0.y, v1.y), v2.y));
}

enum BinningStages {
  kBinning1,  // compute the bin counts
  kBinning2,  // place the triangles into bins
};

template <BinningStages kStage>
static __global__ void BinningProcessingKern(BinningProcessing::Inputs in,
                                             BinningProcessing::Outputs out) {
  extern __shared__ uint block_bin_counts[];

  // Zero out the block bin counts
  for (int i = threadIdx.x; i < in.num_bins; i += blockDim.x) {
    block_bin_counts[i] = 0;
  }
  __syncthreads();

  const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint4 tri;
  float2 v0, v1, v2;
  int min_x, min_y, max_x, max_y;
  int min_tx, min_ty, max_tx, max_ty;

  if (tri_idx < in.tri_count) {
    tri = in.tris[tri_idx];
    v0 = make_float2(in.verts[tri.x].x, in.verts[tri.x].y);
    v1 = make_float2(in.verts[tri.y].x, in.verts[tri.y].y);
    v2 = make_float2(in.verts[tri.z].x, in.verts[tri.z].y);

    // Compute the bounding box of the triangle
    // NOTE: the triangles have already been clipped, so clamping is not needed
    TriangleBoundingBox(v0, v1, v2, min_x, min_y, max_x, max_y);
    min_tx = min_x / in.tile_dim;
    min_ty = min_y / in.tile_dim;
    max_tx = max_x / in.tile_dim;
    max_ty = max_y / in.tile_dim;

    // Update bins of tiles that intersect the bounding box

    // TODO: use multiple threads to process a single triangle

    const int tile_width = max_tx - min_tx + 1;
    const int tile_height = max_ty - min_ty + 1;
    const int workload = tile_width * tile_height;

    for (int wi = 0, tx = min_tx, ty = min_ty; wi < workload; ++wi) {
      if (tx < in.dim_in_tiles.x && ty < in.dim_in_tiles.y) {
        const int bin_idx = ty * in.dim_in_tiles.x + tx;
        atomicAdd(&block_bin_counts[bin_idx], 1);
      }

      if (tx == max_tx) {
        tx = min_tx - 1;
        ++ty;
      }
      ++tx;
    }
  }
  __syncthreads();

  if constexpr (kStage == kBinning1) {
    // Write the block bin counts to global memory
    for (int i = threadIdx.x; i < in.num_bins; i += blockDim.x) {
      atomicAdd(&out.bin_counts[i], block_bin_counts[i]);
    }
  } else if constexpr (kStage == kBinning2) {
    // block_bin_counts will be repurposed to store bin offsets specific to this
    // block
    for (int i = threadIdx.x; i < in.num_bins; i += blockDim.x) {
      // allocate space to be used by this block for this bin
      block_bin_counts[i] = atomicAdd(&out.bin_offsets[i], block_bin_counts[i]);
    }
    __syncthreads();

    if (tri_idx < in.tri_count) {
      // iterate over the intersecting tiles again and write the triangle to the
      // global list
      const int tile_width = (max_tx - min_tx + 1);
      const int tile_height = (max_ty - min_ty + 1);
      const int workload = tile_width * tile_height;

      for (int wi = 0, tx = min_tx, ty = min_ty; wi < workload; ++wi) {
        if (tx < in.dim_in_tiles.x && ty < in.dim_in_tiles.y) {
          const int bin_idx = ty * in.dim_in_tiles.x + tx;
          const uint bin_offset = atomicAdd(&block_bin_counts[bin_idx], 1);
          out.bins[bin_offset] = tri_idx;
        }

        if (tx == max_tx) {
          tx = min_tx - 1;
          ++ty;
        }
        ++tx;
      }
    }
  }
}

static void RequireTempBuf(Pipeline &p, size_t need) {
  if (need > p.bufs_->temp_storage.size) {
    p.bufs_->temp_storage = DeviceBuffer<uchar>(need, p.stream_);
  }
}

static void ClearBinning(Pipeline &p) {
  auto &bufs = *p.bufs_;
  bufs.bins.FillBytes(0);
  bufs.bin_counts.FillBytes(0);
  bufs.bin_offsets.FillBytes(0);
}

template <typename T>
using BinningInPtr = const T *__restrict__;

static void LaunchBinning(Pipeline &p, VertexData<BinningInPtr> verts,
                          BinningInPtr<uint4> tris, size_t tri_count) {
  auto &bufs = *p.bufs_;
  BinningProcessing::Inputs in{
      .tri_count = tri_count,
      .verts = verts.positions,
      .tris = tris,
      .dim_in_tiles = make_int2(p.tile_count_x_, p.tile_count_y_),
      .tile_dim = p.param_.raster_tile_dim,
      .num_bins = p.tile_count_x_ * p.tile_count_y_,
  };
  BinningProcessing::Outputs out{
      .bins = bufs.bins.Raw(),
      .bin_counts = bufs.bin_counts.Raw(),
      .bin_offsets = bufs.bin_offsets.Raw(),
  };
  int num_blocks = hutil::CeilDiv(
      tri_count, static_cast<size_t>(Pipeline::kBinningBlockSize));
  size_t shared_mem_size = p.tile_count_x_ * p.tile_count_y_ * sizeof(uint);

  BinningProcessingKern<kBinning1>
      <<<num_blocks, Pipeline::kBinningBlockSize, shared_mem_size, p.stream_>>>(
          in, out);

  size_t needed_temp = 0;

  // this scan is practically an exclusive scan placed at hip_bin_offsets_ + 1,
  // with an extra element signifying the end offset

  HIP_CHECK(hipcub::DeviceScan::InclusiveSum(
      nullptr, needed_temp, bufs.bin_counts.Raw(), bufs.bin_offsets.Raw() + 2,
      p.tile_count_x_ * p.tile_count_y_, p.stream_));
  RequireTempBuf(p, needed_temp);
  HIP_CHECK(hipcub::DeviceScan::InclusiveSum(
      bufs.temp_storage.Raw(), needed_temp, bufs.bin_counts.Raw(),
      bufs.bin_offsets.Raw() + 2, p.tile_count_x_ * p.tile_count_y_,
      p.stream_));

  // the reason this buffer has an unused first element at this point is because
  // the following stage 2 invocation of the binning kernel utilizes it as
  // temporary storage. by the end of this second stage, the provided slice is
  // effectively shifted by one element to the left, meaning in the end,
  // hip_bin_offsets_ will contain the offsets at location +0
  out.bin_offsets = bufs.bin_offsets.Raw() + 1;

  BinningProcessingKern<kBinning2>
      <<<num_blocks, Pipeline::kBinningBlockSize, shared_mem_size, p.stream_>>>(
          in, out);

  if (!p.param_.order_independent) {
    // sort the triangles in each bin by their index in order to preserve the
    // stability of the binning

    HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, needed_temp, bufs.bins.Raw(), bufs.bins_sorted.Raw(),
        p.MaxItemsBins(), p.tile_count_x_ * p.tile_count_y_,
        bufs.bin_offsets.Raw(), bufs.bin_offsets.Raw() + 1, 0, 8 * sizeof(uint),
        p.stream_));
    RequireTempBuf(p, needed_temp);
    HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(
        bufs.temp_storage.Raw(), needed_temp, bufs.bins.Raw(),
        bufs.bins_sorted.Raw(), p.MaxItemsBins(),
        p.tile_count_x_ * p.tile_count_y_, bufs.bin_offsets.Raw(),
        bufs.bin_offsets.Raw() + 1, 0, 8 * sizeof(uint), p.stream_));
    bufs.bins.swap(bufs.bins_sorted);
  }
}

struct TileProcessing {
  struct Inputs {
    const float4 *__restrict__ verts;
    const float4 *__restrict__ colors;
    const float4 *__restrict__ normals;
    const uint4 *__restrict__ tri_idxs;
    const uint *__restrict__ bin_offsets;
    const uint *__restrict__ bins;
    int tile_dim;
    int2 dim_in_tiles;
    int tex_width, tex_height;
    size_t target_pitch;
    size_t depth_pitch;
  };
  struct Outputs {
    uchar4 *__restrict__ target;
    float *__restrict__ depth;  // This is actually an input as well
  };
};

// Compute barycentric coordinates (u,v,w) of point p with respect to triangle
// v0, v1, v2. Returns true if the point lies inside the triangle (including
// edges)
static __device__ bool PointInTriangle2D(const float2 &p, const float2 &v0,
                                         const float2 &v1, const float2 &v2,
                                         float &u, float &v, float &w) {
  const float tol = 1e-5F;
  const float2 v0v1 = make_float2(v1.x - v0.x, v1.y - v0.y);
  const float2 v0v2 = make_float2(v2.x - v0.x, v2.y - v0.y);
  const float2 v0p = make_float2(p.x - v0.x, p.y - v0.y);
  const float area = 0.5F * (-v0v2.y * v0v1.x + v0v1.y * v0v2.x);
  if (area < tol && area > -tol) return false;

  // TODO: configurable backface behavior
  if (area < 0.F) {
    return false;
  }

  const float d00 = v0v1.x * v0v1.x + v0v1.y * v0v1.y;
  const float d01 = v0v1.x * v0v2.x + v0v1.y * v0v2.y;
  const float d11 = v0v2.x * v0v2.x + v0v2.y * v0v2.y;
  const float d20 = v0p.x * v0v1.x + v0p.y * v0v1.y;
  const float d21 = v0p.x * v0v2.x + v0p.y * v0v2.y;

  const float denom = d00 * d11 - d01 * d01;
  if (denom == 0.0F) return false;  // degenerate triangle

  v = (d11 * d20 - d01 * d21) / denom;
  w = (d00 * d21 - d01 * d20) / denom;
  u = 1.0F - v - w;

  return (u >= -tol) && (v >= -tol) && (w >= -tol);
}

static __device__ float4 InterpBarycentricF4Persp(const float4 &c0,
                                                  const float4 &c1,
                                                  const float4 &c2, float u,
                                                  float v, float w, float q0,
                                                  float q1, float q2) {
  const float qi = u * q0 + v * q1 + w * q2;
  if (qi == 0.F) return make_float4(0.F, 0.F, 0.F, 0.F);
  const float4 ci = make_float4(u * c0.x * q0 + v * c1.x * q1 + w * c2.x * q2,
                                u * c0.y * q0 + v * c1.y * q1 + w * c2.y * q2,
                                u * c0.z * q0 + v * c1.z * q1 + w * c2.z * q2,
                                u * c0.w * q0 + v * c1.w * q1 + w * c2.w * q2);
  return make_float4(ci.x / qi, ci.y / qi, ci.z / qi, ci.w / qi);
}

static __device__ float3 InterpBarycentricF3Persp(const float3 &c0,
                                                  const float3 &c1,
                                                  const float3 &c2, float u,
                                                  float v, float w, float q0,
                                                  float q1, float q2) {
  const float qi = u * q0 + v * q1 + w * q2;
  if (qi == 0.F) return make_float3(0.F, 0.F, 0.F);
  const float3 ci = make_float3(u * c0.x * q0 + v * c1.x * q1 + w * c2.x * q2,
                                u * c0.y * q0 + v * c1.y * q1 + w * c2.y * q2,
                                u * c0.z * q0 + v * c1.z * q1 + w * c2.z * q2);
  return make_float3(ci.x / qi, ci.y / qi, ci.z / qi);
}

static __device__ uchar FloatClampByte(float f) {
  return f < 0.F ? 0 : (f > 1.0F ? 255 : static_cast<uchar>(f * 255.F));
}

static __device__ uchar4 QuantizeColor(float4 c) {
  return make_uchar4(FloatClampByte(c.x), FloatClampByte(c.y),
                     FloatClampByte(c.z), FloatClampByte(c.w));
}

static __device__ float4 Normalize(const float4 v) {
  float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  if (len > 0.F) {
    return make_float4(v.x / len, v.y / len, v.z / len, v.w);
  }
  return make_float4(0.F, 0.F, 0.F, v.w);
}

static __global__ void TileProcessingKern(TileProcessing::Inputs in,
                                          TileProcessing::Outputs out) {
  // TODO: proper parameter
  constexpr bool skip_depth_test = false;

  extern __shared__ float depth_buffer[];

  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int local_id = local_y * in.tile_dim + local_x;
  const int px = tile_x * in.tile_dim + local_x;
  const int py = tile_y * in.tile_dim + local_y;
  if (px >= in.tex_width || py >= in.tex_height) return;

  // Load the depth buffer from the last batch
  float *const global_depth =
      (float *)((char *)out.depth + py * in.depth_pitch) + px;  // NOLINT
  depth_buffer[local_id] = *global_depth;

  const uint tile_idx = tile_y * in.dim_in_tiles.x + tile_x;
  const uint bin_start = in.bin_offsets[tile_idx];
  const uint bin_end = in.bin_offsets[tile_idx + 1];

  bool covered = false;
  float4 interp_col = make_float4(0, 0, 0, 1.0F);
  float4 color;

  for (uint i = bin_start; i < bin_end; ++i) {
    const uint tri_idx = in.bins[i];
    const uint4 tri = in.tri_idxs[tri_idx];
    const float4 v0_raw = in.verts[tri.x];
    const float4 v1_raw = in.verts[tri.y];
    const float4 v2_raw = in.verts[tri.z];
    const float2 v0 = make_float2(v0_raw.x, v0_raw.y);
    const float2 v1 = make_float2(v1_raw.x, v1_raw.y);
    const float2 v2 = make_float2(v2_raw.x, v2_raw.y);
    const float2 p = make_float2(px + 0.5F, py + 0.5F);  // center of pixel
    float u, v, w;
    if (PointInTriangle2D(p, v0, v1, v2, u, v, w)) {
      const float depth = u * v0_raw.z + v * v1_raw.z + w * v2_raw.z;

      // Interpolate vertex colors using barycentric coordinates
      const float4 c0 = in.colors[tri.x];
      const float4 c1 = in.colors[tri.y];
      const float4 c2 = in.colors[tri.z];

      interp_col =
          InterpBarycentricF4Persp(c0, c1, c2, u, v, w, in.verts[tri.x].w,
                                   in.verts[tri.y].w, in.verts[tri.z].w);

      // TODO: blending
      // TODO: clipping stage
      if (depth >= 0.F && depth <= 1.0F &&
          (skip_depth_test || depth < depth_buffer[local_id])) {
        covered = true;
        color = interp_col;
        depth_buffer[local_id] = depth;

        // FIXME: placeholder demo
        color = Normalize(InterpBarycentricF4Persp(
            in.normals[tri.x], in.normals[tri.y], in.normals[tri.z], u, v, w,
            in.verts[tri.x].w, in.verts[tri.y].w, in.verts[tri.z].w));
        color.x = fabsf(color.x);
        color.y = fabsf(color.y);
        color.z = fabsf(color.z);
        color.w = 1.F;
      }
    }
  }

  if (covered) {
    uchar4 *const row =
        (uchar4 *)((char *)out.target + py * in.target_pitch);  // NOLINT
    row[px] = QuantizeColor(color);

    if (!skip_depth_test) {
      // Write the depth buffer back to global memory
      *global_depth = depth_buffer[local_id];
    }
  }
}

static auto &TargetTex(const DrawContext &ctx) {
  return std::get<Resource::TargetTex>(ctx.target->data);
}

void Pipeline::Begin(DrawContext ctx) {
  StateBufferInit(*this);
  draw_context_ = ctx;
  ClearBinning(*this);
  bufs_->target.FillBytes(0);

  // 0x42424242 > 1.F
  constexpr uint8_t k_magic_spray = 0x42;
  bufs_->depth.FillBytes(k_magic_spray);
}

struct VertexProcessing {
  struct Inputs {
    UniformState uniform_state;
    const float4 *__restrict__ positions;
    const float4 *__restrict__ normals;
    const uint *__restrict__ indices;
    size_t vert_count;
    int2 screen_dim;
  };
  struct Outputs {
    float4 *__restrict__ positions;
    float4 *__restrict__ normals;
  };
};

static __device__ float4 operator*(const Mat4 &m, const float4 &v) {
  const auto &a = m.data;
  const float x = a[0] * v.x + a[4] * v.y + a[8] * v.z + a[12];
  const float y = a[1] * v.x + a[5] * v.y + a[9] * v.z + a[13];
  const float z = a[2] * v.x + a[6] * v.y + a[10] * v.z + a[14];
  const float w = a[3] * v.x + a[7] * v.y + a[11] * v.z + a[15];
  return make_float4(x, y, z, w);
}

static __device__ float3 operator*(const Mat3 &m, const float3 &v) {
  const auto &a = m.data;
  const float x = a[0] * v.x + a[3] * v.y + a[6] * v.z;
  const float y = a[1] * v.x + a[4] * v.y + a[7] * v.z;
  const float z = a[2] * v.x + a[5] * v.y + a[8] * v.z;
  return make_float3(x, y, z);
}

static __device__ float4 ToScreen(const float4 &p, const int2 &dim) {
  return make_float4(p.x / p.w * dim.x / 2 + dim.x / 2.F,
                     p.y / p.w * dim.y / 2 + dim.y / 2.F, p.z / p.w + 0.5F,
                     // store q = 1/w in w for perspective-correct interpolation
                     1 / p.w);
}

static __global__ void VertexProcessingKern(VertexProcessing::Inputs in,
                                            VertexProcessing::Outputs out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < in.vert_count) {
    const float4 pos_raw = in.positions[idx];
    const auto pos = make_float4(pos_raw.x, pos_raw.y, pos_raw.z, 1.0F);
    const float4 normal_raw = in.normals[idx];
    const auto normal = make_float3(normal_raw.x, normal_raw.y, normal_raw.z);
    const auto pos_t = in.uniform_state.mvp_mat * pos;
    const auto normal_t = in.uniform_state.normal_mat * normal;
    out.positions[idx] = ToScreen(pos_t, in.screen_dim);
    out.normals[idx] =
        Normalize(make_float4(normal_t.x, normal_t.y, normal_t.z, 1.F));
  }
}

void Pipeline::End() {
  Flush();
  auto &target_tex = TargetTex(*draw_context_);
  hipArray_t tex_buf;
  HIP_CHECK(hipGraphicsMapResources(1, &target_tex.hip_handle, stream_));
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&tex_buf,
                                                 target_tex.hip_handle, 0, 0));

  HIP_CHECK(hipMemcpy2DToArrayAsync(tex_buf, 0, 0, bufs_->target.Raw(),
                                    bufs_->target_pitch, 4 * param_.tex_width,
                                    param_.tex_height, hipMemcpyDeviceToDevice,
                                    stream_));

  HIP_CHECK(hipGraphicsUnmapResources(1, &target_tex.hip_handle, stream_));
  draw_context_ = std::nullopt;
  Sync(*this);
}

void Pipeline::Flush() {}

void Pipeline::SetMvpMat(const glm::mat4 &mvp) {
  uniform_buf_.back().mvp_mat = std::bit_cast<Mat4>(mvp);
}

void Pipeline::SetNormalMat(const glm::mat3 &normal) {
  uniform_buf_.back().normal_mat = std::bit_cast<Mat3>(normal);
}

void Pipeline::DrawTrianglesPadded(Handle<kVertexBuffer> vb_h,
                                   Handle<kIndexBuffer> ib_h) {
  // TODO: draw command queue?

  const auto &vb = std::get<Resource::VertexBuffer>(vb_h->data);
  const auto &ib = std::get<Resource::IndexBuffer>(ib_h->data);

  const UniformState &uniforms = StateBufferCommit(*this);
  size_t vert_count = ib.buf.size;
  assert(vert_count % 4 == 0);
  const size_t unique_vert_count = vb.bufs.positions.size;

  {
    VertexProcessing::Inputs in{
        .uniform_state = uniforms,
        .positions = vb.bufs.positions.Raw(),
        .normals = vb.bufs.normals.Raw(),
        .indices = ib.buf.Raw(),
        .vert_count = unique_vert_count,
        .screen_dim = make_int2(param_.tex_width, param_.tex_height),
    };
    VertexProcessing::Outputs out{
        .positions = bufs_->vert_screen_pos.Raw(),
        .normals = bufs_->vert_screen_normal.Raw(),
    };

    int num_blocks =
        hutil::CeilDiv(unique_vert_count, static_cast<size_t>(256));
    VertexProcessingKern<<<num_blocks, 256, 0, stream_>>>(in, out);
  }

  for (size_t start_idx = 0; start_idx < vert_count;
       start_idx += param_.max_trig_batch * 4) {
    size_t end_idx =
        std::min(start_idx + param_.max_trig_batch * 4, vert_count);
    size_t batch_tris = (end_idx - start_idx) / 4;

    // No primitive assembly needed
    auto tris = reinterpret_cast<BinningInPtr<uint4>>(ib.buf.Raw() + start_idx);
    LaunchBinning(*this,
                  VertexData<BinningInPtr>{
                      .positions = bufs_->vert_screen_pos.Raw(),
                      .normals = bufs_->vert_screen_normal.Raw(),
                      .texcoords = nullptr,
                      .colors = nullptr,
                  },
                  tris, batch_tris);

    {
      TileProcessing::Inputs in{
          .verts = bufs_->vert_screen_pos.Raw(),
          .colors = vb.bufs.colors.Raw(),
          .normals = bufs_->vert_screen_normal.Raw(),
          .tri_idxs = reinterpret_cast<const uint4 *__restrict__>(ib.buf.Raw() +
                                                                  start_idx),
          .bin_offsets = bufs_->bin_offsets.Raw(),
          .bins = bufs_->bins.Raw(),
          .tile_dim = param_.raster_tile_dim,
          .dim_in_tiles = make_int2(tile_count_x_, tile_count_y_),
          .tex_width = param_.tex_width,
          .tex_height = param_.tex_height,
          .target_pitch = bufs_->target_pitch,
          .depth_pitch = bufs_->depth_pitch,
      };
      TileProcessing::Outputs out{
          .target = bufs_->target.Raw(),
          .depth = bufs_->depth.Raw(),
      };
      const size_t shared_mem_size =
          param_.raster_tile_dim * param_.raster_tile_dim * sizeof(float);
      TileProcessingKern<<<dim3(tile_count_x_, tile_count_y_),
                           dim3(param_.raster_tile_dim, param_.raster_tile_dim),
                           shared_mem_size, stream_>>>(in, out);
    }
  }
}

GLuint Pipeline::GetTargetTexGlId(Handle<kTargetTexture> h) {
  return std::get<Resource::TargetTex>(h->data).id;
}

Handle<kTargetTexture> Pipeline::CreateTargetTex() {
  auto &res = *res_.Create();
  auto &target_tex = res.data.emplace<Resource::TargetTex>();
  glGenTextures(1, &target_tex.id);
  glBindTexture(GL_TEXTURE_2D, target_tex.id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, param_.tex_width, param_.tex_height,
               0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  HIP_CHECK(hipGraphicsGLRegisterImage(&target_tex.hip_handle, target_tex.id,
                                       GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsWriteDiscard));
  return &res;
}

Handle<kVertexBuffer> Pipeline::UploadVertexBuffer(VertexData<CSpan> verts) {
  assert(verts.positions.size() <= bufs_->vert_screen_normal.size);
  auto &res = *res_.Create();
  res.data = Resource::VertexBuffer{
      .bufs = {
          .positions = DeviceBuffer(verts.positions, stream_),
          .normals = DeviceBuffer(verts.normals, stream_),
          .texcoords = DeviceBuffer(verts.texcoords, stream_),
          .colors = DeviceBuffer(verts.colors, stream_),
      }};
  return &res;
}

Handle<kIndexBuffer> Pipeline::UploadIndexBuffer(std::span<const uint> idxs) {
  auto &res = *res_.Create();
  res.data = Resource::IndexBuffer{.buf = DeviceBuffer(idxs, stream_)};
  return &res;
}
