#pragma once

#include <boost/noncopyable.hpp>

#include <hip/hip_runtime.h>

#include "hip_util.hpp"

namespace pipeline {

struct PipelineParams {
  int tex_width = 1024;
  int tex_height = 1024;

  int max_binning_batch = 0x200;
  int raster_tile_dim = 32;

  // the texture type is fixed as RGBA8

  bool order_independent = false;
};

/// A batch to pass to the binning stage (bypassing any vertex processing in the
/// pipeline).
struct BinningBatch {
  size_t tris_count;

  // only the first 3 elements of each vertex are used
  device_ptr<float4> verts;

  // only the first 3 elements of each triangle are used
  // these represent indices into the vertex buffer
  device_ptr<uint4> tri_idxs;
};

struct Pipeline : boost::noncopyable {
  // FIXME: change back later
  static constexpr int kBinningBlockSize = 4;

  Pipeline(const PipelineParams &p);
  ~Pipeline();

  inline auto GetStream() { return stream_; }
  inline auto GetHipTargetBuffer() { return hip_target_buf_; }
  inline size_t MaxItemsBins() const {
    return static_cast<size_t>(p_.max_binning_batch) * tile_count_x_ *
           tile_count_y_;
  }

  void FeedBinningAsync(const BinningBatch &batch);

  PipelineParams p_;
  device_ptr<uchar4> hip_target_buf_;
  device_ptr<uint> hip_bins_;
  device_ptr<uint> hip_bins_sorted_;
  device_ptr<uint> hip_bin_counts_;
  device_ptr<uint> hip_bin_offsets_;
  size_t temp_storage_bytes_;
  device_ptr<void> hip_temp_storage_;
  size_t buf_pitch_;
  hipStream_t stream_;
  int tile_count_x_;
  int tile_count_y_;
};

}  // namespace pipeline
