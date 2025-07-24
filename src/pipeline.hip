#include "pipeline.hpp"

#include <thrust/reduce.h>

#include <hipcub/hipcub.hpp>

using namespace pipeline;

Pipeline::Pipeline(const PipelineParams &p) : p_(p) {
  tile_count_x_ = hutil::CeilDiv(p_.tex_width, p_.raster_tile_dim);
  tile_count_y_ = hutil::CeilDiv(p_.tex_height, p_.raster_tile_dim);
  size_t bins_count = tile_count_x_ * tile_count_y_;

  HIP_CHECK(hipStreamCreate(&stream_));

  // TODO: RAII buffer abstraction (or use thrust::device_vector?)
  hip_target_buf_ =
      hutil::DevMallocPitch<uchar4>(buf_pitch_, p_.tex_width, p_.tex_height);
  hip_bins_ = hutil::DevMalloc<uint>(MaxItemsBins());
  hip_bins_sorted_ = hutil::DevMalloc<uint>(MaxItemsBins());
  hip_bin_counts_ = hutil::DevMalloc<uint>(bins_count);
  // the addition of 2 elements is explained in in FeedBinningAsync()
  hip_bin_offsets_ = hutil::DevMalloc<uint>(bins_count + 2);
  hip_temp_storage_ = nullptr;

  HIP_CHECK(hipMemset2DAsync(hip_target_buf_.get(), buf_pitch_, 0,
                             4 * p_.tex_width, p_.tex_height, stream_));
  HIP_CHECK(hipMemsetAsync(hip_bins_.get(), 0, sizeof(uint) * MaxItemsBins(),
                           stream_));
  HIP_CHECK(hipMemsetAsync(hip_bins_sorted_.get(), 0,
                           sizeof(uint) * MaxItemsBins(), stream_));
  HIP_CHECK(hipMemsetAsync(hip_bin_counts_.get(), 0, sizeof(uint) * bins_count,
                           stream_));
  HIP_CHECK(hipMemsetAsync(hip_bin_offsets_.get(), 0,
                           sizeof(uint) * (bins_count + 2)));
  HIP_CHECK(hipStreamSynchronize(stream_));
}

Pipeline::~Pipeline() {
  HIP_CHECK(hipStreamSynchronize(stream_));
  HIP_CHECK(hipStreamDestroy(stream_));
  HIP_CHECK(hipFree(hip_target_buf_.get()));
  HIP_CHECK(hipFree(hip_bins_.get()));
  HIP_CHECK(hipFree(hip_bins_sorted_.get()));
  HIP_CHECK(hipFree(hip_bin_counts_.get()));
  HIP_CHECK(hipFree(hip_bin_offsets_.get()));
  if (hip_temp_storage_.get() != nullptr) {
    HIP_CHECK(hipFree(hip_temp_storage_.get()));
  }
}

struct BinningKernParams {
  const float4 *__restrict__ verts;
  const uint4 *__restrict__ tri_idxs;
  uint *__restrict__ bins;
  uint *__restrict__ bin_counts;
  uint *__restrict__ bin_offsets;
  size_t tris_count;
  int2 dim_in_tiles;
  int tile_dim;
  int num_bins;
};

static inline __host__ __device__ void TriangleBoundingBox(
    const float2 &v0, const float2 &v1, const float2 &v2, int &min_x,
    int &min_y, int &max_x, int &max_y) {
  // floor is not needed for minimums, coordinates are positive
  min_x = (int)fminf(fminf(v0.x, v1.x), v2.x);
  min_y = (int)fminf(fminf(v0.y, v1.y), v2.y);

  max_x = (int)ceilf(fmaxf(fmaxf(v0.x, v1.x), v2.x));
  max_y = (int)ceilf(fmaxf(fmaxf(v0.y, v1.y), v2.y));
}

enum BinningStages {
  Binning1 = 0,  // compute the bin counts
  Binning2,      // place the triangles into bins
};

template <BinningStages Stage>
static __global__ void Binning(const BinningKernParams params) {
  extern __shared__ uint block_bin_counts[];

  // zero out the block bin counts
  for (int i = threadIdx.x; i < params.num_bins; i += blockDim.x) {
    block_bin_counts[i] = 0;
  }
  __syncthreads();

  const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint4 tri;
  float2 v0, v1, v2;
  int min_x, min_y, max_x, max_y;
  int min_tx, min_ty, max_tx, max_ty;

  if (tri_idx < params.tris_count) {
    tri = params.tri_idxs[tri_idx];
    v0 = make_float2(params.verts[tri.x].x, params.verts[tri.x].y);
    v1 = make_float2(params.verts[tri.y].x, params.verts[tri.y].y);
    v2 = make_float2(params.verts[tri.z].x, params.verts[tri.z].y);

    // compute the bounding box of the triangle
    // NOTE: the triangles have already been clipped, so clamping is not needed
    TriangleBoundingBox(v0, v1, v2, min_x, min_y, max_x, max_y);
    min_tx = min_x / params.tile_dim;
    min_ty = min_y / params.tile_dim;
    max_tx = max_x / params.tile_dim;
    max_ty = max_y / params.tile_dim;

    // update bins of tiles that intersect the bounding box
    for (int ty = min_ty; ty <= max_ty; ++ty) {
      for (int tx = min_tx; tx <= max_tx; ++tx) {
        if (tx >= params.dim_in_tiles.x || ty >= params.dim_in_tiles.y) {
          continue;
        }

        const int bin_idx = ty * params.dim_in_tiles.x + tx;
        atomicAdd(&block_bin_counts[bin_idx], 1);
      }
    }
  }
  __syncthreads();

  if constexpr (Stage == Binning1) {
    // write the block bin counts to global memory
    for (int i = threadIdx.x; i < params.num_bins; i += blockDim.x) {
      atomicAdd(&params.bin_counts[i], block_bin_counts[i]);
    }
  } else if constexpr (Stage == Binning2) {
    // block_bin_counts will be repurposed to store bin offsets specific to this
    // block
    for (int i = threadIdx.x; i < params.num_bins; i += blockDim.x) {
      // allocate space to be used by this block for this bin
      block_bin_counts[i] =
          atomicAdd(&params.bin_offsets[i], block_bin_counts[i]);
    }
    __syncthreads();

    if (tri_idx < params.tris_count) {
      // iterate over the intersecting tiles again and write the triangle to the
      // global list
      for (int ty = min_ty; ty <= max_ty; ++ty) {
        for (int tx = min_tx; tx <= max_tx; ++tx) {
          if (tx >= params.dim_in_tiles.x || ty >= params.dim_in_tiles.y) {
            continue;
          }

          const int bin_idx = ty * params.dim_in_tiles.x + tx;
          const uint bin_offset = atomicAdd(&block_bin_counts[bin_idx], 1);
          params.bins[bin_offset] = tri_idx;
        }
      }
    }
  }
}

static void ReallocateDeviceBuffer(device_ptr<void> &ptr, size_t size) {
  HIP_CHECK(hipFree(ptr.get()));
  ptr = hutil::DevMalloc<char>(size);
}

void Pipeline::FeedBinningAsync(const BinningBatch &batch) {
  BinningKernParams p{
      .verts = batch.verts.get(),
      .tri_idxs = batch.tri_idxs.get(),
      .bins = hip_bins_.get(),
      .bin_counts = hip_bin_counts_.get(),
      .bin_offsets = nullptr,
      .tris_count = batch.tris_count,
      .dim_in_tiles = make_int2(tile_count_x_, tile_count_y_),
      .tile_dim = p_.raster_tile_dim,
      .num_bins = tile_count_x_ * tile_count_y_,
  };
  int num_blocks =
      hutil::CeilDiv(batch.tris_count, static_cast<size_t>(kBinningBlockSize));
  size_t shared_mem_size = tile_count_x_ * tile_count_y_ * sizeof(uint);

  Binning<Binning1>
      <<<num_blocks, kBinningBlockSize, shared_mem_size, stream_>>>(p);

  auto nosync_exec_policy = thrust_plat::par.on(stream_);

  // this scan is practically an exclusive scan placed at hip_bin_offsets_ + 1,
  // with an extra element signifying the end offset
  thrust::inclusive_scan(nosync_exec_policy, hip_bin_counts_.get(),
                         hip_bin_counts_.get() + tile_count_x_ * tile_count_y_,
                         hip_bin_offsets_.get() + 2);
  // the reason this buffer has an unused first element at this point is because
  // the following stage 2 invokation of the binning kernel utilizes it as
  // temporary storage. by the end of this second stage, the provided slice is
  // effectively shifted by one element to the left, meaning in the end,
  // hip_bin_offsets_ will contain the offsets at location +0
  p.bin_offsets = hip_bin_offsets_.get() + 1;

  Binning<Binning2>
      <<<num_blocks, kBinningBlockSize, shared_mem_size, stream_>>>(p);

  if (!p_.order_independent) {
    // sort the triangles in each bin by their index in order to preserve the
    // stability of the binning
    size_t temp_storage_bytes = 0;

    HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, temp_storage_bytes, hip_bins_.get(), hip_bins_sorted_.get(),
        MaxItemsBins(), tile_count_x_ * tile_count_y_, hip_bin_offsets_.get(),
        hip_bin_offsets_.get() + 1, 0, 8 * sizeof(uint), stream_));

    if (temp_storage_bytes > temp_storage_bytes_) {
      ReallocateDeviceBuffer(hip_temp_storage_, temp_storage_bytes);
      temp_storage_bytes_ = temp_storage_bytes;
    }

    HIP_CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(
        hip_temp_storage_.get(), temp_storage_bytes, hip_bins_.get(),
        hip_bins_sorted_.get(), MaxItemsBins(), tile_count_x_ * tile_count_y_,
        hip_bin_offsets_.get(), hip_bin_offsets_.get() + 1, 0, 8 * sizeof(uint),
        stream_));
    std::swap(hip_bins_, hip_bins_sorted_);
  }
}
