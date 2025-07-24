#pragma once

#include <hip/hip_runtime.h>

#include <thrust/device_ptr.h>

#define HIP_CHECK(condition)                                             \
  {                                                                      \
    const hipError_t error = condition;                                  \
    if (error != hipSuccess) {                                           \
      std::cerr << "hip error encountered: \"" << hipGetErrorName(error) \
                << "\" at " __FILE__ << ":" << __LINE__ << "\n";         \
      exit(-1);                                                          \
    }                                                                    \
  }

using thrust::device_ptr;
using uint = unsigned int;

#ifdef __HIP_PLATFORM_NVIDIA__
#define thrust_plat ::thrust::cuda
#elif defined(__HIP_PLATFORM_AMD__)
// TODO: test whether this is actually useful for anything or even works
// #define __HIP_MEMORY_SCOPE_WORKGROUP 3
// __device__ inline uint atomicAdd_block(uint *address, uint val) {
//   return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED,
//                                 __HIP_MEMORY_SCOPE_WORKGROUP);
// }
#define atomicAdd_block atomicAdd
#define thrust_plat ::thrust::hip
#else
#error "Unsupported HIP platform"
#endif

namespace hutil {

template <typename T>
inline device_ptr<T> DevMallocPitch(size_t &pitch, size_t width,
                                    size_t height) {
  void *ptr;
  HIP_CHECK(hipMallocPitch(&ptr, &pitch, sizeof(T) * width, height));
  return device_ptr<T>(reinterpret_cast<T *>(ptr));
}

template <typename T>
inline device_ptr<T> DevMalloc(size_t size) {
  void *ptr;
  HIP_CHECK(hipMalloc(&ptr, sizeof(T) * size));
  return device_ptr<T>(reinterpret_cast<T *>(ptr));
}

template <typename T>
__host__ __device__ constexpr auto CeilDiv(const T &dividend,
                                           const T &divisor) {
  return (dividend + divisor - 1) / divisor;
}

}  // namespace hutil
