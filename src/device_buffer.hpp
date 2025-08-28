#pragma once

#include <span>
#include <variant>

#include <boost/noncopyable.hpp>

#include "hip_util.hpp"

template <typename T>
struct DeviceBuffer : private boost::noncopyable {
  device_ptr<T> ptr = nullptr;
  size_t size = 0;
  hipStream_t stream = nullptr;

  DeviceBuffer() {}

  DeviceBuffer(size_t size, hipStream_t stream);

  DeviceBuffer(device_ptr<T> ptr, size_t size, hipStream_t stream)
      : ptr(ptr), size(size), stream(stream) {}

  DeviceBuffer(const T *host_ptr, size_t size, hipStream_t stream);

  DeviceBuffer(std::span<const T> span, hipStream_t stream)
      : DeviceBuffer(span.data(), span.size(), stream) {}

  template <typename F>
  DeviceBuffer(F &&alloc_fn, hipStream_t stream);

  DeviceBuffer(DeviceBuffer &&) noexcept;

  DeviceBuffer &operator=(DeviceBuffer) noexcept;

  ~DeviceBuffer();

  void FillBytes(uint8_t value);

  device_ptr<T> MoveOut() && { return std::exchange(ptr, nullptr); }

  T *Raw() { return ptr.get(); }
  const T *Raw() const { return ptr.get(); }

  void Sync();

  void swap(DeviceBuffer &other) noexcept {
    using std::swap;
    swap(ptr, other.ptr);
    swap(size, other.size);
    swap(stream, other.stream);
  }
};

template <typename T>
void swap(DeviceBuffer<T> &a, DeviceBuffer<T> &b) noexcept {
  a.swap(b);
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(size_t size, hipStream_t stream)
    : size(size), stream(stream) {
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void **>(&ptr), size * sizeof(T),
                           stream));
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const T *host_ptr, size_t size,
                              hipStream_t stream)
    : DeviceBuffer(size, stream) {
  HIP_CHECK(hipMemcpyAsync(Raw(), host_ptr, size * sizeof(T),
                           hipMemcpyHostToDevice, stream));
}

template <typename T>
template <typename F>
DeviceBuffer<T>::DeviceBuffer(F &&alloc_fn, hipStream_t stream)
    : stream(stream) {
  size = alloc_fn(ptr);
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer &&other) noexcept {
  ptr = std::exchange(other.ptr, nullptr);
  size = other.size;
  stream = other.stream;
}

template <typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator=(DeviceBuffer other) noexcept {
  swap(other);
  return *this;
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer() {
  if (ptr) {
    HIP_CHECK(hipFree(Raw()));
  }
}

template <typename T>
void DeviceBuffer<T>::FillBytes(uint8_t value) {
  HIP_CHECK(hipMemsetAsync(Raw(), value, size * sizeof(T), stream));
}

template <typename T>
void DeviceBuffer<T>::Sync() {
  if (stream != nullptr) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }
}
