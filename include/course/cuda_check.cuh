#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <utility>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t course_error__ = (expr);                                         \
    if (course_error__ != cudaSuccess) {                                         \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "      \
                << cudaGetErrorString(course_error__) << '\n';                  \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (false)

namespace course {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ != 0) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = std::exchange(other.ptr_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

  void copy_from_host(const T* src) {
    CUDA_CHECK(cudaMemcpy(ptr_, src, bytes(), cudaMemcpyHostToDevice));
  }

  void copy_to_host(T* dst) const {
    CUDA_CHECK(cudaMemcpy(dst, ptr_, bytes(), cudaMemcpyDeviceToHost));
  }

  void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, bytes())); }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

inline void check_last_kernel(const char* label) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "Kernel launch failed (" << label << "): "
              << cudaGetErrorString(error) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace course
