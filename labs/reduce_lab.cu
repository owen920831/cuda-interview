#include "course/bench.cuh"
#include "lab_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

constexpr int kBlock = 256;

__global__ void student_v0(const float* input, float* output, int n) {
  // TODO(student): shared-memory tree; exactly one global atomic per block.
}

__global__ void student_v1(const float* input, float* output, int n) {
  // TODO(student): first-add-on-load, then a shared-memory tree.
}

__global__ void student_v2(const float* input, float* output, int n) {
  // TODO(student): grid-stride accumulation + warp shuffle + one atomic/block.
  // Add a __device__ warp_reduce_sum helper above if useful.
}

void launch(int version, const float* input, float* output, int n) {
  CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
  const int blocks = std::max(1, std::min((n + kBlock - 1) / kBlock, 4096));
  const size_t shared_bytes = kBlock * sizeof(float);
  if (version == 0) student_v0<<<blocks, kBlock, shared_bytes>>>(input, output, n);
  else if (version == 1) student_v1<<<blocks, kBlock, shared_bytes>>>(input, output, n);
  else student_v2<<<blocks, kBlock, shared_bytes>>>(input, output, n);
  course::check_last_kernel("lab reduce");
}

bool check(int version) {
  constexpr int n = (1 << 20) + 13;
  auto input = course::random_floats(n, 107, 0.0f, 1.0f);
  const std::vector<float> expected{
      static_cast<float>(std::accumulate(input.begin(), input.end(), 0.0))};
  course::DeviceBuffer<float> d_input(n), d_output(1);
  d_input.copy_from_host(input.data());
  launch(version, d_input.get(), d_output.get(), n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(1);
  d_output.copy_to_host(actual.data());
  std::cout << "student reduce v" << version << '\n';
  return course::close_enough(actual, expected, 8.0f, 5e-4f);
}

}  // namespace

int main(int argc, char** argv) {
  course::print_device_banner();
  bool ok = true;
  for (int version : lab_versions(argc, argv, 3)) ok &= check(version);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
