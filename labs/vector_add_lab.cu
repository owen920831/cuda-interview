#include "course/bench.cuh"
#include "lab_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

constexpr int kBlock = 256;

__global__ void student_v0(const float* a, const float* b, float* c, int n) {
  // TODO(student): one thread per element, including a boundary guard.
}

__global__ void student_v1(const float* a, const float* b, float* c, int n) {
  // TODO(student): use a grid-stride loop and coalesced scalar access.
}

__global__ void student_v2(const float* a, const float* b, float* c, int n) {
  // TODO(student): process four aligned floats per thread and handle the tail.
  // You may add a separate tail kernel and change the launcher below.
}

void launch(int version, const float* a, const float* b, float* c, int n) {
  const int blocks = std::max(1, std::min((n + kBlock - 1) / kBlock, 4096));
  if (version == 0) student_v0<<<blocks, kBlock>>>(a, b, c, n);
  else if (version == 1) student_v1<<<blocks, kBlock>>>(a, b, c, n);
  else student_v2<<<blocks, kBlock>>>(a, b, c, n);
  course::check_last_kernel("lab vector_add");
}

bool check(int version) {
  constexpr int n = 100003;
  auto a = course::random_floats(n, 101);
  auto b = course::random_floats(n, 103);
  std::vector<float> expected(n);
  std::transform(a.begin(), a.end(), b.begin(), expected.begin(),
                 [](float x, float y) { return x + y; });
  course::DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
  d_a.copy_from_host(a.data());
  d_b.copy_from_host(b.data());
  d_c.zero();
  launch(version, d_a.get(), d_b.get(), d_c.get(), n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(n);
  d_c.copy_to_host(actual.data());
  std::cout << "student vector_add v" << version << '\n';
  return course::close_enough(actual, expected);
}

}  // namespace

int main(int argc, char** argv) {
  course::print_device_banner();
  bool ok = true;
  for (int version : lab_versions(argc, argv, 3)) ok &= check(version);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
