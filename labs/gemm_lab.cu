#include "course/bench.cuh"
#include "lab_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

namespace {

constexpr int kTile = 16;

__global__ void student_v0(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  // TODO(student): one thread computes one C[row,col] from global memory.
}

__global__ void student_v1(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  // TODO(student): cooperative A/B shared tiles with zero-filled K tail.
}

__global__ void student_v2(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  // TODO(student): one thread computes C[row,col] and C[row,col+16].
  // The launcher gives each block a 16x32 output tile.
}

void launch(int version, const float* a, const float* b, float* c,
            int m, int n, int k) {
  const dim3 block(kTile, kTile);
  const int output_tile_n = version == 2 ? 2 * kTile : kTile;
  const dim3 grid((n + output_tile_n - 1) / output_tile_n,
                  (m + kTile - 1) / kTile);
  if (version == 0) student_v0<<<grid, block>>>(a, b, c, m, n, k);
  else if (version == 1) student_v1<<<grid, block>>>(a, b, c, m, n, k);
  else student_v2<<<grid, block>>>(a, b, c, m, n, k);
  course::check_last_kernel("lab gemm");
}

std::vector<float> cpu_gemm(const std::vector<float>& a,
                            const std::vector<float>& b,
                            int m, int n, int k) {
  std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int inner = 0; inner < k; ++inner) {
      for (int col = 0; col < n; ++col) c[row * n + col] += a[row * k + inner] * b[inner * n + col];
    }
  }
  return c;
}

bool check(int version) {
  constexpr int m = 129;
  constexpr int n = 133;
  constexpr int k = 71;
  auto a = course::random_floats(static_cast<size_t>(m) * k, 113, -0.5f, 0.5f);
  auto b = course::random_floats(static_cast<size_t>(k) * n, 127, -0.5f, 0.5f);
  const auto expected = cpu_gemm(a, b, m, n, k);
  course::DeviceBuffer<float> d_a(a.size()), d_b(b.size()), d_c(expected.size());
  d_a.copy_from_host(a.data());
  d_b.copy_from_host(b.data());
  d_c.zero();
  launch(version, d_a.get(), d_b.get(), d_c.get(), m, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(expected.size());
  d_c.copy_to_host(actual.data());
  std::cout << "student gemm v" << version << '\n';
  return course::close_enough(actual, expected, 3e-3f, 2e-3f);
}

}  // namespace

int main(int argc, char** argv) {
  course::print_device_banner();
  bool ok = true;
  for (int version : lab_versions(argc, argv, 3)) ok &= check(version);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
