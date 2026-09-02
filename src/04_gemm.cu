#include "course/bench.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUBLAS_CHECK(expr)                                                      \
  do {                                                                          \
    const cublasStatus_t course_status__ = (expr);                               \
    if (course_status__ != CUBLAS_STATUS_SUCCESS) {                              \
      std::cerr << "cuBLAS error at " << __FILE__ << ':' << __LINE__             \
                << ": status=" << static_cast<int>(course_status__) << '\n';    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (false)

namespace {

constexpr int kTile = 16;
constexpr int kColumnsPerThread = 4;

__global__ void gemm_v0(const float* a, const float* b, float* c,
                        int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) accumulator += a[row * k + inner] * b[inner * n + col];
  c[row * n + col] = accumulator;
}

__global__ void gemm_v1(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c, int m, int n, int k) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float accumulator = 0.0f;
  for (int base = 0; base < k; base += kTile) {
    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && base + threadIdx.x < k) ? a[row * k + base + threadIdx.x] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (base + threadIdx.y < k && col < n) ? b[(base + threadIdx.y) * n + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int inner = 0; inner < kTile; ++inner) accumulator += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = accumulator;
}

// A 16x16 block produces a 16x64 C tile. Each thread keeps four C values in registers.
__global__ void gemm_v2(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c, int m, int n, int k) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile * kColumnsPerThread];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col_base = blockIdx.x * kTile * kColumnsPerThread + threadIdx.x;
  float accumulator[kColumnsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int base = 0; base < k; base += kTile) {
    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && base + threadIdx.x < k) ? a[row * k + base + threadIdx.x] : 0.0f;
    #pragma unroll
    for (int item = 0; item < kColumnsPerThread; ++item) {
      const int col = col_base + item * kTile;
      tile_b[threadIdx.y][threadIdx.x + item * kTile] =
          (base + threadIdx.y < k && col < n) ? b[(base + threadIdx.y) * n + col] : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
      const float left = tile_a[threadIdx.y][inner];
      #pragma unroll
      for (int item = 0; item < kColumnsPerThread; ++item) {
        accumulator[item] += left * tile_b[inner][threadIdx.x + item * kTile];
      }
    }
    __syncthreads();
  }

  if (row < m) {
    #pragma unroll
    for (int item = 0; item < kColumnsPerThread; ++item) {
      const int col = col_base + item * kTile;
      if (col < n) c[row * n + col] = accumulator[item];
    }
  }
}

void launch(int version, const float* a, const float* b, float* c,
            int m, int n, int k, cublasHandle_t handle) {
  const dim3 block(kTile, kTile);
  if (version == 0) {
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    gemm_v0<<<grid, block>>>(a, b, c, m, n, k);
  } else if (version == 1) {
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    gemm_v1<<<grid, block>>>(a, b, c, m, n, k);
  } else if (version == 2) {
    const dim3 grid((n + kTile * kColumnsPerThread - 1) /
                        (kTile * kColumnsPerThread),
                    (m + kTile - 1) / kTile);
    gemm_v2<<<grid, block>>>(a, b, c, m, n, k);
  } else {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C=A*B is column-major C^T=B^T*A^T without moving data.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k, &alpha, b, n, a, k, &beta, c, n));
  }
  course::check_last_kernel("gemm");
}

std::vector<float> cpu_gemm(const std::vector<float>& a,
                            const std::vector<float>& b,
                            int m, int n, int k) {
  std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int inner = 0; inner < k; ++inner) {
      const float left = a[row * k + inner];
      for (int col = 0; col < n; ++col) c[row * n + col] += left * b[inner * n + col];
    }
  }
  return c;
}

bool check_version(int version, cublasHandle_t handle) {
  constexpr int m = 129;
  constexpr int n = 133;
  constexpr int k = 71;
  auto a = course::random_floats(static_cast<size_t>(m) * k, 3, -0.5f, 0.5f);
  auto b = course::random_floats(static_cast<size_t>(k) * n, 5, -0.5f, 0.5f);
  const auto expected = cpu_gemm(a, b, m, n, k);
  course::DeviceBuffer<float> d_a(a.size()), d_b(b.size()), d_c(expected.size());
  d_a.copy_from_host(a.data());
  d_b.copy_from_host(b.data());
  launch(version, d_a.get(), d_b.get(), d_c.get(), m, n, k, handle);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(expected.size());
  d_c.copy_to_host(actual.data());
  std::cout << "check " << (version == 3 ? "cuBLAS" : "v" + std::to_string(version)) << '\n';
  return course::close_enough(actual, expected, 3e-3f, 2e-3f);
}

void run_benchmarks(const course::Options& options, cublasHandle_t handle) {
  constexpr int m = 1024;
  constexpr int n = 1024;
  constexpr int k = 1024;
  auto a = course::random_floats(static_cast<size_t>(m) * k, 3, -0.5f, 0.5f);
  auto b = course::random_floats(static_cast<size_t>(k) * n, 5, -0.5f, 0.5f);
  course::DeviceBuffer<float> d_a(a.size()), d_b(b.size()), d_c(static_cast<size_t>(m) * n);
  d_a.copy_from_host(a.data());
  d_b.copy_from_host(b.data());
  const double gigaflops = 2.0 * m * n * k / 1e9;
  for (int version : course::versions(options, 4)) {
    const float ms = course::benchmark_ms(
        [&] { launch(version, d_a.get(), d_b.get(), d_c.get(), m, n, k, handle); },
        options.warmup, options.iterations);
    const std::string name = version == 3 ? "gemm cuBLAS" : "gemm v" + std::to_string(version);
    course::print_benchmark(name, ms, gigaflops, "GFLOP/s");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const course::Options options = course::parse_options(argc, argv);
  course::print_device_banner();
  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  bool ok = true;
  if (options.check) {
    for (int version : course::versions(options, 4)) ok &= check_version(version, handle);
  }
  if (options.bench) run_benchmarks(options, handle);
  CUBLAS_CHECK(cublasDestroy(handle));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
