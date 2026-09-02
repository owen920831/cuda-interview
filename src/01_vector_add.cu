#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;

__global__ void vector_add_v0(const float* a, const float* b, float* c, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) c[index] = a[index] + b[index];
}

__global__ void vector_add_v1(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, int n) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < n; index += blockDim.x * gridDim.x) {
    c[index] = a[index] + b[index];
  }
}

__global__ void vector_add_float4(const float4* __restrict__ a,
                                  const float4* __restrict__ b,
                                  float4* __restrict__ c, int n4) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < n4; index += blockDim.x * gridDim.x) {
    const float4 x = a[index];
    const float4 y = b[index];
    c[index] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

__global__ void vector_add_tail(const float* a, const float* b, float* c,
                                int begin, int n) {
  const int index = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) c[index] = a[index] + b[index];
}

void launch(int version, const float* a, const float* b, float* c, int n) {
  if (version == 0) {
    vector_add_v0<<<(n + kBlock - 1) / kBlock, kBlock>>>(a, b, c, n);
  } else if (version == 1) {
    const int blocks = std::min((n + kBlock - 1) / kBlock, 4096);
    vector_add_v1<<<blocks, kBlock>>>(a, b, c, n);
  } else {
    const int n4 = n / 4;
    if (n4 > 0) {
      const int blocks = std::min((n4 + kBlock - 1) / kBlock, 4096);
      vector_add_float4<<<blocks, kBlock>>>(reinterpret_cast<const float4*>(a),
                                            reinterpret_cast<const float4*>(b),
                                            reinterpret_cast<float4*>(c), n4);
    }
    if (n4 * 4 < n) vector_add_tail<<<1, kBlock>>>(a, b, c, n4 * 4, n);
  }
  course::check_last_kernel("vector_add");
}

bool check_version(int version) {
  constexpr int n = 100003;  // intentionally not divisible by block or float4 width
  auto a = course::random_floats(n, 11);
  auto b = course::random_floats(n, 29);
  std::vector<float> expected(n);
  std::transform(a.begin(), a.end(), b.begin(), expected.begin(),
                 [](float x, float y) { return x + y; });

  course::DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
  d_a.copy_from_host(a.data());
  d_b.copy_from_host(b.data());
  launch(version, d_a.get(), d_b.get(), d_c.get(), n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(n);
  d_c.copy_to_host(actual.data());
  std::cout << "check v" << version << '\n';
  return course::close_enough(actual, expected);
}

void run_benchmarks(const course::Options& options) {
  constexpr int n = 1 << 24;
  auto a = course::random_floats(n, 11);
  auto b = course::random_floats(n, 29);
  course::DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
  d_a.copy_from_host(a.data());
  d_b.copy_from_host(b.data());
  const double gigabytes = 3.0 * n * sizeof(float) / 1e9;
  for (int version : course::versions(options, 3)) {
    const float ms = course::benchmark_ms(
        [&] { launch(version, d_a.get(), d_b.get(), d_c.get(), n); },
        options.warmup, options.iterations);
    course::print_benchmark("vector_add v" + std::to_string(version), ms,
                            gigabytes, "GB/s");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const course::Options options = course::parse_options(argc, argv);
  course::print_device_banner();
  bool ok = true;
  if (options.check) {
    for (int version : course::versions(options, 3)) ok &= check_version(version);
  }
  if (options.bench) run_benchmarks(options);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
