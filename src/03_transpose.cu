#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 32;
constexpr int kRows = 8;

__global__ void transpose_v0(const float* input, float* output,
                             int height, int width) {
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
  #pragma unroll
  for (int j = 0; j < kTile; j += kRows) {
    if (x < width && y + j < height) output[x * height + y + j] = input[(y + j) * width + x];
  }
}

template <int Padding>
__global__ void transpose_tiled(const float* __restrict__ input,
                                float* __restrict__ output,
                                int height, int width) {
  __shared__ float tile[kTile][kTile + Padding];
  int x = blockIdx.x * kTile + threadIdx.x;
  int y = blockIdx.y * kTile + threadIdx.y;
  #pragma unroll
  for (int j = 0; j < kTile; j += kRows) {
    if (x < width && y + j < height) tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
  }
  __syncthreads();

  x = blockIdx.y * kTile + threadIdx.x;
  y = blockIdx.x * kTile + threadIdx.y;
  #pragma unroll
  for (int j = 0; j < kTile; j += kRows) {
    if (x < height && y + j < width) output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

void launch(int version, const float* input, float* output, int height, int width) {
  const dim3 block(kTile, kRows);
  const dim3 grid((width + kTile - 1) / kTile,
                  (height + kTile - 1) / kTile);
  if (version == 0) transpose_v0<<<grid, block>>>(input, output, height, width);
  else if (version == 1) transpose_tiled<0><<<grid, block>>>(input, output, height, width);
  else transpose_tiled<1><<<grid, block>>>(input, output, height, width);
  course::check_last_kernel("transpose");
}

bool check_version(int version) {
  constexpr int height = 777;
  constexpr int width = 1003;
  auto input = course::random_floats(static_cast<size_t>(height) * width, 19);
  std::vector<float> expected(input.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) expected[x * height + y] = input[y * width + x];
  }
  course::DeviceBuffer<float> d_input(input.size()), d_output(input.size());
  d_input.copy_from_host(input.data());
  launch(version, d_input.get(), d_output.get(), height, width);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(input.size());
  d_output.copy_to_host(actual.data());
  std::cout << "check v" << version << '\n';
  return course::close_enough(actual, expected, 0.0f, 0.0f);
}

void run_benchmarks(const course::Options& options) {
  constexpr int height = 4096;
  constexpr int width = 4096;
  const size_t count = static_cast<size_t>(height) * width;
  auto input = course::random_floats(count, 19);
  course::DeviceBuffer<float> d_input(count), d_output(count);
  d_input.copy_from_host(input.data());
  const double gigabytes = 2.0 * count * sizeof(float) / 1e9;
  for (int version : course::versions(options, 3)) {
    const float ms = course::benchmark_ms(
        [&] { launch(version, d_input.get(), d_output.get(), height, width); },
        options.warmup, options.iterations);
    course::print_benchmark("transpose v" + std::to_string(version), ms,
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
