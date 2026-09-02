#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// v0: simplest parallel mapping; every element contends on one atomic result.
__global__ void reduce_v0(const float* input, float* output, int n) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < n; index += blockDim.x * gridDim.x) {
    atomicAdd(output, input[index]);
  }
}

// v1: one coalesced load per iteration, shared-memory block tree, one atomic/block.
__global__ void reduce_v1(const float* input, float* output, int n) {
  extern __shared__ float shared[];
  float local = 0.0f;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < n; index += blockDim.x * gridDim.x) {
    local += input[index];
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(output, shared[0]);
}

// v2: first add happens while loading, halving active blocks and shared work.
__global__ void reduce_v2(const float* input, float* output, int n) {
  extern __shared__ float shared[];
  float local = 0.0f;
  for (int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
       index < n; index += blockDim.x * gridDim.x * 2) {
    local += input[index];
    if (index + blockDim.x < n) local += input[index + blockDim.x];
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(output, shared[0]);
}

template <int Block>
__global__ void reduce_v3(const float* __restrict__ input, float* output, int n) {
  static_assert(Block % 32 == 0, "block must contain full warps");
  __shared__ float warp_sums[Block / 32];
  float local = 0.0f;
  for (int index = blockIdx.x * Block * 2 + threadIdx.x;
       index < n; index += Block * gridDim.x * 2) {
    local += input[index];
    if (index + Block < n) local += input[index + Block];
  }
  local = warp_reduce_sum(local);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = local;
  __syncthreads();
  if (warp == 0) {
    local = lane < Block / 32 ? warp_sums[lane] : 0.0f;
    local = warp_reduce_sum(local);
    if (lane == 0) atomicAdd(output, local);
  }
}

template <int Block>
__global__ void reduce_v4(const float4* __restrict__ input4, float* output,
                          int n4, const float* input, int n) {
  __shared__ float warp_sums[Block / 32];
  float local = 0.0f;
  for (int index = blockIdx.x * Block + threadIdx.x;
       index < n4; index += Block * gridDim.x) {
    const float4 value = input4[index];
    local += value.x + value.y + value.z + value.w;
  }
  const int tail_index = n4 * 4 + blockIdx.x * Block + threadIdx.x;
  if (tail_index < n) local += input[tail_index];

  local = warp_reduce_sum(local);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = local;
  __syncthreads();
  if (warp == 0) {
    local = lane < Block / 32 ? warp_sums[lane] : 0.0f;
    local = warp_reduce_sum(local);
    if (lane == 0) atomicAdd(output, local);
  }
}

void launch(int version, const float* input, float* output, int n) {
  CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
  const int blocks = std::max(1, std::min((n + kBlock - 1) / kBlock, 4096));
  if (version == 0) {
    reduce_v0<<<blocks, kBlock>>>(input, output, n);
  } else if (version == 1) {
    reduce_v1<<<blocks, kBlock, kBlock * sizeof(float)>>>(input, output, n);
  } else if (version == 2) {
    const int pair_blocks = std::max(1, std::min((n + 2 * kBlock - 1) / (2 * kBlock), 4096));
    reduce_v2<<<pair_blocks, kBlock, kBlock * sizeof(float)>>>(input, output, n);
  } else if (version == 3) {
    reduce_v3<kBlock><<<blocks, kBlock>>>(input, output, n);
  } else {
    const int n4 = n / 4;
    const int vector_blocks = std::max(1, std::min((n4 + kBlock - 1) / kBlock, 4096));
    reduce_v4<kBlock><<<vector_blocks, kBlock>>>(
        reinterpret_cast<const float4*>(input), output, n4, input, n);
  }
  course::check_last_kernel("reduce");
}

bool check_version(int version) {
  constexpr int n = (1 << 20) + 13;
  auto input = course::random_floats(n, 7, 0.0f, 1.0f);
  const double reference = std::accumulate(input.begin(), input.end(), 0.0);
  const std::vector<float> expected{static_cast<float>(reference)};
  course::DeviceBuffer<float> d_input(n), d_output(1);
  d_input.copy_from_host(input.data());
  launch(version, d_input.get(), d_output.get(), n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(1);
  d_output.copy_to_host(actual.data());
  std::cout << "check v" << version << " (sum=" << actual[0] << ")\n";
  return course::close_enough(actual, expected, 8.0f, 5e-4f);
}

void run_benchmarks(const course::Options& options) {
  constexpr int n = 1 << 24;
  auto input = course::random_floats(n, 7, 0.0f, 1.0f);
  course::DeviceBuffer<float> d_input(n), d_output(1);
  d_input.copy_from_host(input.data());
  const double gigabytes = n * sizeof(float) / 1e9;
  for (int version : course::versions(options, 5)) {
    const float ms = course::benchmark_ms(
        [&] { launch(version, d_input.get(), d_output.get(), n); },
        options.warmup, options.iterations);
    course::print_benchmark("reduce v" + std::to_string(version), ms,
                            gigabytes, "GB/s");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const course::Options options = course::parse_options(argc, argv);
  course::print_device_banner();
  bool ok = true;
  if (options.check) {
    for (int version : course::versions(options, 5)) ok &= check_version(version);
  }
  if (options.bench) run_benchmarks(options);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
