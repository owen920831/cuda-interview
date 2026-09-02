#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;

__global__ void add_bias(const float* input, const float* bias, float* temp,
                         int count, int channels) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    temp[index] = input[index] + bias[index % channels];
  }
}

__global__ void relu(const float* input, float* output, int count) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    output[index] = fmaxf(input[index], 0.0f);
  }
}

__global__ void bias_relu_fused(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int count, int channels) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    output[index] = fmaxf(input[index] + bias[index % channels], 0.0f);
  }
}

__global__ void bias_relu_float4(const float4* __restrict__ input,
                                 const float* __restrict__ bias,
                                 float4* __restrict__ output,
                                 int count4, int channels) {
  for (int index4 = blockIdx.x * blockDim.x + threadIdx.x;
       index4 < count4; index4 += blockDim.x * gridDim.x) {
    const int index = index4 * 4;
    const float4 x = input[index4];
    output[index4] = make_float4(
        fmaxf(x.x + bias[(index + 0) % channels], 0.0f),
        fmaxf(x.y + bias[(index + 1) % channels], 0.0f),
        fmaxf(x.z + bias[(index + 2) % channels], 0.0f),
        fmaxf(x.w + bias[(index + 3) % channels], 0.0f));
  }
}

__global__ void bias_relu_tail(const float* input, const float* bias,
                               float* output, int begin, int count, int channels) {
  const int index = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = fmaxf(input[index] + bias[index % channels], 0.0f);
}

void launch(int version, const float* input, const float* bias,
            float* temp, float* output, int count, int channels) {
  const int blocks = std::max(1, std::min((count + kBlock - 1) / kBlock, 4096));
  if (version == 0) {
    add_bias<<<blocks, kBlock>>>(input, bias, temp, count, channels);
    relu<<<blocks, kBlock>>>(temp, output, count);
  } else if (version == 1) {
    bias_relu_fused<<<blocks, kBlock>>>(input, bias, output, count, channels);
  } else {
    const int count4 = count / 4;
    const int blocks4 = std::max(1, std::min((count4 + kBlock - 1) / kBlock, 4096));
    if (count4 > 0) {
      bias_relu_float4<<<blocks4, kBlock>>>(reinterpret_cast<const float4*>(input),
                                            bias, reinterpret_cast<float4*>(output),
                                            count4, channels);
    }
    if (count4 * 4 < count) bias_relu_tail<<<1, kBlock>>>(input, bias, output, count4 * 4, count, channels);
  }
  course::check_last_kernel("fusion");
}

bool check_version(int version) {
  constexpr int rows = 257;
  constexpr int channels = 1003;
  constexpr int count = rows * channels;
  auto input = course::random_floats(count, 31);
  auto bias = course::random_floats(channels, 37);
  std::vector<float> expected(count);
  for (int i = 0; i < count; ++i) expected[i] = std::max(input[i] + bias[i % channels], 0.0f);
  course::DeviceBuffer<float> d_input(count), d_bias(channels), d_temp(count), d_output(count);
  d_input.copy_from_host(input.data());
  d_bias.copy_from_host(bias.data());
  launch(version, d_input.get(), d_bias.get(), d_temp.get(), d_output.get(), count, channels);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(count);
  d_output.copy_to_host(actual.data());
  std::cout << "check v" << version << '\n';
  return course::close_enough(actual, expected);
}

void run_benchmarks(const course::Options& options) {
  constexpr int rows = 4096;
  constexpr int channels = 4096;
  constexpr int count = rows * channels;
  auto input = course::random_floats(count, 31);
  auto bias = course::random_floats(channels, 37);
  course::DeviceBuffer<float> d_input(count), d_bias(channels), d_temp(count), d_output(count);
  d_input.copy_from_host(input.data());
  d_bias.copy_from_host(bias.data());
  for (int version : course::versions(options, 3)) {
    const float ms = course::benchmark_ms(
        [&] { launch(version, d_input.get(), d_bias.get(), d_temp.get(),
                     d_output.get(), count, channels); },
        options.warmup, options.iterations);
    const double bytes_per_element = version == 0 ? 20.0 : 12.0;
    const double gigabytes = bytes_per_element * count / 1e9;
    course::print_benchmark("bias_relu v" + std::to_string(version), ms,
                            gigabytes, "estimated GB/s");
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
