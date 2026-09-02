#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kOutputX = 16;
constexpr int kOutputY = 8;

__global__ void avg_pool_v0(const float* input, float* output,
                            int batches, int channels, int height, int width,
                            int kernel, int stride, int out_h, int out_w) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;
  const int nc = blockIdx.z;
  if (ox >= out_w || oy >= out_h || nc >= batches * channels) return;
  const int base = nc * height * width;
  float sum = 0.0f;
  for (int ky = 0; ky < kernel; ++ky) {
    for (int kx = 0; kx < kernel; ++kx) {
      sum += input[base + (oy * stride + ky) * width + ox * stride + kx];
    }
  }
  output[(nc * out_h + oy) * out_w + ox] = sum / static_cast<float>(kernel * kernel);
}

__global__ void avg_pool_v1(const float* __restrict__ input,
                            float* __restrict__ output,
                            int batches, int channels, int height, int width,
                            int kernel, int stride, int out_h, int out_w) {
  extern __shared__ float tile[];
  const int tile_w = (kOutputX - 1) * stride + kernel;
  const int tile_h = (kOutputY - 1) * stride + kernel;
  const int nc = blockIdx.z;
  const int in_x0 = blockIdx.x * kOutputX * stride;
  const int in_y0 = blockIdx.y * kOutputY * stride;
  const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y;
  const int input_base = nc * height * width;

  for (int index = linear_thread; index < tile_w * tile_h; index += threads) {
    const int local_y = index / tile_w;
    const int local_x = index - local_y * tile_w;
    const int global_y = in_y0 + local_y;
    const int global_x = in_x0 + local_x;
    tile[index] = (nc < batches * channels && global_y < height && global_x < width)
                      ? input[input_base + global_y * width + global_x]
                      : 0.0f;
  }
  __syncthreads();

  const int ox = blockIdx.x * kOutputX + threadIdx.x;
  const int oy = blockIdx.y * kOutputY + threadIdx.y;
  if (ox >= out_w || oy >= out_h || nc >= batches * channels) return;
  float sum = 0.0f;
  const int local_x0 = threadIdx.x * stride;
  const int local_y0 = threadIdx.y * stride;
  for (int ky = 0; ky < kernel; ++ky) {
    for (int kx = 0; kx < kernel; ++kx) sum += tile[(local_y0 + ky) * tile_w + local_x0 + kx];
  }
  output[(nc * out_h + oy) * out_w + ox] = sum / static_cast<float>(kernel * kernel);
}

// Specialized fast path: 2x2 / stride 2. One thread produces two adjacent outputs.
__global__ void avg_pool_v2_2x2(const float* __restrict__ input,
                                float* __restrict__ output,
                                int batches, int channels, int height, int width,
                                int out_h, int out_w) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int ox = pair * 2;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;
  const int nc = blockIdx.z;
  if (ox >= out_w || oy >= out_h || nc >= batches * channels) return;
  const int ix = ox * 2;
  const int iy = oy * 2;
  const int base = nc * height * width;
  const int out_base = (nc * out_h + oy) * out_w;
  if (ox + 1 < out_w) {
    const float4 top = *reinterpret_cast<const float4*>(input + base + iy * width + ix);
    const float4 bottom = *reinterpret_cast<const float4*>(input + base + (iy + 1) * width + ix);
    output[out_base + ox] = 0.25f * (top.x + top.y + bottom.x + bottom.y);
    output[out_base + ox + 1] = 0.25f * (top.z + top.w + bottom.z + bottom.w);
  } else {
    output[out_base + ox] = 0.25f *
        (input[base + iy * width + ix] + input[base + iy * width + ix + 1] +
         input[base + (iy + 1) * width + ix] + input[base + (iy + 1) * width + ix + 1]);
  }
}

void launch_v1(const float* input, float* output, int batches, int channels,
               int height, int width, int kernel, int stride, int out_h, int out_w) {
  const dim3 block(kOutputX, kOutputY);
  const dim3 grid((out_w + kOutputX - 1) / kOutputX,
                  (out_h + kOutputY - 1) / kOutputY,
                  batches * channels);
  const int tile_w = (kOutputX - 1) * stride + kernel;
  const int tile_h = (kOutputY - 1) * stride + kernel;
  avg_pool_v1<<<grid, block, tile_w * tile_h * sizeof(float)>>>(
      input, output, batches, channels, height, width, kernel, stride, out_h, out_w);
}

void launch(int version, const float* input, float* output,
            int batches, int channels, int height, int width,
            int kernel, int stride) {
  const int out_h = (height - kernel) / stride + 1;
  const int out_w = (width - kernel) / stride + 1;
  if (version == 0) {
    const dim3 block(kOutputX, kOutputY);
    const dim3 grid((out_w + kOutputX - 1) / kOutputX,
                    (out_h + kOutputY - 1) / kOutputY,
                    batches * channels);
    avg_pool_v0<<<grid, block>>>(input, output, batches, channels, height, width,
                                 kernel, stride, out_h, out_w);
  } else if (version == 1 || kernel != 2 || stride != 2 || width % 4 != 0) {
    launch_v1(input, output, batches, channels, height, width,
              kernel, stride, out_h, out_w);
  } else {
    const dim3 block(32, 8);
    const int pairs = (out_w + 1) / 2;
    const dim3 grid((pairs + block.x - 1) / block.x,
                    (out_h + block.y - 1) / block.y,
                    batches * channels);
    avg_pool_v2_2x2<<<grid, block>>>(input, output, batches, channels,
                                     height, width, out_h, out_w);
  }
  course::check_last_kernel("avg_pool");
}

std::vector<float> cpu_avg_pool(const std::vector<float>& input,
                                int batches, int channels, int height, int width,
                                int kernel, int stride) {
  const int out_h = (height - kernel) / stride + 1;
  const int out_w = (width - kernel) / stride + 1;
  std::vector<float> output(static_cast<size_t>(batches) * channels * out_h * out_w);
  for (int nc = 0; nc < batches * channels; ++nc) {
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel; ++ky) {
          for (int kx = 0; kx < kernel; ++kx) sum += input[nc * height * width + (oy * stride + ky) * width + ox * stride + kx];
        }
        output[(nc * out_h + oy) * out_w + ox] = sum / static_cast<float>(kernel * kernel);
      }
    }
  }
  return output;
}

bool check_version(int version) {
  const std::vector<std::tuple<int, int, int, int, int, int>> cases{
      {2, 3, 35, 41, 3, 2},   // general fallback and odd boundaries
      {2, 3, 35, 40, 2, 2}};  // vectorized fast path
  bool ok = true;
  for (const auto& shape : cases) {
    const auto [batches, channels, height, width, kernel, stride] = shape;
    auto input = course::random_floats(static_cast<size_t>(batches) * channels * height * width, 23);
    const auto expected = cpu_avg_pool(input, batches, channels, height, width, kernel, stride);
    course::DeviceBuffer<float> d_input(input.size()), d_output(expected.size());
    d_input.copy_from_host(input.data());
    launch(version, d_input.get(), d_output.get(), batches, channels,
           height, width, kernel, stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(expected.size());
    d_output.copy_to_host(actual.data());
    std::cout << "check v" << version << " k=" << kernel << " s=" << stride << '\n';
    ok &= course::close_enough(actual, expected, 2e-5f, 2e-5f);
  }
  return ok;
}

void benchmark_case(const course::Options& options, int kernel, int stride) {
  constexpr int batches = 4;
  constexpr int channels = 64;
  constexpr int height = 112;
  constexpr int width = 112;
  const int out_h = (height - kernel) / stride + 1;
  const int out_w = (width - kernel) / stride + 1;
  const size_t input_count = static_cast<size_t>(batches) * channels * height * width;
  const size_t output_count = static_cast<size_t>(batches) * channels * out_h * out_w;
  auto input = course::random_floats(input_count, 23);
  course::DeviceBuffer<float> d_input(input_count), d_output(output_count);
  d_input.copy_from_host(input.data());
  const double logical_gigabytes = output_count * (kernel * kernel + 1.0) * sizeof(float) / 1e9;
  const int versions = (kernel == 2 && stride == 2) ? 3 : 2;
  if (options.version >= versions) return;
  std::cout << "case: k=" << kernel << ", stride=" << stride << '\n';
  for (int version = 0; version < versions; ++version) {
    if (options.version >= 0 && options.version != version) continue;
    const float ms = course::benchmark_ms(
        [&] { launch(version, d_input.get(), d_output.get(), batches, channels,
                     height, width, kernel, stride); },
        options.warmup, options.iterations);
    course::print_benchmark("avg_pool v" + std::to_string(version), ms,
                            logical_gigabytes, "logical GB/s");
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
  if (options.bench) {
    benchmark_case(options, 3, 1);
    benchmark_case(options, 2, 2);
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
