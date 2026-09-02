#include "course/bench.cuh"
#include "lab_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

namespace {

constexpr int kTileX = 16;
constexpr int kTileY = 8;

__global__ void student_v0(const float* input, float* output, int nc,
                           int height, int width, int kernel, int stride,
                           int out_h, int out_w) {
  // TODO(student): one thread per NCHW output; implement valid average pooling.
}

__global__ void student_v1(const float* input, float* output, int nc,
                           int height, int width, int kernel, int stride,
                           int out_h, int out_w) {
  // TODO(student): cooperative dynamic-shared input tile including halo.
}

__global__ void student_v2(const float* input, float* output, int nc,
                           int height, int width, int out_h, int out_w) {
  // TODO(student): specialized 2x2/S2 path; each thread produces two x outputs.
}

void launch(int version, const float* input, float* output, int nc,
            int height, int width, int kernel, int stride) {
  const int out_h = (height - kernel) / stride + 1;
  const int out_w = (width - kernel) / stride + 1;
  if (version == 2 && kernel == 2 && stride == 2 && width % 4 == 0) {
    const dim3 block(32, 8);
    const int pairs = (out_w + 1) / 2;
    const dim3 grid((pairs + block.x - 1) / block.x,
                    (out_h + block.y - 1) / block.y, nc);
    student_v2<<<grid, block>>>(input, output, nc, height, width, out_h, out_w);
  } else {
    const dim3 block(kTileX, kTileY);
    const dim3 grid((out_w + kTileX - 1) / kTileX,
                    (out_h + kTileY - 1) / kTileY, nc);
    if (version == 0) {
      student_v0<<<grid, block>>>(input, output, nc, height, width,
                                  kernel, stride, out_h, out_w);
    } else {
      const int tile_w = (kTileX - 1) * stride + kernel;
      const int tile_h = (kTileY - 1) * stride + kernel;
      student_v1<<<grid, block, tile_w * tile_h * sizeof(float)>>>(
          input, output, nc, height, width, kernel, stride, out_h, out_w);
    }
  }
  course::check_last_kernel("lab avg_pool");
}

std::vector<float> cpu_pool(const std::vector<float>& input, int nc,
                            int height, int width, int kernel, int stride) {
  const int out_h = (height - kernel) / stride + 1;
  const int out_w = (width - kernel) / stride + 1;
  std::vector<float> output(static_cast<size_t>(nc) * out_h * out_w);
  for (int plane = 0; plane < nc; ++plane) {
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel; ++ky) {
          for (int kx = 0; kx < kernel; ++kx) sum += input[plane * height * width + (oy * stride + ky) * width + ox * stride + kx];
        }
        output[(plane * out_h + oy) * out_w + ox] = sum / (kernel * kernel);
      }
    }
  }
  return output;
}

bool check(int version) {
  constexpr int nc = 6;
  constexpr int height = 35;
  constexpr int width = 40;
  const int kernel = version == 2 ? 2 : 3;
  constexpr int stride = 2;
  auto input = course::random_floats(static_cast<size_t>(nc) * height * width, 131);
  const auto expected = cpu_pool(input, nc, height, width, kernel, stride);
  course::DeviceBuffer<float> d_input(input.size()), d_output(expected.size());
  d_input.copy_from_host(input.data());
  d_output.zero();
  launch(version, d_input.get(), d_output.get(), nc, height, width, kernel, stride);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(expected.size());
  d_output.copy_to_host(actual.data());
  std::cout << "student avg_pool v" << version << '\n';
  return course::close_enough(actual, expected, 2e-5f, 2e-5f);
}

}  // namespace

int main(int argc, char** argv) {
  course::print_device_banner();
  bool ok = true;
  for (int version : lab_versions(argc, argv, 3)) ok &= check(version);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
