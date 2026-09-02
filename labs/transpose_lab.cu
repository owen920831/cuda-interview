#include "course/bench.cuh"
#include "lab_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

namespace {

constexpr int kTile = 32;
constexpr int kRows = 8;

__global__ void student_v0(const float* input, float* output, int height, int width) {
  // TODO(student): direct transpose; cover a 32x32 region with a 32x8 block.
}

__global__ void student_v1(const float* input, float* output, int height, int width) {
  // TODO(student): shared [32][32] tile; keep the bank-conflicted version on purpose.
}

__global__ void student_v2(const float* input, float* output, int height, int width) {
  // TODO(student): padded [32][33] tile; preserve all boundary checks.
}

void launch(int version, const float* input, float* output, int height, int width) {
  const dim3 block(kTile, kRows);
  const dim3 grid((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  if (version == 0) student_v0<<<grid, block>>>(input, output, height, width);
  else if (version == 1) student_v1<<<grid, block>>>(input, output, height, width);
  else student_v2<<<grid, block>>>(input, output, height, width);
  course::check_last_kernel("lab transpose");
}

bool check(int version) {
  constexpr int height = 777;
  constexpr int width = 1003;
  const size_t count = static_cast<size_t>(height) * width;
  auto input = course::random_floats(count, 109);
  std::vector<float> expected(count);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) expected[x * height + y] = input[y * width + x];
  }
  course::DeviceBuffer<float> d_input(count), d_output(count);
  d_input.copy_from_host(input.data());
  d_output.zero();
  launch(version, d_input.get(), d_output.get(), height, width);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> actual(count);
  d_output.copy_to_host(actual.data());
  std::cout << "student transpose v" << version << '\n';
  return course::close_enough(actual, expected, 0.0f, 0.0f);
}

}  // namespace

int main(int argc, char** argv) {
  course::print_device_banner();
  bool ok = true;
  for (int version : lab_versions(argc, argv, 3)) ok &= check(version);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
