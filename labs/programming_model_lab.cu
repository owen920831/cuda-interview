#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

struct Coord1D {
  int block;
  int thread;
  int global;
};

struct Coord2D {
  int block_x;
  int block_y;
  int thread_x;
  int thread_y;
  int global_x;
  int global_y;
};

__global__ void student_record_1d(Coord1D* coordinates, int n) {
  // TODO 1: compute global x from blockIdx/threadIdx/blockDim.
  // TODO 2: guard the tail and record all three coordinates.
}

__global__ void student_record_2d(Coord2D* coordinates, int width, int height) {
  // TODO 3: map the 2D block and thread coordinates to global (x, y).
  // TODO 4: store at row-major offset y * width + x with a 2D guard.
}

__global__ void student_reverse_within_block(const int* input, int* output, int n) {
  extern __shared__ int tile[];
  // TODO 5: cooperatively load one item/thread, including a safe tail value.
  // TODO 6: add the block-wide barrier in a location reached by every thread.
  // TODO 7: reverse only the active portion of each block into output.
  (void)input;
  (void)output;
  (void)n;
  (void)tile;
}

bool check_1d() {
  constexpr int n = 19;
  constexpr int threads = 8;
  course::DeviceBuffer<Coord1D> device(n);
  device.zero();
  student_record_1d<<<(n + threads - 1) / threads, threads>>>(device.get(), n);
  course::check_last_kernel("student_record_1d");
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<Coord1D> actual(n);
  device.copy_to_host(actual.data());
  for (int i = 0; i < n; ++i) {
    if (actual[i].block != i / threads || actual[i].thread != i % threads ||
        actual[i].global != i) return false;
  }
  return true;
}

bool check_2d() {
  constexpr int width = 7;
  constexpr int height = 5;
  const dim3 threads(4, 3);
  const dim3 blocks((width + threads.x - 1) / threads.x,
                    (height + threads.y - 1) / threads.y);
  course::DeviceBuffer<Coord2D> device(width * height);
  device.zero();
  student_record_2d<<<blocks, threads>>>(device.get(), width, height);
  course::check_last_kernel("student_record_2d");
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<Coord2D> actual(width * height);
  device.copy_to_host(actual.data());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto& c = actual[y * width + x];
      if (c.global_x != x || c.global_y != y ||
          c.block_x != x / static_cast<int>(threads.x) ||
          c.block_y != y / static_cast<int>(threads.y) ||
          c.thread_x != x % static_cast<int>(threads.x) ||
          c.thread_y != y % static_cast<int>(threads.y)) return false;
    }
  }
  return true;
}

bool check_barrier() {
  constexpr int n = 19;
  constexpr int threads = 8;
  std::vector<int> input(n), expected(n), actual(n);
  for (int i = 0; i < n; ++i) input[i] = i;
  for (int begin = 0; begin < n; begin += threads) {
    const int end = std::min(begin + threads, n);
    for (int i = begin; i < end; ++i) expected[i] = end - 1 - (i - begin);
  }
  course::DeviceBuffer<int> device_input(n), device_output(n);
  device_input.copy_from_host(input.data());
  device_output.zero();
  student_reverse_within_block<<<(n + threads - 1) / threads, threads,
                                  threads * sizeof(int)>>>(
      device_input.get(), device_output.get(), n);
  course::check_last_kernel("student_reverse_within_block");
  CUDA_CHECK(cudaDeviceSynchronize());
  device_output.copy_to_host(actual.data());
  return actual == expected;
}

}  // namespace

int main() {
  course::print_device_banner();
  const bool mapping_1d = check_1d();
  const bool mapping_2d = check_2d();
  const bool barrier = check_barrier();
  std::cout << "1D mapping: " << (mapping_1d ? "PASS" : "FAIL") << '\n'
            << "2D mapping: " << (mapping_2d ? "PASS" : "FAIL") << '\n'
            << "block barrier: " << (barrier ? "PASS" : "FAIL") << '\n';
  return mapping_1d && mapping_2d && barrier ? EXIT_SUCCESS : EXIT_FAILURE;
}
