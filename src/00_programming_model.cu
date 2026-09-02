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

__global__ void record_1d(Coord1D* coordinates, int n) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global < n) {
    coordinates[global] = {
        static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x), global};
  }
}

__global__ void record_2d(Coord2D* coordinates, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    coordinates[y * width + x] = {
        static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
        static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y), x, y};
  }
}

__global__ void reverse_within_block(const int* input, int* output, int n) {
  extern __shared__ int tile[];
  const int block_begin = blockIdx.x * blockDim.x;
  const int global = block_begin + threadIdx.x;
  const int active = min(static_cast<int>(blockDim.x), n - block_begin);

  // Every thread writes a defined value and reaches the block-wide barrier.
  tile[threadIdx.x] = global < n ? input[global] : -1;
  __syncthreads();

  if (global < n) {
    output[global] = tile[active - 1 - static_cast<int>(threadIdx.x)];
  }
}

bool check_1d(bool print_mapping) {
  constexpr int n = 19;
  constexpr int threads = 8;
  const int blocks = (n + threads - 1) / threads;
  course::DeviceBuffer<Coord1D> device(n);
  record_1d<<<blocks, threads>>>(device.get(), n);
  course::check_last_kernel("record_1d");
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Coord1D> actual(n);
  device.copy_to_host(actual.data());
  for (int global = 0; global < n; ++global) {
    if (actual[global].block != global / threads ||
        actual[global].thread != global % threads ||
        actual[global].global != global) {
      std::cerr << "1D mapping mismatch at " << global << '\n';
      return false;
    }
  }
  if (print_mapping) {
    std::cout << "\n1D mapping (N=19, blockDim.x=8):\n";
    for (const auto& coordinate : actual) {
      std::cout << "  blockIdx.x=" << coordinate.block
                << " threadIdx.x=" << coordinate.thread
                << " -> global=" << coordinate.global << '\n';
    }
  }
  return true;
}

bool check_2d(bool print_mapping) {
  constexpr int width = 7;
  constexpr int height = 5;
  const dim3 threads(4, 3);
  const dim3 blocks((width + threads.x - 1) / threads.x,
                    (height + threads.y - 1) / threads.y);
  course::DeviceBuffer<Coord2D> device(width * height);
  record_2d<<<blocks, threads>>>(device.get(), width, height);
  course::check_last_kernel("record_2d");
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Coord2D> actual(width * height);
  device.copy_to_host(actual.data());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto& coordinate = actual[y * width + x];
      if (coordinate.global_x != x || coordinate.global_y != y ||
          coordinate.block_x != x / static_cast<int>(threads.x) ||
          coordinate.block_y != y / static_cast<int>(threads.y) ||
          coordinate.thread_x != x % static_cast<int>(threads.x) ||
          coordinate.thread_y != y % static_cast<int>(threads.y)) {
        std::cerr << "2D mapping mismatch at (" << x << ", " << y << ")\n";
        return false;
      }
    }
  }
  if (print_mapping) {
    const auto& sample = actual[4 * width + 6];
    std::cout << "\n2D sample (x=6,y=4, block=(4,3)):\n"
              << "  blockIdx=(" << sample.block_x << ',' << sample.block_y
              << ") threadIdx=(" << sample.thread_x << ',' << sample.thread_y
              << ") -> global=(" << sample.global_x << ',' << sample.global_y
              << ")\n";
  }
  return true;
}

bool check_barrier(bool print_mapping) {
  constexpr int n = 19;
  constexpr int threads = 8;
  std::vector<int> input(n);
  for (int i = 0; i < n; ++i) input[i] = i;
  course::DeviceBuffer<int> device_input(n), device_output(n);
  device_input.copy_from_host(input.data());
  const int blocks = (n + threads - 1) / threads;
  reverse_within_block<<<blocks, threads, threads * sizeof(int)>>>(
      device_input.get(), device_output.get(), n);
  course::check_last_kernel("reverse_within_block");
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> actual(n), expected(n);
  device_output.copy_to_host(actual.data());
  for (int begin = 0; begin < n; begin += threads) {
    const int end = std::min(begin + threads, n);
    for (int i = begin; i < end; ++i) expected[i] = end - 1 - (i - begin);
  }
  if (actual != expected) {
    std::cerr << "shared-memory synchronization check failed\n";
    return false;
  }
  if (print_mapping) {
    std::cout << "\nblock-local reverse with shared memory + __syncthreads():\n  ";
    for (int value : actual) std::cout << value << ' ';
    std::cout << '\n';
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const course::Options options = course::parse_options(argc, argv);
  course::print_device_banner();
  const bool print_mapping = options.bench;
  const bool ok = check_1d(print_mapping) && check_2d(print_mapping) &&
                  check_barrier(print_mapping);
  std::cout << (ok ? "programming model checks PASS\n"
                   : "programming model checks FAIL\n");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
