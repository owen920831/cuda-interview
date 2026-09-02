#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  (void)course::parse_options(argc, argv);
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::cout << "CUDA devices: " << count << "\n";
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    std::cout << "\n[" << device << "] " << p.name << '\n'
              << "  compute capability: " << p.major << '.' << p.minor << '\n'
              << "  global memory:      " << std::fixed << std::setprecision(1)
              << p.totalGlobalMem / static_cast<double>(1ULL << 30) << " GiB\n"
              << "  SM count:           " << p.multiProcessorCount << '\n'
              << "  warp size:          " << p.warpSize << '\n'
              << "  max threads/block:  " << p.maxThreadsPerBlock << '\n'
              << "  shared mem/block:   " << p.sharedMemPerBlock / 1024 << " KiB\n"
              << "  registers/block:    " << p.regsPerBlock << '\n'
              << "  async engines:      " << p.asyncEngineCount << '\n'
              << "  concurrent kernels: " << (p.concurrentKernels ? "yes" : "no") << '\n';
  }
  return count > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
