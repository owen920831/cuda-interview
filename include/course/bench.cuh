#pragma once

#include "course/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace course {

struct Options {
  bool check = true;
  bool bench = true;
  int warmup = 5;
  int iterations = 50;
  int version = -1;  // -1 means all versions
};

inline Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--check-only") options.bench = false;
    else if (arg == "--bench-only") options.check = false;
    else if (arg == "--iters" && i + 1 < argc) options.iterations = std::stoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc) options.warmup = std::stoi(argv[++i]);
    else if (arg == "--version" && i + 1 < argc) {
      std::string value = argv[++i];
      if (value == "all") options.version = -1;
      else {
        if (!value.empty() && value[0] == 'v') value.erase(0, 1);
        options.version = std::stoi(value);
      }
    }
    else if (arg == "--help") {
      std::cout << "Options: --check-only | --bench-only | --version all|vN|N"
                   " | --iters N | --warmup N\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown option: " << arg << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

inline std::vector<int> versions(const Options& options, int count) {
  if (options.version >= count) {
    std::cerr << "Version " << options.version << " is invalid; expected 0.."
              << count - 1 << " or all\n";
    std::exit(EXIT_FAILURE);
  }
  if (options.version >= 0) return {options.version};
  std::vector<int> selected(count);
  for (int i = 0; i < count; ++i) selected[i] = i;
  return selected;
}

template <typename Launch>
float benchmark_ms(Launch&& launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / static_cast<float>(iterations);
}

inline std::vector<float> random_floats(size_t count, uint32_t seed = 42,
                                        float low = -1.0f, float high = 1.0f) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(low, high);
  std::vector<float> values(count);
  std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
  return values;
}

inline bool close_enough(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float absolute_tolerance = 1e-4f,
                         float relative_tolerance = 1e-4f) {
  if (actual.size() != expected.size()) return false;
  float max_absolute = 0.0f;
  float max_relative = 0.0f;
  size_t worst = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float absolute = std::abs(actual[i] - expected[i]);
    const float relative = absolute / std::max(std::abs(expected[i]), 1e-6f);
    if (absolute > max_absolute) {
      max_absolute = absolute;
      max_relative = relative;
      worst = i;
    }
    if (absolute > absolute_tolerance && relative > relative_tolerance) {
      std::cerr << "Mismatch at " << i << ": actual=" << actual[i]
                << ", expected=" << expected[i] << ", abs=" << absolute
                << ", rel=" << relative << '\n';
      return false;
    }
  }
  std::cout << "  max error: abs=" << max_absolute << ", rel=" << max_relative
            << " (index " << worst << ")\n";
  return true;
}

inline void print_benchmark(const std::string& name, float milliseconds,
                            double work = 0.0, const char* unit = "") {
  std::cout << std::left << std::setw(22) << name << std::right << std::fixed
            << std::setprecision(4) << std::setw(10) << milliseconds << " ms";
  if (work > 0.0) {
    std::cout << "  " << std::setprecision(2) << work / (milliseconds * 1e-3)
              << ' ' << unit;
  }
  std::cout << '\n';
}

inline void print_device_banner() {
  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::cout << "GPU: " << properties.name << " (sm_" << properties.major
            << properties.minor << ")\n";
}

}  // namespace course
