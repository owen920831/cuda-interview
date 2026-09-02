#include "course/bench.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace {

constexpr int kBlock = 256;

__global__ void affine(const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = 2.0f * input[index] + 1.0f;
}

void enqueue_sequential(const float* host_input, float* host_output,
                        float* device_input, float* device_output,
                        int count, cudaStream_t stream) {
  const size_t bytes = static_cast<size_t>(count) * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(device_input, host_input, bytes, cudaMemcpyHostToDevice, stream));
  affine<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(device_input, device_output, count);
  CUDA_CHECK(cudaMemcpyAsync(host_output, device_output, bytes, cudaMemcpyDeviceToHost, stream));
  course::check_last_kernel("streams sequential");
}

void enqueue_chunked(const float* host_input, float* host_output,
                     float* device_input, float* device_output,
                     int count, const std::vector<cudaStream_t>& streams,
                     int chunks) {
  const int chunk_size = (count + chunks - 1) / chunks;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int begin = chunk * chunk_size;
    const int size = std::min(chunk_size, count - begin);
    if (size <= 0) break;
    const cudaStream_t stream = streams[chunk % streams.size()];
    const size_t bytes = static_cast<size_t>(size) * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(device_input + begin, host_input + begin, bytes,
                               cudaMemcpyHostToDevice, stream));
    affine<<<(size + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        device_input + begin, device_output + begin, size);
    CUDA_CHECK(cudaMemcpyAsync(host_output + begin, device_output + begin, bytes,
                               cudaMemcpyDeviceToHost, stream));
  }
  course::check_last_kernel("streams chunked");
}

template <typename Enqueue>
double wall_time_ms(Enqueue&& enqueue, int iterations) {
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    enqueue();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count() / iterations;
}

}  // namespace

int main(int argc, char** argv) {
  const course::Options options = course::parse_options(argc, argv);
  course::print_device_banner();
  constexpr int count = 1 << 24;
  constexpr int stream_count = 4;
  constexpr int chunks = 8;
  float* host_input = nullptr;
  float* host_output = nullptr;
  CUDA_CHECK(cudaMallocHost(&host_input, static_cast<size_t>(count) * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&host_output, static_cast<size_t>(count) * sizeof(float)));
  for (int i = 0; i < count; ++i) host_input[i] = static_cast<float>(i % 101) / 101.0f;
  course::DeviceBuffer<float> device_input(count), device_output(count);
  std::vector<cudaStream_t> streams(stream_count);
  for (auto& stream : streams) CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  bool ok = true;
  if (options.check) {
    enqueue_chunked(host_input, host_output, device_input.get(), device_output.get(),
                    count, streams, chunks);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < count; ++i) {
      const float expected = 2.0f * host_input[i] + 1.0f;
      if (std::abs(host_output[i] - expected) > 1e-6f) {
        std::cerr << "Mismatch at " << i << '\n';
        ok = false;
        break;
      }
    }
    std::cout << "chunked stream correctness: " << (ok ? "PASS" : "FAIL") << '\n';
  }

  if (options.bench) {
    const int iterations = std::min(options.iterations, 20);
    const double sequential = wall_time_ms(
        [&] { enqueue_sequential(host_input, host_output, device_input.get(),
                                 device_output.get(), count, streams[0]); },
        iterations);
    const double chunked = wall_time_ms(
        [&] { enqueue_chunked(host_input, host_output, device_input.get(),
                              device_output.get(), count, streams, chunks); },
        iterations);
    std::cout << "sequential pinned: " << sequential << " ms\n"
              << "4 streams / 8 chunks: " << chunked << " ms\n"
              << "speedup: " << sequential / chunked << "x\n";
  }

  for (auto stream : streams) CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(host_output));
  CUDA_CHECK(cudaFreeHost(host_input));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
