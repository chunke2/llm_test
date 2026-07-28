#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s failed: %s\n", #call,                     \
                         cudaGetErrorString(error));                              \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void saxpy(const float* x, float* y, float alpha, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        y[index] = alpha * x[index] + y[index];
    }
}

int main() {
    constexpr int count = 1 << 20;
    constexpr float alpha = 2.0f;
    constexpr size_t bytes = count * sizeof(float);

    std::vector<float> host_x(count, 1.5f);
    std::vector<float> host_y(count, 0.5f);
    float* device_x = nullptr;
    float* device_y = nullptr;

    CUDA_CHECK(cudaMalloc(&device_x, bytes));
    CUDA_CHECK(cudaMalloc(&device_y, bytes));
    CUDA_CHECK(cudaMemcpy(device_x, host_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y, host_y.data(), bytes, cudaMemcpyHostToDevice));

    saxpy<<<(count + 255) / 256, 256>>>(device_x, device_y, alpha, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_y.data(), device_y, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_x));
    CUDA_CHECK(cudaFree(device_y));

    const float expected = alpha * host_x[0] + 0.5f;
    if (std::fabs(host_y[0] - expected) > 1e-6f) {
        std::fprintf(stderr, "unexpected result: got %.6f, expected %.6f\n",
                     host_y[0], expected);
        return 2;
    }

    std::printf("cuda_smoke_ok result=%.6f elements=%d\n", host_y[0], count);
    return 0;
}
