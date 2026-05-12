#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// C[i] = A[i] + B[i]
// 完全 memory-bound：每元素 3 × 4B (读 A、读 B、写 C) = 12 B
// H100 HBM ~3 TB/s，理论上限 = 3000 / 12 = 250 G elem/s
//
// v1 naive  : 一线程一元素
// v2 float4 : 一线程一个 float4，128-bit load/store，减少访存指令 4×

#define BLOCK_SIZE 256


__global__ void vec_add_v1(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}


__global__ void vec_add_v2_f4(const float4* __restrict__ a,
                              const float4* __restrict__ b,
                              float4* __restrict__ c, int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n4; i += stride) {
        float4 x = a[i], y = b[i];
        float4 z;
        z.x = x.x + y.x;
        z.y = x.y + y.y;
        z.z = x.z + y.z;
        z.w = x.w + y.w;
        c[i] = z;
    }
}


float bench(const char* name, void (*fn)(), int n, int bytes_per_elem = 12) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    fn();  // warmup
    int iters = 50;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    ms /= iters;
    float gbps = (float)bytes_per_elem * n / (ms * 1e6f);
    printf("[%-14s] %.4f ms  %.1f GB/s\n", name, ms, gbps);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}


int main() {
    int n = 1 << 24;            // 16M
    size_t nBytes = n * sizeof(float);
    printf("vector_add: N=%d (%.1f MB per array)\n", n, nBytes / 1024.f / 1024.f);

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float *h_a = (float*)malloc(nBytes);
    float *h_b = (float*)malloc(nBytes);
    float *h_c = (float*)malloc(nBytes);
    for (int i = 0; i < n; i++) { h_a[i] = i * 0.001f; h_b[i] = i * 0.002f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, nBytes); cudaMalloc(&d_b, nBytes); cudaMalloc(&d_c, nBytes);
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    // ---- v1 ----
    dim3 b1(BLOCK_SIZE), g1((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    bench("v1 naive", [&](){ vec_add_v1<<<g1, b1>>>(d_a, d_b, d_c, n); }, n);
    cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);
    bool ok1 = true;
    for (int i = 0; i < n; i++) if (h_c[i] != h_a[i] + h_b[i]) { ok1 = false; break; }
    printf("  v1 correctness: %s\n", ok1 ? "OK" : "FAIL");

    // ---- v2 ----
    int n4 = n / 4;
    int g2x = prop.multiProcessorCount * 16;     // 让 grid 远大于 SM 数
    if (g2x > (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE) g2x = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 b2(BLOCK_SIZE), g2(g2x);
    bench("v2 float4", [&](){
        vec_add_v2_f4<<<g2, b2>>>((float4*)d_a, (float4*)d_b, (float4*)d_c, n4);
    }, n);
    cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);
    bool ok2 = true;
    for (int i = 0; i < n; i++) if (h_c[i] != h_a[i] + h_b[i]) { ok2 = false; break; }
    printf("  v2 correctness: %s\n", ok2 ? "OK" : "FAIL");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
