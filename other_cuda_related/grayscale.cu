#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// RGB -> grayscale
// 公式：Y = 0.299 R + 0.587 G + 0.114 B (ITU-R BT.601)
// 用 8-bit 整数版本：Y ≈ (77*R + 150*G + 29*B) >> 8
//   77/256=0.300  150/256=0.586  29/256=0.113   误差 < 1
//
// 思路：
//   一线程一个像素，完全 embarrassingly parallel
//   v1 标量    : 读 RGB 三个 byte，写 1 个 byte
//   v2 uchar4  : 输入打包成 RGBA (4 byte)，一次 32-bit load，吞吐更高

#define BLOCK_SIZE 256


// ---------- v1: scalar 8-bit ----------
__global__ void rgb2gray_v1(const unsigned char* __restrict__ rgb,
                            unsigned char* __restrict__ gray, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int r = rgb[3 * i + 0];
    unsigned int g = rgb[3 * i + 1];
    unsigned int b = rgb[3 * i + 2];
    gray[i] = (unsigned char)((77u * r + 150u * g + 29u * b) >> 8);
}


// ---------- v2: uchar4 (RGBA) 输入，一次 32-bit load ----------
// 注意：这要求输入按 RGBA 4 字节打包，不是 RGB 3 字节
__global__ void rgb2gray_v2_u4(const uchar4* __restrict__ rgba,
                               unsigned char* __restrict__ gray, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        uchar4 p = rgba[i];        // 一次 32-bit load
        unsigned int v = (77u * p.x + 150u * p.y + 29u * p.z) >> 8;
        gray[i] = (unsigned char)v;
    }
}


// ---------- host helpers ----------
void rgb2grayCPU(const unsigned char* rgb, unsigned char* gray, int n) {
    for (int i = 0; i < n; i++) {
        unsigned int r = rgb[3 * i + 0];
        unsigned int g = rgb[3 * i + 1];
        unsigned int b = rgb[3 * i + 2];
        gray[i] = (unsigned char)((77u * r + 150u * g + 29u * b) >> 8);
    }
}

bool checkResult(const unsigned char* ref, const unsigned char* gpu, int n, int tol = 1) {
    for (int i = 0; i < n; i++) {
        int d = (int)ref[i] - (int)gpu[i];
        if (d < -tol || d > tol) {
            printf("  Mismatch at %d: cpu=%u gpu=%u\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}


int main() {
    int W = 4096, H = 4096;
    int n = W * H;
    printf("grayscale: %dx%d (%d pixels, %.1f MB RGB)\n",
           W, H, n, n * 3 / 1024.f / 1024.f);

    cudaSetDevice(0);

    // ---- v1 input: RGB packed (3 byte / pixel) ----
    size_t rgbBytes = n * 3;
    unsigned char* h_rgb  = (unsigned char*)malloc(rgbBytes);
    unsigned char* h_gray = (unsigned char*)malloc(n);
    unsigned char* h_ref  = (unsigned char*)malloc(n);
    for (int i = 0; i < n; i++) {
        h_rgb[3 * i + 0] = (unsigned char)(i & 0xff);
        h_rgb[3 * i + 1] = (unsigned char)((i >> 1) & 0xff);
        h_rgb[3 * i + 2] = (unsigned char)((i >> 2) & 0xff);
    }
    rgb2grayCPU(h_rgb, h_ref, n);

    unsigned char *d_rgb, *d_gray;
    cudaMalloc(&d_rgb,  rgbBytes);
    cudaMalloc(&d_gray, n);
    cudaMemcpy(d_rgb, h_rgb, rgbBytes, cudaMemcpyHostToDevice);

    auto bench = [&](const char* name, auto fn, size_t bytes_moved) {
        cudaMemset(d_gray, 0, n);
        fn();   // warmup
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        int iters = 50;
        cudaEventRecord(s);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
        float gbps = (float)bytes_moved / (ms * 1e6f);
        cudaMemcpy(h_gray, d_gray, n, cudaMemcpyDeviceToHost);
        bool ok = checkResult(h_ref, h_gray, n);
        printf("[%-14s] %.4f ms  %.1f GB/s  %s\n", name, ms, gbps, ok ? "OK" : "FAIL");
        cudaEventDestroy(s); cudaEventDestroy(e);
    };

    // ---- v1 ----
    dim3 b1(BLOCK_SIZE), g1((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    bench("v1 rgb scalar",
          [&](){ rgb2gray_v1<<<g1, b1>>>(d_rgb, d_gray, n); },
          rgbBytes + n);   // 读 3N + 写 1N

    // ---- v2 (RGBA 输入) ----
    size_t rgbaBytes = n * 4;
    unsigned char* h_rgba = (unsigned char*)malloc(rgbaBytes);
    for (int i = 0; i < n; i++) {
        h_rgba[4 * i + 0] = h_rgb[3 * i + 0];
        h_rgba[4 * i + 1] = h_rgb[3 * i + 1];
        h_rgba[4 * i + 2] = h_rgb[3 * i + 2];
        h_rgba[4 * i + 3] = 0;   // alpha 不用
    }
    unsigned char* d_rgba;
    cudaMalloc(&d_rgba, rgbaBytes);
    cudaMemcpy(d_rgba, h_rgba, rgbaBytes, cudaMemcpyHostToDevice);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    dim3 b2(BLOCK_SIZE), g2(prop.multiProcessorCount * 16);
    bench("v2 rgba uchar4",
          [&](){ rgb2gray_v2_u4<<<g2, b2>>>((uchar4*)d_rgba, d_gray, n); },
          rgbaBytes + n);  // 读 4N + 写 1N

    cudaFree(d_rgb); cudaFree(d_rgba); cudaFree(d_gray);
    free(h_rgb); free(h_rgba); free(h_gray); free(h_ref);
    return 0;
}
