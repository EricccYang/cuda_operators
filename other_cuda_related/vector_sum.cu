#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Vector sum (reduce -> scalar)
// 思路：两阶段 reduction，一趟 kernel 搞定
//   1) 每个线程 grid-stride 累加自己负责的若干元素
//   2) warp 内 __shfl_down_sync 树形 reduce
//   3) 每 warp 的 partial 写到 smem
//   4) 第 0 warp 再 reduce 一次，由 lane 0 atomicAdd 到全局结果
//
// 与 reduce/ 目录里多版本不同，这里写一个"现代"的精简版作为对照。

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)


__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}


__global__ void vec_sum(const float* __restrict__ in, float* out, int n) {
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 1) 每线程的 partial sum (grid-stride)
    float v = 0.f;
    for (int i = tid; i < n; i += stride) v += in[i];

    // 2) warp 内 reduce
    v = warp_reduce_sum(v);

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    // 3) 第 0 warp 把 WARPS_PER_BLOCK 个 partial 再 reduce
    if (wid == 0) {
        v = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}


float sumCPU(const float* a, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i];
    return (float)s;
}


int main() {
    int n = 1 << 24;
    size_t nBytes = n * sizeof(float);

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float *h_in = (float*)malloc(nBytes);
    for (int i = 0; i < n; i++) h_in[i] = 1.f;     // 期望 sum = n
    float ref = sumCPU(h_in, n);

    float *d_in, *d_out;
    cudaMalloc(&d_in, nBytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    int grid_size = prop.multiProcessorCount * 16;
    dim3 block(BLOCK_SIZE), grid(grid_size);

    cudaMemset(d_out, 0, sizeof(float));
    vec_sum<<<grid, block>>>(d_in, d_out, n);    // warmup
    cudaMemset(d_out, 0, sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 50;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) {
        cudaMemsetAsync(d_out, 0, sizeof(float));
        vec_sum<<<grid, block>>>(d_in, d_out, n);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    float gbps = (float)nBytes / (ms * 1e6f);

    float h_out = 0;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("[vec_sum] grid=%d block=%d  %.4f ms  %.1f GB/s\n", grid_size, BLOCK_SIZE, ms, gbps);
    printf("  cpu=%.1f  gpu=%.1f  %s\n", ref, h_out,
           (fabsf(ref - h_out) / ref < 1e-4f) ? "OK" : "FAIL");

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}
