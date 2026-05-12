#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 256-bin histogram over an array of bytes (stored in unsigned int for simplicity).
// 优化路径：
//   v1 naive : 每个线程对 global histogram 直接 atomicAdd        -> 全局原子争用爆炸
//   v2 smem  : block 内 shared memory 私有直方图，再合并回 global -> 本文件实现
//   v3 repl  : 每 block 维护 N 份子直方图，进一步打散冲突
//   v4 vec   : uchar4 / int4 向量化 load 提高带宽利用

#define NUM_BINS    256
#define BLOCK_SIZE  256


// ---------- v1: naive global atomic ----------
__global__ void histogram_v1_global(const unsigned int* __restrict__ input,
                                    unsigned int* __restrict__ hist,
                                    int n) {

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x* gridDim.x;
    for(int i = id ;i < n;  i+= stride){
        int index = input[i] & (NUM_BINS -1);
        atomicAdd(&hist[index] ,1u); 
    }    
}




// int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = gridDim.x * blockDim.x;
//     for (int i = tid; i < n; i += stride) {
//         unsigned int b = input[i] & (NUM_BINS - 1);
//         atomicAdd(&hist[b], 1u);
//     }


// ---------- v2: shared memory privatized histogram ----------
__global__ void histogram_v2_smem(const unsigned int* __restrict__ input,
                                  unsigned int* __restrict__ hist,
                                  int n) {
    __shared__ int s_bins[NUM_BINS];
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    
    #pragma unroll
    for(int i = threadIdx.x; i < NUM_BINS; i+= blockDim.x){
        s_bins[i] = 0u;
    }
    __syncthreads();
    

    int stride = gridDim.x * blockDim.x;

    for(int i = id ;i < n ;i+=stride){
        int index = input[i] &( NUM_BINS - 1);
        atomicAdd(&s_bins[index],1);
    }

    __syncthreads();

    #pragma unroll
    for(int i = threadIdx.x; i < NUM_BINS; i+= blockDim.x){
        atomicAdd(&hist[i] , s_bins[i]);
    }    
}



__global__ void histogram_v2_smem(const unsigned int* __restrict__ input,
    unsigned int* __restrict__ hist,
    int n) {

    __shared__ sm[NUM_BINS];
    
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    #pragma unroll
    for(int i = tid ;i < NUM_BINS; i+= blockDim.x){
        sm[i] = 0u;
    }
    __syncthreads();


    int stride = gridDim.x * blockDim.x;
    for(int i = tid; i<  n ;i+=stride){

    }

    __syncthreads();
    #pragma unroll
    for(int i = tid ;i < NUM_BINS; i+= blockDim.x){
        atomicAdd(&hist[i],sm[i]);
    }

}    




// __shared__ unsigned int s_hist[NUM_BINS];

// // 1) clear shared histogram. block 256 threads 正好对应 256 bins，一人清一个。
// //    若 BLOCK_SIZE != NUM_BINS 需要 stride 循环。
// #pragma unroll
// for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
//     s_hist[i] = 0u;
// }
// __syncthreads();

// // 2) grid-stride loop, atomicAdd 落在 smem 上，比 gmem 快一个数量级
// int tid = blockIdx.x * blockDim.x + threadIdx.x;
// int stride = gridDim.x * blockDim.x;
// for (int i = tid; i < n; i += stride) {
//     unsigned int b = input[i] & (NUM_BINS - 1);
//     atomicAdd(&s_hist[b], 1u);
// }
// __syncthreads();

// // 3) merge per-block histogram back to global
// for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
//     atomicAdd(&hist[i], s_hist[i]);
// }


// ---------- host helpers ----------
void initData(unsigned int* p, int n, unsigned int seed = 1234u) {
    // 简单 LCG，避免 rand() 的低位偏置
    unsigned int s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = s;  // kernel 内会 & 0xFF
    }
}

void histogramCPU(const unsigned int* input, unsigned int* hist, int n) {
    memset(hist, 0, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < n; i++) {
        hist[input[i] & (NUM_BINS - 1)]++;
    }
}

bool checkResult(const unsigned int* ref, const unsigned int* gpu) {
    for (int i = 0; i < NUM_BINS; i++) {
        if (ref[i] != gpu[i]) {
            printf("Mismatch at bin %d: cpu=%u gpu=%u\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

float timeKernel(void (*launch)(const unsigned int*, unsigned int*, int, dim3, dim3),
                 const unsigned int* d_in, unsigned int* d_hist, int n,
                 dim3 grid, dim3 block, const char* name) {
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // warmup
    launch(d_in, d_hist, n, grid, block);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    cudaEventRecord(s);
    int iters = 10;
    for (int i = 0; i < iters; i++) {
        cudaMemsetAsync(d_hist, 0, NUM_BINS * sizeof(unsigned int));
        launch(d_in, d_hist, n, grid, block);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    ms /= iters;

    float bytes = (float)n * sizeof(unsigned int);
    float gbps = bytes / (ms * 1e6f);
    printf("[%-18s] grid=%d block=%d  time=%.4f ms  eff_BW=%.1f GB/s\n",
           name, grid.x, block.x, ms, gbps);

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms;
}

void launch_v1(const unsigned int* d_in, unsigned int* d_hist, int n, dim3 g, dim3 b) {
    histogram_v1_global<<<g, b>>>(d_in, d_hist, n);
}
void launch_v2(const unsigned int* d_in, unsigned int* d_hist, int n, dim3 g, dim3 b) {
    histogram_v2_smem<<<g, b>>>(d_in, d_hist, n);
}


int main() {
    printf("Histogram (NUM_BINS=%d, BLOCK=%d)\n", NUM_BINS, BLOCK_SIZE);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);

    int n = 1 << 26;  // 64M elements ≈ 256 MB
    size_t nBytes = (size_t)n * sizeof(unsigned int);

    unsigned int *h_in = (unsigned int*)malloc(nBytes);
    unsigned int  h_ref[NUM_BINS];
    unsigned int  h_gpu[NUM_BINS];
    initData(h_in, n);
    histogramCPU(h_in, h_ref, n);

    unsigned int *d_in, *d_hist;
    cudaMalloc(&d_in, nBytes);
    cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    // grid-stride: 让线程数 ~ 4 × SM × max_threads，足够覆盖延迟又不至于过多
    int grid_size = prop.multiProcessorCount * 32;
    dim3 grid(grid_size);

    // ---- v1 ----
    timeKernel(launch_v1, d_in, d_hist, n, grid, block, "v1_global_atomic");
    cudaMemcpy(h_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("  v1 correctness: %s\n", checkResult(h_ref, h_gpu) ? "OK" : "FAIL");

    // ---- v2 ----
    timeKernel(launch_v2, d_in, d_hist, n, grid, block, "v2_smem_privatized");
    cudaMemcpy(h_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("  v2 correctness: %s\n", checkResult(h_ref, h_gpu) ? "OK" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_hist);
    free(h_in);
    return 0;
}
