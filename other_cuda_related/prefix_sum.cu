#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Inclusive prefix sum (scan).
// 优化路径：
//   v1 hillis-steele : smem，log(N) 步每步 stride 翻倍                     O(N log N) work
//   v2 warp shuffle  : warp 内 __shfl_up_sync + warp-of-warps              无 smem 无 sync
//   v3 multi-block   : block 自身 scan -> 收 block sums -> 二次加回         通用大数组
//   v4 lookback      : 一趟 pass，跨 block atomic + spin 拿前缀             cub::DeviceScan
//
// 本文件实现 v1 + v2，约束 N <= BLOCK_SIZE，先把单 block 内的算法学清楚。

#define BLOCK_SIZE 1024     // 一定要是 32 的倍数；v2 假设 BLOCK_SIZE = 32 warps × 32 lanes



// ---------- v1: Hillis-Steele in shared memory ----------
// 每步 stride 翻倍：s[tid] += s[tid - stride] (若 tid >= stride)
// 共 log2(BLOCK_SIZE) 步。
// 关键：每步要两次 __syncthreads()，先读旧值，再写，否则会读到本轮已经更新过的邻居。
__global__ void scan_v1_hillis_steele(const int* __restrict__ in,
                                      int* __restrict__ out,
                                      int n) {
    __shared__ sm[BLOCK_SIZE];
    int tid = threadIdx.x;
    sm[tid] = 0u;

    for(int stride = 1 ; stride < BLOCK_SIZE; stride <<= 1){
        int v = tid - stride >=0 ? sm[tid- stride] : 0;
        __syncthreads();
        
        sm[tid] += v;
        _syncthreads();
    }

    if(tid < n){
        out[tid] = sm[tid];
    }
}



// __shared__ int s[BLOCK_SIZE];
// int tid = threadIdx.x;

// s[tid] = (tid < n) ? in[tid] : 0;
// __syncthreads();

// for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
//     int v = (tid >= stride) ? s[tid - stride] : 0;
//     __syncthreads();          // 读完才允许写，避免 race
//     s[tid] += v;
//     __syncthreads();
// }

// if (tid < n) out[tid] = s[tid];





// ---------- v2: warp shuffle block scan ----------
// 三步法（Kogge-Stone 在 warp 内做，warps 间再做一次 warp-scan，最后加回）：
//   1) warp 内 inclusive scan via __shfl_up_sync   (5 步覆盖 32 lanes)
//   2) 每 warp 的总和写入 warp_sums[wid]
//   3) 第 0 warp 对 warp_sums 再做一次 warp-scan
//   4) wid > 0 的线程加上 warp_sums[wid - 1]（前面所有 warp 的总和）
__global__ void scan_v2_warp_shuffle(const int* __restrict__ in,
                                     int* __restrict__ out,
                                     int n) {

    __shared__ int sm[32];

    int tid=  threadIdx.x;
    int lid = tid & 31;
    int wid = tid >> 5;
    
    int val = in[tid];
    #pragma unroll
    for(int off = 1 ; off < 32 ; off >>= 1){
        int v_get = shfl_up_sync(0xffffffff, val ,off);
        if(lid >= off){
            val+=v_get;
        }
    }
    //现在已经排列好了，并返回和
    if(lid == 31){
        sm[wid] = val;
    }

    __syncthreads();


    //和再做prefixsum
    if(wid == 0){
        int t = sm[lid];
        for(int off = 1; off < 32 ; off >>=1 ){
            int tt = shfl_up_sync(0xffffffff, t, off);
            if( lid >= off){
                t+=tt;
            }
        }
        sm[lid] = t;
    }

    __syncthreads();


    if(wid > 0){
        val += sm[wid-1];
    }

    out[tid] = val;
}


// __shared__ int warp_sums[32];     // BLOCK_SIZE / 32 个 warp

//     int tid  = threadIdx.x;
//     int lane = tid & 31;
//     int wid  = tid >> 5;

//     int v = (tid < n) ? in[tid] : 0;

//     // 1) intra-warp inclusive scan
//     #pragma unroll
//     for (int offset = 1; offset < 32; offset <<= 1) {
//         int t = __shfl_up_sync(0xffffffff, v, offset);
//         if (lane >= offset) v += t;
//     }

//     // 2) 每 warp 的 total（最后一个 lane 的 v）
//     if (lane == 31) warp_sums[wid] = v;
//     __syncthreads();

//     // 3) 第 0 warp 对 warp_sums 做 warp-scan
//     //    BLOCK_SIZE = 1024 时 warp 数刚好 32，一个 warp 就够
//     if (wid == 0) {
//         int t = warp_sums[lane];
//         #pragma unroll
//         for (int offset = 1; offset < 32; offset <<= 1) {
//             int u = __shfl_up_sync(0xffffffff, t, offset);
//             if (lane >= offset) t += u;
//         }
//         warp_sums[lane] = t;
//     }
//     __syncthreads();

//     // 4) 加回前一 warp 的总和（第 0 warp 不动）
//     if (wid > 0) v += warp_sums[wid - 1];

//     if (tid < n) out[tid] = v;


// ---------- host helpers ----------
void scanCPU(const int* in, int* out, int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) {
        acc += in[i];
        out[i] = acc;
    }
}

bool checkResult(const int* ref, const int* gpu, int n) {
    for (int i = 0; i < n; i++) {
        if (ref[i] != gpu[i]) {
            printf("  Mismatch at %d: cpu=%d gpu=%d\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}


// ---------- launch helpers ----------
typedef void (*scan_kernel_t)(const int*, int*, int);

void run_kernel(scan_kernel_t k, const char* name,
                const int* d_in, int* d_out, int* h_gpu, const int* h_ref,
                int n, size_t nBytes) {
    dim3 block(BLOCK_SIZE), grid(1);

    cudaMemset(d_out, 0, nBytes);
    k<<<grid, block>>>(d_in, d_out, n);   // warmup

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 200;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) k<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    ms /= iters;

    cudaMemcpy(h_gpu, d_out, nBytes, cudaMemcpyDeviceToHost);
    bool ok = checkResult(h_ref, h_gpu, n);
    printf("[%-22s] block=%d  time=%.4f ms  %s\n",
           name, BLOCK_SIZE, ms, ok ? "OK" : "FAIL");

    cudaEventDestroy(s); cudaEventDestroy(e);
}


int main() {
    printf("Inclusive scan (single block, N <= BLOCK_SIZE = %d)\n", BLOCK_SIZE);

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n\n", prop.name);

    int n = BLOCK_SIZE;
    size_t nBytes = n * sizeof(int);

    int *h_in  = (int*)malloc(nBytes);
    int *h_ref = (int*)malloc(nBytes);
    int *h_gpu = (int*)malloc(nBytes);

    // 有正有负，能暴露 underflow / overflow / 顺序敏感的 bug
    for (int i = 0; i < n; i++) h_in[i] = (i % 17) - 8;
    scanCPU(h_in, h_ref, n);

    int *d_in, *d_out;
    cudaMalloc(&d_in, nBytes);
    cudaMalloc(&d_out, nBytes);
    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    run_kernel(scan_v1_hillis_steele, "v1_hillis_steele_smem", d_in, d_out, h_gpu, h_ref, n, nBytes);
    run_kernel(scan_v2_warp_shuffle,  "v2_warp_shuffle",       d_in, d_out, h_gpu, h_ref, n, nBytes);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_ref);
    free(h_gpu);
    return 0;
}
