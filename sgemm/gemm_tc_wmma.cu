// nvcc -arch=sm_90 gemm_tc_wmma.cu -o gemm_tc_wmma
//
// ============================================================
//  TensorCore SGEMM  (TF32 via WMMA, sm_80+ 通用, 包括 sm_90)
// ============================================================
//
//  和之前 SIMT 版本的关键差别:
//
//  1. **A 不再转置**: s_a[STAGES][BM][BK] (M 慢 K 快, 和 gmem 一致)
//     → cp.async 一发就是 16B (float4), 不再拆 4B, gmem coalescing 也回来了
//
//  2. **compute 不再手写 outer product**: 改用 wmma MMA 指令 m16n16k8 TF32
//     → 一条 mma.sync 吃整块 16x16x8 tile, 吞吐比 FFMA 高一大截
//
//  3. **TF32 精度**: float 输入在 mma 前被舍到 19-bit (符号 1 + 指数 8 + 尾数 10),
//     FP32 累加. 误差会比 pure FP32 大几十倍, 但 ~1e-3 级别还在合理范围.
//
//  Tile 规划:
//     BM = 128, BN = 128, BK = 8      (与 SIMT 版保持一致, 方便横向比较)
//     256 threads = 8 warps
//     warp tile = 64 x 32 (M x N)     ← 2 warps 在 M, 4 warps 在 N = 8 warps
//     mma shape = m16n16k8            ← TF32 的 wmma 标准形状
//     每 warp 每 k-slice 做 (64/16) x (32/16) = 4 x 2 = 8 条 mma 指令
//     BK = MMA_K = 8 所以 1 个 bk 迭代对应 1 个 k-slice
//
//  会不会再遇到之前的坑?
//     - 4x cp.async.4B : 不会, s_a 不转置
//     - smem bank conflict: wmma::load_matrix_sync 内部处理了 tile 内的 LDS pattern,
//       如果 ncu 之后发现仍有冲突, 再给 s_a 最内维加 padding
//     - uncoalesced gmem  : 不会, 16B cp.async 命中 cache line
//
//  Hopper 原生路线 (未走):
//     wgmma.mma_async + TMA + mbarrier + warp specialize.  那套复杂得多,
//     这里用 wmma 看 TensorCore 基本概念, wgmma 留到下一轮.
// ============================================================

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);
float testPerformance(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float psum = 0.0f;
            for (int k = 0; k < K; k++) psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            c[OFFSET(m, n, N)] = psum;
        }
}


__global__ void sgemm_tc_wmma(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    // ----- Tile constants -----
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int STAGES = 3;

    const int MMA_M = 16;
    const int MMA_N = 16;
    const int MMA_K = 8;

    // 8 warps per block, arranged 2 (along M) x 4 (along N)
    const int WARP_ROWS = 2;                     // warps stacked in M
    const int WARP_COLS = 4;                     // warps spread in N
    const int WM = BM / WARP_ROWS;               // 64 per warp in M
    const int WN = BN / WARP_COLS;               // 32 per warp in N
    const int MCNT = WM / MMA_M;                 // 4
    const int NCNT = WN / MMA_N;                 // 2

    int tid     = threadIdx.x;                   // 1D, 256 threads
    int warp_id = tid >> 5;                      // 0..7
    int warp_m  = warp_id / WARP_COLS;           // 0..1
    int warp_n  = warp_id % WARP_COLS;           // 0..3

    // ----- Shared memory (natural, non-transposed) -----
    __shared__ float s_a[STAGES][BM][BK];
    __shared__ float s_b[STAGES][BK][BN];

    // ----- Load partitioning (256 threads, 1 float4 each per operand per stage) -----
    // A: 128 M rows x 8 K cols = 1024 floats; 2 threads per M row, each loads 4 K cols
    int load_a_m = tid >> 1;
    int load_a_k = (tid & 1) << 2;
    // B: 8 K rows x 128 N cols = 1024 floats; 32 threads per K row, each loads 4 N cols
    int load_b_k = tid >> 5;
    int load_b_n = (tid & 31) << 2;

    int load_gmem_a_m = blockIdx.y * BM + load_a_m;
    int load_gmem_b_n = blockIdx.x * BN + load_b_n;

    int step = (K + BK - 1) / BK;

    // ----- Accumulator fragments (one fragment per 16x16 warp-tile cell) -----
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_frag[MCNT][NCNT];
    #pragma unroll
    for (int i = 0; i < MCNT; i++)
        #pragma unroll
        for (int j = 0; j < NCNT; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // ----- Prologue: issue first STAGES-1 stages of cp.async -----
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < step) {
            int gk_a = s * BK + load_a_k;
            int gk_b = s * BK + load_b_k;
            __pipeline_memcpy_async(
                &s_a[s][load_a_m][load_a_k],
                &a[load_gmem_a_m * K + gk_a],
                sizeof(float4));                 // 16B, 一发搞定, 不用再拆 4B
            __pipeline_memcpy_async(
                &s_b[s][load_b_k][load_b_n],
                &b[gk_b * N + load_gmem_b_n],
                sizeof(float4));
        }
        __pipeline_commit();
    }

    // ----- Main loop -----
    for (int bk = 0; bk < step; bk++) {
        int issue_bk    = bk + STAGES - 1;
        int issue_stage = issue_bk % STAGES;
        if (issue_bk < step) {
            int gk_a = issue_bk * BK + load_a_k;
            int gk_b = issue_bk * BK + load_b_k;
            __pipeline_memcpy_async(
                &s_a[issue_stage][load_a_m][load_a_k],
                &a[load_gmem_a_m * K + gk_a],
                sizeof(float4));
            __pipeline_memcpy_async(
                &s_b[issue_stage][load_b_k][load_b_n],
                &b[gk_b * N + load_gmem_b_n],
                sizeof(float4));
        }
        __pipeline_commit();
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();

        int compute_stage = bk % STAGES;

        // Load A fragments: MCNT=4 个 16x8 瓦片 (row major, K 方向)
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K,
                       wmma::precision::tf32, wmma::row_major> a_frag[MCNT];
        #pragma unroll
        for (int i = 0; i < MCNT; i++) {
            int m_off = warp_m * WM + i * MMA_M;
            wmma::load_matrix_sync(
                a_frag[i],
                &s_a[compute_stage][m_off][0],
                BK);                              // row stride
        }

        // Load B fragments: NCNT=2 个 8x16 瓦片 (row major, N 方向)
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K,
                       wmma::precision::tf32, wmma::row_major> b_frag[NCNT];
        #pragma unroll
        for (int j = 0; j < NCNT; j++) {
            int n_off = warp_n * WN + j * MMA_N;
            wmma::load_matrix_sync(
                b_frag[j],
                &s_b[compute_stage][0][n_off],
                BN);
        }

        // 把 float 元素舍到 TF32 (清零低 13 bit 尾数). 否则 mma.sync 行为未定义.
        #pragma unroll
        for (int i = 0; i < MCNT; i++) {
            #pragma unroll
            for (int t = 0; t < a_frag[i].num_elements; t++)
                a_frag[i].x[t] = wmma::__float_to_tf32(a_frag[i].x[t]);
        }
        #pragma unroll
        for (int j = 0; j < NCNT; j++) {
            #pragma unroll
            for (int t = 0; t < b_frag[j].num_elements; t++)
                b_frag[j].x[t] = wmma::__float_to_tf32(b_frag[j].x[t]);
        }

        // MMA: 每 warp MCNT * NCNT = 8 条 mma.sync.m16n16k8 指令
        #pragma unroll
        for (int i = 0; i < MCNT; i++)
            #pragma unroll
            for (int j = 0; j < NCNT; j++)
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);

        __syncthreads();
    }

    // ----- Store accumulators back to C -----
    #pragma unroll
    for (int i = 0; i < MCNT; i++) {
        #pragma unroll
        for (int j = 0; j < NCNT; j++) {
            int m_off = blockIdx.y * BM + warp_m * WM + i * MMA_M;
            int n_off = blockIdx.x * BN + warp_n * WN + j * MMA_N;
            wmma::store_matrix_sync(
                &c[m_off * N + n_off],
                c_frag[i][j],
                N,
                wmma::mem_row_major);
        }
    }
}


int main(void) {
    printf("\nKernel = sgemm_tc_wmma (TF32)\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128;
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm_tc_wmma;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(256);                                   // 1D, 8 warps
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f  (TF32 精度, ~1e-3 级别正常)\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024};

    for (int i = 0; i < 15; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        dim3 blockDim(256);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        double max_sec = 0.0, min_sec = DBL_MAX, total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG = %10.4lf Gflops\n",
               M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}


float testError(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *h_a=(float*)malloc(size_a),*h_b=(float*)malloc(size_b),*h_c=(float*)malloc(size_c),*h_d_c=(float*)malloc(size_c);
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size_a); cudaMalloc(&d_b,size_b); cudaMalloc(&d_c,size_c);
    srand(time(0));
    for(int i=0;i<M*K;i++) h_a[i]=rand()/float(RAND_MAX);
    for(int i=0;i<K*N;i++) h_b[i]=rand()/float(RAND_MAX);
    cudaMemset(d_c,0,size_c);
    cpuSgemm(h_a,h_b,h_c,M,N,K);
    cudaMemcpy(d_a,h_a,size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_b,cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim,blockDim>>>(d_a,d_b,d_c,M,N,K);
    cudaMemcpy(h_d_c,d_c,size_c,cudaMemcpyDeviceToHost);
    float max_error=0.0f;
    for(int i=0;i<M*N;i++){float e=fabsf(h_d_c[i]-h_c[i]); if(e!=e||max_error!=max_error) max_error=-NAN; else max_error=max(max_error,e);}
    free(h_a);free(h_b);free(h_c);free(h_d_c);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    return max_error;
}

float testPerformance(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size_a); cudaMalloc(&d_b,size_b); cudaMalloc(&d_c,size_c);
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<repeat;i++) gpuSgemm<<<gridDim,blockDim>>>(d_a,d_b,d_c,M,N,K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    return ms/1000.0/repeat;
}
