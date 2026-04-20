// nvcc -arch=sm_80 gemm_multistage.cu -o gemm_multistage
//
// ============================================================
//  Multistage SGEMM (基于 gemm_double_buffer.cu 的最小化改动)
// ============================================================
//
//  相比 double buffer 只有 3 处本质变化，compute 部分完全不变：
//
//  (1) smem 第一维从 [2] 变成 [STAGES]  (这里 STAGES=3)
//  (2) gmem -> smem 改用 cp.async (__pipeline_memcpy_async)
//  (3) 每个 bk 迭代: 发射下一个 stage 的 cp.async, 然后
//      __pipeline_wait_prior(STAGES-2) 等当前 stage 就绪
//
//  思想: double buffer 只能 "算一块 / 搬一块", 同一时刻最多 1 个
//        cp.async 在飞.  STAGES=3 允许 2 个 cp.async 同时在飞,
//        更好地隐藏 global memory 延迟.
// ============================================================

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>          // <-- 新增: cp.async 的 C++ 封装

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0f;
            for (int k = 0; k < K; k++)
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


__global__ void sgemm_multistage(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int STAGES = 3;                          // (1) stage 数改为 3

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float s_a[STAGES][BK][BM];          // (1) stage 维度
    __shared__ float s_b[STAGES][BK][BN];

    // 和 double buffer 完全一样的加载坐标
    int load_smem_a_m = tid >> 1;
    int load_smem_a_k = (tid & 1) << 2;
    int load_smem_b_k = tid >> 5;
    int load_smem_b_n = (tid & 31) << 2;

    int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m;
    int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n;

    int step = (K + BK - 1) / BK;

    // ---------- (2) Prologue: 预先发射 STAGES-1 = 2 个 stage 的 cp.async ----------
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < step) {
            int gk_a = s * BK + load_smem_a_k;
            int gk_b = s * BK + load_smem_b_k;

            // A 是 transposed 存法 (s_a[k][m] = a[m][k]), 没法一发 float4 搞定,
            // 所以拆 4 个 4-byte cp.async, 每个负责一个 k.
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __pipeline_memcpy_async(
                    &s_a[s][load_smem_a_k + i][load_smem_a_m],
                    &a[load_gmem_a_m * K + gk_a + i],
                    sizeof(float));
            }
            // B 按自然布局, 一发 float4 cp.async.
            __pipeline_memcpy_async(
                &s_b[s][load_smem_b_k][load_smem_b_n],
                &b[gk_b * N + load_gmem_b_n],
                sizeof(float4));
        }
        __pipeline_commit();                       // 每一 stage 作为一个 commit group
    }

    float r_c[TM][TN] = {0};
    float compute_a[TM];
    float compute_b[TN];

    // ---------- 主循环 ----------
    for (int bk = 0; bk < step; bk++) {

        // (2) 发射 bk+STAGES-1 这块的 cp.async, 写到 stage (bk+STAGES-1)%STAGES
        int issue_bk    = bk + STAGES - 1;
        int issue_stage = issue_bk % STAGES;
        if (issue_bk < step) {
            int gk_a = issue_bk * BK + load_smem_a_k;
            int gk_b = issue_bk * BK + load_smem_b_k;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __pipeline_memcpy_async(
                    &s_a[issue_stage][load_smem_a_k + i][load_smem_a_m],
                    &a[load_gmem_a_m * K + gk_a + i],
                    sizeof(float));
            }
            __pipeline_memcpy_async(
                &s_b[issue_stage][load_smem_b_k][load_smem_b_n],
                &b[gk_b * N + load_gmem_b_n],
                sizeof(float4));
        }
        __pipeline_commit();                       // 保证 commit 计数一致 (越界时是 empty commit)

        // (3) 等到 "最老的那个 commit group 完成", 也就是当前 bk 要用的那个 stage.
        //     发完后有 STAGES 个在飞, wait_prior(STAGES-2) 等到只剩 STAGES-2 个 = 最老 1 个完成.
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();

        // ---------- 以下 compute 部分和 double buffer 完全一样 ----------
        int compute_stage = bk % STAGES;
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            FLOAT4(compute_a[0]) = FLOAT4(s_a[compute_stage][i][ty * TM / 2]);
            FLOAT4(compute_a[4]) = FLOAT4(s_a[compute_stage][i][ty * TM / 2 + BM / 2]);
            FLOAT4(compute_b[0]) = FLOAT4(s_b[compute_stage][i][tx * TN / 2]);
            FLOAT4(compute_b[4]) = FLOAT4(s_b[compute_stage][i][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += compute_a[m] * compute_b[n];
                }
            }
        }
        __syncthreads();                           // 下轮会覆盖同一 stage 槽位, 先 sync
    }

    // ---------- 写回 C ----------
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int g_m = blockIdx.y * BM + ty * TM / 2 + i;
        int g_n = blockIdx.x * BN + tx * TN / 2;
        FLOAT4(c[g_m * N + g_n])          = FLOAT4(r_c[i][0]);
        FLOAT4(c[g_m * N + g_n + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int g_m = blockIdx.y * BM + ty * TM / 2 + BM / 2 + i;
        int g_n = blockIdx.x * BN + tx * TN / 2;
        FLOAT4(c[g_m * N + g_n])          = FLOAT4(r_c[TM / 2 + i][0]);
        FLOAT4(c[g_m * N + g_n + BN / 2]) = FLOAT4(r_c[TM / 2 + i][4]);
    }
}


int main(void) {
    printf("\nKernel = sgemm_multistage\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm_multistage;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024};

    for (int i = 0; i < 15; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        dim3 blockDim(BN / TN, BM / TM);
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

    float *h_a = (float *)malloc(size_a);
    float *h_b = (float *)malloc(size_b);
    float *h_c = (float *)malloc(size_c);
    float *h_d_c = (float *)malloc(size_c);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++) h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++) h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 0, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(h_d_c[i] - h_c[i]);
        if (e != e || max_error != max_error) max_error = -NAN;
        else max_error = max(max_error, e);
    }

    free(h_a); free(h_b); free(h_c); free(h_d_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return max_error;
}


float testPerformance(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return msec / 1000.0 / repeat;
}
