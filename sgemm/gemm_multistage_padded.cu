// nvcc -arch=sm_90 gemm_multistage_padded.cu -o gemm_multistage_padded
//
// ============================================================
//  Multistage SGEMM + smem padding (照 CUTLASS SIMT 的 thinking)
//
//  相比 gemm_multistage.cu 只多一处改动:
//    s_a[STAGES][BK][BM]  →  s_a[STAGES][BK][BM + PAD_M]
//
//  目的: 避 smem 写入的 bank conflict.
//
//  为什么原 multistage 有 conflict:
//    - 每个线程对 A 做 4 条 4B cp.async, 写到 s_a[k..k+3][m]
//    - warp 内偶数 tid 的 i=0 写 s_a[0][0..15], 奇数 tid 写 s_a[4][0..15]
//    - 无 padding 时 4*BM = 512 = 16 * 32, 行间偏移是 bank 对齐整数倍
//      → 两组写落到同一批 bank 上, 2-way 冲突
//
//  为什么 PAD_M = 4 正好:
//    - 新 stride = 128 + 4 = 132
//    - 4 * 132 = 528 = 16 * 32 + 16
//    - 行 0 与行 4 的 bank 偏差正好 16 (半圈), 两组填满 32 个 bank, 零冲突
//
//  CUTLASS simt_transpose_padding(256, 8, 32) = 32, 是针对他们自己 thread
//  map 的最佳值; 对咱们这种 (m=tid/2, k=(tid&1)*4) 的 mapping, 4 就够了,
//  也更省 smem.
// ============================================================

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

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


__global__ void sgemm_multistage_padded(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int STAGES = 3;
    const int PAD_M = 4;                       // <-- 关键: s_a 的 M 维加 4 个 float padding
    const int BM_PAD = BM + PAD_M;             // 132

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float s_a[STAGES][BK][BM_PAD];  // <-- 唯一差别: 最内维带 padding
    __shared__ float s_b[STAGES][BK][BN];      //     s_b 不改 (float4 写基本 conflict-free)

    int load_smem_a_m = tid >> 1;
    int load_smem_a_k = (tid & 1) << 2;
    int load_smem_b_k = tid >> 5;
    int load_smem_b_n = (tid & 31) << 2;

    int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m;
    int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n;

    int step = (K + BK - 1) / BK;

    // prologue
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < step) {
            int gk_a = s * BK + load_smem_a_k;
            int gk_b = s * BK + load_smem_b_k;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __pipeline_memcpy_async(
                    &s_a[s][load_smem_a_k + i][load_smem_a_m],
                    &a[load_gmem_a_m * K + gk_a + i],
                    sizeof(float));
            }
            __pipeline_memcpy_async(
                &s_b[s][load_smem_b_k][load_smem_b_n],
                &b[gk_b * N + load_gmem_b_n],
                sizeof(float4));
        }
        __pipeline_commit();
    }

    float r_c[TM][TN] = {0};
    float compute_a[TM];
    float compute_b[TN];

    // main loop
    for (int bk = 0; bk < step; bk++) {
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
        __pipeline_commit();

        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();

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
        __syncthreads();
    }

    // store
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
    printf("\nKernel = sgemm_multistage_padded\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm_multistage_padded;

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
