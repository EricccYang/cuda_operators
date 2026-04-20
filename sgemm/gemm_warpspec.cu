// nvcc -arch=sm_80 gemm_warpspec.cu -o gemm_warpspec
//
// ============================================================
//  Warp-Specialized SGEMM (在 multistage 基础上做 role split)
// ============================================================
//
//  核心改动 (相比 gemm_multistage.cu):
//
//  (1) blockDim 从 256 变成 256 + 64 = 320 threads (10 warps).
//         - thread 0..255 : CONSUMER  (compute)
//         - thread 256..319: PRODUCER (只负责 cp.async)
//
//  (2) 所有 cp.async 只由 PRODUCER 发, 写法和 multistage 一样
//      (__pipeline_memcpy_async + commit + wait_prior).
//
//  (3) CONSUMER 的 compute 分支和 multistage / double buffer
//      "inner 两重 loop" 一模一样, 不动.
//
//  (4) 同步还是用 __syncthreads() —— 最简单, 全 block 所有 320
//      threads 都会参与. 真正的 "producer 比 consumer 领先多
//      stage" 需要 named barrier / mbarrier, 这里先不引入, 这版
//      主要是把 "角色分离" 的结构看清楚. 跑通之后再换 named bar.
//
//  为什么要 WS:
//    - PRODUCER warp 不持有 r_c[TM][TN] 这 64 个累加器寄存器,
//      寄存器压力低很多, 占用率 (occupancy) 可以更高.
//    - PRODUCER 专心做 memory, CONSUMER 专心做 math, 指令流
//      更干净, dual-issue/warp scheduling 更容易重叠.
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


__global__ void sgemm_warpspec(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int STAGES = 3;
    const int CONSUMER_THREADS = 256;     // 8 warps 算 compute
    const int PRODUCER_THREADS = 64;      // 2 warps 专门搬数

    int tid = threadIdx.x;                // 1D blockDim = 320
    bool isProducer = (tid >= CONSUMER_THREADS);
    int ptid = tid - CONSUMER_THREADS;    // producer 内部编号, 0..63

    // consumer 沿用 multistage 里 (ty, tx) 的切分, 只不过从 1D tid 恢复
    int cty = tid >> 4;                   // tid / 16, 0..15
    int ctx = tid & 15;                   // tid % 16, 0..15

    __shared__ float s_a[STAGES][BK][BM];
    __shared__ float s_b[STAGES][BK][BN];

    int step = (K + BK - 1) / BK;

    // PRODUCER 的 load 切分 (64 threads 各自搬 16 float A + 16 float B):
    //   A (128 m × 8 k): ptid 负责 m = ptid*2, ptid*2+1 这两行, 共 8*2 = 16 float.
    //   B (8 k × 128 n): ptid 负责 k = ptid/8, n = (ptid%8)*16 开始的 16 列.
    int prod_a_m0 = ptid * 2;
    int prod_b_k  = ptid >> 3;
    int prod_b_n0 = (ptid & 7) * 16;

    // -------- 封装一下: producer 发射一个 stage 的 cp.async --------
    auto producer_issue = [&](int stage, int bk_idx) {
        int gm = blockIdx.y * BM;
        int gk = bk_idx * BK;
        int gn = blockIdx.x * BN;

        // A: 每个 m 行 8 个 k, 因为 s_a transposed, 拆成 8 个 4-byte cp.async
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m = prod_a_m0 + mi;
            #pragma unroll
            for (int ki = 0; ki < BK; ki++) {
                __pipeline_memcpy_async(
                    &s_a[stage][ki][m],
                    &a[(gm + m) * K + gk + ki],
                    sizeof(float));
            }
        }
        // B: 16 个 n 拆成 4 个 float4 cp.async
        #pragma unroll
        for (int ni = 0; ni < 16; ni += 4) {
            __pipeline_memcpy_async(
                &s_b[stage][prod_b_k][prod_b_n0 + ni],
                &b[(gk + prod_b_k) * N + gn + prod_b_n0 + ni],
                sizeof(float4));
        }
    };

    // ---------- Prologue: producer 预发 STAGES-1 个 stage ----------
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (isProducer) {
            if (s < step) producer_issue(s, s);
            __pipeline_commit();
        }
    }

    float r_c[TM][TN] = {0};
    float compute_a[TM];
    float compute_b[TN];

    // ---------- 主循环 ----------
    for (int bk = 0; bk < step; bk++) {
        int issue_bk    = bk + STAGES - 1;
        int issue_stage = issue_bk % STAGES;

        // PRODUCER: 发下一块的 cp.async (不 block), 然后 wait 最老那块
        if (isProducer) {
            if (issue_bk < step) producer_issue(issue_stage, issue_bk);
            __pipeline_commit();
            __pipeline_wait_prior(STAGES - 2);
        }

        __syncthreads();   // 全 block 同步: 让 consumer 看见 producer 写好的 smem

        // CONSUMER: compute. 这段和 multistage / double buffer 的 inner loop 一模一样
        if (!isProducer) {
            int compute_stage = bk % STAGES;
            #pragma unroll
            for (int i = 0; i < BK; i++) {
                FLOAT4(compute_a[0]) = FLOAT4(s_a[compute_stage][i][cty * TM / 2]);
                FLOAT4(compute_a[4]) = FLOAT4(s_a[compute_stage][i][cty * TM / 2 + BM / 2]);
                FLOAT4(compute_b[0]) = FLOAT4(s_b[compute_stage][i][ctx * TN / 2]);
                FLOAT4(compute_b[4]) = FLOAT4(s_b[compute_stage][i][ctx * TN / 2 + BN / 2]);

                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    #pragma unroll
                    for (int n = 0; n < TN; n++) {
                        r_c[m][n] += compute_a[m] * compute_b[n];
                    }
                }
            }
        }

        __syncthreads();   // 保证 consumer 读完再让 producer 覆盖这个 stage
    }

    // ---------- 写回 C (只有 consumer 写) ----------
    if (!isProducer) {
        #pragma unroll
        for (int i = 0; i < TM / 2; i++) {
            int g_m = blockIdx.y * BM + cty * TM / 2 + i;
            int g_n = blockIdx.x * BN + ctx * TN / 2;
            FLOAT4(c[g_m * N + g_n])          = FLOAT4(r_c[i][0]);
            FLOAT4(c[g_m * N + g_n + BN / 2]) = FLOAT4(r_c[i][4]);
        }
        #pragma unroll
        for (int i = 0; i < TM / 2; i++) {
            int g_m = blockIdx.y * BM + cty * TM / 2 + BM / 2 + i;
            int g_n = blockIdx.x * BN + ctx * TN / 2;
            FLOAT4(c[g_m * N + g_n])          = FLOAT4(r_c[TM / 2 + i][0]);
            FLOAT4(c[g_m * N + g_n + BN / 2]) = FLOAT4(r_c[TM / 2 + i][4]);
        }
    }
}


int main(void) {
    printf("\nKernel = sgemm_warpspec\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128;
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm_warpspec;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(320);                                       // 1D, 256 + 64
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024};

    for (int i = 0; i < 15; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        dim3 blockDim(320);
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
