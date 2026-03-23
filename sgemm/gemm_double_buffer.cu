#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


__global__ void sgemm_V3(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    __shared__ float a_s[2][BK][BM];
    __shared__ float b_s[2][BK][BN];

    
    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid &1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n=  (tid & 31) << 2;


    float r_c[TM][TN] = {0.f};
    float load_r_a[4];
    
    //先load一个放在0里
    {
        int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m* K + load_a_gmem_k;
        FLOAT4(load_r_a[0]) = FLOAT4(a[load_a_gmem_addr]);

        s_a[0][load_a_smem_k][load_a_smem_m] = load_r_a[0];
        s_a[0][load_a_smem_k+1][load_a_smem_m] = load_r_a[1];
        s_a[0][load_a_smem_k+2][load_a_smem_m] = load_r_a[2];
        s_a[0][load_a_smem_k+3][load_a_smem_m] = load_r_a[3];

        int load_b_gmem_k =  load_b_smem_k;
        int load_b_gmem_n =  blockIdx.x * BN + load_b_smem_n;
        int load_b_gmem_addr = load_b_gmem_k* N  + load_b_gmem_n;
        FLOAT4(s_b[0][load_b_gmem_k][load_b_gmem_n]) = FLOAT4(b[load_b_gmem_addr]);
    }

    __syncthreads();

    float compute_a[TM];
    float compute_b[TM];
    int step =   (K + BK-1)/BK;
    for(int bk = 1; bk < step; bk++){
        //compute 0 and load 1 

        int compute_index =  (bk-1)%2;
        int load_index = bk%2;

        int tx= threadIdx.x;
        int ty = threadIdx.y;
        //compute
        #pragma unroll
        for(int i =0; i  < BK; i++){
            FLOAT4(compute_a[0]) =  FLOAT4(s_a[compute_index][i][tx*TN/2]);
            FLOAT4(compute_a[4]) =  FLOAT4(s_b[compute_index][i][tx*TN/2 + BN/2])
            FLOAT4(compute_b[0]) =  FLOAT4(s_a[compute_index][i][tx*TN/2]);
            FLOAT4(compute_b[4]) =  FLOAT4(s_b[compute_index][i][tx*TN/2 + BN/2]);

            for(int m = 0;m < TM; m++){
                for(int n = 0; n < TN ;n++){
                    r_c[m][n] += compute_a[m]* compute_b[n];
                }
            }
        }

        //load
        int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
        int load_a_gmem_k = load_a_smem_k + BK * bk;
        int load_a_gmem_addr = load_a_gmem_m* K + load_a_gmem_k;
        FLOAT4(load_r_a[0]) = FLOAT4(a[load_a_gmem_addr]);

        s_a[load_index][load_a_smem_k][load_a_smem_m] = load_r_a[0];
        s_a[load_index][load_a_smem_k+1][load_a_smem_m] = load_r_a[1];
        s_a[load_index][load_a_smem_k+2][load_a_smem_m] = load_r_a[2];
        s_a[load_index][load_a_smem_k+3][load_a_smem_m] = load_r_a[3];

        int load_b_gmem_k =  load_b_smem_k + BK* bk;
        int load_b_gmem_n =  blockIdx.x * BN + load_b_smem_n;
        int load_b_gmem_addr = load_b_gmem_k* N  + load_b_gmem_n;
        FLOAT4(s_b[load_index][load_b_gmem_k][load_b_gmem_n]) = FLOAT4(b[load_b_gmem_addr]);
        
        __syncthreads();

    }


    {
        int tx= threadIdx.x;
        int ty = threadIdx.y;
        //compute
        #pragma unroll
        for(int i =0; i  < BK; i++){
            FLOAT4(compute_a[0]) =  FLOAT4(s_a[1][i][tx*TN/2]);
            FLOAT4(compute_a[4]) =  FLOAT4(s_b[1][i][tx*TN/2 + BN/2])
            FLOAT4(compute_b[0]) =  FLOAT4(s_a[1][i][tx*TN/2]);
            FLOAT4(compute_b[4]) =  FLOAT4(s_b[1][i][tx*TN/2 + BN/2]);

            for(int m = 0;m < TM; m++){
                for(int n = 0; n < TN ;n++){
                    r_c[m][n] += compute_a[m]* compute_b[n];
                }
            }
        }
    }

    //store TM/2 rows
    #pragma unroll
    for(int i =0;i < TM/2;i++){
        int store_gmem_m = blockIdx.y * BM + threadIdx.y * TM + i;
        int store_gmem_n = blockIdx.x * BN + threadIdx.x * TN/2;
        int addr = store_gmem_m*  N + store_gmem_n;
        FLOAT4(c[addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[addr + BN/2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for(int i =0;i < TM/2;i++){
        int store_gmem_m = blockIdx.y * BM + threadIdx.y * TM + i + BM/2;
        int store_gmem_n = blockIdx.x * BN + threadIdx.x * TN/2;
        int addr = store_gmem_m*  N + store_gmem_n;
        FLOAT4(c[addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[addr + BN/2]) = FLOAT4(r_c[i][4]);
    }

}

//1. 算之前一定要sync
//2.  为什么每次for循环最后需要加sync？ 不加会有数据竞争吧，比如跑的快的线程会直接进入下个循环，但是load tile的时候跟计算的时候每个线程负责的部分不同
//3.  双buffer index的控制应该需要比原先写的更好
//4，线程怎么算sm还是略模糊
//sync的开销有多大，如何实现的？



int main(void) {
    printf("\nKernal = sgemm_V3\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = sgemm_V3;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    
    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}


float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}


float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
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

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}






