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

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];
    
    int a_smem_m = tid/2;
    int a_smem_k =  (tid %2 ) << 2;
    int b_smem_k = tid >> 5;
    int b_smem_n = (tid % 32) << 2;

    float r_a_load[4]; 
    float r_b_load[4];
    
    
    //先load好一个    load在0里面
    {
        int a_gmem_m =  by * BM + a_smem_m;
        int a_gmem_k =  0 * BK + a_smem_k;
        int a_gmem_addr = a_gmem_m* K + a_gmem_k;

        int b_gmem_k =  0 * BK + b_smem_k;
        int b_gmem_n =  bx * BN  + b_smem_n;
        int b_gmem_addr = b_gmem_k* N  + b_gmem_n;

        FLOAT4(r_a_load[0]) = FLOAT4( a[a_gmem_addr] ); 
        FLOAT4(r_b_load[0]) = FLOAT4( b[b_gmem_addr] ); 

        s_a[0][a_smem_k][a_smem_m] = r_a_load[0];
        s_a[0][a_smem_k+1][a_smem_m] = r_a_load[1];
        s_a[0][a_smem_k+2][a_smem_m] = r_a_load[2];
        s_a[0][a_smem_k+3][a_smem_m] = r_a_load[3];
        FLOAT4(s_b[0][b_smem_k][b_smem_n]) = FLOAT4(r_b_load[0]);

    }
    __syncthreads();
    

    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] ={0.0};

    //再开始for循环
    int step = (K+BK-1)/BK ;
    for(int i =1; i <step   ;i++){

        int load_bk = i;
        int s_load_index = load_bk % 2;
        int s_compute_index = (i-1)%2;

        //load
        int a_gmem_m =  by * BM + a_smem_m;
        int a_gmem_k =  load_bk * BK + a_smem_k;
        int a_gmem_addr = a_gmem_m* K + a_gmem_k;
        

        int b_gmem_k =  load_bk * BK + b_smem_k;
        int b_gmem_n =  bx * BN  + b_smem_n;
        int b_gmem_addr = b_gmem_k* N  + b_gmem_n;
        FLOAT4(r_b_load[0]) = FLOAT4( b[b_gmem_addr] ); 
        FLOAT4(r_a_load[0]) = FLOAT4( a[a_gmem_addr] ); 


        //compute
        #pragma unroll
        for(int i = 0; i< BK; i++){
            //shared到register
            FLOAT4(r_b[0]) =  FLOAT4( s_b[s_compute_index][i][tx* TN/2  ]);
            FLOAT4(r_b[4]) = FLOAT4( s_b[s_compute_index][i][tx* TN/2 + BN/2] );

            FLOAT4(r_a[0]) =  FLOAT4( s_a[s_compute_index][i][ty* TM/2  ]);
            FLOAT4(r_a[4]) = FLOAT4( s_a[s_compute_index][i][ ty * TM/2 + BM/2] );

            #pragma unroll
            for(int m = 0 ; m  < TM; m++){
                for(int n = 0 ; n< TN ;n++){
                    r_c[m][n] +=  r_a[m] * r_b[n];
                }
            }

        }

        s_a[s_load_index][a_smem_k][a_smem_m] = r_a_load[0];
        s_a[s_load_index][a_smem_k+1][a_smem_m] = r_a_load[1];
        s_a[s_load_index][a_smem_k+2][a_smem_m] = r_a_load[2];
        s_a[s_load_index][a_smem_k+3][a_smem_m] = r_a_load[3];
        FLOAT4(s_b[ s_load_index ][b_smem_k][b_smem_n]) = FLOAT4(r_b_load[0]);
        
        __syncthreads();
    }


    //然后再算最后一个
    int last_load_index = 1 ;
    #pragma unroll
    for(int i = 0; i< BK; i++){
        //shared到register
        FLOAT4(r_b[0]) =  FLOAT4( s_b[last_load_index][i][tx* TN/2  ]);
        FLOAT4(r_b[4]) = FLOAT4( s_b[last_load_index][i][tx* TN/2 + BN/2] );

        FLOAT4(r_a[0]) =  FLOAT4( s_a[last_load_index][i][ty* TM/2  ]);
        FLOAT4(r_a[4]) = FLOAT4( s_a[last_load_index][i][ ty * TM/2 + BM/2] );

        #pragma unroll
        for(int m = 0 ; m  < TM; m++){
            for(int n = 0 ; n< TN ;n++){
                r_c[m][n] +=  r_a[m] * r_b[n];
            }
        }
    }

    //然后写回数据
    #pragma unroll
    for(int i = 0; i < TM/2 ;i++){
        int c_gmem_m = blockIdx.y * BM + threadIdx.y * TM/2 + i;
        int c_gmem_n = blockIdx.x * BN + threadIdx.x * TN/2;
        int c_gmem_addr =  c_gmem_m * N + c_gmem_n;
        FLOAT4(c[c_gmem_addr] ) = FLOAT4( r_c[i][0] );
        FLOAT4(c[c_gmem_addr + BN/2 ] ) = FLOAT4( r_c[i][4] ); 
    }
    #pragma unroll
    for(int i = 0; i < TM/2 ;i++){
        int c_gmem_m = blockIdx.y * BM + threadIdx.y * TM/2 + i + BM/2 ;
        int c_gmem_n = blockIdx.x * BN + threadIdx.x * TN/2;
        int c_gmem_addr =  c_gmem_m * N + c_gmem_n;
        FLOAT4(c[c_gmem_addr] ) = FLOAT4( r_c[i + TM/2 ][0] );
        FLOAT4(c[c_gmem_addr + BN/2 ] ) = FLOAT4( r_c[ i+ TM / 2 ][4] ); 
    }    
}

//1.  为什么第一次load完不加sync，是因为有block框了一下吗？应该不是 
//2.  为什么每次for循环最后需要加sync？ 不加会有数据竞争吧，比如跑的快的线程会直接进入下个循环，但是load tile的时候跟计算的时候每个线程负责的部分不同
//3.  双buffer index的控制应该需要比原先写的更好



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






