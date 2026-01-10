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


//先写float
//一个block处理q的一部分

//同一时间：q的1/N，for(K){K的1/N，V的的1/N,  各种计算}
// 
//  K
//512线程load 2k个数据，
//K = 1024 BM =2; BK = 2;
__global__ void SingleHeadAttensionKernel(
    float * __restrict__ q, float * __restrict__ k, float * __restrict__ v,
    const int seq_len, const int d, const int M, const int K, const int BM, const int BK) {
    

    int tx= threadIdx.x;
    int ty = threadIdx.y;
    //最终结果的矩阵维度
    int gx=  tx + blockDim.x * blockIdx.x;
    int gy = ty + blockDim.y * blockIdx.y;


    __shared__ float s_q[BM][K];
    __shared__ float s_k[BM][K];
    __shared__ float s_v[BM][K];
    __shared__ float r_t[BM][BM];

     int tid = tx+ ty* K;
    //load partial q
    {
        int data_per_thread =  K/blockDim.x;
        int load_q_sm_m =  ty;
        int load_q_sm_k = data_per_thread * tx;
        
        int load_q_gm_m = blockDim.y * blockIdx.y + load_q_sm_m;
        int load_q_gm_k =  blockDim.x * blockIdx.x + load_q_sm_k;
        int load_q_g_addr = load_q_gm_m * K + load_q_gm_k;
        
        //bank conflict
        //todo vector load
        for(int i =0 ;i < data_per_thread; i++){
            s_q[load_q_sm_m][load_q_sm_k+i] = q[load_q_g_addr+i];
        }
    }
    __syncthreads();


    __shared__ float s_out[BM][K]; 
    __shared__ float new_block_out[BM][K]; 
    __shared__ float block_denom[BM];
    __shared__ float block_max[BM];
    __shared__ float new_max[BM];

    __shared__ float acc[BM][K];
    
    int step = (K +BK- 1)/ BK;
    for( int i =0 ; i < step; i++){
        //1. load partial k
        {
            int load_k_sm_m =  tid * (data_per_thread/BK) ;  
            int load_k_sm_n =  0;
            
            int load_k_gm_m =  (blockDim.y * blockIdx.y + load_k_sm_m ) * (data_per_thread/BK);
            int load_k_gm_n =  blockDim.x * blockIdx.x + i* BK + load_k_sm_n;

            //一个线程负责四个数据
            int load_k_gm_addr = load_k_gm_m * K + load_k_gm_n;
            int load_k_gm_addr_next_line = ( load_k_gm_m + 1 ) * K +  load_j_gm_n;
            
            //一个线程load4个数据
            s_k[load_k_sm_n][load_k_sm_m] = k[load_k_gm_addr];
            s_k[load_k_sm_n+1][load_k_sm_m] = k[load_k_gm_addr+1];
            s_k[load_k_sm_n][load_k_sm_m+1] = k[load_k_gm_addr_next_line];  
            s_k[load_k_sm_n+1][load_k_sm_m+1] = k[load_k_gm_addr_next_line+1];       
        }
        __syncthreads();


        __shared__ float exp_score[BM][BK] = {0.0};
        __shared__ float exp_score_sum[BM][BK];
        __shared__ float tmp[BM][BK] = {0.0};
        //2. compute  : result r_t[BM][BM];
        //这个算法得是原子的
        {
            for(int i= 0 ; i < data_per_thread;i++ ){
                atomicAdd(r_t[load_q_sm_m][load_k_sm_n],  s_q[load_q_sm_m][load_q_sm_n] * s_k[load_k_sm_n][load_k_sm_m]);
            }

            if(ty < BM && tx < BK){
                r_t[ty][tx]/= sqrtf(d); //其实就是blockDim.x
                tmp[ty][tx]=  r_t[ty][tx];
            }
            

            //用作求最大值
            for(int stride = BK/2;stride > 0; stride/=2){
                if(tx < stride){
                    tmp[ty][tx] = max( tmp[ty][tx+stride], tmp[ty][tx]);
                }
            }

            //得出最大值
            if(tx == 0){
                new_max[ty] = max(block_max[ty], tmp[ty][0]);            
            }

            //根据最大值求exp分数, 计算expscore sum
            exp_score[ty][tx] = exp(r_t[ty][tx]  -  new_max[ty]);
            exp_score_sum[ty][tx] = exp_score[ty][tx];
            for(int stride = BK/2;stride > 0; stride/=2){
                if(tx < stride){
                    exp_score_sum[ty][tx] += exp_score_sum[ty][tx+stride];
                }
            }
            
            //block_denom
            if(tx == 0){
                block_denom[ty] =  block_denom[ty] * exp(block_max[ty] - new_max[ty]) +  exp_score_sum[ty][tx];
            }
            
        }
        
        
        //3. load partial v
        {
            int load_v_sm_m =  i * BM;        //这里和q矩阵是不一样的
            int load_v_sm_k = data_per_thread * tx;
            
            int load_v_gm_m = blockDim.y * blockIdx.y + load_v_sm_m;
            int load_v_gm_k =  blockDim.x * blockIdx.x + load_v_sm_k;
            int load_v_g_addr = load_v_gm_m * K + load_v_gm_k;
            
            //bank conflict
            //todo vector load
            for(int i =0 ;i < data_per_thread; i++){
                s_v[load_v_sm_m][load_v_sm_k+i] = v[load_v_g_addr+i];
            }
        }
        __syncthreads();
        

        //4. compute exp_score @ v[BM][K]
        if(tx < K && ty < BK)
        {
            s_out[ty][K] = 0.0;
            #pragma unroll
            for(int i = 0 ; i < BK; i++){
                acc[ty][tx] +=  exp_score[ty+i] * s_v[ty+i][tx];
            }
            new_block_out[ty][tx] = s_out[ty][tx]  * exp( block_max[ty] - new_max[ty])  + acc[ty][tx];
        }
        __syncthreads():

        
        //5. update: softmax reduce
        //针对  s_out做 更新，那段逻辑
        {
           block_max[ty] = new_max[ty];
           s_out[ty][tx] = new_block_out[ty][tx];
        }
    } 


    //然后最后这是算出来部分行的结果数据
    //deno有用了
    //一行的最后除以deno，等到最终输出的一部分
    s_out[ty][tx] = s_out[ty][tx]/ block_denom[ty];
}


int main(void) {
    printf("\nKernal = sgemm_V2\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM =  2 , BN = 1024 , TM = 1, TN =  4;          //512线程load 2k个数据，2*128, 2*2的结果矩阵
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = sgemm_V2;

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

