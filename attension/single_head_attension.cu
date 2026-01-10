#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(
    void (*gpuSgemm) (float *, float *, float *, float*,  const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N );
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, float*, const int, const int ),
    dim3 gridDim, dim3 blockDim, const int M, const int N,  const int repeat);

void cpuAttension(
    float *a, float *b, float *c, float* output, const int M, const int N) {
    // a = Q (M x N), b = K (N x M, 转置存储), c = V (M x N)
    // output = Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
    
    float sqrt_d = sqrtf((float)N);
    
    // 1. 计算 QK^T，结果 scores 是 M x M
    // 注意：b (K) 是 N x M 转置存储的，所以 b[k * M + j] 对应 K^T[k][j] = K[j][k]
    // QK^T[i][j] = sum_k(Q[i][k] * K^T[k][j]) = sum_k(a[i * N + k] * b[k * M + j])
    
    float* scores = (float*)malloc(M * M * sizeof(float));
    
    // 计算 QK^T
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            float sum = 0.0f;
            for(int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * M + j];  // Q[i][k] * K^T[k][j]
            }
            scores[i * M + j] = sum / sqrt_d;  // 除以 sqrt(d)
        }
    }
    
    // 2. 计算 softmax(scores) - 对每一行应用 softmax
    for(int i = 0; i < M; i++) {
        // 先找最大值（数值稳定性）
        float max_val = scores[i * M + 0];
        for(int j = 1; j < M; j++) {
            if(scores[i * M + j] > max_val) {
                max_val = scores[i * M + j];
            }
        }
        
        // 计算 exp 和 sum
        float sum_exp = 0.0f;
        for(int j = 0; j < M; j++) {
            scores[i * M + j] = expf(scores[i * M + j] - max_val);
            sum_exp += scores[i * M + j];
        }
        
        // 归一化
        for(int j = 0; j < M; j++) {
            scores[i * M + j] /= sum_exp;
        }
    }
    
    // 3. 计算 softmax(scores) * V，结果 output 是 M x N
    // output[i][k] = sum_j(softmax[i][j] * V[j][k])
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < N; k++) {
            float sum = 0.0f;
            for(int j = 0; j < M; j++) {
                sum += scores[i * M + j] * c[j * N + k];  // softmax[i][j] * V[j][k]
            }
            output[i * N + k] = sum;
        }
    }
    
    free(scores);
}


//先写float
//一个block处理q的一部分

//同一时间：q的1/N，for(K){K的1/N，V的的1/N,  各种计算}
// 
//  K
//512线程load 2k个数据，
//K = 1024 BM =2; BK = 2;
__global__ void SingleHeadAttensionKernel(
    float * __restrict__ q, float * __restrict__ k, float * __restrict__ v, float* out_put,
    const int seq_len, const int d) {


    const int BM = 2;
    const int BK = 2;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ float s_q[BM][512];  // 最大特征维度512
    __shared__ float s_k[BK][512];
    __shared__ float s_v[BK][512];
    __shared__ float r_t[BM][BK];

    // 检查参数有效性
    if(d > 512 || seq_len <= 0 || d <= 0){
        return;  // 如果d超过512，直接返回（需要修改共享内存大小）
    }
    
    int data_per_thread = (d + blockDim.x - 1) / blockDim.x;  // 向上取整
    int load_q_sm_m = ty;
    int load_q_sm_k = data_per_thread * tx;
    
    int load_q_gm_m = blockDim.y * blockIdx.y + load_q_sm_m;
    int load_q_g_addr = load_q_gm_m * d + load_q_sm_k;
    
    // load partial q - 并行加载
    if(load_q_gm_m < seq_len && ty < BM){
        for(int j = 0; j < data_per_thread && load_q_sm_k + j < d && load_q_sm_k + j < 512; j++){
            s_q[load_q_sm_m][load_q_sm_k + j] = q[load_q_g_addr + j];
        }
    }
    __syncthreads();

    __shared__ float s_out[BM][512]; 
    __shared__ float new_block_out[BM][512]; 
    __shared__ float block_denom[BM];
    __shared__ float block_max[BM];
    __shared__ float new_max[BM];
    __shared__ float acc[BM][512];
    
    // 初始化 - 需要处理 d > 512 的情况
    for(int idx = tx; idx < 512 && idx < d; idx += blockDim.x){
        if(ty < BM){
            s_out[ty][idx] = 0.0f;
            acc[ty][idx] = 0.0f;
        }
    }
    if(ty < BM && tx == 0){
        block_denom[ty] = 0.0f;
        block_max[ty] = -1e30f;
    }
    __syncthreads();
    
    int step = (seq_len + BK - 1) / BK;
    for(int step_idx = 0; step_idx < step; step_idx++){
        // 1. load partial k (K是转置存储的，所以是 d x seq_len)
        int load_k_gm_n = step_idx * BK;  // seq_len维度
        for(int k_idx = 0; k_idx < BK && load_k_gm_n + k_idx < seq_len; k_idx++){
            int load_k_sm_n = k_idx;
            for(int d_idx = tx; d_idx < d && d_idx < 512; d_idx += blockDim.x){  // 限制在512范围内
                if(ty == 0 && load_k_sm_n < BK){
                    // k是转置存储：k[d_idx * seq_len + load_k_gm_n + k_idx]
                    s_k[load_k_sm_n][d_idx] = k[d_idx * seq_len + load_k_gm_n + k_idx];
                }
            }
        }
        __syncthreads();


        __shared__ float exp_score[BM][BK];
        __shared__ float exp_score_sum[BM][BK];
        __shared__ float tmp[BM][BK];
        
        // 2. compute QK^T: r_t[BM][BK]
        // 初始化r_t
        if(ty < BM && tx < BK){
            r_t[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        // 计算矩阵乘法：Q[i][k] * K^T[k][j] = Q[i][k] * K[j][k]
        // Q是当前block的行，K是当前step的行
        if(ty < BM && tx < BK && step_idx * BK + tx < seq_len){
            float sum = 0.0f;
            // 确保k不超过d和512
            int max_k = (d < 512) ? d : 512;
            for(int k = 0; k < max_k && k < d; k++){
                // s_q[ty][k] 是 Q[current_row + ty][k]
                // s_k[tx][k] 是 K[step_idx*BK + tx][k] (转置存储)
                if(k < 512){  // 额外的边界检查
                    sum += s_q[ty][k] * s_k[tx][k];
                }
            }
            r_t[ty][tx] = (d > 0) ? sum / sqrtf((float)d) : 0.0f;  // 避免除零
            tmp[ty][tx] = r_t[ty][tx];
        }
        __syncthreads();

        // 求最大值
        if(ty < BM && tx < BK && step_idx * BK + tx < seq_len){
            for(int stride = BK / 2; stride > 0; stride /= 2){
                if(tx < stride && step_idx * BK + tx + stride < seq_len){
                    tmp[ty][tx] = fmaxf(tmp[ty][tx + stride], tmp[ty][tx]);
                }
                __syncthreads();
            }
        }
        __syncthreads();

        // 得出最大值
        if(ty < BM && tx == 0 && step_idx * BK < seq_len){
            new_max[ty] = fmaxf(block_max[ty], tmp[ty][0]);
        }
        __syncthreads();

        // 根据最大值求exp分数, 计算exp_score_sum
        if(ty < BM && tx < BK && step_idx * BK + tx < seq_len){
            exp_score[ty][tx] = expf(r_t[ty][tx] - new_max[ty]);
            exp_score_sum[ty][tx] = exp_score[ty][tx];
        }
        __syncthreads();
        
        if(ty < BM && tx < BK && step_idx * BK + tx < seq_len){
            for(int stride = BK / 2; stride > 0; stride /= 2){
                if(tx < stride && step_idx * BK + tx + stride < seq_len){
                    exp_score_sum[ty][tx] += exp_score_sum[ty][tx + stride];
                }
                __syncthreads();
            }
        }
        __syncthreads();
        
        // block_denom更新
        if(ty < BM && tx == 0 && step_idx * BK < seq_len){
            block_denom[ty] = block_denom[ty] * expf(block_max[ty] - new_max[ty]) + exp_score_sum[ty][0];
        }
        __syncthreads();
        
        // 3. load partial v
        int load_v_gm_m = step_idx * BK;  // seq_len维度
        for(int v_idx = 0; v_idx < BK && load_v_gm_m + v_idx < seq_len; v_idx++){
            int load_v_sm_m = v_idx;
            for(int d_idx = tx; d_idx < d && d_idx < 512; d_idx += blockDim.x){  // 限制在512范围内
                if(ty == 0 && load_v_sm_m < BK){
                    // v是 M x N: v[load_v_gm_m + v_idx][d_idx]
                    s_v[load_v_sm_m][d_idx] = v[(load_v_gm_m + v_idx) * d + d_idx];
                }
            }
        }
        __syncthreads();

        // 4. compute exp_score @ V: acc[BM][d]
        if(tx < d && tx < 512 && ty < BM){  // 添加边界检查
            float sum = 0.0f;
            for(int j = 0; j < BK && step_idx * BK + j < seq_len && j < BK; j++){
                if(j < BK && tx < 512){  // 额外边界检查
                    sum += exp_score[ty][j] * s_v[j][tx];
                }
            }
            acc[ty][tx] += sum;
            new_block_out[ty][tx] = s_out[ty][tx] * expf(block_max[ty] - new_max[ty]) + acc[ty][tx];
        }
        __syncthreads();

        // 5. update: softmax reduce
        if(tx < d && tx < 512 && ty < BM){  // 添加边界检查
            if(tx == 0){  // 只在第一个线程更新block_max
                block_max[ty] = new_max[ty];
            }
            s_out[ty][tx] = new_block_out[ty][tx];
        }
        __syncthreads();
    } 


    // 最后归一化并写回全局内存
    if(tx < d && tx < 512 && ty < BM){  // 添加边界检查
        int output_row = blockDim.y * blockIdx.y + ty;
        if(output_row < seq_len && block_denom[ty] > 1e-10f){  // 避免除零，使用更小的阈值
            float normalized = s_out[ty][tx] / block_denom[ty];
            out_put[output_row * d + tx] = normalized;
        } else if(output_row < seq_len){
            out_put[output_row * d + tx] = 0.0f;  // 如果denom为0，输出0
        }
    }
}


int main(void) {
    printf("\nKernal = SingleHeadAttensionKernel\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM =  2 , BN = 512 , TM = 1, TN =  2;          //512线程load 2k个数据，2*128, 2*2的结果矩阵
    void (*gpuSgemm) (float *, float *, float *, float* , const int, const int) = SingleHeadAttensionKernel;

    {
        const int M = 512, N = 512;
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim( 1, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    
    const int TESTNUM = 15;
    for (int i = 0; i < 1; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim(1, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, inner_repeat);
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
    void (*gpuSgemm) (float *, float *, float *, float* ,  const int, const int ),
    dim3 gridDim, dim3 blockDim, const int M, const int N) {

    size_t size_q = M * N * sizeof(float);
    size_t size_k = N * M * sizeof(float);
    size_t size_v = M * N * sizeof(float);
    size_t size_output = M * N * sizeof(float);

    float* h_q, * h_k, * h_v, * h_output, * d_q, * d_k, * d_v, * d_output;
    
    h_q = (float *)malloc(size_q);
    h_k = (float *)malloc(size_k);
    h_v = (float *)malloc(size_v);
    h_output = (float *)malloc(size_output);

    
    cudaMalloc(&d_q, size_q);
    cudaMalloc(&d_k, size_k);
    cudaMalloc(&d_v, size_v);
    cudaMalloc(&d_output, size_output);
    float* output_ref = (float *)malloc(size_output);

    srand(time(0));
    for (int i = 0; i < M * N; i++)
        h_q[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < N * M; i++)
        h_k[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < M * N; i++)
        h_v[i] = rand() / float(RAND_MAX);
    cudaMemset(d_output, 0, size_output);

    cpuAttension(h_q, h_k, h_v, h_output, M, N);

    cudaMemcpy(d_q, h_q, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, size_k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, size_v, cudaMemcpyHostToDevice);
    SingleHeadAttensionKernel<<<gridDim, blockDim>>>(d_q, d_k, d_v, d_output, M, N);
    cudaMemcpy(output_ref, d_output, size_output, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = fabs(h_output[i] - output_ref[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_q);
    free(h_k);
    free(h_v);
    free(h_output);
    free(output_ref);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);

    return max_error;
}


float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, float* ,  const int, const int ),
    dim3 gridDim, dim3 blockDim, const int M, const int N,  const int repeat) {

    size_t size_q = M * N * sizeof(float);
    size_t size_k = N * M * sizeof(float);
    size_t size_v = M * N * sizeof(float);
    size_t size_output = M * N * sizeof(float);

    float *d_q, *d_k, *d_v, *d_output;
    cudaMalloc(&d_q, size_q);
    cudaMalloc(&d_k, size_k);
    cudaMalloc(&d_v, size_v);
    cudaMalloc(&d_output, size_output);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        SingleHeadAttensionKernel<<<gridDim, blockDim>>>(d_q, d_k, d_v, d_output, M, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);

    return sec;
}

