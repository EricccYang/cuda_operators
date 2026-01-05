#include <cuda_runtime.h>
#include <stdio.h>




#define BLOCK_SIZE 512

//出入结果是一样的
//传入M*K
//传出M*K
//一个block管一行 ？ 先这么写 这是最好写的应该， block是一维的
__global__  void RMSNorm(float* g_idata, unsigned int n){

    
    //一个block管一行，一个线程管一个数 版本
    //512 数 BLOCK_SIZE = 512， 512线程 16warp版本  
    //先 乘方， 再规约
    //寄存器存什么？shared mem存什么？
    //寄存器存 单个值，shared做规约？ 
    int tid = threadIdx.x;
    int data_index =  blockDim.x* blockIdx.x +tid;

    if(data_index >= n){
        return;
    }

    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = g_idata[data_index];
    float cur = s_data[tid];
    s_data[tid] *= cur;

    __syncthreads();
    
    //然后开始规约
    for(int stride = blockDim.x /2; stride > 0 ; stride/=2 ){
        if(tid < stride){
            s_data[tid]+= s_data[tid+stride];
        }
        __syncthreads();
    }

    //和
    if(tid == 0){
        s_data[0] = sqrtf(s_data[0] / blockDim.x);
    }
    __syncthreads();

    g_idata[data_index]  = cur / s_data[0];

}


void initData(float *ip, int size){
    for(int i = 0; i < size; i++){
        // 使用更小的初始值，避免数值溢出
        // 归一化到 [-1, 1] 范围内
        ip[i] = (float)((i % 2 == 0) ? 0.5f : -0.5f);
    }
}


// CPU版本的RMSNorm计算函数 - 验证GPU结果
void RMSNormCPU(float *input, float *output, unsigned int N){
    // 第一步：计算平方和
    float sum_sq = 0.0f;
    for(int i = 0; i < N; i++){
        sum_sq += input[i] * input[i];
    }
    
    // 第二步：计算RMS值
    float rms = sqrtf(sum_sq / N);
    
    // 第三步：归一化
    for(int i = 0; i < N; i++){
        output[i] = input[i] / rms;
    }
}

// 检查RMSNorm结果
void checkRMSNormResult(float *hostRef, float *gpuRef, const int N){
    float epsilon = 1.0E-5;  // 允许的误差范围
    int mismatch_count = 0;
    float max_diff = 0.0f;
    
    for(int i = 0; i < N; i++){
        float diff = fabs(hostRef[i] - gpuRef[i]);
        if(diff > max_diff){
            max_diff = diff;
        }
        if(diff > epsilon){
            if(mismatch_count < 10){  // 只打印前10个不匹配的结果
                printf("Mismatch at index %d: cpu %.6f, gpu %.6f, diff %.6e\n", 
                       i, hostRef[i], gpuRef[i], diff);
            }
            mismatch_count++;
        }
    }
    
    printf("=== RMSNorm Verification ===\n");
    printf("Max difference: %.6e\n", max_diff);
    if(mismatch_count == 0){
        printf("✓ All values match!\n");
    } else {
        printf("✗ Total %d mismatches out of %d values (%.2f%%)\n", 
               mismatch_count, N, (float)mismatch_count/N*100);
    }
}




int main(){

    printf("Starting...\n");
    int dev =0;
    cudaSetDevice(dev);

    int block_size = BLOCK_SIZE;
    int n = 2 << 24;

    int total = ((n+ block_size -1)/block_size) * block_size;
    printf(" total is %d \n" ,total);
    size_t nBytes = total * sizeof(float);
    
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);
    memset(h_A, 0, nBytes);
    memset(h_B, 0, nBytes);
    initData(h_A, n );
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    //这里为啥是指针指针来着？？？ 
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);


    dim3 block(block_size);
    int grid_size =  (n+block.x-1)/block.x;
    dim3 grid(grid_size);


    // 先用CPU计算参考结果
    printf("\n=== Computing CPU Reference ===\n");
    RMSNormCPU(h_A, hostRef, n);
    printf("CPU reference computed.\n");

    // 运行GPU核函数
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    RMSNorm<<<grid,block>>>(d_A , n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0;
    
    printf("\n=== GPU Execution ===\n");
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
    printf("Execution time: %f ms (%f s)\n", msec, sec);

    // 将GPU结果拷回主机
    cudaMemcpy(gpuRef, d_A, nBytes, cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("\n");
    checkRMSNormResult(hostRef, gpuRef, n);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(gpuRef);
    free(hostRef);

    return 0;
}





