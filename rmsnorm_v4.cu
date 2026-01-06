#include <cuda_runtime.h>
#include <stdio.h>



#define BLOCK_SIZE 512
#define DATA_PER_THREAD 2

//出入结果是一样的
//传入M*K
//传出M*K
//一个block管一行 ？ 先这么写 这是最好写的应该， block是一维的
__global__  void RMSNorm(float* g_idata, unsigned int n){

    
    //一个block管BLOCK_SIZE* DATA_PER_THREAD个数据

    int tid = threadIdx.x;
    int data_begin =  blockDim.x* blockIdx.x * DATA_PER_THREAD;
    
    int data_index_first_block =  data_begin+ tid;

    //其实有问题，先不管了, 理论上外部需要保证能整除
    if(data_index_first_block >= n){
        return;
    }


    float r_data[DATA_PER_THREAD];
    float squ_sum[DATA_PER_THREAD] = {0.0};
    //线程负责多少数据，sm就得多大，1K数据*4 最多20左右的block   ,  100K/SM  
    //一个block最多1～2K个数据差不多了，再多没有那么的sm
    //2k数据，8kBytes内存/block，   512t/block, 1536/512= 3个block，24kB,应该还是能装下 sm应该还没有问题
    //其实不用，sm能少则少，直接寄存器存了，sm只用作共享数据就行
    __shared__ float s_data[BLOCK_SIZE];

    #pragma unroll
    for(int i= 0 ;i <DATA_PER_THREAD ;i++ ){
        r_data[i] = g_idata[data_index_first_block + i* BLOCK_SIZE];
        squ_sum[i] += r_data[i]  *  r_data[i];
    }
    #pragma unroll
    for(int i= 1 ; i < DATA_PER_THREAD; i++){
        squ_sum[0] += squ_sum[i];
    }
    s_data[tid] = squ_sum[0];

    __syncthreads(); //只要碰sm就得sync



    for(int stride =  blockDim.x /2 ; stride >= 32 ; stride/=2 ){
        if(tid < stride){
            s_data[tid]+= s_data[tid+stride];
        }
        __syncthreads();
    }

    //stride == 32的情况  每个线程都做这些事
    float sum = s_data[tid];
    if(tid < 32){
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    

    //加和， 这里一个线程除一下应该比所有线程都除要好
    if(tid == 0){
        s_data[0] = sqrtf( sum / (DATA_PER_THREAD * BLOCK_SIZE) );
    }
    __syncthreads();

    #pragma unroll
    for(int i= 0 ;i < DATA_PER_THREAD; i++){
        g_idata[data_index_first_block + i * BLOCK_SIZE  ] =  r_data[i]/s_data[0];
    }

}


void initData(float *ip, int size){
    for(int i = 0; i < size; i++){
        // 使用更小的初始值，避免数值溢出
        // 归一化到 [-1, 1] 范围内
        ip[i] = (float)((i % 2 == 0) ? 0.5f : -0.5f);
    }
}




// CPU版本的RMSNorm计算函数 - 按BLOCK_SIZE分组
void RMSNormCPU(float *input, float *output, unsigned int N, const int block_size){
    int num_blocks = (N + block_size - 1) / block_size;
    
    // 按block分组处理
    for(int b = 0; b < num_blocks; b++){
        int start = b * block_size;
        int end = (b + 1) * block_size < N ? (b + 1) * block_size : N;
        int actual_size = end - start;
        
        // 第一步：计算该block内的平方和
        float sum_sq = 0.0f;
        for(int i = start; i < end; i++){
            sum_sq += input[i] * input[i];
        }
        
        // 第二步：计算该block的RMS值
        float rms = sqrtf(sum_sq / actual_size);
        
        // 第三步：对该block内的数据进行归一化
        for(int i = start; i < end; i++){
            output[i] = input[i] / rms;
        }
    }
}

// 检查RMSNorm结果 - 按BLOCK_SIZE分组
void checkRMSNormResult(float *hostRef, float *gpuRef, const int N, const int block_size){
    float epsilon = 1.0E-5;  // 允许的误差范围
    int mismatch_count = 0;
    float max_diff = 0.0f;
    int num_blocks = (N + block_size - 1) / block_size;
    
    // 按block分组检查
    for(int b = 0; b < num_blocks; b++){
        int start = b * block_size;
        int end = (b + 1) * block_size < N ? (b + 1) * block_size : N;
        
        for(int i = start; i < end; i++){
            float diff = fabs(hostRef[i] - gpuRef[i]);
            if(diff > max_diff){
                max_diff = diff;
            }
            if(diff > epsilon){
                if(mismatch_count < 10){  // 只打印前10个不匹配的结果
                    printf("Block %d, Mismatch at index %d: cpu %.6f, gpu %.6f, diff %.6e\n", 
                           b, i, hostRef[i], gpuRef[i], diff);
                }
                mismatch_count++;
            }
        }
    }
    
    printf("=== RMSNorm Verification ===\n");
    printf("Total blocks: %d (block_size: %d)\n", num_blocks, block_size);
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
    initData(h_B, n );
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
    dim3 grid(grid_size/DATA_PER_THREAD);


    // 先用CPU计算参考结果
    printf("\n=== Computing CPU Reference ===\n");
    RMSNormCPU(h_A, hostRef, n, block_size);
    printf("CPU reference computed.\n");

    RMSNorm<<<grid,block>>>(d_B , n);

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
    checkRMSNormResult(hostRef, gpuRef, n, BLOCK_SIZE * DATA_PER_THREAD);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(gpuRef);
    free(hostRef);

    return 0;
}





