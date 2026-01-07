#include <cuda_runtime.h>
#include <stdio.h>



#define BLOCK_SIZE 512
#define DATA_PER_THREAD 4
#define WARP_SIZE 32


//512个数一个emb， 一个线程处理4个数，4个warp一个emb，先这么写


__global__  void RMSNorm(float* g_idata, unsigned int n, unsigned int d){
    const int warp_size = 32;
    int tx= threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    int threads_count = blockDim.x * blockDim.y;
    int datas_per_block = threads_count * DATA_PER_THREAD;
    float* block_begin_data_ptr = g_idata +  blockIdx.y * datas_per_block;
    float* warp_begin_data_ptr = block_begin_data_ptr + ty * warp_size* DATA_PER_THREAD;

    
    float sum = 0.0f;

    #pragma unroll
    for(int i =0 ;i < DATA_PER_THREAD; i++){
        sum += warp_begin_data_ptr[ tx + i * warp_size] * warp_begin_data_ptr[ tx + i * warp_size];
    }
    

    //一个warp把管理的 warp_size* DATA_PER_THREAD 的数据加完，分散在warp_size个线程里
    //然后shuf
    sum+=__shfl_down_sync(0xffffffff,sum,16);
    sum+=__shfl_down_sync(0xffffffff,sum,8);
    sum+=__shfl_down_sync(0xffffffff,sum,4);
    sum+=__shfl_down_sync(0xffffffff,sum,2);
    sum+=__shfl_down_sync(0xffffffff,sum,1);
    

    __shared__ float sm[16];
    //然后每个warp存一个sm的位置
    //用每个warp的第一个线程 目前是
    if(tx == 0){
        sm[ty] = sum;
    }
    __syncthreads();

    //用一个warp规约sm - 只让每个warp的第一个线程参与
    for(int stride = blockDim.y/2 ; stride > 0 ; stride/=2){
        if(tx == 0 && ty < stride){
            sm[ty] += sm[ty + stride];
        }
        __syncthreads();
    }
    

    float reduced =  sqrtf( sm[0] / (blockDim.x * DATA_PER_THREAD*  blockDim.y));
    

    //把自己负责的数字读出来再除以这个值再放回去
    #pragma unroll
    for(int i =0 ;i < DATA_PER_THREAD; i++){
        warp_begin_data_ptr[tx + i * warp_size] = warp_begin_data_ptr[tx + i * warp_size]/reduced;
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


    dim3 block( WARP_SIZE , block_size/(WARP_SIZE * DATA_PER_THREAD)  );
    int grid_size = n/ block_size;
    dim3 grid(1, grid_size);


    // 先用CPU计算参考结果
    printf("\n=== Computing CPU Reference ===\n");
    RMSNormCPU(h_A, hostRef, n, block_size);
    printf("CPU reference computed.\n");

    RMSNorm<<<grid,block>>>(d_B , n, block_size);

    // 运行GPU核函数
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    RMSNorm<<<grid,block>>>(d_A , n, block_size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0;
    
    printf("\n=== GPU Execution ===\n");
    printf("Execution configuration grid<%d, %d>\n", grid.x, grid.y);
    printf("Execution configuration block<%d, %d>\n", block.x, block.y);
    printf("Execution time: %f ms (%f s)\n", msec, sec);

    // 将GPU结果拷回主机
    cudaMemcpy(gpuRef, d_A, nBytes, cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("\n");
    checkRMSNormResult(hostRef, gpuRef, n, BLOCK_SIZE);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(gpuRef);
    free(hostRef);

    return 0;
}





