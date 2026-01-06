#include <cuda_runtime.h>
#include <stdio.h>



//把后一个block的数据先挪到前一个block，一个线程block管两个数据block的意思

__global__  void reduce(int* g_idata, int* g_odata, unsigned int n){

    int tid =  threadIdx.x;
    int idx = blockDim.x * (blockIdx.x* 8)+ tid;
    int* idata= g_idata+ blockDim.x * (blockIdx.x* 8);

        
    //挪数据
    #pragma unroll
    for(int iter = 1 ;iter < 8 ; iter++ ){
        idata[tid] += idata[tid + iter* blockDim.x];
    }

    __syncthreads();

    for(int stride = blockDim.x /2; stride > 32 ; stride/=2 ){
        if(tid < stride){
            idata[tid]+= idata[tid+stride];
        }
        __syncthreads();
    }

    //为啥是这个来着？
    if(tid < 32){
        volatile int* vmem = idata;
        vmem[tid]+=vmem[tid+32];
        vmem[tid]+=vmem[tid+16];
        vmem[tid]+=vmem[tid+8];
        vmem[tid]+=vmem[tid+4];
        vmem[tid]+=vmem[tid+2];
        vmem[tid]+=vmem[tid+1];
    }
        
    if(tid == 0){
        g_odata[blockIdx.x]= idata[0];
    }

}


void initData(int *ip, int size){
    for(int i = 1; i <= size; i++){
        ip[i-1] = (int)(i);
    }
}

void sumArraysOnHost(int *A, int *B, int *C, const int N){
    for(int idx = 0; idx < N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void sumArraysOnDevice(int *A, int *B, int *C){
    int i = threadIdx.x;
    C[i] = A[i]+B[i];
}

void checkResult(int *hostRef, int *gpuRef, const int N){
    double epsilon = 1.0E-8;
    for(int i = 0; i < N; i++){
        if(fabs(hostRef[i] - gpuRef[i]) > epsilon){
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            return;
        }
    }
    printf("Arrays match.\n");
    return;
}

int getSumCpuRes(int* input , int n ){
    int res = 0;
    for(int i= 0; i< n ;i++){
        res+= input[i];
    }
    return res;
}
int main(){

    printf("Starting...\n");
    int dev =0;
    cudaSetDevice(dev);

    int block_size = 512;
    int n = 2 << 24;

    int total = ((n+ block_size -1)/block_size) * block_size;
    printf(" total is %d \n" ,total);
    size_t nBytes = total * sizeof(int);
    
    int *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int*)malloc(nBytes);
    h_B = (int*)malloc(nBytes);
    hostRef = (int*)malloc(nBytes);
    gpuRef = (int*)malloc(nBytes);
    memset(h_A, 0, nBytes);
    memset(h_B, 0, nBytes);
    initData(h_A, n );
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    int *d_A, *d_B, *d_C;
    //这里为啥是指针指针来着？？？ 
    cudaMalloc((int**)&d_A, nBytes);
    cudaMalloc((int**)&d_B, nBytes);
    cudaMalloc((int**)&d_C, nBytes);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);


    dim3 block(block_size);
    int grid_size =  (n+block.x-1)/block.x;
    dim3 grid(grid_size/8);

    //warmup
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    reduce<<<grid,block>>>(d_A, d_B, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0;
    
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
    printf(" time  %f s \n", sec );

    cudaMemcpy(gpuRef, d_B, nBytes, cudaMemcpyDeviceToHost);
    int gpu_res  = 0 ;
    for(int i= 0; i < grid_size/8; i++){
        //printf("  gpu res %d \n", gpuRef[i] );
        gpu_res+= gpuRef[i];
    }
    printf(" gpu result %d \n", gpu_res );
    printf(" cpu result %d \n",  getSumCpuRes(h_A, n));


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(gpuRef);
    free(hostRef);

    return 0;
}





