#include "iostream"
#include "cuda_runtime.h"
#include <thread>
// #include ""



#define NUM_EXPERTS 128



__forceinline__  __device__ float warp_reduce_max(float val){

    #pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        val = fmax(val,  __shfl_down_sync(0xffffffff, offset, val));
    }
    return val;
};



__device__ void select_topk(float* logits, int num_tokens, int num_experts, int* out ,int topk){
    
    
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;

    const int warp_size = 32;

    float* block_start = blockIdx.y *  NUM_EXPERTS +  logits;
    float* warp_start  = block_start + wid * warp_size;

    
    //怎么比较呢？
    //fmax ,然后跑k次，应该也可以，先这么写着再说
    //对啊，怎么拿到index呢哥,
    //怎么把最大mask掉呢？
    //置成最小值，
    

    //max
    float cur_value = warp_start[lid];


    int k = 0;
    while(k < topk){
        __shared__ float sm[blockDim.x/warp_size];
        float warp_max = warp_reduce_max(cur_value);
        if(lid == 0){
            sm[lid] = warp_sum;
        }

        __syncthreads();

        float block_max =  0.f;

        if(wid == 0){
            float val = lid < blockDim.x/warp_size ?  sm[lid] : 0.f;
            block_max = warp_reduce_max(val);
        }

        __syncthreads();

        if(fabs(cur_value - block_max) < 1e-9 ){
            out[k] = lid + wid * warp_size;
            cur_value = -INFINITY;
        }

        k++;
    }

    return;


};


#define N 1024


int main(void){


    
    
    dim3 block_size(128); 
    dim3 grid_size(1, (N+block_size.x-1)/block_size.x);

    

    float* a;
    float* b;
    select_topk(a, N/128 ,128, b);


    return 0;


}