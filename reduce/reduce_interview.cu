#include <cuda_runtime.h>
#include <stdio.h>



__forceinline__  __device__ float warp_reduce_sum(float val){

    #pragma unroll
    for(int off = 16; off > 0 ; off >>= 1){
        val += shfl_down_sync(0xffffffff, val, off);
    }

    return val;
}

__device__  void block_reduce_sum(float val,  float* sm_ptr){
    int tid =  threadIdx.x ;
    int lid = tid & 31;
    int wid= tid >> 5;


    int w_v = warp_reduce_sum(val);
    if(lid == 0){
        sm_ptr[wid] = w_v;
    }

    __syncthreads();


    if(wid == 0){
        float v =   lid < blockDim.x/32 ? sm_ptr[lid] : 0.;
        float sum = warp_reduce_sum(v);
        if(lid == 0){
            sm_ptr[0] =sum;
        }
    }
    __syncthreads();
}




template <typename T>
__forceinline__ __device__ T warp_reduce_sum2(T val){
    #pragma unroll
    for(int off = 16;off > 0 ;  off>>=1){
        val += __shfl_xor_sync(0xffffffff, val, off, 32);
    }
    return val;
}




__device__ float block_reduce_sum(float val, float* s){

    int lid = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    float v = warp_reduce_sum(val);
    if(lid == 0)
        s[wid] = v;

    __syncthreads();

    int nw = blockDim.x / 32;
    float warp_v  = threadIdx.x < nw ? s[threadIdx.x] : 0.0f;

    if(wid == 0){
        float sum_v = warp_reduce_sum(warp_v);
        if(threadIdx.x == 0 )
            s[0] = sum_v;
    }

    __syncthreads();
    return s[0];
}



//rmsnorm

//一个block处理D个数据
__device__ void rmsnorm(float* in, float* out , int D, float eps, float gamma){

    int tid = threadIdx.x;

    float* block_start  = blockIdx.x  * D  + in;


    float ss =0 ;
    for(int i = tid; i <  D ;i+= blockDim.x){
        ss += start[i] * start[i];
    }

    __shared__ sm[32];
    float block_sum = block_reduce_sum(ss, sm);
    float a = rsqrt(block_sum);


    float* out_start = blockIdx.x  * D  + out;
    for(int i = tid; i <  D ;i+= blockDim.x){
        out_start[i] = start[i]* a *   gamma;
    }

}



//或者可以测试一下一个warp做很多数据和 一个block处理这些数据的对比





__forceinline__  __device__ float warp_reduce_max(float val){

    #pragma unroll
    for(int off = 16; off > 0 ; off >>= 1){
        val =  fmaxf(val,shfl_down_sync(0xffffffff, val, off));
    }
    return val;
}

__device__ float block_reduce_max(float val, float* sm){

    int lid = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    float warp_max = warp_reduce_max(val);
    if(lid == 0){
        sm[wid] = warp_max;
    }

    __syncthreads();

    if(wid == 0){
        float v = tid < blockDim.x/32 ? sm[tid] :  0.0;
        float block_max = warp_reduce_max(v);
        if(lid == 0){
            s[0] = block_max;
        }
    }

    __syncthreads();
    return s[0];


}



//softmax
//一个block处理一个样本
__device__ void softmax(float* in, float* out, int D){

    //find max
    int tid = threadIdx.x;
    float* block_start = in + blockIdx.x * D;

    float max_in_group = -FLT_MAX;
    //D是变量所以应该展不开吧
    for(int i = tid; i < D; i+= blockDim.x){
        max_in_group = fmaxf(max_in_group, block_start[i]);
    }
    __shared__ sm[32];
    float block_max= block_reduce_max(max_in_group, sm);


    //get sum;
    float sum_group =0 ;
    for(int i = tid; i < D; i+= blockDim.x)
        sum_group+= expf(block_start[i]- block_max);
    float block_sum = block_reduce_sum(sum_group,sm);


    float* out_start= out + blockIdx.x * D;
    for(int i = tid; i < D; i+= blockDim.x){
        out[tid] =  expf(block_start[i] - block_max)/block_sum;
    }
}



__device__ void online_update(float& m ,float& s, float& x){
    float m_new = fmaxf(m,x);
    s = s*expf(m-m_new) + expf(x-m_new);
    m = m_new;
}

__device__ void combine(float& m1, float& s1, float& m2, float& s2){
    float m_new = fmaxf(m1,m2);
    s1 = s1*expf(m1-m_new) + s2*expf(m2-m_new);
    m1 = m_new;
}


