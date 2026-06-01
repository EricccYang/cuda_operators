#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <iterator>
#include <utility>



#define NUM_EXPERTS 128
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


__forceinline__  __device__ float warp_reduce_max(float val){
    #pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        val = fmaxf(val,  __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
};



namespace my_util{

    template <typename T>
    __forceinline__ __device__ void swap(T* a, T* b){
        T temp = *a;
        *a = *b;
        *b = temp;
    }
    
}


//sort,  max value first
template <int num>
__forceinline__ __device__ void sort_elements(float* value){
    static_assert(num == 1 || num == 2 || num == 4, "num must be 1, 2 or 4");
    return;
}


template <>
__forceinline__ __device__ void sort_elements<1>(float* value){
    return;
}


template <>
__forceinline__ __device__ void sort_elements<2>(float* value){
    float a = value[0];
    float b = value[1];
    if(a < b){
        my_util::swap(&value[0],&value[1]);
    }
    return;
}

template <>
__forceinline__ __device__ void sort_elements<4>(float* value){

    sort_elements<2>(value);
    sort_elements<2>(value + 2);
    if(value[0] < value[2]){
        my_util::swap(&value[0], &value[2]);
    }
    if(value[1] < value[3]){
        my_util::swap(&value[1], &value[3]);
    }
    if(value[1] < value[2]){
        my_util::swap(&value[1], &value[2]);
    }
}



template <int num>
__forceinline__ __device__ void sort_elements_and_index(float* value, int* index){
    static_assert(num == 1 || num == 2 || num == 4, "num must be 1, 2 or 4");
    return;
}


template <>
__forceinline__ __device__ void sort_elements_and_index<1>(float* value, int* index){
    return;
}

template <>
__forceinline__ __device__ void sort_elements_and_index<2>(float* value, int* index){
    if(value[0] < value[1]){
        my_util::swap(&value[0], &value[1]);
        my_util::swap(&index[0], &index[1]);
    }
    return;
}

template <>
__forceinline__ __device__ void sort_elements_and_index<4>(float* value, int* index){
    sort_elements_and_index<2>(value, index);
    sort_elements_and_index<2>(value + 2, index + 2);
    if(value[0] < value[2]){
        my_util::swap(&value[0], &value[2]);
        my_util::swap(&index[0], &index[2]);
    }
    if(value[1] < value[3]){
        my_util::swap(&value[1], &value[3]);
        my_util::swap(&index[1], &index[3]);
    }
    if(value[1] < value[2]){
        my_util::swap(&value[1], &value[2]);
        my_util::swap(&index[1], &index[2]);
    }
}

//warp per token
//先排序自己的一些数据
__global__ void select_topk(float* logits, int num_tokens, int num_experts, int* out ,int topk){
    
    
    int tid = threadIdx.x;
    int lid = tid & 31;
    int wid = tid >> 5;
    int block_index = blockIdx.y;

    int warp_per_block = 4;
    int warp_index =  wid + block_index * warp_per_block;
    constexpr int items_per_thread = 4;

    float* block_start =  logits + block_index * num_experts * warp_per_block;
    float* warp_start = block_start + wid * num_experts;
    

    //load
    float r_num[4];
    int r_index[4] = {0, 1, 2, 3};
    FLOAT4(r_num[0]) = FLOAT4(warp_start[lid*items_per_thread]);

    //sort step 1
    //single function
    sort_elements_and_index<items_per_thread>(r_num, r_index);
    

    //sort step 2
    //how to decide bigger topk between threads
    //还是可以warp_reduce_max来做，
    //这个还真是不好用fabsf来算，只能把数字返回回来。就是调换的时候换回来，一个新的数组吧，
    int k = 0;
    int cur_index =  0;
    while(k < topk){

        if(cur_index == items_per_thread){
            continue;;
        }
        float value  = r_num[cur_index];
        float warp_max = warp_reduce_max(value);
        printf(" warp_max: %f, value: %f, index: %d \n", warp_max, value, r_index[cur_index]+ lid * items_per_thread);
        
        //per thread to execute
        if(fabsf(value - warp_max) < 1e-9){
            out[warp_index * topk + k] = r_index[cur_index]+ lid * items_per_thread;
        }
        k++;
    }
    

    return;
    
    
};


#define TOPK 4
// logits layout: [num_tokens, num_experts]
void init_logits(float* logits, int num_tokens, int num_experts) {
    for (int t = 0; t < num_tokens; ++t) {
        for (int e = 0; e < num_experts; ++e) {
            // 让每个 token 的起始值不同，每个 expert 连续递增
            logits[t * num_experts + e] = 1.0f * (t * num_experts + e);
        }
    }
}

// CPU reference: iterative argmax with lower-index tie-break.
// out layout: [num_tokens, topk]
void select_topk_cpu_ref(const float* logits, int num_tokens, int num_experts,
                         int* out, int topk) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* row = logits + t * num_experts;
        bool picked[NUM_EXPERTS] = {};

        for (int k = 0; k < topk; ++k) {
            int best_e = -1;
            float best_v = -INFINITY;
            for (int e = 0; e < num_experts; ++e) {
                if (picked[e]) continue;
                if (best_e < 0 || row[e] > best_v ||
                    (row[e] == best_v && e < best_e)) {
                    best_v = row[e];
                    best_e = e;
                }
            }
            picked[best_e] = true;
            out[t * topk + k] = best_e;
        }
    }
}

// Compare GPU output against CPU reference.
// Returns true if every token's top-k indices match.
bool check_topk_result(const int* gpu_out, const int* cpu_ref, int num_tokens,
                       int topk) {
    bool ok = true;
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < topk; ++k) {
            int g = gpu_out[t * topk + k];
            int c = cpu_ref[t * topk + k];
            if (g != c) {
                printf("Mismatch at token=%d k=%d: gpu=%d cpu=%d\n", t, k, g, c);
                ok = false;
            }
        }
    }
    if (ok) {
        printf("Top-K result matches CPU reference.\n");
    }
    return ok;
}

int main(void) {
    printf("Starting...\n");
    cudaSetDevice(0);

    const int num_experts = NUM_EXPERTS;
    const int num_tokens = 8;
    const int topk = TOPK;

    const int threads_per_block = 256;
    const int token_per_block = threads_per_block/32;
    dim3 block_size(threads_per_block);
    dim3 grid_size(1, (num_tokens+token_per_block-1)/token_per_block);

    const size_t logits_bytes = (size_t)num_tokens * num_experts * sizeof(float);
    const size_t out_bytes = (size_t)num_tokens * topk * sizeof(int);

    float* h_logits = (float*)malloc(logits_bytes);
    int* h_gpu_out = (int*)malloc(out_bytes);
    int* h_cpu_ref = (int*)malloc(out_bytes);
    if (!h_logits || !h_gpu_out || !h_cpu_ref) {
        printf("Host malloc failed.\n");
        return 1;
    }

    init_logits(h_logits, num_tokens, num_experts);
    select_topk_cpu_ref(h_logits, num_tokens, num_experts, h_cpu_ref, topk);
    memset(h_gpu_out, -1, out_bytes);

    float* d_logits = nullptr;
    int* d_out = nullptr;
    cudaMalloc(&d_logits, logits_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_logits, h_logits, logits_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, -1, out_bytes);

    select_topk<<<grid_size, block_size>>>(d_logits, num_tokens, num_experts,
                                           d_out, topk);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_gpu_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    bool passed = check_topk_result(h_gpu_out, h_cpu_ref, num_tokens, topk);

    cudaFree(d_logits);
    cudaFree(d_out);
    free(h_logits);
    free(h_gpu_out);
    free(h_cpu_ref);

    return passed ? 0 : 1;
}