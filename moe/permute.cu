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

__forceinline__ __device__  float warp_reduce_max(float val ,int index, int* index_out){
    
    #pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        float other_value = __shfl_xor_sync(0xffffffff, val, offset);
        int other_index = __shfl_xor_sync(0xffffffff, index, offset);
        if(other_value > val || (other_value == val && other_index < index)){
            val = other_value;
            index = other_index;
        }
        *index_out = index;
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
__global__ void permute_topk(float* table, int seq_len, int topk, int* expert_token_count_arr){
    
    
    __shared__ int count[1];
    if(threadIdx.x == 0){
        count[0] = 0;
    }
    
    
    
    int experts_index = blockIdx.y;
    //暂时先不处理多的情况

    int token_id = threadIdx.x;

    float* block_start = table +  blockDim.x*topk *  blockIdx.y;
    float* thread_start = block_start+ token_id* topk;

    bool matched = false;
    for(int k = 0; k <  topk; k++){
        if(thread_start[k] ==  experts_index){
            matched = true;
            break;
        }
    }

    
    //sm，有个sm做聚合？
    //好像一个数字就可以，不就是这个experts的数量吗？
    if(matched){
        atomicAdd(&count[0],1);
    }

    __syncthreads();

    
    //写global呗，貌似是这样的吧？
    //写啥呢？当然是写自己这个block的对应的数字写进去，先出一个表呗
    expert_token_count_arr[experts_index] = count[0];



    //一个block处理一部分的seq吧那就
    // 然后一个线程分到几个token，也就是几行，然后遍历 
    // 然后能确定是不是有这个experts？
    // 那好像是不成一个线程不要遍历太多吧我理解？
    // 需要设置的是什么？ 这个东西很有意思
    // 但是其实就是只用一次吧
    // 4个float，32个warp 128个数据，8个experts的话就是16个token，
    // 但是因为其实grid是按experts分的？？？？，所以相当于需要for循环便利所有的token，for循环seqlen/16 次？
    // 这好像不太对吧， 
    // ～～～～～～～～～～～～问题来了，这种情况下怎么设计
    //一个线程一个token， 但是应该是一个token的experts的一部分
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

    permute_topk<<<grid_size, block_size>>>(d_logits, num_tokens, num_experts,
                                           d_out);

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