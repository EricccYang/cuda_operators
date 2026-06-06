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
__global__ void permute_topk(int* table, int seq_len, int topk, int* expert_token_count_arr){


    __shared__ int count[1];
    if(threadIdx.x == 0){
        count[0] = 0;
    }

    __syncthreads();


    int experts_index = blockIdx.y;
    //暂时先不处理多的情况

    int token_id = threadIdx.x;
    int* block_start = table ;
    int* thread_start = block_start+ token_id* topk;

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

// 构造路由表 table[seq_len][topk]：每个 token 选 topk 个“互不相同”的 expert。
// 用确定性的方式生成，方便和 CPU 参考对拍。
void init_routing_table(int* table, int seq_len, int topk, int num_experts) {
    for (int t = 0; t < seq_len; ++t) {
        for (int k = 0; k < topk; ++k) {
            int e = (t * topk + k) % num_experts;   // 同一 token 内尽量错开
            table[t * topk + k] = e;
        }
    }
}

// CPU 参考：和 kernel 语义一致 —— 同一 token 命中某 expert 只算一次。
// counts layout: [num_experts]
void count_experts_cpu_ref(const int* table, int seq_len, int topk,
                           int num_experts, int* counts) {
    for (int e = 0; e < num_experts; ++e) {
        int c = 0;
        for (int t = 0; t < seq_len; ++t) {
            const int* row = table + t * topk;
            bool matched = false;
            for (int k = 0; k < topk; ++k) {
                if (row[k] == e) { matched = true; break; }
            }
            if (matched) ++c;
        }
        counts[e] = c;
    }
}

// 对比 GPU 直方图和 CPU 参考。
bool check_count_result(const int* gpu_counts, const int* cpu_ref,
                        int num_experts) {
    bool ok = true;
    for (int e = 0; e < num_experts; ++e) {
        if (gpu_counts[e] != cpu_ref[e]) {
            printf("Mismatch at expert=%d: gpu=%d cpu=%d\n",
                   e, gpu_counts[e], cpu_ref[e]);
            ok = false;
        }
    }
    if (ok) {
        printf("Expert token-count matches CPU reference.\n");
    }
    return ok;
}

    printf("Starting...\n");
    cudaSetDevice(0);

    const int num_experts = NUM_EXPERTS;
    const int topk = TOPK;

    // kernel 意图：token_id = threadIdx.x，一个线程恰好一个 token，
    // 没有 stride 循环也没有越界判断 —— 所以全部 token 必须装进一个 block。
    // => blockDim.x == seq_len，且 seq_len <= 1024（单 block 线程上限）。
    const int seq_len = 256*8;
    const int threads_per_block = 256;   // 一线程一 token
    dim3 block_size(threads_per_block); //确实是横着的
    // gridDim.y = num_experts：一个 block 负责一个 expert（blockIdx.y）
    //gridDim.x  
    dim3 grid_size(  (seq_len + threads_per_block - 1)/threads_per_block , num_experts); 


    const size_t table_bytes = (size_t)seq_len * topk * sizeof(int);
    const size_t count_bytes = (size_t)num_experts * sizeof(int);

    int* h_table = (int*)malloc(table_bytes);
    int* h_gpu_counts = (int*)malloc(count_bytes);
    int* h_cpu_ref = (int*)malloc(count_bytes);
    if (!h_table || !h_gpu_counts || !h_cpu_ref) {
        printf("Host malloc failed.\n");
        return 1;
    }

    init_routing_table(h_table, seq_len, topk, num_experts);
    count_experts_cpu_ref(h_table, seq_len, topk, num_experts, h_cpu_ref);
    memset(h_gpu_counts, 0, count_bytes);

    int* d_table = nullptr;
    int* d_counts = nullptr;
    cudaMalloc(&d_table, table_bytes);
    cudaMalloc(&d_counts, count_bytes);
    cudaMemcpy(d_table, h_table, table_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_counts, 0, count_bytes);

    permute_topk<<<grid_size, block_size>>>(d_table, seq_len, topk, d_counts);

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

    cudaMemcpy(h_gpu_counts, d_counts, count_bytes, cudaMemcpyDeviceToHost);

    bool passed = check_count_result(h_gpu_counts, h_cpu_ref, num_experts);

    cudaFree(d_table);
    cudaFree(d_counts);
    free(h_table);
    free(h_gpu_counts);
    free(h_cpu_ref);

    return passed ? 0 : 1;
}