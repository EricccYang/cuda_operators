#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>



#define NUM_EXPERTS 128



__forceinline__  __device__ float warp_reduce_max(float val){

    #pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        val = fmaxf(val,  __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
};



__global__ void select_topk(float* logits, int num_tokens, int num_experts, int* out ,int topk){
    
    
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;

    constexpr int warp_size = 32;

    float* block_start = logits + blockIdx.y *  NUM_EXPERTS ;
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
        __shared__  float sm[NUM_EXPERTS/warp_size];
        float warp_max = warp_reduce_max(cur_value);
        if(lid == 0){
            sm[wid] = warp_max;
            printf("warp_max: %f , warp index: %d , token index: %d \n", warp_max, wid, blockIdx.y);
        }  

        __syncthreads();

        float block_max =  0.f;

        if(wid == 0){
            float val = lid < blockDim.x/warp_size ?  sm[lid] : 0.f;
            block_max = warp_reduce_max(val);
        }

        __syncthreads();

        if(fabs(cur_value - block_max) < 1e-4 ){
            out[k] = lid + wid * warp_size;
            cur_value = -INFINITY;
        }

        k++;
    }

    return;


};


#define TOPK 1

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

    dim3 block_size(num_experts);
    dim3 grid_size(1, num_tokens);

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