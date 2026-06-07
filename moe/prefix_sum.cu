#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define NUM_EXPERTS 128

__global__ void prefix_sum(int* source, int len){

    
    int tid = threadIdx.x;
    int lid =  tid & 31;
    int wid =  tid >> 5;

    int* s = source + (blockDim.x  * blockIdx.x + tid);
    int val = s[0];
    
    __shared__ int sm[32];
    
    for(int offset = 1; offset < 32; offset <<= 1){
        val += __shfl_up_sync(0xffffffff, val, offset);
    }

    if(lid  == 31){
        sm[wid] = val;
    }
    
    __syncthreads();

    if(wid  == 0){
        int t = sm[lid];
        for(int offset = 1; offset < 32; offset <<= 1){
            t += __shfl_up_sync(0xffffffff, t, offset);
        }
        sm[tid] = t;
    }
    __syncthreads();

    if(wid > 0){
        int sum_val = sm[wid-1];
        s[0] = val + sum_val;
    }


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

// CPU 参考 scan —— inclusive，和当前 kernel 的语义对齐
// （out[i] = in[0] + ... + in[i]）。
// 注意：MoE 真正要的 offset 是 exclusive；等 kernel 跑通后把 kernel 改成
// 写 exclusive，这里同步改成 out[i] = (i==0)?0:out[i-1]+in[i-1] 即可。
void prefix_sum_cpu_ref(const int* in, int* out, int n) {
    int acc = 0;
    for (int i = 0; i < n; ++i) {
        acc += in[i];
        out[i] = acc;   // inclusive
    }
}

bool check_scan_result(const int* gpu, const int* ref, int n) {
    for (int i = 0; i < n; ++i) {
        if (gpu[i] != ref[i]) {
            printf("Mismatch at i=%d: gpu=%d ref=%d\n", i, gpu[i], ref[i]);
            return false;
        }
    }
    printf("Prefix-sum matches CPU reference.\n");
    return true;
}

int main() {
    printf("Starting...\n");
    cudaSetDevice(0);

    const int num_experts = NUM_EXPERTS;
    const int topk = TOPK;

    // 路由表规模：只用来在 CPU 端造一份真实的 per-expert count 当 scan 输入。
    const int seq_len = 256*8;

    // prefix_sum 是 block 内 scan：num_experts(=128) 个元素全塞进一个 block。
    // blockDim.x == num_experts，单 block，gridDim.x == 1。
    dim3 block_size(num_experts);
    dim3 grid_size(1);


    const size_t table_bytes = (size_t)seq_len * topk * sizeof(int);
    const size_t count_bytes = (size_t)num_experts * sizeof(int);

    int* h_table    = (int*)malloc(table_bytes);
    int* h_counts   = (int*)malloc(count_bytes);   // scan 的输入：每个 expert 的行数
    int* h_scan_ref = (int*)malloc(count_bytes);   // CPU 参考 scan
    int* h_gpu_scan = (int*)malloc(count_bytes);   // GPU 跑出来的 scan
    if (!h_table || !h_counts || !h_scan_ref || !h_gpu_scan) {
        printf("Host malloc failed.\n");
        return 1;
    }

    // 造输入 + CPU 参考
    init_routing_table(h_table, seq_len, topk, num_experts);
    count_experts_cpu_ref(h_table, seq_len, topk, num_experts, h_counts);
    prefix_sum_cpu_ref(h_counts, h_scan_ref, num_experts);

    // device: 把 count 拷上去，原地做 scan
    int* d_counts = nullptr;
    cudaMalloc(&d_counts, count_bytes);
    cudaMemcpy(d_counts, h_counts, count_bytes, cudaMemcpyHostToDevice);

    prefix_sum<<<grid_size, block_size>>>(d_counts, num_experts);

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

    cudaMemcpy(h_gpu_scan, d_counts, count_bytes, cudaMemcpyDeviceToHost);

    bool passed = check_scan_result(h_gpu_scan, h_scan_ref, num_experts);

    cudaFree(d_counts);
    free(h_table);
    free(h_counts);
    free(h_scan_ref);
    free(h_gpu_scan);

    return passed ? 0 : 1;
}