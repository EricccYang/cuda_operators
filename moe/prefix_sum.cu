#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define N 128   // 元素个数：单 block，blockDim==N，且为 32 的倍数

__global__ void prefix_sum(int* source, int len){

    
    int tid = threadIdx.x;
    int lid =  tid & 31;
    int wid =  tid >> 5;

    int* s = source + (blockDim.x  * blockIdx.x + tid);
    int val = s[0];
    
    __shared__ int sm[32];
    
    for(int offset = 1; offset < 32; offset <<= 1){
        int v = __shfl_up_sync(0xffffffff, val, offset);
        if(lid >= offset){
            val += v;
        }
    }

    if(lid  == 31){
        sm[wid] = val;
    }
    
    __syncthreads();

    if(wid  == 0){
        int t = sm[lid];
        for(int offset = 1; offset < 32; offset <<= 1){
            int v = __shfl_up_sync(0xffffffff, t, offset);
            if(lid >= offset){
                t += v;
            }
        }
        sm[tid] = t;
    }
    __syncthreads();

    int sum_val = 0;
    if(wid > 0){
        sum_val = sm[wid-1];  
    }
    s[0] = val + sum_val;

};


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

    const int n = N;

    // 单 block scan：blockDim==n，gridDim==1。
    dim3 block_size(n);
    dim3 grid_size(1);

    const size_t bytes = (size_t)n * sizeof(int);

    int* h_in       = (int*)malloc(bytes);   // scan 输入
    int* h_scan_ref = (int*)malloc(bytes);   // CPU 参考
    int* h_gpu_scan = (int*)malloc(bytes);   // GPU 结果
    if (!h_in || !h_scan_ref || !h_gpu_scan) {
        printf("Host malloc failed.\n");
        return 1;
    }

    // 造各不相同的输入，让每个前缀唯一，错位立刻暴露
    for (int i = 0; i < n; ++i) h_in[i] = i + 1;   // 1,2,3,... -> 前缀是三角数
    prefix_sum_cpu_ref(h_in, h_scan_ref, n);

    // device: 把输入拷上去，原地做 scan
    int* d_in = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    prefix_sum<<<grid_size, block_size>>>(d_in, n);

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

    cudaMemcpy(h_gpu_scan, d_in, bytes, cudaMemcpyDeviceToHost);

    bool passed = check_scan_result(h_gpu_scan, h_scan_ref, n);

    cudaFree(d_in);
    free(h_in);
    free(h_scan_ref);
    free(h_gpu_scan);

    return passed ? 0 : 1;
}