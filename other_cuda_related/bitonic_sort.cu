#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Bitonic sort, 单 block 版本，约束 N 是 2 的幂、N <= BLOCK_SIZE × 2
//
// 思路：bitonic 网络是一组 compare-and-swap，结构静态、无分支冲突，非常适合 GPU。
//   外层循环 k：构建 / 排序的"子序列大小"，从 2 翻倍到 N
//   内层循环 j：当前 stage 的"比较距离"，从 k/2 减半到 1
//   对每对 (tid, tid ^ j)：根据 (tid & k) 决定升序还是降序，然后比较交换
//
// 总步数 = log2(N) × (log2(N) + 1) / 2 ≈ O(log² N)
// 每步并行度 = N / 2 个比较对，warp 内 lockstep 没有 divergence
//
// 推广到大 N (跨 block) 需要多 kernel + 全局 stage，暂不实现。

#define N 1024     // 必须是 2 的幂


__device__ inline void cmp_swap(int* a, int* b, bool ascending) {
    if ((*a > *b) == ascending) {
        int t = *a; *a = *b; *b = t;
    }
}


// 假设 blockDim.x = N，一线程一元素
__global__ void bitonic_sort_block(int* data) {
    __shared__ int s[N];
    int tid = threadIdx.x;
    s[tid] = data[tid];
    __syncthreads();

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                cmp_swap(&s[tid], &s[ixj], ascending);
            }
            __syncthreads();
        }
    }

    data[tid] = s[tid];
}


__global__ void bitonic_sort(int* data){
    __shared__ int s[N];
    int tid = threadIdx.x;
    s[tid] = data+tid;
    __syncthreads();

    
    for(int k = 2; k <= N; k <<= 1){
        for(int  j = k >> 1; j > 0 ;j >>= 1){

            int ixj = tid ^ j;
            if(ixj > tid){

                bool ascending =  ((tid & k) == 0);
                cmp_swap(&s[tid],&s[ixj], ascending);

 
            }  
            __syncthreads();
        }
    }
    data[tid] = s[tid];
}




// ---------- host helpers ----------
int cmp_int(const void* a, const void* b) { return *(int*)a - *(int*)b; }


int main() {
    printf("bitonic sort: N=%d (single block)\n", N);
    cudaSetDevice(0);

    size_t nBytes = N * sizeof(int);
    int* h_in  = (int*)malloc(nBytes);
    int* h_ref = (int*)malloc(nBytes);
    int* h_gpu = (int*)malloc(nBytes);

    srand(1234);
    for (int i = 0; i < N; i++) h_in[i] = rand();
    memcpy(h_ref, h_in, nBytes);
    qsort(h_ref, N, sizeof(int), cmp_int);

    int* d_data;
    cudaMalloc(&d_data, nBytes);
    cudaMemcpy(d_data, h_in, nBytes, cudaMemcpyHostToDevice);

    dim3 block(N), grid(1);   // 一线程一元素
    bitonic_sort_block<<<grid, block>>>(d_data);
    cudaMemcpy(h_gpu, d_data, nBytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; i++) if (h_gpu[i] != h_ref[i]) { ok = false; break; }
    printf("  correctness: %s\n", ok ? "OK" : "FAIL");

    // 计时
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 200;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) {
        cudaMemcpyAsync(d_data, h_in, nBytes, cudaMemcpyHostToDevice);
        bitonic_sort_block<<<grid, block>>>(d_data);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    printf("  time per sort (incl. H2D): %.4f ms\n", ms);

    cudaFree(d_data);
    free(h_in); free(h_ref); free(h_gpu);
    return 0;
}



#include <torch/extension.h>

// Even-odd transposition sort kernel for small arrays
__global__ void sort_small_kernel(float* data, int n, int phase) {
    // Calculate global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each thread compares and potentially swaps adjacent elements
    // Based on even-odd transposition sort (parallel bubble sort)
    int i = 2 * tid + (phase & 1); // Alternate between even and odd indices
    
    if (i < n - 1) {
        if (data[i] > data[i + 1]) {
            // Swap elements
            float temp = data[i];
            data[i] = data[i + 1];
            data[i + 1] = temp;
        }
    }
}

// Bitonic sort kernel for more efficiency with larger arrays
__global__ void bitonic_sort_kernel(float* data, int n, int j, int k) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate the indices for comparison
    int ixj = i ^ j;
    
    // Check bounds and if the thread should do work
    if (ixj > i && i < n && ixj < n) {
        // Check direction of sort (increasing or decreasing)
        if ((i & k) == 0) {
            // Increasing
            if (data[i] > data[ixj]) {
                // Swap
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Decreasing
            if (data[i] < data[ixj]) {
                // Swap
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

torch::Tensor sort(torch::Tensor data) {
    if (!data.is_cuda()) {
        throw std::runtime_error("Input tensor must be a CUDA tensor");
    }
    if (data.scalar_type() != torch::kFloat) {
        throw std::runtime_error("Input tensor must be of type float32");
    }
    if (data.dim() != 1) {
        throw std::runtime_error("Input tensor must be 1-dimensional");
    }

    int n = data.numel();
    
    // Create a copy of the input tensor for our sorting
    torch::Tensor output = data.clone();
    float* d_output = output.data_ptr<float>();
    
    // Use the simpler sort for very small arrays
    if (n < 1024) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        // For even-odd sort, we need n phases to ensure the array is sorted
        for (int phase = 0; phase < n; phase++) {
            sort_small_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, n, phase);
            cudaDeviceSynchronize();
        }
    } 
    else {
        // For larger arrays, use bitonic sort
        // Pad the array to the next power of 2 for bitonic sort
        int size = 1;
        while (size < n) size *= 2;
        
        // Create a padded tensor with high values for the padding
        torch::Tensor padded = torch::full({size}, INFINITY, 
                               torch::TensorOptions()
                               .dtype(torch::kFloat)
                               .device(data.device()));
                               
        // Copy the original data to the beginning of the padded tensor
        padded.slice(0, 0, n).copy_(output);
        
        // Get pointer to the padded data
        float* d_padded = padded.data_ptr<float>();
        
        // Set thread and block configuration
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
        // Perform bitonic sort
        for (int k = 2; k <= size; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                bitonic_sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_padded, size, j, k);
                cudaDeviceSynchronize();
            }
        }
        
        // Copy the sorted data back to the output tensor (only the original size)
        output = padded.slice(0, 0, n).clone();
    }
    
    return output;
}