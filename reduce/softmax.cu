#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// =====================================================================
// 配置：rows 个样本，每个样本长度 D，softmax 沿 D 做。
// 一个 block 处理一个样本（blockIdx.x = 样本号）。
// 这些都做成宏，方便你自己改规模、调 block 大小。
// =====================================================================
#define ROWS    8192
#define D_LEN   1024
#define THREADS 256

// =====================================================================
// ↓↓↓ 以下 kernel 区域是从 reduce_interview.cu 原样搬过来的，按要求“不改错误”。
//     里面已知的 bug（shfl 少 __、tid/s 未定义、__shared__ 缺类型、out[tid]
//     应为 out_start[i] 等）都保留，留给你自己 debug + 优化。
//     softmax 由 __device__ 改成 __global__，这样 main 能直接 launch。
// =====================================================================

__forceinline__  __device__ float warp_reduce_sum(float val){

    #pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        val += __shfl_down_sync(0xffffffff,val, offset);
    }
    return val;
}

__forceinline__  __device__ float warp_reduce_max(float val){
    #pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff,val, offset));
    }
    return val;
    
}


__forceinline__ __device__  float block_reduce_sum(float val, float* sm){
    
    int lid = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    float warp_sum = warp_reduce_sum(val);
    if(lid == 0){
        sm[wid] = warp_sum;
    }
    __syncthreads();

    if(wid == 0){
        float cur_val = lid <  blockDim.x/32 ?  sm[lid] : 0.0f;
        float block_sum  = warp_reduce_sum(cur_val);
        if(lid == 0){
            sm[0] = block_sum;
        }
    }

    __syncthreads();

    return sm[0];
}

__forceinline__ __device__ float block_reduce_max(float val, float* sm){
        
    int lid = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    float warp_sum = warp_reduce_max(val);
    if(lid == 0){
        sm[wid] = warp_sum;
    }
    __syncthreads();

    if(wid == 0){
        float cur_val = lid <  blockDim.x/32 ?  sm[lid] : 0.0f;
        float block_sum  = warp_reduce_max(cur_val);
        if(lid == 0){
            sm[0] = block_sum;
        }
    }

    __syncthreads();
    return sm[0];
}

//softmax
//一个block处理一个样本
//D样本的长度，
__global__ void softmax(float* in, float* out, int D){
    

    __shared__ float sm[32];
    //get max
    float max_in_group = -FLT_MAX;
    int tid = threadIdx.x;
    float* block_start =  in + blockIdx.x * D;
    for(int i = tid; i < D ;i+= blockDim.x){
        max_in_group = fmaxf(max_in_group, block_start[i]);
    }    
    float block_max = block_reduce_max(max_in_group,sm);


    //get sum
    float sum_group = 0.0;
    for(int i = tid; i < D ;i+= blockDim.x){
        sum_group += expf(block_start[i] - block_max);
    }    
    float block_sum = block_reduce_sum(sum_group,sm);

    //write_back
    float* out_start = out + blockIdx.x * D;
    for(int i = tid; i < D ;i+= blockDim.x){
        out_start[i] = expf(block_start[i] - block_max)/block_sum;
    }    
}






// =====================================================================
// ↑↑↑ kernel 区域结束。下面是配套的测试框架（CPU 参考 + main）。
// =====================================================================

// CPU 参考：逐行 numerically-stable softmax。
//   m = max(row); out[i] = exp(x[i]-m) / Σ exp(x[j]-m)
void softmax_cpu_ref(const float* in, float* out, int rows, int D){
    for(int r = 0; r < rows; ++r){
        const float* x = in  + (size_t)r * D;
        float*       y = out + (size_t)r * D;

        float m = -FLT_MAX;
        for(int i = 0; i < D; ++i) m = fmaxf(m, x[i]);

        double sum = 0.0;                       // 用 double 累加，参考值更准
        for(int i = 0; i < D; ++i) sum += expf(x[i] - m);

        for(int i = 0; i < D; ++i) y[i] = expf(x[i] - m) / (float)sum;
    }
}

// 对拍：逐元素比较，atol + rtol 容忍浮点误差，报告最大误差和首个越界位置。
bool check_softmax_result(const float* gpu, const float* ref,
                          int rows, int D, float atol, float rtol){
    double max_abs = 0.0;
    int    bad_r = -1, bad_i = -1;
    bool   ok = true;

    for(int r = 0; r < rows; ++r){
        for(int i = 0; i < D; ++i){
            size_t idx = (size_t)r * D + i;
            double g = gpu[idx], e = ref[idx];
            double abs_err = fabs(g - e);
            if(abs_err > max_abs){ max_abs = abs_err; bad_r = r; bad_i = i; }
            if(abs_err > atol + rtol * fabs(e)){
                if(ok){   // 只打印第一个越界点，避免刷屏
                    printf("Mismatch at row=%d i=%d: gpu=%.8f ref=%.8f (abs_err=%.3e)\n",
                           r, i, g, e, abs_err);
                }
                ok = false;
            }
        }
    }

    printf("max_abs_err = %.3e (at row=%d i=%d)\n", max_abs, bad_r, bad_i);
    if(ok) printf("Softmax matches CPU reference.\n");
    return ok;
}

int main(){
    printf("Starting... rows=%d D=%d threads=%d\n", ROWS, D_LEN, THREADS);
    cudaSetDevice(0);

    const int rows = ROWS;
    const int D    = D_LEN;
    const size_t n = (size_t)rows * D;
    const size_t bytes = n * sizeof(float);

    float* h_in      = (float*)malloc(bytes);
    float* h_ref     = (float*)malloc(bytes);
    float* h_gpu_out = (float*)malloc(bytes);
    if(!h_in || !h_ref || !h_gpu_out){
        printf("Host malloc failed.\n");
        return 1;
    }

    // 造确定性输入：[-10, 10] 均匀，含较大量级以考验 max 减法的数值稳定性。
    srand(1234);
    for(size_t i = 0; i < n; ++i){
        h_in[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }

    // CPU 参考
    softmax_cpu_ref(h_in, h_ref, rows, D);

    // device 内存
    float* d_in  = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 grid(rows);
    dim3 block(THREADS);

    // ---- 正确性：跑一次，拷回对拍 ----
    softmax<<<grid, block>>>(d_in, d_out, D);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_gpu_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool passed = check_softmax_result(h_gpu_out, h_ref, rows, D, 1e-5f, 1e-4f);

    // ---- 性能：warmup + 多次取平均，报告耗时和有效带宽 ----
    const int WARMUP = 5;
    const int ITERS  = 50;
    for(int i = 0; i < WARMUP; ++i) softmax<<<grid, block>>>(d_in, d_out, D);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < ITERS; ++i) softmax<<<grid, block>>>(d_in, d_out, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_ms = ms / ITERS;
    // softmax 是访存型：读一遍 in + 写一遍 out = 2 * bytes
    double gbps = (2.0 * bytes) / (avg_ms * 1e-3) / 1e9;
    printf("avg %.4f ms/iter, effective BW %.1f GB/s\n", avg_ms, gbps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_ref);
    free(h_gpu_out);

    return passed ? 0 : 1;
}
