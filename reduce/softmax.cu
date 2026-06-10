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

// 静默对拍：返回是否通过，并带出最大绝对误差。
static bool compare_quiet(const float* gpu, const float* ref, size_t cnt,
                          float atol, float rtol, double* max_err_out){
    double max_abs = 0.0;
    bool ok = true;
    for(size_t i = 0; i < cnt; ++i){
        double g = gpu[i], e = ref[i];
        double abs_err = fabs(g - e);
        if(abs_err > max_abs) max_abs = abs_err;
        if(abs_err > atol + rtol * fabs(e)) ok = false;
    }
    *max_err_out = max_abs;
    return ok;
}

// 用小 rows 校验某个 D 的正确性（CPU 参考成本可控）。d_in 已含 h_in 的同份数据。
static bool verify_case(float* d_in, float* d_out, const float* h_in,
                        int D, int threads, double* max_err_out){
    const int rows_v = 64;
    const size_t cnt = (size_t)rows_v * D;
    float* h_ref = (float*)malloc(cnt * sizeof(float));
    float* h_gpu = (float*)malloc(cnt * sizeof(float));

    softmax_cpu_ref(h_in, h_ref, rows_v, D);
    softmax<<<dim3(rows_v), dim3(threads)>>>(d_in, d_out, D);
    if(cudaDeviceSynchronize() != cudaSuccess){
        free(h_ref); free(h_gpu); *max_err_out = -1.0; return false;
    }
    cudaMemcpy(h_gpu, d_out, cnt * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = compare_quiet(h_gpu, h_ref, cnt, 1e-5f, 1e-4f, max_err_out);
    free(h_ref); free(h_gpu);
    return ok;
}

// 计时单个 case：warmup + iters 次，返回平均 ms（出错返回 -1）。
static double bench_ms(float* d_in, float* d_out, int rows, int D,
                       int threads, int iters){
    dim3 grid(rows), block(threads);
    for(int i = 0; i < 5; ++i) softmax<<<grid, block>>>(d_in, d_out, D);
    if(cudaDeviceSynchronize() != cudaSuccess) return -1.0;

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i = 0; i < iters; ++i) softmax<<<grid, block>>>(d_in, d_out, D);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

// 终端 ASCII 柱状图：柱长按 GB/s 归一化。
static void print_chart(const char* title, const char* xname,
                        const long* xs, const double* gbps,
                        const double* ms, const bool* ok, int n){
    const int BARW = 40;
    double maxv = 0.0;
    for(int i = 0; i < n; ++i) if(gbps[i] > maxv) maxv = gbps[i];

    printf("\n%s   (柱长 ∝ 有效带宽)\n", title);
    printf("%8s | %-*s %10s %11s %6s\n", xname, BARW, "", "GB/s", "ms/iter", "check");
    for(int i = 0; i < n; ++i){
        int len = (maxv > 0.0) ? (int)(gbps[i] / maxv * BARW + 0.5) : 0;
        char bar[BARW + 1];
        for(int j = 0; j < BARW; ++j) bar[j] = (j < len) ? '#' : ' ';
        bar[BARW] = '\0';
        printf("%8ld | %s %10.1f %11.4f %6s\n",
               xs[i], bar, gbps[i], ms[i], ok[i] ? "ok" : "FAIL");
    }
}

int main(){
    cudaSetDevice(0);
    const int threads = THREADS;
    const int ITERS   = 50;

    // 两个 sweep 里最大的元素数：8192*8192 == 65536*1024 == 67M，按它一次性分配。
    const size_t MAX_ELEMS = (size_t)65536 * 1024;
    const size_t max_bytes = MAX_ELEMS * sizeof(float);

    float* h_in = (float*)malloc(max_bytes);
    if(!h_in){ printf("Host malloc failed.\n"); return 1; }
    srand(1234);
    for(size_t i = 0; i < MAX_ELEMS; ++i)
        h_in[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;

    float* d_in  = nullptr;
    float* d_out = nullptr;
    if(cudaMalloc(&d_in,  max_bytes) != cudaSuccess ||
       cudaMalloc(&d_out, max_bytes) != cudaSuccess){
        printf("cudaMalloc failed (需要 ~%.1f GB x2)\n", max_bytes / 1e9);
        return 1;
    }
    cudaMemcpy(d_in, h_in, max_bytes, cudaMemcpyHostToDevice);

    printf("Softmax sweep | threads=%d, iters=%d\n", threads, ITERS);

    // ---------- Sweep 1: 固定 rows，扫 D ----------
    {
        const int  rows = 8192;
        const long Ds[]  = {128, 256, 512, 1024, 2048, 4096, 8192};
        const int  N = (int)(sizeof(Ds) / sizeof(Ds[0]));
        long xs[16]; double gbps[16], ms[16]; bool ok[16];

        for(int k = 0; k < N; ++k){
            int D = (int)Ds[k];
            double err = 0.0;
            ok[k] = verify_case(d_in, d_out, h_in, D, threads, &err);
            double t = bench_ms(d_in, d_out, rows, D, threads, ITERS);
            xs[k] = D;
            ms[k] = t;
            // 访存型：读 in + 写 out = 2 * rows * D * 4 字节
            gbps[k] = (t > 0.0) ? (2.0 * rows * D * sizeof(float)) / (t * 1e-3) / 1e9 : 0.0;
        }
        print_chart("Sweep 1: 固定 rows=8192，扫 D", "D", xs, gbps, ms, ok, N);
    }

    // ---------- Sweep 2: 固定 D，扫 rows ----------
    {
        const int  D = 1024;
        const long Rs[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        const int  N = (int)(sizeof(Rs) / sizeof(Rs[0]));
        long xs[16]; double gbps[16], ms[16]; bool ok[16];

        for(int k = 0; k < N; ++k){
            int rows = (int)Rs[k];
            double err = 0.0;
            ok[k] = verify_case(d_in, d_out, h_in, D, threads, &err);
            double t = bench_ms(d_in, d_out, rows, D, threads, ITERS);
            xs[k] = rows;
            ms[k] = t;
            gbps[k] = (t > 0.0) ? (2.0 * rows * D * sizeof(float)) / (t * 1e-3) / 1e9 : 0.0;
        }
        print_chart("Sweep 2: 固定 D=1024，扫 rows", "rows", xs, gbps, ms, ok, N);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
