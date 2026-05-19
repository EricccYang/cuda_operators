// ============================================================================
// HBM Peak Bandwidth Benchmark
//
// 测四种 streaming workload，用 float4 向量化访问、grid-stride loop 充分覆盖：
//   read-only : sum  = Σ a[i]           — 只读   (1N bytes)
//   write-only: a[i] = const            — 只写   (1N bytes)
//   copy      : b[i] = a[i]             — 读+写  (2N bytes)
//   triad     : a[i] = b[i] + s*c[i]    — STREAM (3N bytes)
//
// 数组大小默认 1 GiB，远超 H100 50MB L2，保证 cache miss 必打 HBM。
// 计时用 cudaEvent，每 pattern 跑 warmup + repeats 次取最好。
//
// 编译: nvcc -O3 -arch=sm_90 hbm_load.cu -o hbm_load
// 运行: ./hbm_load  [N_in_MB]  [repeats]
// ============================================================================

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA err %s @ %s:%d\n",                          \
                    cudaGetErrorString(e), __FILE__, __LINE__);               \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ----------- kernels (all use float4 = 16B per thread per iter) -------------

// read-only: 每个线程把 float4 加进 acc，最后只写一个值防止 DCE
__global__ void k_read(const float4* __restrict__ a, float* sink, size_t n4) {
    size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = tid; i < n4; i += stride) {
        float4 v = a[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    // 这条 if 编译期不可知，但运行期永远 false，编译器无法消除 acc
    if ((acc.x + acc.y + acc.z + acc.w) == -1.f) {
        sink[tid] = acc.x;
    }
}

__global__ void k_write(float4* __restrict__ a, size_t n4, float val) {
    size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float4 v = make_float4(val, val, val, val);
    for (size_t i = tid; i < n4; i += stride) {
        a[i] = v;
    }
}

__global__ void k_copy(const float4* __restrict__ a,
                       float4*       __restrict__ b, size_t n4) {
    size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid; i < n4; i += stride) {
        b[i] = a[i];
    }
}

__global__ void k_triad(float4*       __restrict__ a,
                        const float4* __restrict__ b,
                        const float4* __restrict__ c,
                        float s, size_t n4) {
    size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid; i < n4; i += stride) {
        float4 x = b[i], y = c[i];
        float4 r;
        r.x = x.x + s * y.x;
        r.y = x.y + s * y.y;
        r.z = x.z + s * y.z;
        r.w = x.w + s * y.w;
        a[i] = r;
    }
}

// ---------------------------------- main -----------------------------------

struct Stat {
    const char* name;
    double      bytes;   // 单次 kernel 访问的总字节数
    float       ms_best; // 最快的一次 (ms)
};

static void print_stat(const Stat& s, double peak_TBs) {
    double gbps = s.bytes / (s.ms_best * 1.0e-3) / 1.0e9;
    printf("  %-10s  %8.3f ms   %8.1f GB/s   (%.1f%% of %.2f TB/s)\n",
           s.name, s.ms_best, gbps, gbps / (peak_TBs * 1000.0) * 100.0,
           peak_TBs);
}

int main(int argc, char** argv) {
    size_t N_MB   = (argc > 1) ? std::atoll(argv[1]) : 1024;  // default 1 GiB
    int    repeat = (argc > 2) ? std::atoi(argv[2])  : 10;
    int    warmup = 3;

    size_t N_bytes = N_MB * (1ull << 20);
    size_t N       = N_bytes / sizeof(float);     // 单数组元素数 (float)
    size_t n4      = N / 4;                       // float4 数量

    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));

    // 理论显存峰值 (TB/s)：根据 device 名字粗略选一个；实测时只是个对比基准
    // 注: 4090/5090 是 GDDR6X/GDDR7, 不是 HBM, 这里只是统一报个理论值方便对比
    double peak_TBs = 3.35;                                          // H100 SXM5 HBM3
    if (strstr(prop.name, "H100 PCIe"))    peak_TBs = 2.0;
    if (strstr(prop.name, "A100"))         peak_TBs = 2.0;
    if (strstr(prop.name, "B200"))         peak_TBs = 8.0;
    if (strstr(prop.name, "4090"))         peak_TBs = 1.008;         // GDDR6X 21Gbps x 384bit
    if (strstr(prop.name, "5090"))         peak_TBs = 1.792;         // GDDR7

    printf("Device : %s   (SM count = %d, L2 = %.1f MB)\n",
           prop.name, prop.multiProcessorCount,
           prop.l2CacheSize / 1024.0 / 1024.0);
    printf("Buffer : %.1f MiB per array, float4 elems = %zu, repeat = %d\n\n",
           N_bytes / 1024.0 / 1024.0, n4, repeat);

    // 3 个数组：triad 用到 a,b,c；其他用 a,b
    float4 *dA, *dB, *dC;
    float  *dSink;
    CK(cudaMalloc(&dA, N_bytes));
    CK(cudaMalloc(&dB, N_bytes));
    CK(cudaMalloc(&dC, N_bytes));
    CK(cudaMalloc(&dSink, sizeof(float) * 1024));
    CK(cudaMemset(dA, 0, N_bytes));
    CK(cudaMemset(dB, 0, N_bytes));
    CK(cudaMemset(dC, 0, N_bytes));

    int block = 256;
    int grid  = prop.multiProcessorCount * 4;
    // 也别开太多 block：grid * block 足以让 LDG 饱和即可
    if ((size_t)grid * block > n4) grid = (int)((n4 + block - 1) / block);

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    auto bench = [&](const char* name, double bytes, auto launch) -> Stat {
        for (int i = 0; i < warmup; ++i) launch();
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < repeat; ++i) {
            CK(cudaEventRecord(e0));
            launch();
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        return {name, bytes, best};
    };

    Stat s_read = bench("read", (double)N_bytes, [&] {
        k_read<<<grid, block>>>(dA, dSink, n4);
    });
    Stat s_write = bench("write", (double)N_bytes, [&] {
        k_write<<<grid, block>>>(dA, n4, 1.0f);
    });
    Stat s_copy = bench("copy", 2.0 * N_bytes, [&] {
        k_copy<<<grid, block>>>(dA, dB, n4);
    });
    Stat s_triad = bench("triad", 3.0 * N_bytes, [&] {
        k_triad<<<grid, block>>>(dA, dB, dC, 2.0f, n4);
    });

    printf("Pattern       time         BW             utilization\n");
    printf("------------------------------------------------------\n");
    print_stat(s_read,  peak_TBs);
    print_stat(s_write, peak_TBs);
    print_stat(s_copy,  peak_TBs);
    print_stat(s_triad, peak_TBs);

    CK(cudaFree(dA));
    CK(cudaFree(dB));
    CK(cudaFree(dC));
    CK(cudaFree(dSink));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return 0;
}
