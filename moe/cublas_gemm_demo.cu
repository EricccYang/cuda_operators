// ============================================================================
// 手写 Tiled GEMM — 模拟 cuBLAS 内部核心计算逻辑
//
// 计算: C[M,N] = A[M,K] × B[K,N]
// 对应 MoE gating: logits[num_tokens, num_experts] = X × W_gate
//
// cuBLAS 内部做了什么? 核心是三层 tiling:
//
//   Level 1 — Block Tile (grid 级):
//     把 C[M,N] 切成 BM×BN 的块, 每个 thread block 负责一块
//     grid: (N/BN, M/BM)
//
//   Level 2 — K-Loop (时间维):
//     沿 K 维度每次搬 BK 列/行到 shared memory
//     A 的一个 tile: sA[BM, BK], B 的一个 tile: sB[BK, BN]
//     循环 K/BK 次, 每次做一步 partial GEMM
//
//   Level 3 — Thread Tile (register 级):
//     每个 thread 负责 C 中 TM×TN 的微块
//     从 shared memory 取 A 的 TM 个元素和 B 的 TN 个元素
//     在寄存器中做外积累加 (outer product)
//
// 本文件实现的 tiling 参数:
//   BM=128, BN=128, BK=8, TM=8, TN=8
//   每 block 256 threads, 每 thread 算 8×8 = 64 个输出
//   shared memory: (128+128) × 8 × 4B = 8 KB
//
// 参考: DeepSeek-V2/V3 风格参数
//   num_tokens=4096, hidden_size=4096, num_experts=128, top_k=8
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
               cudaGetErrorString(err));                            \
        exit(1);                                                    \
    }                                                               \
} while(0)

#define CHECK_CUBLAS(call) do {                                     \
    cublasStatus_t stat = call;                                     \
    if (stat != CUBLAS_STATUS_SUCCESS) {                            \
        printf("cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,     \
               (int)stat);                                          \
        exit(1);                                                    \
    }                                                               \
} while(0)

// ============================================================================
// 三层 Tiling GEMM Kernel
// ============================================================================
//
// 内存层级对应:
//   Global Memory  →  A[M,K], B[K,N]           (HBM, 带宽高但延迟大)
//   Shared Memory  →  sA[BM,BK], sB[BK,BN]    (片上, ~100× 低延迟)
//   Registers      →  regA[TM], regB[TN]       (最快, 每 thread 私有)
//                      regC[TM][TN]             (累加器)
//
// 数据流:
//   Global → Shared: 协作加载, 所有 thread 搬数据 (coalesced access)
//   Shared → Register: 每个 thread 只读自己需要的行/列
//   Register: 外积累加 regC[i][j] += regA[i] * regB[j]
//
// ============================================================================

constexpr int BM = 128;   // block tile M
constexpr int BN = 128;   // block tile N
constexpr int BK = 8;     // block tile K (每次沿 K 搬多少)
constexpr int TM = 8;     // thread tile M (每个 thread 算 8 行)
constexpr int TN = 8;     // thread tile N (每个 thread 算 8 列)

__global__ void sgemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    // ── block 负责 C 的哪一块 ──
    //   blockIdx.x → N 方向, blockIdx.y → M 方向
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // ── thread 在 block 内负责哪个微块 ──
    //   256 threads 排成 16×16 的 thread grid
    //   每个 thread 负责 TM×TN = 8×8 的输出
    //   16×8 = 128 = BM, 16×8 = 128 = BN ✓
    const int tx = threadIdx.x;  // 0..255
    const int thread_row = tx / (BN / TN);  // 0..15, 微块在 M 方向的编号
    const int thread_col = tx % (BN / TN);  // 0..15, 微块在 N 方向的编号

    // ── Shared memory: A tile 和 B tile ──
    __shared__ float sA[BM * BK];   // 128 × 8 = 1024 floats
    __shared__ float sB[BK * BN];   // 8 × 128 = 1024 floats

    // ── 寄存器: 累加器 + 临时向量 ──
    float regC[TM * TN] = {0.0f};   // 8×8 = 64 个累加器, 全部在寄存器
    float regA[TM];                  // 从 sA 取出的一列
    float regB[TN];                  // 从 sB 取出的一行

    // ── 协作加载的索引 ──
    //  加载 sA[BM, BK]: 1024 元素 / 256 threads = 4 元素/thread
    //    每个 thread 在一个 "行" 中对应 (row, col) 位置
    //    stride = 256/BK = 32, 循环 BM/32 = 4 次
    const int load_a_row = tx / BK;                    // 0..31
    const int load_a_col = tx % BK;                    // 0..7
    const int load_a_stride = blockDim.x / BK;         // 32

    //  加载 sB[BK, BN]: 1024 元素 / 256 threads = 4 元素/thread
    //    stride = 256/BN = 2, 循环 BK/2 = 4 次
    const int load_b_row = tx / BN;                    // 0..1
    const int load_b_col = tx % BN;                    // 0..127
    const int load_b_stride = blockDim.x / BN;         // 2

    // ── 指向当前 block 处理的 A、B 起始位置 ──
    //   A: 从第 by*BM 行开始
    //   B: 从第 bx*BN 列开始
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;

    // ═══════════════════════════════════════════════════════════════
    // K-Loop: 沿 K 维度滑动, 每次搬 BK 到 shared memory
    // ═══════════════════════════════════════════════════════════════
    for (int k = 0; k < K; k += BK) {

        // ── Step 1: Global → Shared (协作加载) ──
        //   所有 256 个 thread 一起搬, 保证 coalesced access
        //   A tile: 从 A[by*BM, k] 搬 BM×BK
        //   B tile: 从 B[k, bx*BN] 搬 BK×BN
        #pragma unroll
        for (int i = 0; i < BM; i += load_a_stride) {
            int row = load_a_row + i;
            sA[row * BK + load_a_col] = A[row * K + (k + load_a_col)];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += load_b_stride) {
            int row = load_b_row + i;
            sB[row * BN + load_b_col] = B[(k + row) * N + load_b_col];
        }

        __syncthreads();

        // ── Step 2: Shared → Register → Accumulate ──
        //   对 shared memory 中的 BK 个 "slice" 逐一做外积
        //   这是计算密集部分: 每个 thread 做 TM×TN = 64 次 FMA / slice
        #pragma unroll
        for (int dot = 0; dot < BK; dot++) {

            // 从 sA 加载 TM 个元素到寄存器
            //   对应 sA 的第 dot 列中, 当前 thread 负责的 TM 行
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = sA[(thread_row * TM + i) * BK + dot];
            }
            // 从 sB 加载 TN 个元素到寄存器
            //   对应 sB 的第 dot 行中, 当前 thread 负责的 TN 列
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = sB[dot * BN + thread_col * TN + j];
            }

            // 外积累加: regC[i][j] += regA[i] * regB[j]
            //   这就是 GEMM 的核心 — rank-1 update
            //   TM × TN = 8 × 8 = 64 次 FMA, 全部在寄存器中完成
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    regC[i * TN + j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // ── Step 3: Register → Global (写回结果) ──
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            C[(thread_row * TM + i) * N + (thread_col * TN + j)] = regC[i * TN + j];
        }
    }
}

// ── CPU 参考 (只算前 verify_rows 行) ──
void gemm_cpu(const float* A, const float* B, float* C,
              int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    // ── 真实 MoE 参数 (DeepSeek-V2/V3 风格) ──
    constexpr int num_tokens = 4096;
    constexpr int hidden_size = 4096;
    constexpr int num_experts = 128;
    constexpr int top_k = 8;

    constexpr int M = num_tokens;     // 4096
    constexpr int K = hidden_size;    // 4096
    constexpr int N = num_experts;    // 128

    constexpr int warmup_iters = 10;
    constexpr int bench_iters = 50;
    constexpr int verify_rows = 32;

    printf("=== Tiled GEMM — MoE Gating ===\n");
    printf("C[%d,%d] = A[%d,%d] × B[%d,%d]\n", M, N, M, K, K, N);
    printf("num_tokens=%d  hidden=%d  experts=%d  top_k=%d\n",
           num_tokens, hidden_size, num_experts, top_k);
    printf("tiling: BM=%d BN=%d BK=%d TM=%d TN=%d  threads/block=%d\n",
           BM, BN, BK, TM, TN, (BM / TM) * (BN / TN));
    printf("FLOPs/call: %.2f GFLOP\n\n", 2.0 * M * N * K / 1e9);

    // ── Host 数据 ──
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_tiled(M * N), h_C_cublas(M * N);

    srand(42);
    auto randf = []() { return (rand() / (float)RAND_MAX - 0.5f) * 0.1f; };
    for (auto& v : h_A) v = randf();
    for (auto& v : h_B) v = randf();

    printf("Memory: A=%.1f MB  B=%.1f MB  C=%.1f MB\n\n",
           sizeA / 1e6, sizeB / 1e6, sizeC / 1e6);

    // ── Device 内存 ──
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));

    dim3 block(BM / TM * (BN / TN));  // 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ═══════════════════════════════════════════════════════════════
    // Benchmark 手写 tiled kernel
    // ═══════════════════════════════════════════════════════════════
    for (int i = 0; i < warmup_iters; i++) {
        sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_tiled = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tiled, start, stop));
    ms_tiled /= bench_iters;
    double gflops_tiled = 2.0 * M * N * K / (ms_tiled * 1e6);

    CHECK_CUDA(cudaMemcpy(h_C_tiled.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    // ═══════════════════════════════════════════════════════════════
    // Benchmark cuBLAS (对比)
    // ═══════════════════════════════════════════════════════════════
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmup_iters; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_cublas = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cublas, start, stop));
    ms_cublas /= bench_iters;
    double gflops_cublas = 2.0 * M * N * K / (ms_cublas * 1e6);

    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    // ═══════════════════════════════════════════════════════════════
    // 正确性: 手写 vs cuBLAS
    // ═══════════════════════════════════════════════════════════════
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_C_tiled[i] - h_C_cublas[i]);
        if (err > max_err) max_err = err;
    }

    // 额外: CPU 验证前 verify_rows 行
    std::vector<float> h_ref(verify_rows * N);
    gemm_cpu(h_A.data(), h_B.data(), h_ref.data(), verify_rows, N, K);
    float max_err_cpu = 0.0f;
    for (int i = 0; i < verify_rows * N; i++) {
        float err = fabsf(h_C_tiled[i] - h_ref[i]);
        if (err > max_err_cpu) max_err_cpu = err;
    }

    // ═══════════════════════════════════════════════════════════════
    // 输出
    // ═══════════════════════════════════════════════════════════════
    printf("── Performance (%d iters, %d warmup) ──\n", bench_iters, warmup_iters);
    printf("  tiled kernel: %7.3f ms  %8.2f GFLOPS\n", ms_tiled, gflops_tiled);
    printf("  cuBLAS:       %7.3f ms  %8.2f GFLOPS\n", ms_cublas, gflops_cublas);
    printf("  ratio:        %.1f%%\n\n", gflops_tiled / gflops_cublas * 100);

    printf("── Correctness ──\n");
    printf("  tiled vs cuBLAS (全部): max_err = %e  %s\n",
           max_err, max_err < 1e-3f ? "PASS" : "FAIL");
    printf("  tiled vs CPU (前%d行): max_err = %e  %s\n",
           verify_rows, max_err_cpu, max_err_cpu < 1e-3f ? "PASS" : "FAIL");

    printf("\nlogits 样例 (token 0, 前8个expert):\n");
    printf("  tiled:  [");
    for (int e = 0; e < 8; e++) printf("%7.4f%s", h_C_tiled[e], e < 7 ? "," : "");
    printf("]\n  cuBLAS: [");
    for (int e = 0; e < 8; e++) printf("%7.4f%s", h_C_cublas[e], e < 7 ? "," : "");
    printf("]\n");

    // ── 清理 ──
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
