// nvcc -arch=sm_80 -std=c++17 -O3 -I $HOME/cutlass/include gemm_cute_simple.cu -o gemm_cute_simple
//
// ============================================================
//  最简 CuTe SGEMM —— 学习 CuTe 抽象用
// ============================================================
//
//  目标: 用最少的概念把 CuTe 的 4 个核心抽象串起来跑通一次 SGEMM:
//    1. Layout / Tensor          —— shape+stride 的视图
//    2. local_tile               —— 把全局 tensor 切成 block tile
//    3. TiledCopy + partition_S/D—— gmem ↔ smem 的搬运分派
//    4. TiledMma  + partition_*  —— smem → reg 的 fragment 分派 + gemm()
//
//  刻意省略:
//    - TensorCore (用 UniversalFMA, 普通 FFMA, 一个线程一个 MMA)
//    - cp.async / 异步管线 / multi-stage / TMA
//    - swizzle (smem 没用 swizzle, 会有 bank conflict, 先不管)
//
//  矩阵约定 (CuTe 默认 col-major):
//    A: M × K, col-major, lda = M
//    B: K × N, col-major, ldb = K
//    C: M × N, col-major, ldc = M
//    C = A × B
//
//  依赖: CUTLASS headers 在 ~/cutlass/include/
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

using namespace cute;


// ============================================================
//  Kernel
// ============================================================
template <class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class TiledCopyA,
          class BStride, class BSmemLayout, class TiledCopyB,
          class CStride, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 float const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                 float const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                 float      * C, CStride dC, TiledMma mma)
{
    // ----- (1) 把裸指针 + shape + stride 包成 Tensor -----
    // 这就是 CuTe 的"全局视图", 还没切, 还没分派, 只是带了 layout 的指针
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    

    // Tensor g_A = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);



    // ----- (2) local_tile: 切 block tile -----
    // cta_tiler = (BLK_M, BLK_N, BLK_K)
    // cta_coord = (blockIdx.x, blockIdx.y, _)  最后一维的 _ 表示 K 方向不切, 留作 k-loop
    // Step<_1,X,_1> = "对 cta_tiler 的第 0 维 (BLK_M) 和第 2 维 (BLK_K) 起作用, 跳过第 1 维 (N)"
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M, BLK_K, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N, BLK_K, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M, BLK_N)


    


    // ----- (3) shared memory tensors -----
    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K)

    // ----- (4) TiledCopy: gmem → smem 的线程分派 -----
    // copy_a 是个 tiled-copy 描述子, 知道 256 个线程怎么瓜分一个 (BLK_M, BLK_K) tile
    // get_slice(tid) 返回"我这个线程负责哪一片", partition_S/D 把视图给我
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)  从 gmem 读
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)     往 smem 写

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

    // ----- (5) TiledMma: smem → reg fragment 的线程分派 -----
    // mma 知道一组线程怎么共同算一个 (BLK_M, BLK_N) 的 MMA
    // partition_A/B/C 给我"做 MMA 时这线程负责哪几个元素"的 view
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA, MMA_N, MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    // 寄存器版的 fragment, 形状和上面对应的 partition 一致
    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // 寄存器 A 块
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // 寄存器 B 块
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // 寄存器累加器

    clear(tCrC); // C 累加器清零

    // ----- (6) K-loop: 每次拿一个 BLK_K 切片做一次 mainloop -----
    auto K_TILE_MAX = size<3>(tAgA); // 对应 local_tile 出来的最后一维 'k'
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        // gmem → smem (整个 block 协同搬一个 BLK_M*BLK_K 和 BLK_N*BLK_K)
        copy(copy_a, tAgA(_,_,_,k_tile), tAsA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBsB);
        __syncthreads();

        // smem → reg (TiledMma 的 partition 已经决定每个线程读哪几个元素)
        copy(tCsA, tCrA);
        copy(tCsB, tCrB);

        // reg-level MMA: tCrC += tCrA * tCrB
        gemm(mma, tCrA, tCrB, tCrC);

        __syncthreads();
    }

    // ----- (7) Epilogue: 累加器写回 gmem -----
    copy(tCrC, tCgC);
}





// ============================================================
//  Host 侧: 把 layout / tile / copy / mma 配好后调 kernel
// ============================================================
void gemm_cute(int M, int N, int K,
               float const* A, float const* B, float* C,
               cudaStream_t stream = 0)
{
    // ----- problem shape & strides (col-major) -----
    auto prob_shape = make_shape(M, N, K);

    // A: (M,K) col-major → 内存里 (1, M)
    // B: (N,K) col-major (B 看成 K×N, 但视为 (N,K) 后 stride 变成 (K, 1)? 不对)
    //
    // 这里用 CuTe tutorial 的 "TN" 风格, 但为了最简, A/B 都按 col-major 处理:
    //   A: (M,K), stride = (1, M)
    //   B: 物理上 K×N col-major, 我们当 (N,K) 看, stride = (M? 不) ...
    //
    // 简化做法: 让 A 和 B 都是 col-major 的 (rows, cols):
    //   mA shape=(M,K) stride=(1,M)  ← 内存里就是 col-major M×K
    //   mB shape=(N,K) stride=(K,1)? 不对, 这样不是 col-major
    //
    // 最干净的是: B 物理上是 K×N col-major (lda=K), 我们看作 (N,K) 视图, stride=(K, 1)
    auto dA = make_stride(Int<1>{}, M);   // A col-major,  (M,K), stride=(1, M)
    auto dB = make_stride(K,        Int<1>{}); // B 物理 K×N col-major, 看作 (N,K), stride=(K, 1)
    auto dC = make_stride(Int<1>{}, M);   // C col-major, (M,N), stride=(1, M)

    // ----- block tile -----
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<  8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    // ----- smem layouts (无 swizzle, 朴素 col-major) -----
    auto sA_layout = make_layout(make_shape(bM, bK)); // (128,8) col-major default
    auto sB_layout = make_layout(make_shape(bN, bK));

    // ----- TiledCopy: 256 线程协同搬一个 (128×8 = 1024) 元素的 tile -----
    // 每线程搬 4 个 float, 用 128-bit 向量化 LD/ST
    // 线程在 (M,K) 维度的排布: 32×8 (M 方向 32 线程, K 方向 8 线程) → 256 线程
    using CopyOpA = SM80_CP_ASYNC_CACHEALWAYS<uint128_t>; // 不一定要 cp.async, 用普通的也行
    TiledCopy copy_a = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{},
        make_layout(make_shape(Int<32>{}, Int<8>{}),  // 线程排布 (T_M, T_K) = 32×8
                    make_stride(Int<8>{}, Int<1>{})), // K 方向连续, M 方向跳 8
        make_layout(make_shape(Int<4>{}, Int<1>{}))); // 每线程 4 个元素 (在 M 维向量化)

    TiledCopy copy_b = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{},
        make_layout(make_shape(Int<32>{}, Int<8>{}),
                    make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<4>{}, Int<1>{})));

    // ----- TiledMma: 用 UniversalFMA, 一个线程做一次 1×1×1 的 MMA, 共 256 线程 -----
    // 排布成 16×16 = 256 个线程, 每个负责 C 的一个元素 (M 方向 16 线程 × N 方向 16 线程)
    // 注意: 一个 block tile 是 128×128, 256 线程 → 每线程负责 8×8 = 64 个 C 元素
    //   这通过 MMA atom 的 layout-tv 自动展开
    TiledMMA mma = make_tiled_mma(
        UniversalFMA<float, float, float, float>{},
        make_layout(make_shape(Int<16>{}, Int<16>{})), // 256 线程的二维排布
        Tile<Int<128>, Int<128>, Int<8>>{});          // 这组线程要覆盖的 (M,N,K) tile

    // ----- launch -----
    dim3 dimBlock(size(mma));                  // 256
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)));

    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA_layout, copy_a,
        B, dB, sB_layout, copy_b,
        C, dC, mma);
}


// ============================================================
//  CPU reference + verify + 简单 timing
// ============================================================
static void cpu_gemm(int M, int N, int K, float const* A, float const* B, float* C)
{
    // A: (M,K) col-major,  A[m + k*M]
    // B: (K,N) col-major,  B[k + n*K]
    // C: (M,N) col-major,  C[m + n*M]
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[m + k * M] * B[k + n * K];
            }
            C[m + n * M] = acc;
        }
    }
}

int main()
{
    int M = 256, N = 256, K = 64;

    size_t szA = size_t(M) * K, szB = size_t(K) * N, szC = size_t(M) * N;

    float *hA = (float*)malloc(szA * sizeof(float));
    float *hB = (float*)malloc(szB * sizeof(float));
    float *hC = (float*)malloc(szC * sizeof(float));
    float *hC_ref = (float*)malloc(szC * sizeof(float));

    srand(0);
    for (size_t i = 0; i < szA; ++i) hA[i] = (rand() % 100) / 100.f;
    for (size_t i = 0; i < szB; ++i) hB[i] = (rand() % 100) / 100.f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA * sizeof(float));
    cudaMalloc(&dB, szB * sizeof(float));
    cudaMalloc(&dC, szC * sizeof(float));
    cudaMemcpy(dA, hA, szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB * sizeof(float), cudaMemcpyHostToDevice);

    gemm_cute(M, N, K, dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, szC * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_gemm(M, N, K, hA, hB, hC_ref);

    double max_err = 0.0;
    for (size_t i = 0; i < szC; ++i) {
        double e = fabs(double(hC[i]) - double(hC_ref[i]));
        if (e > max_err) max_err = e;
    }
    printf("M=%d N=%d K=%d  max abs err = %.6f  %s\n",
           M, N, K, max_err, max_err < 1e-2 ? "OK" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}
