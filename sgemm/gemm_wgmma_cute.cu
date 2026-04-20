// nvcc -arch=sm_90a -std=c++17 -O3 -I $HOME/cutlass/include gemm_wgmma_cute.cu -o gemm_wgmma_cute
//
// ============================================================
//  SGEMM with wgmma + swizzled smem, via CUTLASS CuTe
// ============================================================
//
//  走法 B: 换上 Hopper 原生路径 —— wgmma SS 操作数都在 smem, 配
//          Layout_K_SW128_Atom 的 XOR swizzle, 彻底绕开 wmma 的 LDS bank conflict.
//
//  本版改动:
//    1. 不再用 wmma, 改用 cute 包的 SM90_64x64x8_F32TF32TF32_SS (wgmma TF32)
//    2. smem 布局不再是 "自然 row-major", 而是 GMMA::Layout_K_SW128_Atom,
//       物理上按 128B XOR 打散, bank conflict 硬件免费消除
//    3. compute 走 warpgroup_arrive / warpgroup_wait 异步同步模型,
//       不再是 wmma 的 synchronous mma.sync
//    4. 一个 block = 1 个 warp group = 128 threads (不是之前 256)
//    5. 用 cp.async 做 gmem→smem (TMA 留到下一版), 3-stage pipeline
//    6. TN 风格 (A row-major, B row-major viewed as (N,K)):
//       kernel 计算 C[m,n] = Σ_k A[m,k] * B[n,k]
//       与我们标准 C = A @ B (B 是 K×N) 等价: 把 B 当作 (N,K) 查看即可
//
//  测试策略:
//    - 主 harness 与之前版本一致 (M_list, N_list, 10次平均取 Gflops)
//    - cpuSgemm 的参考也用 "C = A × B^T, B is (N,K)" 变体,
//      避免在 benchmark 里多做一次 CPU 转置
//
//  依赖: CUTLASS headers 在 ~/cutlass/include/
// ============================================================

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using TA = cutlass::tfloat32_t;   // wgmma atom 要求的元素类型, bit-wise 和 float 一致
using TB = cutlass::tfloat32_t;
using TC = float;

// -------- shared memory storage --------
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};


// ============================================================
//  Kernel (来自 wgmma_sm90.cu 的 gemm_device, 只换类型与 MMA atom)
// ============================================================
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void wgmma_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC      * C, CStride dC, TiledMma mma)
{
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

    extern __shared__ char shared_memory[];
    using SStor = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SStor& smem = *reinterpret_cast<SStor*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

    // Copy partitioning
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor sA_  = as_position_independent_swizzle_tensor(sA);
    Tensor tAsA = thr_copy_a.partition_D(sA_);

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor sB_  = as_position_independent_swizzle_tensor(sB);
    Tensor tBsB = thr_copy_b.partition_D(sB_);

    // MMA partitioning
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_PIPE_MAX = size<3>(tAsA);

    // Prologue: 预取 K_PIPE_MAX-1 块
    CUTE_UNROLL
    for (int k = 0; k < K_PIPE_MAX - 1; ++k) {
        copy(copy_a, tAgA(_,_,_,k), tAsA(_,_,_,k));
        copy(copy_b, tBgB(_,_,_,k), tBsB(_,_,_,k));
        cp_async_fence();
    }
    __syncthreads();

    int k_pipe_read  = 0;
    int k_pipe_write = K_PIPE_MAX - 1;

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        int k_tile_next = k_tile + (K_PIPE_MAX - 1);
        k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX - 1 : k_tile_next;

        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe_write));
        cp_async_fence();

        ++k_pipe_write;
        k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

        cp_async_wait<0>();

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(mma, tCrA(_,_,_,k_pipe_read), tCrB(_,_,_,k_pipe_read), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        ++k_pipe_read;
        k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
    }

    // Epilogue: 写回 C
    copy(tCrC, tCgC);
}


// ============================================================
//  Host launcher
// ============================================================
static void launch_wgmma(TA* d_a, TB* d_b, TC* d_c,
                         int M, int N, int K, cudaStream_t stream = 0)
{
    auto prob = make_shape(M, N, K);

    // TN strides: A row-major (K 快), B 读作 (N,K) 其实 physical 是 (K,N)
    // make_stride(K, Int<1>) 表示 (m-stride=K, k-stride=1), 匹配 A row-major
    // make_stride(K, Int<1>) 在 (N,K) 视角下表示 (n-stride=K, k-stride=1),
    //   实际物理是 K×N row-major, 但这时 B[n][k] = B_ptr[n*K+k]
    //   要求 N×K row-major 布局. 我们 testError 里 h_b 按 (N,K) row-major 填,
    //   cpuSgemm 也按此语义 (A @ B^T) 计算.
    auto dA = make_stride(K, Int<1>{});
    auto dB = make_stride(K, Int<1>{});
    auto dC = make_stride(N, Int<1>{});

    // Tile shapes
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<32>{};              // 32 floats = 128B, SW128 swizzle 的最小 K
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<2>{};               // 2-stage pipeline

    // smem layouts: K-major + SW128 XOR swizzle (GMMA atom)
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    // gmem→smem 用 cp.async.cachealways, 16B 每次 (4 个 TF32 / float)
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
        Layout<Shape<_16,_8>, Stride<_8,_1>>{},
        Layout<Shape< _1,_4>>{});
    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
        Layout<Shape<_16,_8>, Stride<_8,_1>>{},
        Layout<Shape< _1,_4>>{});

    // wgmma atom: SS variant (A, B 都在 smem), TF32 m64n64k8
    // CUTLASS TF32 的 SS atom 名字已固定 layout 为 _TN (A/B 都 K-major)
    TiledMMA tiled_mma = make_tiled_mma(
        SM90_64x64x8_F32TF32TF32_SS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});

    int smem_size = sizeof(SharedStorage<float, float, decltype(sA), decltype(sB)>);

    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid((M + int(bM) - 1) / int(bM),
                 (N + int(bN) - 1) / int(bN));

    auto* kptr = &wgmma_kernel<decltype(prob), decltype(cta_tiler),
                                TA, decltype(dA), decltype(sA), decltype(copyA),
                                TB, decltype(dB), decltype(sB), decltype(copyB),
                                TC, decltype(dC), decltype(tiled_mma)>;

    cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kptr<<<dimGrid, dimBlock, smem_size, stream>>>(
        prob, cta_tiler,
        d_a, dA, sA, copyA,
        d_b, dB, sB, copyB,
        d_c, dC, tiled_mma);
}


// ============================================================
//  Reference: C[m,n] = Σ_k A[m,k] * B[n,k]   (B 存为 N×K row-major)
// ============================================================
static void cpuSgemm_TN(float* a, float* b, float* c,
                        int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0f;
            for (int k = 0; k < K; k++)
                psum += a[m * K + k] * b[n * K + k];    // B 按 (N,K) 读
            c[m * N + n] = psum;
        }
    }
}


static float testError(int M, int N, int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = N * K * sizeof(float);                // B 按 (N,K) 布局
    size_t size_c = M * N * sizeof(float);

    float *h_a=(float*)malloc(size_a), *h_b=(float*)malloc(size_b);
    float *h_c=(float*)malloc(size_c), *h_d_c=(float*)malloc(size_c);
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size_a); cudaMalloc(&d_b,size_b); cudaMalloc(&d_c,size_c);
    srand(time(0));
    for (int i = 0; i < M*K; i++) h_a[i] = rand()/float(RAND_MAX);
    for (int i = 0; i < N*K; i++) h_b[i] = rand()/float(RAND_MAX);
    cudaMemset(d_c, 0, size_c);
    cpuSgemm_TN(h_a, h_b, h_c, M, N, K);
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    launch_wgmma(reinterpret_cast<TA*>(d_a), reinterpret_cast<TB*>(d_b), d_c, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
        return -1.f;
    }
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(h_d_c[i] - h_c[i]);
        if (e != e || max_error != max_error) max_error = -NAN;
        else max_error = fmaxf(max_error, e);
    }
    free(h_a); free(h_b); free(h_c); free(h_d_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return max_error;
}

static float testPerformance(int M, int N, int K, int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = N * K * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a, size_a); cudaMalloc(&d_b, size_b); cudaMalloc(&d_c, size_c);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < repeat; i++)
        launch_wgmma(reinterpret_cast<TA*>(d_a), reinterpret_cast<TB*>(d_b), d_c, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return ms / 1000.0 / repeat;
}


int main(void) {
    printf("\nKernel = sgemm_wgmma_cute (TF32, wgmma SS, SW128 swizzle)\n");
    const int outer_repeat = 10, inner_repeat = 1;
    {
        const int M = 512, N = 512, K = 512;
        float me = testError(M, N, K);
        printf("Max Error = %f  (TF32 精度, ~1e-2 级别正常)\n", me);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024};

    for (int i = 0; i < 15; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        double max_sec = 0.0, min_sec = DBL_MAX, total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++) {
            double t = testPerformance(M, N, K, inner_repeat);
            max_sec = fmax(max_sec, t); min_sec = fmin(min_sec, t); total_sec += t;
        }
        double avg = total_sec / outer_repeat;
        double g = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG = %10.4lf Gflops\n",
               M, N, K, min_sec, avg, max_sec, g);
    }
    return 0;
}
