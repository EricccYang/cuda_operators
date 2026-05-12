// nvcc -arch=sm_90a -std=c++17 -O3 -I $HOME/cutlass/include \
//   gemm_wgmma_tma_simple.cu -o gemm_wgmma_tma_simple
//
// A compact "semantics first" Hopper GEMM:
//
//   host builds CUtensorMap descriptors
//   one CTA / warp-group issues TMA global->shared copies
//   shared memory uses GMMA's 128B swizzled layout
//   cute::gemm emits WGMMA from shared memory
//
// It computes C = A @ B^T, where:
//   A is row-major (M, K)
//   B is row-major (N, K)
//   C is row-major (M, N)

#include <cuda.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cutlass/numeric_types.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cute;
namespace cde = cuda::device::experimental;

using TA = cutlass::tfloat32_t;
using TB = cutlass::tfloat32_t;
using TC = float;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
    alignas(8) barrier_t tma_barrier;
};

template <class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout,
          class BStride, class BSmemLayout,
          class CStride, class TiledMma>
__global__ __launch_bounds__(128)
void wgmma_tma_simple_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    AStride dA, ASmemLayout sA_layout,
    BStride dB, BSmemLayout sB_layout,
    TC* C, CStride dC, TiledMma mma)
{
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using SStor = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SStor& smem = *reinterpret_cast<SStor*>(shared_memory);

    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BM,BK)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BN,BK), B^T view

    if (threadIdx.x == 0) {
        init(&smem.tma_barrier, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    int k_tiles = ceil_div(get<2>(shape_MNK), BK);
    uint32_t phase = 0;
    constexpr uint32_t tx_bytes = (BM * BK + BN * BK) * sizeof(float);

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        // One thread issues two bulk tensor copies. The mbarrier tracks both
        // TMA completion and the arrival of all CTA threads.
        if (threadIdx.x == 0) {
            (void)cuda::device::barrier_arrive_tx(smem.tma_barrier, 1, tx_bytes);

            // A tensor map uses (x,y) = (K,M), so this loads A[M tile, K tile].
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                smem.A.begin(), &tma_a,
                k_tile * BK, blockIdx.x * BM,
                smem.tma_barrier);

            // B is viewed as (N,K), so (x,y) = (K,N).
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                smem.B.begin(), &tma_b,
                k_tile * BK, blockIdx.y * BN,
                smem.tma_barrier);
        } else {
            (void)smem.tma_barrier.arrive();
        }

        smem.tma_barrier.wait_parity(phase & 1);
        ++phase;

        // The SM90_64x64x8 atom consumes the 64x32 shared tile as four K=8
        // WGMMA steps. CuTe owns that decomposition.
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        // Keep the next TMA write from racing with any late shared-memory read.
        __syncthreads();
    }

    copy(tCrC, tCgC);
}

static void setup_tma_a(CUtensorMap* tma, float* d_a, int M, int K)
{
    // A is row-major (M,K). TMA coordinates are (x,y) = (K,M).
    cuuint64_t global_dim[2] = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t global_stride[1] = {(cuuint64_t)K * sizeof(float)};
    cuuint32_t box_dim[2] = {(cuuint32_t)BK, (cuuint32_t)BM};
    cuuint32_t elem_stride[2] = {1, 1};

    CUresult r = cuTensorMapEncodeTiled(
        tma,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        d_a,
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled(A) failed: %d\n", r);
        exit(1);
    }
}

static void setup_tma_b(CUtensorMap* tma, float* d_b, int N, int K)
{
    // B is row-major (N,K). TMA coordinates are (x,y) = (K,N).
    cuuint64_t global_dim[2] = {(cuuint64_t)K, (cuuint64_t)N};
    cuuint64_t global_stride[1] = {(cuuint64_t)K * sizeof(float)};
    cuuint32_t box_dim[2] = {(cuuint32_t)BK, (cuuint32_t)BN};
    cuuint32_t elem_stride[2] = {1, 1};

    CUresult r = cuTensorMapEncodeTiled(
        tma,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        d_b,
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled(B) failed: %d\n", r);
        exit(1);
    }
}

static void launch_wgmma_tma_simple(float* d_a, float* d_b, float* d_c,
                                    int M, int N, int K, cudaStream_t stream = 0)
{
    CUtensorMap tma_a;
    CUtensorMap tma_b;
    setup_tma_a(&tma_a, d_a, M, K);
    setup_tma_b(&tma_b, d_b, N, K);

    auto problem = make_shape(M, N, K);
    auto dA = make_stride(K, Int<1>{});
    auto dB = make_stride(K, Int<1>{});
    auto dC = make_stride(N, Int<1>{});

    auto bM = Int<BM>{};
    auto bN = Int<BN>{};
    auto bK = Int<BK>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK));

    TiledMMA mma = make_tiled_mma(
        SM90_64x64x8_F32TF32TF32_SS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});

    auto* kernel = &wgmma_tma_simple_kernel<
        decltype(problem), decltype(cta_tiler),
        decltype(dA), decltype(sA),
        decltype(dB), decltype(sB),
        decltype(dC), decltype(mma)>;

    int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 block(size(mma)); // 128 threads: one warp-group
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    kernel<<<grid, block, smem_size, stream>>>(
        problem, cta_tiler,
        tma_a, tma_b,
        dA, sA,
        dB, sB,
        d_c, dC, mma);
}

static void cpu_gemm_tn(float* A, float* B, float* C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

int main()
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 256;

    size_t bytes_a = M * K * sizeof(float);
    size_t bytes_b = N * K * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);

    float *h_a = (float*)malloc(bytes_a);
    float *h_b = (float*)malloc(bytes_b);
    float *h_ref = (float*)malloc(bytes_c);
    float *h_out = (float*)malloc(bytes_c);

    for (int i = 0; i < M * K; ++i) h_a[i] = (i % 17) * 0.01f;
    for (int i = 0; i < N * K; ++i) h_b[i] = (i % 13) * 0.02f;

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, bytes_c);

    launch_wgmma_tma_simple(d_a, d_b, d_c, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_out, d_c, bytes_c, cudaMemcpyDeviceToHost);
    cpu_gemm_tn(h_a, h_b, h_ref, M, N, K);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = fmaxf(max_error, fabsf(h_out[i] - h_ref[i]));
    }

    printf("simple WGMMA+TMA max error = %f (TF32 path)\n", max_error);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_ref);
    free(h_out);
    return 0;
}
