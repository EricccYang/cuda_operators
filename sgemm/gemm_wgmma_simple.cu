// nvcc -arch=sm_90a -std=c++17 -O3 -I $HOME/cutlass/include \
//   gemm_wgmma_simple.cu -o gemm_wgmma_simple
//
// A deliberately small WGMMA example for reading the SM90/CuTe semantics.
// It computes C = A @ B^T, where:
//   A is stored as row-major (M, K)
//   B is stored as row-major (N, K)
//   C is stored as row-major (M, N)
//
// One CTA is one warp-group: 128 threads.
// One CTA computes one 64x64 output tile.

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cutlass/numeric_types.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cute;

using TA = cutlass::tfloat32_t;
using TB = cutlass::tfloat32_t;
using TC = float;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class TiledCopyA,
          class BStride, class BSmemLayout, class TiledCopyB,
          class CStride, class TiledMma>
__global__ __launch_bounds__(128)
void wgmma_simple_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC* C, CStride dC, TiledMma mma)
{
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K), i.e. B^T view
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using SStor = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SStor& smem = *reinterpret_cast<SStor*>(shared_memory);

    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    Tensor sA_swizzled = as_position_independent_swizzle_tensor(sA);
    Tensor sB_swizzled = as_position_independent_swizzle_tensor(sB);
    Tensor tAsA = thr_copy_a.partition_D(sA_swizzled);
    Tensor tBsB = thr_copy_b.partition_D(sB_swizzled);

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    int k_tiles = size<3>(tAgA);

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        // global -> shared. This is cp.async under the CuTe copy atom.
        copy(copy_a, tAgA(_, _, _, k_tile), tAsA);
        copy(copy_b, tBgB(_, _, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // shared -> WGMMA -> accumulator registers.
        // The atom is m64n64k8, while the CTA K tile is 32, so CuTe emits
        // the required K inner steps for this shared-memory tile.
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        __syncthreads();
    }

    copy(tCrC, tCgC);
}

static void launch_wgmma_simple(TA* d_a, TB* d_b, TC* d_c,
                                int M, int N, int K, cudaStream_t stream = 0)
{
    auto problem = make_shape(M, N, K);

    // A(m,k) and B(n,k) are both row-major in their logical tensors.
    auto dA = make_stride(K, Int<1>{});
    auto dB = make_stride(K, Int<1>{});
    auto dC = make_stride(N, Int<1>{});

    auto bM = Int<64>{};
    auto bN = Int<64>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK));

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _4>>{});
    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _4>>{});

    TiledMMA mma = make_tiled_mma(
        SM90_64x64x8_F32TF32TF32_SS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});

    auto* kernel = &wgmma_simple_kernel<
        decltype(problem), decltype(cta_tiler),
        decltype(dA), decltype(sA), decltype(copyA),
        decltype(dB), decltype(sB), decltype(copyB),
        decltype(dC), decltype(mma)>;

    int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 block(size(mma)); // 128 threads: one warp-group
    dim3 grid((M + int(bM) - 1) / int(bM),
              (N + int(bN) - 1) / int(bN));

    kernel<<<grid, block, smem_size, stream>>>(
        problem, cta_tiler,
        d_a, dA, sA, copyA,
        d_b, dB, sB, copyB,
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

    launch_wgmma_simple(reinterpret_cast<TA*>(d_a),
                        reinterpret_cast<TB*>(d_b),
                        d_c, M, N, K);

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

    printf("simple WGMMA max error = %f (TF32 path)\n", max_error);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_ref);
    free(h_out);
    return 0;
}
