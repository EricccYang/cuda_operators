// nvcc -arch=sm_90 -std=c++17 gemm_tma_wmma.cu -o gemm_tma_wmma
//
// ============================================================
//  SGEMM with TMA + wmma  (sm_90 only)
// ============================================================
//
//  相比 gemm_tc_wmma.cu 的本质差别:
//
//  1. **gmem→smem 用 TMA** (cp.async.bulk.tensor), 不再用 cp.async
//     - 一条 TMA 指令搬整个 128×32 的 A tile (原来 256 thread 每人发 4 条 cp.async)
//     - TMA 地址计算/对齐/out-of-bounds 全走硬件
//     - 不占 warp issue slot (走独立 Copy Engine)
//
//  2. **BK 从 8 扩到 32**: TMA 的规模优势在大 tile 上才明显
//     - 每 bk 迭代做 4 个 MMA k-slice (32/8) × 8 MMAs/warp = 32 MMAs/warp, ILP 深一些
//     - 也能多摊点 TMA 的启动开销
//
//  3. **STAGES = 2**: TMA + mbarrier 的 async 已经够强, 双 buffer 足矣
//     - 再多 stage smem 会溢出 (BM×BK=128×32 每 stage 16KB, 双 stage A+B = 64KB)
//
//  4. **同步机制用 cuda::barrier (mbarrier)**:
//     - TMA 完成后硬件自动在 mbarrier 上计 tx_count
//     - 消费端 `bar.wait(phase)` 等 tx 齐了才放行
//     - 比 cp.async 的 commit_group / wait_group 更整齐
//
//  compute 侧完全沿用 gemm_tc_wmma.cu 的 wmma m16n16k8 TF32, 只是内层多了
//  k_slice loop (每 bk 做 4 个 slice).
//
//  Host 侧新增:
//    - `setup_tma_a/b` 构造 CUtensorMap (描述 tensor 的形状、步长、box 大小)
//    - 通过 __grid_constant__ 把 CUtensorMap 传进 kernel
// ============================================================

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>

using namespace nvcuda;
namespace cde = cuda::device::experimental;

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// -------- Tile constants (shared by host TMA setup and kernel) --------
static const int BM     = 128;
static const int BN     = 128;
static const int BK     = 16;         // 32 会让 2-stage smem 超过 48KB 的默认上限
static const int STAGES = 2;          // 2×(128×16 + 16×128)×4B = 32KB, 够用

// -------- CPU reference --------
void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float psum = 0.0f;
            for (int k = 0; k < K; k++) psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            c[OFFSET(m, n, N)] = psum;
        }
}


// ============================================================
//  Kernel
// ============================================================
__global__ __launch_bounds__(256)
void sgemm_tma_wmma(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    float * __restrict__ c,
    const int M, const int N, const int K) {

    const int MMA_M = 16, MMA_N = 16, MMA_K = 8;
    const int K_SLICES = BK / MMA_K;               // 4

    const int WARP_ROWS = 2;
    const int WARP_COLS = 4;
    const int WM = BM / WARP_ROWS;                 // 64
    const int WN = BN / WARP_COLS;                 // 32
    const int MCNT = WM / MMA_M;                   // 4
    const int NCNT = WN / MMA_N;                   // 2

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int warp_m = warp_id / WARP_COLS;
    int warp_n = warp_id % WARP_COLS;

    // -------- shared memory, 128B 对齐是 TMA 的硬性要求 --------
    __shared__ alignas(128) float s_a[STAGES][BM][BK];
    __shared__ alignas(128) float s_b[STAGES][BK][BN];
    __shared__ alignas(8)   barrier_t bar[STAGES];

    // 每 stage TMA 要搬的字节数 (A + B)
    const uint32_t tx_bytes = BM * BK * 4 + BK * BN * 4;

    // 初始化 mbarrier
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) init(&bar[s], blockDim.x);
        cde::fence_proxy_async_shared_cta();       // 保证 init 对 TMA 可见
    }
    __syncthreads();

    // 累加器 fragment
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_frag[MCNT][NCNT];
    #pragma unroll
    for (int i = 0; i < MCNT; i++)
        #pragma unroll
        for (int j = 0; j < NCNT; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int step = (K + BK - 1) / BK;

    // mbarrier 的 phase parity, 每 stage 各自一份
    uint32_t phase0 = 0, phase1 = 0;

    // -------- Prologue: 发射 stage 0 的 TMA --------
    {
        if (tid == 0) {
            (void)cuda::device::barrier_arrive_tx(bar[0], 1, tx_bytes);
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &s_a[0][0][0], &tma_a,
                /*x=*/0,                             /*y=*/blockIdx.y * BM,
                bar[0]);
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &s_b[0][0][0], &tma_b,
                /*x=*/blockIdx.x * BN,               /*y=*/0,
                bar[0]);
        } else {
            (void)bar[0].arrive();
        }
    }

    // -------- Main loop --------
    for (int bk = 0; bk < step; bk++) {
        int compute_stage = bk & 1;
        int load_stage    = (bk + 1) & 1;

        // 等当前 stage 数据就绪 (TMA 完成 + 所有 thread arrive)
        if (compute_stage == 0) { bar[0].wait_parity(phase0 & 1); phase0++; }
        else                    { bar[1].wait_parity(phase1 & 1); phase1++; }

        // 发射 bk+1 的 TMA (非阻塞)
        if (bk + 1 < step) {
            if (tid == 0) {
                cuda::device::barrier_arrive_tx(bar[load_stage], 1, tx_bytes);
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &s_a[load_stage][0][0], &tma_a,
                    (bk + 1) * BK,                    blockIdx.y * BM,
                    bar[load_stage]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &s_b[load_stage][0][0], &tma_b,
                    blockIdx.x * BN,                  (bk + 1) * BK,
                    bar[load_stage]);
            } else {
                (void)bar[load_stage].arrive();
            }
        }

        // -------- 在 compute_stage 上做 K_SLICES = 4 个 MMA k-slice --------
        #pragma unroll
        for (int k_slice = 0; k_slice < K_SLICES; k_slice++) {
            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K,
                           wmma::precision::tf32, wmma::row_major> a_frag[MCNT];
            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K,
                           wmma::precision::tf32, wmma::row_major> b_frag[NCNT];

            #pragma unroll
            for (int i = 0; i < MCNT; i++) {
                int m_off = warp_m * WM + i * MMA_M;
                int k_off = k_slice * MMA_K;
                wmma::load_matrix_sync(a_frag[i],
                                       &s_a[compute_stage][m_off][k_off],
                                       BK);
            }
            #pragma unroll
            for (int j = 0; j < NCNT; j++) {
                int n_off = warp_n * WN + j * MMA_N;
                int k_off = k_slice * MMA_K;
                wmma::load_matrix_sync(b_frag[j],
                                       &s_b[compute_stage][k_off][n_off],
                                       BN);
            }

            // float → tf32 舍入
            #pragma unroll
            for (int i = 0; i < MCNT; i++) {
                #pragma unroll
                for (int t = 0; t < a_frag[i].num_elements; t++)
                    a_frag[i].x[t] = wmma::__float_to_tf32(a_frag[i].x[t]);
            }
            #pragma unroll
            for (int j = 0; j < NCNT; j++) {
                #pragma unroll
                for (int t = 0; t < b_frag[j].num_elements; t++)
                    b_frag[j].x[t] = wmma::__float_to_tf32(b_frag[j].x[t]);
            }

            #pragma unroll
            for (int i = 0; i < MCNT; i++)
                #pragma unroll
                for (int j = 0; j < NCNT; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }

        // 保证所有 thread 读完 compute_stage 再让下一轮 (thread 0) 往这块 stage
        // 发 TMA. 不加这行会和上面 "issue next TMA (line 148)" 产生跨迭代写读竞争.
        __syncthreads();
    }

    // -------- Store C --------
    #pragma unroll
    for (int i = 0; i < MCNT; i++) {
        #pragma unroll
        for (int j = 0; j < NCNT; j++) {
            int m_off = blockIdx.y * BM + warp_m * WM + i * MMA_M;
            int n_off = blockIdx.x * BN + warp_n * WN + j * MMA_N;
            wmma::store_matrix_sync(&c[m_off * N + n_off], c_frag[i][j],
                                    N, wmma::mem_row_major);
        }
    }
}


// ============================================================
//  Host: 构造 CUtensorMap
//  TMA 把 gmem 的 2D tile 描述成 "rank-2 tensor + box size + stride",
//  之后 device 端引用 tma_map 就能一发 TMA 搬一个 box.
// ============================================================
static void setup_tma_a(CUtensorMap* tma, float* d_a, int M, int K) {
    // A 是 row-major, K 快. 对 TMA 而言 (x, y) = (K 方向, M 方向).
    cuuint64_t globalDim[2]    = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t globalStride[1] = {(cuuint64_t)K * sizeof(float)};   // 只有 (rank-1) 个 stride
    cuuint32_t boxDim[2]       = {(cuuint32_t)BK, (cuuint32_t)BM};
    cuuint32_t elemStride[2]   = {1, 1};

    CUresult r = cuTensorMapEncodeTiled(
        tma,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        /*rank=*/2,
        d_a,
        globalDim,
        globalStride,
        boxDim,
        elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,          // 先不加 swizzle, 看看裸的效果
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled(A) failed: %d\n", r);
        exit(1);
    }
}

static void setup_tma_b(CUtensorMap* tma, float* d_b, int K, int N) {
    // B 是 row-major, N 快. (x, y) = (N 方向, K 方向).
    cuuint64_t globalDim[2]    = {(cuuint64_t)N, (cuuint64_t)K};
    cuuint64_t globalStride[1] = {(cuuint64_t)N * sizeof(float)};
    cuuint32_t boxDim[2]       = {(cuuint32_t)BN, (cuuint32_t)BK};
    cuuint32_t elemStride[2]   = {1, 1};

    CUresult r = cuTensorMapEncodeTiled(
        tma,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        /*rank=*/2,
        d_b,
        globalDim,
        globalStride,
        boxDim,
        elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled(B) failed: %d\n", r);
        exit(1);
    }
}


// ============================================================
//  Host launcher (测试 + benchmark 走这个)
// ============================================================
static float testError(int M, int N, int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *h_a=(float*)malloc(size_a),*h_b=(float*)malloc(size_b);
    float *h_c=(float*)malloc(size_c),*h_d_c=(float*)malloc(size_c);
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size_a); cudaMalloc(&d_b,size_b); cudaMalloc(&d_c,size_c);
    srand(time(0));
    for(int i=0;i<M*K;i++) h_a[i]=rand()/float(RAND_MAX);
    for(int i=0;i<K*N;i++) h_b[i]=rand()/float(RAND_MAX);
    cudaMemset(d_c,0,size_c);
    cpuSgemm(h_a,h_b,h_c,M,N,K);
    cudaMemcpy(d_a,h_a,size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_b,cudaMemcpyHostToDevice);

    CUtensorMap tma_a, tma_b;
    setup_tma_a(&tma_a, d_a, M, K);
    setup_tma_b(&tma_b, d_b, K, N);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(256);
    sgemm_tma_wmma<<<grid, block>>>(tma_a, tma_b, d_c, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }
    cudaMemcpy(h_d_c,d_c,size_c,cudaMemcpyDeviceToHost);

    float max_error=0.0f;
    for(int i=0;i<M*N;i++){
        float e=fabsf(h_d_c[i]-h_c[i]);
        if(e!=e||max_error!=max_error) max_error=-NAN;
        else max_error=max(max_error,e);
    }
    free(h_a);free(h_b);free(h_c);free(h_d_c);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    return max_error;
}

static float testPerformance(int M, int N, int K, int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size_a); cudaMalloc(&d_b,size_b); cudaMalloc(&d_c,size_c);

    CUtensorMap tma_a, tma_b;
    setup_tma_a(&tma_a, d_a, M, K);
    setup_tma_b(&tma_b, d_b, K, N);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(256);

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<repeat;i++)
        sgemm_tma_wmma<<<grid, block>>>(tma_a, tma_b, d_c, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    return ms/1000.0/repeat;
}


int main(void) {
    printf("\nKernel = sgemm_tma_wmma (TF32, TMA, sm_90)\n");
    const int outer_repeat = 10, inner_repeat = 1;

    {
        const int M = 512, N = 512, K = 512;
        float me = testError(M, N, K);
        printf("Max Error = %f  (TF32 精度, ~1e-3 级别正常)\n", me);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024};

    for (int i = 0; i < 15; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        double max_sec = 0.0, min_sec = DBL_MAX, total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG = %10.4lf Gflops\n",
               M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}
