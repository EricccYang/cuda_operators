#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 2D convolution, 单 channel float, KERNEL = 3x3, 'same' padding (zero pad)
//
// 思路：
//   v1 naive  : 一线程一个输出像素，每个线程从 global 读 9 个输入
//                相邻 9 个线程会读重叠数据，重复 load 系数 ~9
//   v2 smem   : block 处理 TILE × TILE 输出，先把 (TILE+2) × (TILE+2) 区域加载到 smem
//                包含 halo (周围一圈)，每个输入像素只 global load 一次
//                重复 load 系数从 9 降到 ~1
//
// arithmetic intensity 很低 (9 FLOP / 4B load = 2.25 FLOP/B)，本质 memory-bound，
// smem tiling 主要靠"减少 global load 次数"赚收益。

#define KS    3
#define R     (KS / 2)        // 半径 = 1
#define TILE  16              // 每 block 输出 16x16 区域


// ---------- v1: 直接卷积 ----------
__global__ void conv2d_v1(const float* __restrict__ in,
                          const float* __restrict__ ker,
                          float* __restrict__ out,
                          int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.f;
    #pragma unroll
    for (int ky = 0; ky < KS; ky++) {
        #pragma unroll
        for (int kx = 0; kx < KS; kx++) {
            int yy = y + ky - R;
            int xx = x + kx - R;
            float v = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[yy * W + xx] : 0.f;
            acc += v * ker[ky * KS + kx];
        }
    }
    out[y * W + x] = acc;
}


// ---------- v2: shared memory tiling with halo ----------
__global__ void conv2d_v2_smem(const float* __restrict__ in,
                               const float* __restrict__ ker,
                               float* __restrict__ out,
                               int H, int W) {
    constexpr int SH = TILE + 2 * R;        // smem tile 含 halo
    __shared__ float tile[SH][SH];

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 协作加载 SH × SH 到 smem，每线程负责至少一个，可能多个
    for (int dy = ty; dy < SH; dy += TILE) {
        for (int dx = tx; dx < SH; dx += TILE) {
            int yy = by + dy - R;
            int xx = bx + dx - R;
            tile[dy][dx] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[yy * W + xx] : 0.f;
        }
    }
    __syncthreads();

    int x = bx + tx;
    int y = by + ty;
    if (x >= W || y >= H) return;

    float acc = 0.f;
    #pragma unroll
    for (int ky = 0; ky < KS; ky++) {
        #pragma unroll
        for (int kx = 0; kx < KS; kx++) {
            acc += tile[ty + ky][tx + kx] * ker[ky * KS + kx];
        }
    }
    out[y * W + x] = acc;
}


// ---------- host helpers ----------
void conv2dCPU(const float* in, const float* ker, float* out, int H, int W) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float acc = 0.f;
            for (int ky = 0; ky < KS; ky++) {
                for (int kx = 0; kx < KS; kx++) {
                    int yy = y + ky - R;
                    int xx = x + kx - R;
                    float v = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[yy * W + xx] : 0.f;
                    acc += v * ker[ky * KS + kx];
                }
            }
            out[y * W + x] = acc;
        }
    }
}

bool checkResult(const float* ref, const float* gpu, int n, float tol = 1e-3f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - gpu[i]) > tol) {
            printf("  Mismatch at %d: cpu=%f gpu=%f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}


int main() {
    int H = 2048, W = 2048;
    int n = H * W;
    size_t nBytes = n * sizeof(float);
    printf("conv2d: %dx%d, kernel %dx%d\n", H, W, KS, KS);

    cudaSetDevice(0);

    float *h_in  = (float*)malloc(nBytes);
    float *h_out = (float*)malloc(nBytes);
    float *h_ref = (float*)malloc(nBytes);
    for (int i = 0; i < n; i++) h_in[i] = (float)((i * 37) % 256) / 255.f;

    // 经典 Sobel-x 算子
    float h_ker[KS * KS] = {
        -1.f, 0.f, 1.f,
        -2.f, 0.f, 2.f,
        -1.f, 0.f, 1.f,
    };
    conv2dCPU(h_in, h_ker, h_ref, H, W);

    float *d_in, *d_out, *d_ker;
    cudaMalloc(&d_in,  nBytes);
    cudaMalloc(&d_out, nBytes);
    cudaMalloc(&d_ker, KS * KS * sizeof(float));
    cudaMemcpy(d_in,  h_in,  nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, KS * KS * sizeof(float), cudaMemcpyHostToDevice);

    auto bench = [&](const char* name, auto fn) {
        cudaMemset(d_out, 0, nBytes);
        fn();   // warmup
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        int iters = 50;
        cudaEventRecord(s);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
        cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);
        bool ok = checkResult(h_ref, h_out, n);
        printf("[%-12s] %.4f ms  %s\n", name, ms, ok ? "OK" : "FAIL");
        cudaEventDestroy(s); cudaEventDestroy(e);
    };

    // ---- v1 ----
    {
        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        bench("v1 naive", [&](){ conv2d_v1<<<grid, block>>>(d_in, d_ker, d_out, H, W); });
    }

    // ---- v2 ----
    {
        dim3 block(TILE, TILE);
        dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
        bench("v2 smem", [&](){ conv2d_v2_smem<<<grid, block>>>(d_in, d_ker, d_out, H, W); });
    }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ker);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
