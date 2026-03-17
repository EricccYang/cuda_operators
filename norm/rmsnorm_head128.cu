/**
 * RMSNorm 针对 128 维 attention head 的优化版本
 * 设计: 1 warp (32 threads) × 4 elements/thread = 128
 * 只用 warp shuffle reduce，不用 shared memory 做 tree reduce
 *
 * 输入/输出: [batch*heads*seq, 128]，每行 128 维
 */
#include <cuda_runtime.h>

#define HEAD_DIM 128
#define WARP_SIZE 32
#define ELEMS_PER_THREAD (HEAD_DIM / WARP_SIZE)  // 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void RMSNormHead128(
    float* __restrict__ io,  // in-place, [rows, 128]
    int rows
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;  // 0..31
    const int base = row * HEAD_DIM;

    float r_data[ELEMS_PER_THREAD];
    float squ_sum = 0.0f;

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        r_data[i] = io[base + tid + i * WARP_SIZE];
        squ_sum += r_data[i] * r_data[i];
    }

    squ_sum = warp_reduce_sum(squ_sum);

    float rms;
    if (tid == 0) {
        rms = sqrtf(squ_sum / HEAD_DIM);
    }
    rms = __shfl_sync(0xffffffff, rms, 0);  // broadcast lane0 to all

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        io[base + tid + i * WARP_SIZE] = r_data[i] / rms;
    }
}

// 256 维版本：32 threads × 8
#define HEAD_DIM_256 256
#define ELEMS_PER_THREAD_256 (HEAD_DIM_256 / WARP_SIZE)  // 8

__global__ void RMSNormHead256(
    float* __restrict__ io,
    int rows
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int base = row * HEAD_DIM_256;

    float r_data[ELEMS_PER_THREAD_256];
    float squ_sum = 0.0f;

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD_256; i++) {
        r_data[i] = io[base + tid + i * WARP_SIZE];
        squ_sum += r_data[i] * r_data[i];
    }

    squ_sum = warp_reduce_sum(squ_sum);

    float rms;
    if (tid == 0) {
        rms = sqrtf(squ_sum / HEAD_DIM_256);
    }
    rms = __shfl_sync(0xffffffff, rms, 0);

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD_256; i++) {
        io[base + tid + i * WARP_SIZE] = r_data[i] / rms;
    }
}

// 启动配置示例
// 128: block=(32,1,1), grid=(batch*num_heads*seq_len, 1)
// 256: 同上
