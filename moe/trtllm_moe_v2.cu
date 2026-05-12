// ==========================================================================
// TRT-LLM Style MoE — Educational Version (V2: Warp Shuffle + Shared Memory)
//
// V1 → V2 核心变化:
// ┌─────────────────┬─────────────────────┬───────────────────────────────┐
// │     Kernel      │        V1           │           V2                  │
// ├─────────────────┼─────────────────────┼───────────────────────────────┤
// │ Gating          │ 1 thread/expert     │ 1 warp(32线程)/expert         │
// │                 │ hidden 串行累加     │ warp shuffle 并行归约         │
// │                 │ 4 threads/block     │ shared mem 复用 x[t]          │
// ├─────────────────┼─────────────────────┼───────────────────────────────┤
// │ Softmax + TopK  │ 1 thread/token      │ 1 warp/token                  │
// │                 │ 串行 O(E*K) 贪心    │ 并行 softmax + butterfly      │
// │                 │                     │ __shfl_xor argmax 做 top-k    │
// └─────────────────┴─────────────────────┴───────────────────────────────┘
//
// 对应 TRT-LLM 源码:
//   kernels/moeTopKFuncs.cuh — warp shuffle 做 top-k:
//     把 (value, index) pack 成 64-bit, cg::reduce(warp, val, greater<>())
//     V2 用 butterfly __shfl_xor argmax 模拟同样的思想
//   kernels/customMoeRoutingKernels.cu — warp-based softmax + top-k 路由
//
// 优化路线图:
//   V1: 正确性优先, 朴素实现 ✓
//   V2 (当前): warp shuffle + shared memory (gating & top-k)
//   V3: block-aligned padding + device-side prefix sum (CUB)
//   V4: shared memory tiled GEMM for FC1/FC2
//   V5: FP16 + Tensor Core (WMMA / CUTLASS)
// ==========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// ============================================================================
// Stage 0: Gating GEMM V2 — 一个 warp 处理一个 expert 的 dot product
//
//   TRT-LLM 对应: cuBLAS SGEMM (MoE 外部)
//
//   V1 问题: 每个 expert 只有 1 个线程, hidden_size 维度串行累加
//     → block 只有 num_experts(4) 个线程, GPU 利用率极低
//
//   V2 方案:
//     - 每个 warp (32线程) 负责一个 expert
//     - warp 内线程按 stride 分摊 hidden_size 的乘加
//     - __shfl_down_sync 做 warp 内归约, 无需 shared memory
//     - x[t] 通过 shared memory 加载一次, 被所有 warp 复用
//       (num_experts 个 warp 都要读 x[t], 不复用就读 num_experts 次)
//
//   TRT-LLM 生产环境: 这一步直接用 cuBLAS GEMM
//   但 warp-level reduction 是 CUDA 基本功, 值得手写理解
// ============================================================================
__global__ void gating_logits_kernel_v2(const float* __restrict__ x,
                                        const float* __restrict__ w_gate,
                                        float* __restrict__ logits,
                                        int num_tokens, int hidden_size,
                                        int num_experts) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    // ── 将 x[t] 加载到 shared memory, 所有 warp 复用 ──
    extern __shared__ float s_x[];  // 大小 = hidden_size
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        s_x[h] = x[t * hidden_size + h];
    }
    __syncthreads();

    // ── 每个 warp 处理一个 expert ──
    int warp_id = threadIdx.x / 32;  // expert index
    int lane = threadIdx.x % 32;     // lane within warp
    if (warp_id >= num_experts) return;
    int e = warp_id;

    // 每个 lane 累加一部分 hidden 维度
    // lane 0: h=0,32,64,...  lane 1: h=1,33,65,...
    float partial = 0.0f;
    for (int h = lane; h < hidden_size; h += 32) {
        partial += s_x[h] * w_gate[h * num_experts + e];
    }

    // warp shuffle tree reduction: 32 → 16 → 8 → 4 → 2 → 1
    //
    //  __shfl_down_sync(mask, val, offset):
    //    lane_i 收到 lane_{i+offset} 的 val
    //    offset=16: lane 0 += lane 16, lane 1 += lane 17, ...
    //    offset=8:  lane 0 += lane 8, ...
    //    ...
    //    最终 lane 0 持有完整 sum
    //
    //  对比 shared memory reduction:
    //    warp shuffle 无需 __syncthreads(), 延迟更低
    //    寄存器直传, 不经过 shared memory bank
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    if (lane == 0) {
        logits[t * num_experts + e] = partial;
    }
}

// ============================================================================
// Stage 1: Routing V2 — 一个 warp (32线程) 处理一个 token
//
//   TRT-LLM 对应: customMoeRoutingKernel (customMoeRoutingKernels.cu)
//                  └─ reduceTopK           (moeTopKFuncs.cuh)
//
//   V1 问题: 单线程/token, softmax 和 top-k 都是串行的
//
//   V2 方案 (对应 TRT-LLM moeTopKFuncs.cuh):
//      - 每个 lane 负责一个 expert (要求 num_experts ≤ 32)
//      - Softmax: butterfly __shfl_xor 求 max 和 exp_sum
//      - TopK:    迭代 k 次 butterfly argmax, 每次选出最大的 expert
//
//    ★ 核心技巧: Butterfly Reduction ★
//    __shfl_xor_sync(mask, val, offset):
//      lane_i 与 lane_{i^offset} 交换数据
//
//      offset=16: 0↔16, 1↔17, 2↔18, ...
//      offset=8:  0↔8,  1↔9,  2↔10, ...
//      offset=4:  0↔4,  1↔5,  2↔6,  ...
//      offset=2:  0↔2,  1↔3, ...
//      offset=1:  0↔1, ...
//
//      5 轮后 **所有** lane 都持有全局结果 (vs __shfl_down 只有 lane 0)
//      这对 top-k 很重要: 每个 lane 都需要知道谁被选中, 以便将其 mask 掉
//
//    TRT-LLM 用 TopKRedType 把 (float val, int idx) pack 成 uint64_t,
//    再用 cooperative_groups::reduce(warp, packed, greater<>()) 做归约.
//    V2 用显式 __shfl_xor + 比较实现同样的逻辑, 更容易理解.
// ============================================================================
__global__ void softmax_topk_kernel_v2(const float* __restrict__ logits,
                                       float* __restrict__ topk_weight,
                                       int* __restrict__ topk_idx,
                                       int num_tokens, int num_experts,
                                       int top_k) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    int lane = threadIdx.x;  // 0..31

    // ── Step 1: 每个 lane 加载一个 expert 的 logit ──
    // lane >= num_experts 的线程填 -INF, 不影响 softmax/top-k
    float my_logit = (lane < num_experts)
                         ? logits[t * num_experts + lane]
                         : -INFINITY;

    // ── Step 2: Butterfly max (数值稳定 softmax 的第一步) ──
    float max_val = my_logit;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val,
                        __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    // 所有 lane 现在都持有 max_val

    // ── Step 3: exp + butterfly sum → softmax ──
    float my_exp = (lane < num_experts) ? expf(my_logit - max_val) : 0.0f;
    float sum_exp = my_exp;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    }
    float my_prob = my_exp / sum_exp;

    // ── Step 4: 迭代 Butterfly Argmax 实现 Top-K ──
    //
    //  每轮选出当前最大的 expert:
    //    1. butterfly __shfl_xor 归约, 每个 lane 与对手比较 (val, lane_id)
    //    2. 所有 lane 得到相同的 (best_val, best_lane)
    //    3. best_lane 的线程将自己的 prob 置为 -1, 下轮不会被选中
    //
    //  tie-breaking: 相同 prob 时选 lane_id 更小的 (保证确定性)
    //
    float prob_remaining = my_prob;

    for (int k = 0; k < top_k; ++k) {
        float best_val = prob_remaining;
        int best_lane = lane;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            int other_lane = __shfl_xor_sync(0xffffffff, best_lane, offset);
            // 选较大者; 相等时选 lane_id 小的
            if (other_val > best_val ||
                (other_val == best_val && other_lane < best_lane)) {
                best_val = other_val;
                best_lane = other_lane;
            }
        }
        // 所有 lane 现在都知道 (best_val, best_lane)

        if (lane == 0) {
            topk_idx[t * top_k + k] = best_lane;    // expert id = lane id
            topk_weight[t * top_k + k] = best_val;  // softmax probability
        }

        // mask 掉已选的 expert, 使其下轮不被选中
        if (lane == best_lane) {
            prob_remaining = -1.0f;
        }
    }

    // ── Step 5: 归一化 top-k 权重 ──
    // (只需 lane 0 操作, 因为 top_k 通常很小: 2~4)
    if (lane == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < top_k; ++k)
            wsum += topk_weight[t * top_k + k];
        for (int k = 0; k < top_k; ++k)
            topk_weight[t * top_k + k] /= wsum;
    }
}

// ============================================================================
// Stage 2~7: 与 V1 相同, V3/V4 再优化
// ============================================================================

// Stage 2: Align — TRT-LLM: moe_align_block_size_kernel (moeAlignKernels.cu)
__global__ void count_tokens_per_expert_kernel(
    const int* __restrict__ topk_idx, int* __restrict__ expert_count,
    int num_tokens, int top_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * top_k) return;
    atomicAdd(&expert_count[topk_idx[idx]], 1);
}

// Stage 3: Permute — TRT-LLM: moePermuteKernel (moeUtils.cu) / expandInputRowsKernel (moe_kernels.cu)
__global__ void permute_tokens_kernel(
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_weight,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_write_pos,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ sorted_expert_ids,
    float* __restrict__ sorted_weights,
    int* __restrict__ token_to_sorted,
    int num_tokens, int top_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * top_k) return;

    int t = idx / top_k;
    int e = topk_idx[idx];
    float w = topk_weight[idx];

    int slot = atomicAdd(&expert_write_pos[e], 1);
    int pos = expert_offsets[e] + slot;

    sorted_token_ids[pos] = t;
    sorted_expert_ids[pos] = e;
    sorted_weights[pos] = w;
    token_to_sorted[idx] = pos;
}

// Stage 4+5: Expert GEMM1 + Activation — TRT-LLM: CUTLASS moeGemm + moeActivationKernel
__global__ void expert_gemm1_act_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ sorted_expert_ids,
    float* __restrict__ fc1_out,
    int total_permuted, int hidden_size, int intermediate_size) {
    int p = blockIdx.x;
    int j = threadIdx.x;
    if (p >= total_permuted || j >= intermediate_size) return;

    int t = sorted_token_ids[p];
    int e = sorted_expert_ids[p];

    float sum = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
        sum += x[t * hidden_size + h] *
               w1[(e * hidden_size + h) * intermediate_size + j];
    }
    float silu = sum / (1.0f + expf(-sum));
    fc1_out[p * intermediate_size + j] = silu;
}

// Stage 6: Expert GEMM2 — TRT-LLM: CUTLASS moeGemm (FC2)
__global__ void expert_gemm2_kernel(
    const float* __restrict__ fc1_out,
    const float* __restrict__ w2,
    const int* __restrict__ sorted_expert_ids,
    float* __restrict__ permuted_output,
    int total_permuted, int intermediate_size, int hidden_size) {
    int p = blockIdx.x;
    int h = threadIdx.x;
    if (p >= total_permuted || h >= hidden_size) return;

    int e = sorted_expert_ids[p];

    float sum = 0.0f;
    for (int j = 0; j < intermediate_size; ++j) {
        sum += fc1_out[p * intermediate_size + j] *
               w2[(e * intermediate_size + j) * hidden_size + h];
    }
    permuted_output[p * hidden_size + h] = sum;
}

// Stage 7: Unpermute+Finalize — TRT-LLM: moeUnpermuteKernel / finalizeMoeRoutingKernel
__global__ void unpermute_and_combine_kernel(
    const float* __restrict__ permuted_output,
    const float* __restrict__ topk_weight,
    const int* __restrict__ token_to_sorted,
    float* __restrict__ y,
    int num_tokens, int top_k, int hidden_size) {
    int t = blockIdx.x;
    int h = threadIdx.x;
    if (t >= num_tokens || h >= hidden_size) return;

    float val = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int pos = token_to_sorted[t * top_k + k];
        float w = topk_weight[t * top_k + k];
        val += w * permuted_output[pos * hidden_size + h];
    }
    y[t * hidden_size + h] = val;
}

// ============================================================================
// CPU reference (同 V1, 用于验证正确性)
// ============================================================================
void moe_cpu_reference(const std::vector<float>& x,
                       const std::vector<float>& w_gate,
                       const std::vector<float>& w1,
                       const std::vector<float>& w2, std::vector<float>& y_ref,
                       int num_tokens, int hidden_size, int intermediate_size,
                       int num_experts, int top_k) {
    std::vector<float> logits(num_tokens * num_experts);
    for (int t = 0; t < num_tokens; ++t)
        for (int e = 0; e < num_experts; ++e) {
            float s = 0;
            for (int h = 0; h < hidden_size; ++h)
                s += x[t * hidden_size + h] * w_gate[h * num_experts + e];
            logits[t * num_experts + e] = s;
        }

    std::vector<int> tk_idx(num_tokens * top_k);
    std::vector<float> tk_w(num_tokens * top_k);
    for (int t = 0; t < num_tokens; ++t) {
        float max_v = logits[t * num_experts];
        for (int e = 1; e < num_experts; ++e)
            max_v = std::max(max_v, logits[t * num_experts + e]);
        float sum_e = 0;
        for (int e = 0; e < num_experts; ++e)
            sum_e += std::exp(logits[t * num_experts + e] - max_v);

        std::vector<float> probs(num_experts);
        for (int e = 0; e < num_experts; ++e)
            probs[e] = std::exp(logits[t * num_experts + e] - max_v) / sum_e;

        std::vector<bool> used(num_experts, false);
        for (int k = 0; k < top_k; ++k) {
            int be = -1;
            float bp = -1;
            for (int e = 0; e < num_experts; ++e)
                if (!used[e] && probs[e] > bp) {
                    bp = probs[e];
                    be = e;
                }
            tk_idx[t * top_k + k] = be;
            tk_w[t * top_k + k] = bp;
            used[be] = true;
        }
        float s = 0;
        for (int k = 0; k < top_k; ++k) s += tk_w[t * top_k + k];
        for (int k = 0; k < top_k; ++k) tk_w[t * top_k + k] /= s;
    }

    y_ref.assign(num_tokens * hidden_size, 0.0f);
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int e = tk_idx[t * top_k + k];
            float w = tk_w[t * top_k + k];
            std::vector<float> inter(intermediate_size);
            for (int j = 0; j < intermediate_size; ++j) {
                float s = 0;
                for (int h = 0; h < hidden_size; ++h)
                    s += x[t * hidden_size + h] *
                         w1[(e * hidden_size + h) * intermediate_size + j];
                inter[j] = s / (1.0f + std::exp(-s));
            }
            for (int h = 0; h < hidden_size; ++h) {
                float s = 0;
                for (int j = 0; j < intermediate_size; ++j)
                    s += inter[j] *
                         w2[(e * intermediate_size + j) * hidden_size + h];
                y_ref[t * hidden_size + h] += w * s;
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    constexpr int num_tokens = 8;
    constexpr int hidden_size = 64;
    constexpr int intermediate_size = 128;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int max_permuted = num_tokens * top_k;

    // ─── Host 数据初始化 ───
    std::vector<float> h_x(num_tokens * hidden_size);
    std::vector<float> h_w_gate(hidden_size * num_experts);
    std::vector<float> h_w1(num_experts * hidden_size * intermediate_size);
    std::vector<float> h_w2(num_experts * intermediate_size * hidden_size);

    srand(42);
    auto randf = []() { return (rand() / (float)RAND_MAX - 0.5f) * 0.1f; };
    for (auto& v : h_x) v = randf();
    for (auto& v : h_w_gate) v = randf();
    for (auto& v : h_w1) v = randf();
    for (auto& v : h_w2) v = randf();

    // ─── Device 内存 ───
    float *d_x, *d_w_gate, *d_w1, *d_w2;
    float *d_logits, *d_topk_w, *d_y;
    float *d_sorted_weights, *d_fc1_out, *d_permuted_output;
    int *d_topk_idx, *d_expert_count, *d_expert_write_pos;
    int *d_sorted_token_ids, *d_sorted_expert_ids, *d_token_to_sorted;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_gate, h_w_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w1, h_w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2, h_w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_w, num_tokens * top_k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_idx, num_tokens * top_k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_count, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_write_pos, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_token_ids, max_permuted * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_expert_ids, max_permuted * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_weights, max_permuted * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_token_to_sorted, max_permuted * sizeof(int)));
    CHECK_CUDA(
        cudaMalloc(&d_fc1_out, max_permuted * intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_permuted_output,
                          max_permuted * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, num_tokens * hidden_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w_gate, h_w_gate.data(),
                          h_w_gate.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_expert_count, 0, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_expert_write_pos, 0, num_experts * sizeof(int)));

    // ═══════════════════════════════════════════════════════════════
    // Pipeline 执行
    // ═══════════════════════════════════════════════════════════════

    std::cout << "=== TRT-LLM Style MoE Pipeline (V2: Warp Shuffle) ===\n";
    std::cout << "num_tokens=" << num_tokens << " hidden=" << hidden_size
              << " inter=" << intermediate_size << " experts=" << num_experts
              << " top_k=" << top_k << "\n\n";

    // ── Stage 0: Gating GEMM V2 ──
    //    TRT-LLM: cuBLAS (外部)
    //    V2: 每个 warp 算一个 expert 的 dot product + shared mem 复用 x[t]
    int gating_threads = num_experts * 32;  // 4 experts × 32 = 128 threads
    int gating_smem = hidden_size * sizeof(float);
    gating_logits_kernel_v2<<<num_tokens, gating_threads, gating_smem>>>(
        d_x, d_w_gate, d_logits, num_tokens, hidden_size, num_experts);

    // ── Stage 1: Routing V2 (Softmax + TopK) ──
    //    TRT-LLM: customMoeRoutingKernel
    //    V2: 一个 warp/token, butterfly shuffle softmax + argmax top-k
    softmax_topk_kernel_v2<<<num_tokens, 32>>>(
        d_logits, d_topk_w, d_topk_idx, num_tokens, num_experts, top_k);

    // ── Stage 2: Align (Count + Prefix Sum) ──
    //    TRT-LLM: moe_align_block_size_kernel (fused count + CUB prefix sum + padding)
    int threads_s = 128;
    int blocks_s = (num_tokens * top_k + threads_s - 1) / threads_s;
    count_tokens_per_expert_kernel<<<blocks_s, threads_s>>>(
        d_topk_idx, d_expert_count, num_tokens, top_k);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Host-side prefix sum (V3 → CUB DeviceScan)
    std::vector<int> h_expert_count(num_experts);
    CHECK_CUDA(cudaMemcpy(h_expert_count.data(), d_expert_count,
                          num_experts * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<int> h_expert_offsets(num_experts);
    h_expert_offsets[0] = 0;
    for (int e = 1; e < num_experts; ++e)
        h_expert_offsets[e] = h_expert_offsets[e - 1] + h_expert_count[e - 1];
    int total_permuted =
        h_expert_offsets[num_experts - 1] + h_expert_count[num_experts - 1];

    int* d_expert_offsets;
    CHECK_CUDA(cudaMalloc(&d_expert_offsets, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_expert_offsets, h_expert_offsets.data(),
                          num_experts * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ── Stage 3: Permute ──
    //    TRT-LLM: moePermuteKernel / expandInputRowsKernel
    permute_tokens_kernel<<<blocks_s, threads_s>>>(
        d_topk_idx, d_topk_w, d_expert_offsets, d_expert_write_pos,
        d_sorted_token_ids, d_sorted_expert_ids, d_sorted_weights,
        d_token_to_sorted, num_tokens, top_k);

    // ── Stage 4+5: Expert GEMM1 + SiLU (V1, V4 再优化) ──
    //    TRT-LLM: CUTLASS moeGemm (FC1) + moeActivationKernel
    expert_gemm1_act_kernel<<<total_permuted, intermediate_size>>>(
        d_x, d_w1, d_sorted_token_ids, d_sorted_expert_ids, d_fc1_out,
        total_permuted, hidden_size, intermediate_size);

    // ── Stage 6: Expert GEMM2 (V1, V4 再优化) ──
    //    TRT-LLM: CUTLASS moeGemm (FC2)
    expert_gemm2_kernel<<<total_permuted, hidden_size>>>(
        d_fc1_out, d_w2, d_sorted_expert_ids, d_permuted_output,
        total_permuted, intermediate_size, hidden_size);

    // ── Stage 7: Unpermute + Finalize ──
    //    TRT-LLM: moeUnpermuteKernel / finalizeMoeRoutingKernel
    unpermute_and_combine_kernel<<<num_tokens, hidden_size>>>(
        d_permuted_output, d_topk_w, d_token_to_sorted, d_y, num_tokens, top_k,
        hidden_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ═══════════════════════════════════════════════════════════════
    // 验证
    // ═══════════════════════════════════════════════════════════════

    std::vector<int> h_topk_idx(num_tokens * top_k);
    std::vector<float> h_topk_w(num_tokens * top_k);
    std::vector<float> h_y(num_tokens * hidden_size);

    CHECK_CUDA(cudaMemcpy(h_topk_idx.data(), d_topk_idx,
                          h_topk_idx.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_topk_w.data(), d_topk_w,
                          h_topk_w.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "--- Routing ---\n";
    for (int t = 0; t < num_tokens; ++t) {
        std::cout << "token " << t << " -> ";
        for (int k = 0; k < top_k; ++k)
            std::cout << "expert " << h_topk_idx[t * top_k + k]
                      << " (w=" << h_topk_w[t * top_k + k] << ") ";
        std::cout << "\n";
    }

    std::cout << "\n--- Expert token counts ---\n";
    for (int e = 0; e < num_experts; ++e)
        std::cout << "expert " << e << ": " << h_expert_count[e]
                  << " tokens (offset=" << h_expert_offsets[e] << ")\n";

    std::vector<float> y_ref;
    moe_cpu_reference(h_x, h_w_gate, h_w1, h_w2, y_ref, num_tokens,
                      hidden_size, intermediate_size, num_experts, top_k);

    float max_err = 0.0f;
    for (int i = 0; i < num_tokens * hidden_size; ++i)
        max_err = std::max(max_err, std::abs(h_y[i] - y_ref[i]));

    std::cout << "\n--- Verification ---\n";
    std::cout << "max |GPU - CPU| = " << max_err
              << (max_err < 1e-3f ? "  PASS" : "  FAIL") << "\n";

    std::cout << "\n--- Output y[0][:8] ---\n";
    for (int h = 0; h < std::min(8, hidden_size); ++h)
        std::cout << h_y[h] << (h == 7 ? "\n" : ", ");

    // ─── 释放 ───
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w_gate));
    CHECK_CUDA(cudaFree(d_w1));
    CHECK_CUDA(cudaFree(d_w2));
    CHECK_CUDA(cudaFree(d_logits));
    CHECK_CUDA(cudaFree(d_topk_w));
    CHECK_CUDA(cudaFree(d_topk_idx));
    CHECK_CUDA(cudaFree(d_expert_count));
    CHECK_CUDA(cudaFree(d_expert_write_pos));
    CHECK_CUDA(cudaFree(d_expert_offsets));
    CHECK_CUDA(cudaFree(d_sorted_token_ids));
    CHECK_CUDA(cudaFree(d_sorted_expert_ids));
    CHECK_CUDA(cudaFree(d_sorted_weights));
    CHECK_CUDA(cudaFree(d_token_to_sorted));
    CHECK_CUDA(cudaFree(d_fc1_out));
    CHECK_CUDA(cudaFree(d_permuted_output));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}
