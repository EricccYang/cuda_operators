// ==========================================================================
// TRT-LLM Style MoE — Educational Version (V1: Correct Baseline)
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │                    Pipeline 严格对齐 TRT-LLM                        │
// │                                                                     │
// │  Stage 0  Gating GEMM       ── logits = x @ W_gate                 │
// │     ↓                                                               │
// │  Stage 1  Routing           ── softmax + top-k 选 expert            │
// │     ↓                                                               │
// │  Stage 2  Align             ── 统计 / prefix sum / block padding    │
// │     ↓                                                               │
// │  Stage 3  Permute           ── 按 expert 重排 token 数据            │
// │     ↓                                                               │
// │  Stage 4  Expert GEMM1      ── FC1 grouped GEMM                    │
// │     ↓                                                               │
// │  Stage 5  Activation        ── SiLU / SwiGLU / GELU               │
// │     ↓                                                               │
// │  Stage 6  Expert GEMM2      ── FC2 grouped GEMM                    │
// │     ↓                                                               │
// │  Stage 7  Unpermute+Finalize── 还原顺序, 加权求和                    │
// └─────────────────────────────────────────────────────────────────────┘
//
// ┌──────────────┬──────────────────────────┬──────────────────────────────────────────────────────────┐
// │    Stage     │  V1 (本文件)              │  TRT-LLM 对应                                           │
// ├──────────────┼──────────────────────────┼──────────────────────────────────────────────────────────┤
// │ 0 Gating     │ gating_logits_kernel     │ cuBLAS SGEMM (MoE 外部, 非 MoE kernel)                  │
// │ 1 Routing    │ softmax_topk_kernel      │ customMoeRoutingKernel   (customMoeRoutingKernels.cu)    │
// │              │                          │  └─ reduceTopK           (moeTopKFuncs.cuh)              │
// │ 2 Align      │ count_and_prefix_sum     │ moe_align_block_size_kernel       (moeAlignKernels.cu)  │
// │              │  (host-side prefix sum)  │  └─ CUB BlockScan (device prefix sum + block padding)   │
// │              │                          │ blockExpertPrefixSumKernel         (moe_kernels.cu)      │
// │              │                          │ globalExpertPrefixSumKernel        (moe_kernels.cu)      │
// │              │                          │ mergeExpertPrefixSumKernel         (moe_kernels.cu)      │
// │ 3 Permute    │ permute_tokens_kernel    │ moePermuteKernel         (cuteDslKernels/moeUtils.cu)   │
// │              │                          │ expandInputRowsKernel    (moe_kernels.cu)                │
// │ 4 GEMM1      │ expert_gemm1_kernel      │ CUTLASS MoeGemmRunner::moeGemm()                        │
// │ 5 Activation  │ (fused in GEMM1)        │ moeActivationKernel     (cuteDslKernels/moeUtils.cu)    │
// │              │                          │ doGatedActivationKernel  (moe_kernels.cu)                │
// │              │                          │ doActivationKernel       (moe_kernels.cu)                │
// │ 6 GEMM2      │ expert_gemm2_kernel      │ CUTLASS MoeGemmRunner::moeGemm()                        │
// │ 7 Unpermute   │ unpermute_and_combine    │ moeUnpermuteKernel      (cuteDslKernels/moeUtils.cu)    │
// │  +Finalize   │                          │ finalizeMoeRoutingKernel (moe_kernels.cu)                │
// └──────────────┴──────────────────────────┴──────────────────────────────────────────────────────────┘
//
// 说明:
//   - TRT-LLM 有两条代码路径:
//     (A) cuteDsl 路径 (新, SM90+): moePermuteKernel → CUTLASS GEMM → moeActivationKernel → CUTLASS GEMM → moeUnpermuteKernel
//     (B) legacy 路径:             expandInputRowsKernel → CUTLASS GEMM → doActivation → CUTLASS GEMM → finalizeMoeRoutingKernel
//     表中两条路径的函数都列出, V1 对应的是简化版本
//   - Stage 0 (Gating) 在 TRT-LLM 中由前置的 Linear 层完成, 不属于 MoE kernel 本身
//     V1 手写了一个 naive GEMM 用于教学
//   - Stage 4+5 在 V1 中融合为一个 kernel (FC1+SiLU)
//     TRT-LLM 中 cuteDsl 路径将 activation 独立为一个 kernel; legacy 路径可融合到 GEMM epilogue
//
// 优化路线图:
//   V1 (当前): 正确性优先, 朴素实现
//   V2: warp shuffle routing + shared memory gating
//   V3: device-side CUB prefix sum + block-aligned padding
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
// Stage 0: Gating GEMM
//   logits[t][e] = dot(x[t], W_gate[:, e])
//
//   TRT-LLM 对应: cuBLAS SGEMM (MoE kernel 外部)
//     router logits 由前置 Linear 层产生, 作为 MoE plugin 的输入 tensor
//     这里手写 naive GEMM 用于教学, 生产环境不会出现
//
//   V1: 每线程处理一个 (token, expert) 对, hidden 维度串行累加
// ============================================================================
__global__ void gating_logits_kernel(const float* __restrict__ x,
                                     const float* __restrict__ w_gate,
                                     float* __restrict__ logits,
                                     int num_tokens, int hidden_size,
                                     int num_experts) {
    int t = blockIdx.x;   // token
    int e = threadIdx.x;  // expert
    if (t >= num_tokens || e >= num_experts) return;

    float sum = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
        sum += x[t * hidden_size + h] * w_gate[h * num_experts + e];
    }
    logits[t * num_experts + e] = sum;
}

// ============================================================================
// Stage 1: Routing (Softmax + Top-K)
//
//   TRT-LLM 对应:
//     customMoeRoutingKernel<InputT, OutputT, IdxT, MaxNumExperts,
//                            MaxNumTopExperts, DoSoftmaxBeforeTopK>
//     文件: kernels/customMoeRoutingKernels.cu
//
//     TRT-LLM 实现细节:
//       - 每个 warp (32线程) 处理一个 token
//       - warp 内线程并行加载 expert logits
//       - 可选先 softmax 再 top-k (DoSoftmaxBeforeTopK 模板参数)
//       - Top-K 核心: reduceTopK() in moeTopKFuncs.cuh
//         把 (float value, int index) pack 成 TopKRedType (uint64_t)
//         用 cg::reduce(warp, packed_val, greater<>()) 做 warp-level 归约
//         支持 K=1,2,4,8, 用 swap network 做 K≤4 的排序
//
//   V1: 单线程/token, O(E*K) 贪心选取
// ============================================================================
__global__ void softmax_topk_kernel(const float* __restrict__ logits,
                                    float* __restrict__ topk_weight,
                                    int* __restrict__ topk_idx,
                                    int num_tokens, int num_experts,
                                    int top_k) {
    int t = blockIdx.x;
    if (t >= num_tokens || threadIdx.x != 0) return;

    const float* row = logits + t * num_experts;

    // 数值稳定 softmax
    float max_val = row[0];
    for (int e = 1; e < num_experts; ++e) max_val = fmaxf(max_val, row[e]);

    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; ++e) sum_exp += expf(row[e] - max_val);

    // 贪心 top-k
    for (int k = 0; k < top_k; ++k) {
        int best_e = -1;
        float best_p = -1.0f;
        for (int e = 0; e < num_experts; ++e) {
            float p = expf(row[e] - max_val) / sum_exp;
            bool used = false;
            for (int kk = 0; kk < k; ++kk) {
                if (topk_idx[t * top_k + kk] == e) {
                    used = true;
                    break;
                }
            }
            if (!used && p > best_p) {
                best_p = p;
                best_e = e;
            }
        }
        topk_idx[t * top_k + k] = best_e;
        topk_weight[t * top_k + k] = best_p;
    }

    // 归一化 top-k 权重
    float topk_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) topk_sum += topk_weight[t * top_k + k];
    for (int k = 0; k < top_k; ++k) topk_weight[t * top_k + k] /= topk_sum;
}

// ============================================================================
// Stage 2: Align (统计 + Prefix Sum + Block Padding)
//
//   TRT-LLM 对应 (两条路径):
//
//   路径 A — moe_align_block_size_kernel (moeAlignKernels.cu)
//     一个 kernel 融合三步:
//       1. shared memory atomicAdd 统计每个 expert 的 token 数
//       2. CUB BlockScan 做 exclusive prefix sum
//       3. 将 token 数 pad 到 GEMM tile 的倍数 (ceil_div(count, block_size) * block_size)
//     输出: sorted_token_ids, expert_ids, cumsum, total_tokens_post_pad
//     小 batch 有专用: moe_align_block_size_small_batch_expert_kernel
//
//   路径 B — 三个独立 kernel (moe_kernels.cu)
//       blockExpertPrefixSumKernel   → 每个 block 内统计 expert token 数
//       globalExpertPrefixSumKernel  → CUB DeviceScan 全局 prefix sum
//       mergeExpertPrefixSumKernel   → 合并 block-level 和 global prefix sum
//
//   V1: 分成两步
//     Step 2a: count_tokens_per_expert_kernel (device-side atomicAdd 计数)
//     Step 2b: host-side exclusive prefix sum (V3 会改为 CUB device scan)
// ============================================================================
__global__ void count_tokens_per_expert_kernel(
    const int* __restrict__ topk_idx, int* __restrict__ expert_count,
    int num_tokens, int top_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * top_k) return;
    atomicAdd(&expert_count[topk_idx[idx]], 1);
}

// ============================================================================
// Stage 3: Permute (按 expert 重排 token)
//
//   TRT-LLM 对应 (两条路径):
//
//   路径 A — moePermuteKernel (cuteDslKernels/moeUtils.cu)
//     template <typename InputType, typename SFType, int kSFVecSize, int kThreadsPerBlock>
//     __global__ void moePermuteKernel(input, permuted_output,
//         input_sf, permuted_sf,                     // FP4 scale factor
//         tile_idx_to_mn_limit,                      // tile→(M,N) 边界
//         permuted_idx_to_expanded_idx,              // 排列映射表
//         num_non_exiting_tiles,
//         hidden_size, top_k, tile_size)
//     特点: 支持 FP4 scale factor 随数据一起 permute; tile-based 设计
//
//   路径 B — expandInputRowsKernel (moe_kernels.cu)
//     将每个 token 按 top-k 展开 (复制 K 份), 按 expert 分组排列
//
//   ★ 核心思想 ★
//   排列前: token 路由分散, 比如 t0→{e1,e3}, t1→{e0,e2}, t2→{e1,e0} ...
//   排列后: [e0 的所有 token | e1 的所有 token | e2 ... | e3 ...]
//   这样后续 expert FFN 对同一 expert 的 token 做 GEMM 时,
//   输入在内存中是连续的 → grouped GEMM / batched GEMM 高效执行
//
//   V1: atomicAdd 游标写入, 同时构建反向映射 token_to_sorted
// ============================================================================
__global__ void permute_tokens_kernel(
    const int* __restrict__ topk_idx,          // [num_tokens, top_k]
    const float* __restrict__ topk_weight,     // [num_tokens, top_k]
    const int* __restrict__ expert_offsets,     // [num_experts] exclusive prefix sum
    int* __restrict__ expert_write_pos,         // [num_experts] 原子写入游标
    int* __restrict__ sorted_token_ids,         // [total_permuted] → 原始 token id
    int* __restrict__ sorted_expert_ids,        // [total_permuted] → expert id
    float* __restrict__ sorted_weights,         // [total_permuted] → routing 权重
    int* __restrict__ token_to_sorted,          // [num_tokens * top_k] 反向映射
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
    token_to_sorted[idx] = pos;  // 反向映射: (t, k) → sorted position
}

// ============================================================================
// Stage 4+5: Expert GEMM1 + Activation (V1 融合为一个 kernel)
//
//   TRT-LLM 对应:
//
//   GEMM1:
//     CUTLASS MoeGemmRunner::moeGemm()
//     文件: kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h
//     实现: CUTLASS TMA warp-specialized grouped GEMM (SM90+)
//           支持 FP16/BF16/FP8/FP4/INT8/INT4 等多种精度组合
//
//   Activation:
//     路径 A — moeActivationKernel (cuteDslKernels/moeUtils.cu)
//       template <typename InputType, typename OutputType, typename SFType,
//                 int kSFVecSize, typename ActFn, int kThreadsPerBlock>
//       支持 SiLU, GELU, SwiGLU, GeGLU 等; 处理 FP4 scale factor
//     路径 B — doGatedActivationKernel / doActivationKernel (moe_kernels.cu)
//       gated: output = Act(gemm_result) * gemm_result_gate  (SwiGLU, GeGLU)
//       non-gated: output = Act(gemm_result)                 (ReLU, GELU, SiLU)
//
//   V1: 融合 FC1 + SiLU
//     fc1_out[p][j] = SiLU( Σ_h x[token][h] * W1[expert][h][j] )
//     每 block 处理一个 sorted position, 每线程算一个 intermediate 元素
// ============================================================================
__global__ void expert_gemm1_act_kernel(
    const float* __restrict__ x,                // [num_tokens, hidden_size]
    const float* __restrict__ w1,               // [num_experts, hidden_size, inter_size]
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ sorted_expert_ids,
    float* __restrict__ fc1_out,                // [total_permuted, inter_size]
    int total_permuted, int hidden_size, int intermediate_size) {
    int p = blockIdx.x;   // sorted position
    int j = threadIdx.x;  // intermediate dimension
    if (p >= total_permuted || j >= intermediate_size) return;

    int t = sorted_token_ids[p];
    int e = sorted_expert_ids[p];

    // GEMM: 遍历 hidden_size 做点积
    float sum = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
        sum += x[t * hidden_size + h] *
               w1[(e * hidden_size + h) * intermediate_size + j];
    }

    // SiLU activation: x * sigmoid(x)
    float silu = sum / (1.0f + expf(-sum));
    fc1_out[p * intermediate_size + j] = silu;
}

// ============================================================================
// Stage 6: Expert GEMM2
//
//   TRT-LLM 对应:
//     CUTLASS MoeGemmRunner::moeGemm()  (同 GEMM1, 不同的权重矩阵)
//
//   V1: permuted_out[p][h] = Σ_j fc1_out[p][j] * W2[expert][j][h]
// ============================================================================
__global__ void expert_gemm2_kernel(
    const float* __restrict__ fc1_out,          // [total_permuted, inter_size]
    const float* __restrict__ w2,               // [num_experts, inter_size, hidden_size]
    const int* __restrict__ sorted_expert_ids,
    float* __restrict__ permuted_output,        // [total_permuted, hidden_size]
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

// ============================================================================
// Stage 7: Unpermute + Finalize (还原顺序, 加权求和)
//
//   TRT-LLM 对应 (两条路径):
//
//   路径 A — moeUnpermuteKernel (cuteDslKernels/moeUtils.cu)
//     template <typename InputType, typename TopKScaleType, int kThreadsPerBlock>
//     __global__ void moeUnpermuteKernel(
//         permuted_input, output,
//         expanded_idx_to_permuted_idx,   // 排列映射
//         topk_scales,                    // routing 权重
//         hidden_size, top_k)
//     每个 block 处理一个原始 token, 遍历 top_k 个 expert 加权求和
//
//   路径 B — finalizeMoeRoutingKernel (moe_kernels.cu)
//     template <typename GemmOutputType, typename ScaleType>
//     __global__ void finalizeMoeRoutingKernel(
//         expanded_permuted_rows, token_final_scales, output,
//         unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
//         num_rows, num_valid_rows, hidden_size, experts_per_token)
//     支持 deterministic 模式 (无 atomicAdd)
//     SM90+ 可融合到 GEMM2 的 epilogue (FusedFinalizeEpilogue)
//
//   V1: y[t][h] = Σ_k  weight[t,k] * permuted_output[sorted_pos(t,k)][h]
//       不用 atomicAdd! 通过 token_to_sorted 反向映射直接读取
// ============================================================================
__global__ void unpermute_and_combine_kernel(
    const float* __restrict__ permuted_output,  // [total_permuted, hidden_size]
    const float* __restrict__ topk_weight,      // [num_tokens, top_k]
    const int* __restrict__ token_to_sorted,    // [num_tokens * top_k]
    float* __restrict__ y,                      // [num_tokens, hidden_size]
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
// CPU reference: 用于验证 GPU 结果的正确性
// ============================================================================
void moe_cpu_reference(const std::vector<float>& x,
                       const std::vector<float>& w_gate,
                       const std::vector<float>& w1,
                       const std::vector<float>& w2, std::vector<float>& y_ref,
                       int num_tokens, int hidden_size, int intermediate_size,
                       int num_experts, int top_k) {
    // Stage 0: Gating
    std::vector<float> logits(num_tokens * num_experts);
    for (int t = 0; t < num_tokens; ++t)
        for (int e = 0; e < num_experts; ++e) {
            float s = 0;
            for (int h = 0; h < hidden_size; ++h)
                s += x[t * hidden_size + h] * w_gate[h * num_experts + e];
            logits[t * num_experts + e] = s;
        }

    // Stage 1: Routing (Softmax + TopK)
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

    // Stage 3~7: Permute → Expert FFN → Unpermute+Combine
    y_ref.assign(num_tokens * hidden_size, 0.0f);
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int e = tk_idx[t * top_k + k];
            float w = tk_w[t * top_k + k];

            // GEMM1 + SiLU
            std::vector<float> inter(intermediate_size);
            for (int j = 0; j < intermediate_size; ++j) {
                float s = 0;
                for (int h = 0; h < hidden_size; ++h)
                    s += x[t * hidden_size + h] *
                         w1[(e * hidden_size + h) * intermediate_size + j];
                inter[j] = s / (1.0f + std::exp(-s));  // SiLU
            }
            // GEMM2 + weighted combine
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
// Main: 编排 MoE Pipeline (Stage 0 → 7)
// ============================================================================
int main() {
    // ─── 配置 ───
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

    // ─── Device 内存分配 ───
    float *d_x, *d_w_gate, *d_w1, *d_w2;
    float *d_logits, *d_topk_w, *d_y;
    float *d_sorted_weights, *d_fc1_out, *d_permuted_output;
    int *d_topk_idx, *d_expert_count, *d_expert_write_pos;
    int *d_sorted_token_ids, *d_sorted_expert_ids, *d_token_to_sorted;

    // 输入权重
    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_gate, h_w_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w1, h_w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2, h_w2.size() * sizeof(float)));

    // 中间结果
    CHECK_CUDA(cudaMalloc(&d_logits, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_w, num_tokens * top_k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_idx, num_tokens * top_k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_count, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_write_pos, num_experts * sizeof(int)));

    // 排列相关
    CHECK_CUDA(cudaMalloc(&d_sorted_token_ids, max_permuted * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_expert_ids, max_permuted * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_weights, max_permuted * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_token_to_sorted, max_permuted * sizeof(int)));

    // FFN 中间结果
    CHECK_CUDA(
        cudaMalloc(&d_fc1_out, max_permuted * intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_permuted_output,
                          max_permuted * hidden_size * sizeof(float)));

    // 输出
    CHECK_CUDA(cudaMalloc(&d_y, num_tokens * hidden_size * sizeof(float)));

    // H2D 拷贝
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
    // Pipeline 执行 (Stage 0 → 7)
    // ═══════════════════════════════════════════════════════════════

    std::cout << "=== TRT-LLM Style MoE Pipeline (V1) ===\n";
    std::cout << "num_tokens=" << num_tokens << " hidden=" << hidden_size
              << " inter=" << intermediate_size << " experts=" << num_experts
              << " top_k=" << top_k << "\n\n";

    // ── Stage 0: Gating GEMM ──
    //    TRT-LLM: cuBLAS (外部)
    gating_logits_kernel<<<num_tokens, num_experts>>>(
        d_x, d_w_gate, d_logits, num_tokens, hidden_size, num_experts);

    // ── Stage 1: Routing (Softmax + TopK) ──
    //    TRT-LLM: customMoeRoutingKernel
    softmax_topk_kernel<<<num_tokens, 1>>>(d_logits, d_topk_w, d_topk_idx,
                                           num_tokens, num_experts, top_k);

    // ── Stage 2: Align (Count + Prefix Sum) ──
    //    TRT-LLM: moe_align_block_size_kernel (fused count + CUB prefix sum + padding)
    //    V1: split into device count + host prefix sum
    int threads_s = 128;
    int blocks_s = (num_tokens * top_k + threads_s - 1) / threads_s;
    count_tokens_per_expert_kernel<<<blocks_s, threads_s>>>(
        d_topk_idx, d_expert_count, num_tokens, top_k);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Host-side exclusive prefix sum (V3 → CUB DeviceScan)
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

    // ── Stage 4+5: Expert GEMM1 + SiLU Activation ──
    //    TRT-LLM: CUTLASS moeGemm (FC1) + moeActivationKernel
    expert_gemm1_act_kernel<<<total_permuted, intermediate_size>>>(
        d_x, d_w1, d_sorted_token_ids, d_sorted_expert_ids, d_fc1_out,
        total_permuted, hidden_size, intermediate_size);

    // ── Stage 6: Expert GEMM2 ──
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
    // 结果验证
    // ═══════════════════════════════════════════════════════════════

    // 回传 GPU 结果
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

    // 打印路由信息
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

    // CPU reference 验证
    std::vector<float> y_ref;
    moe_cpu_reference(h_x, h_w_gate, h_w1, h_w2, y_ref, num_tokens,
                      hidden_size, intermediate_size, num_experts, top_k);

    float max_err = 0.0f;
    for (int i = 0; i < num_tokens * hidden_size; ++i)
        max_err = std::max(max_err, std::abs(h_y[i] - y_ref[i]));

    std::cout << "\n--- Verification ---\n";
    std::cout << "max |GPU - CPU| = " << max_err
              << (max_err < 1e-3f ? "  PASS" : "  FAIL") << "\n";

    // 打印 token 0 的输出前 8 个元素
    std::cout << "\n--- Output y[0][:8] ---\n";
    for (int h = 0; h < std::min(8, hidden_size); ++h)
        std::cout << h_y[h] << (h == std::min(7, hidden_size - 1) ? "\n" : ", ");

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
