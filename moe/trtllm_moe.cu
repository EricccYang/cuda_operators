
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ------------------------------
// 1) Gating: logits = x * W_gate
// ------------------------------
__global__ void gating_logits_kernel(const float* x, const float* w_gate,
                                     float* logits, int num_tokens,
                                     int hidden_size, int num_experts) {
    int t = blockIdx.x;
    int e = threadIdx.x;
    if (t >= num_tokens || e >= num_experts) return;

    float sum = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
        sum += x[t * hidden_size + h] * w_gate[h * num_experts + e];
    }
    logits[t * num_experts + e] = sum;
}

// --------------------------------------------------------
// 2) Softmax + TopK
//    简版：1 block 处理 1 token，仅用于演示
// --------------------------------------------------------
__global__ void softmax_topk_kernel(const float* logits, float* topk_weight,
                                    int* topk_idx, int num_tokens,
                                    int num_experts, int top_k) {
    int t = blockIdx.x;
    if (t >= num_tokens || threadIdx.x != 0) return;

    const float* row = logits + t * num_experts;

    float max_logit = row[0];
    for (int e = 1; e < num_experts; ++e) max_logit = fmaxf(max_logit, row[e]);

    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; ++e) sum_exp += expf(row[e] - max_logit);

    // 取 top-k（极简 O(E*K)）
    for (int k = 0; k < top_k; ++k) {
        int best_e = -1;
        float best_p = -1.0f;
        for (int e = 0; e < num_experts; ++e) {
            float p = expf(row[e] - max_logit) / sum_exp;
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

    // 常见做法：对 top-k 权重再归一化
    float topk_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) topk_sum += topk_weight[t * top_k + k];
    for (int k = 0; k < top_k; ++k) topk_weight[t * top_k + k] /= topk_sum;
}

// -------------------------------------------------------
// 3) Dispatch: 将 token 分桶到各个 expert
//    类似 TRT-LLM 里 prefix-sum/permute 的前置目标
// -------------------------------------------------------
__global__ void dispatch_kernel(const int* topk_idx, const float* topk_weight,
                                int* expert_count, int* expert_token,
                                float* expert_token_w, int num_tokens,
                                int num_experts, int top_k,
                                int max_tokens_per_expert) {
    int t = blockIdx.x;
    int k = threadIdx.x;
    if (t >= num_tokens || k >= top_k) return;

    int e = topk_idx[t * top_k + k];
    float w = topk_weight[t * top_k + k];

    int slot = atomicAdd(&expert_count[e], 1);
    if (slot < max_tokens_per_expert) {
        int offset = e * max_tokens_per_expert + slot;
        expert_token[offset] = t;
        expert_token_w[offset] = w;
    }
}

// ----------------------------------------------------
// 4) Expert compute + combine
//    这里用 "逐元素缩放" 代替 MLP，便于聚焦 MoE 流程
// ----------------------------------------------------
__global__ void expert_compute_combine_kernel(
    const float* x, const float* expert_scale, const int* expert_count,
    const int* expert_token, const float* expert_token_w, float* y,
    int hidden_size, int num_experts, int max_tokens_per_expert) {
    int e = blockIdx.x;
    int slot = blockIdx.y;
    int h = threadIdx.x;
    if (e >= num_experts || h >= hidden_size) return;

    int count = expert_count[e];
    if (slot >= count || slot >= max_tokens_per_expert) return;

    int token = expert_token[e * max_tokens_per_expert + slot];
    float w = expert_token_w[e * max_tokens_per_expert + slot];

    float in_v = x[token * hidden_size + h];
    float expert_v = in_v * expert_scale[e * hidden_size + h];
    atomicAdd(&y[token * hidden_size + h], w * expert_v);
}

int main() {
    // -------- demo 配置 --------
    constexpr int num_tokens = 4;
    constexpr int hidden_size = 8;
    constexpr int num_experts = 3;
    constexpr int top_k = 2;
    constexpr int max_tokens_per_expert = num_tokens * top_k;

    // -------- host 数据 --------
    std::vector<float> h_x(num_tokens * hidden_size);
    std::vector<float> h_w_gate(hidden_size * num_experts);
    std::vector<float> h_expert_scale(num_experts * hidden_size);

    for (int i = 0; i < static_cast<int>(h_x.size()); ++i) {
        h_x[i] = 0.1f * (i % 13 + 1);
    }
    for (int i = 0; i < static_cast<int>(h_w_gate.size()); ++i) {
        h_w_gate[i] = 0.05f * ((i % 7) - 3);
    }
    for (int e = 0; e < num_experts; ++e) {
        for (int h = 0; h < hidden_size; ++h) {
            h_expert_scale[e * hidden_size + h] = 0.8f + 0.2f * (e + 1);
        }
    }

    // -------- device 内存 --------
    float *d_x = nullptr, *d_w_gate = nullptr, *d_expert_scale = nullptr;
    float *d_logits = nullptr, *d_topk_w = nullptr, *d_y = nullptr;
    int *d_topk_idx = nullptr, *d_expert_count = nullptr, *d_expert_token = nullptr;
    float* d_expert_token_w = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_gate, h_w_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_scale, h_expert_scale.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_w, num_tokens * top_k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_idx, num_tokens * top_k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_count, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_token,
                          num_experts * max_tokens_per_expert * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_token_w,
                          num_experts * max_tokens_per_expert * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, num_tokens * hidden_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w_gate, h_w_gate.data(), h_w_gate.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_scale, h_expert_scale.data(),
                          h_expert_scale.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_expert_count, 0, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_y, 0, num_tokens * hidden_size * sizeof(float)));

    // -------- pipeline --------
    gating_logits_kernel<<<num_tokens, num_experts>>>(d_x, d_w_gate, d_logits,
                                                      num_tokens, hidden_size,
                                                      num_experts);
    softmax_topk_kernel<<<num_tokens, 1>>>(d_logits, d_topk_w, d_topk_idx,
                                           num_tokens, num_experts, top_k);
    dispatch_kernel<<<num_tokens, top_k>>>(d_topk_idx, d_topk_w, d_expert_count,
                                           d_expert_token, d_expert_token_w,
                                           num_tokens, num_experts, top_k,
                                           max_tokens_per_expert);
    dim3 grid(num_experts, max_tokens_per_expert);
    expert_compute_combine_kernel<<<grid, hidden_size>>>(
        d_x, d_expert_scale, d_expert_count, d_expert_token, d_expert_token_w,
        d_y, hidden_size, num_experts, max_tokens_per_expert);
    CHECK_CUDA(cudaDeviceSynchronize());

    // -------- 回传并打印 --------
    std::vector<int> h_topk_idx(num_tokens * top_k);
    std::vector<float> h_topk_w(num_tokens * top_k);
    std::vector<int> h_expert_count(num_experts);
    std::vector<float> h_y(num_tokens * hidden_size);

    CHECK_CUDA(cudaMemcpy(h_topk_idx.data(), d_topk_idx,
                          h_topk_idx.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_topk_w.data(), d_topk_w, h_topk_w.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_expert_count.data(), d_expert_count,
                          h_expert_count.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "=== TopK routing ===\n";
    for (int t = 0; t < num_tokens; ++t) {
        std::cout << "token " << t << ": ";
        for (int k = 0; k < top_k; ++k) {
            std::cout << "(e=" << h_topk_idx[t * top_k + k]
                      << ", w=" << h_topk_w[t * top_k + k] << ") ";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Expert token counts ===\n";
    for (int e = 0; e < num_experts; ++e) {
        std::cout << "expert " << e << ": " << h_expert_count[e] << "\n";
    }

    std::cout << "\n=== MoE output y (token 0 shown) ===\n";
    for (int h = 0; h < hidden_size; ++h) {
        std::cout << h_y[h] << (h + 1 == hidden_size ? "\n" : ", ");
    }

    // -------- 释放 --------
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w_gate));
    CHECK_CUDA(cudaFree(d_expert_scale));
    CHECK_CUDA(cudaFree(d_logits));
    CHECK_CUDA(cudaFree(d_topk_w));
    CHECK_CUDA(cudaFree(d_topk_idx));
    CHECK_CUDA(cudaFree(d_expert_count));
    CHECK_CUDA(cudaFree(d_expert_token));
    CHECK_CUDA(cudaFree(d_expert_token_w));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}

