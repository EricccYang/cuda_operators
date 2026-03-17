# CUDA 算子手写实现

手写 CUDA 算子实现，并与现有推理/基础库（如 FlashInfer）做性能对比。每个子模块包含多版逐步优化及 NCU 指标说明。

---

## 模块概览

| 模块 | 说明 | README |
|------|------|--------|
| **reduce** | 一维整型数组 block 内求和归约 | [reduce/README.md](reduce/README.md) |
| **sgemm** | 单精度矩阵乘 C = A×B | [sgemm/README.md](sgemm/README.md) |
| **norm** | RMSNorm 等归一化算子 | [norm/README.md](norm/README.md) |
| **attension / fused_qknorm_rope / other_cuda_related** | Attention、融合算子及其它 CUDA 相关 | — |

---

## Reduce：优化与 NCU 指标概要

| 版本 | 主要优化 | NCU 关注指标 |
|------|----------|----------------------|
| v1 | 朴素归约，global in-place，warp 分化 | Warp Execution Efficiency 低 |
| v2 | 连续索引去分化 | Warp Efficiency ↑ |
| v3 | 二分树、递减 stride | Global 访存更规整 |
| v4 | 每 block 管 2 段，grid 减半 | Grid size ↓，延迟隐藏更好 |
| v5-unroll8 | 每 block 管 8 段，最后 32 用 warp 归约 | Grid ↓，Sync 开销 ↓ |
| v*_sm | 用 shared memory 做归约 | DRAM/Global ↓，Shared 使用 ↑ |

详见 [reduce/README.md](reduce/README.md)。

---

## SGEMM：优化与 NCU 指标概要

| 版本 | 主要优化 | NCU 关注指标 |
|------|----------|----------------------|
| naive | 每线程一元素，全 global | DRAM 高、无复用 |
| gemm_sm (V1) | 分块 + shared + FLOAT4 | Global ↓，算术强度 ↑ |
| gemm_bank_cf (V2) | s_a 按 K 存，消 bank 冲突 | Shared 冲突 ↓ |
| gemm_double_buffer (V3) | 双缓冲，算与加载重叠 | 延迟隐藏、SM 利用率 ↑ |

详见 [sgemm/README.md](sgemm/README.md)。

---

## Norm (RMSNorm)：优化与 NCU 指标概要

| 版本 | 主要优化 | NCU 关注指标 |
|------|----------|----------------------|
| v1 | 1 线程 1 元素，shared 树归约 | 基线 |
| v2 | 每线程 4 元素，寄存器部分和 | Grid ↓，shared 压力 ↓ |
| v3/v4 | 最后 32 用 volatile / warp shuffle | __syncthreads ↓ |
| v5 | 树归约 stride 展开 | 分支与 sync 略降 |
| head128/256 | 固定 head 维，1 warp 一行，仅 shuffle | 无 shared、无 block sync |
| flashinfer 风格 | 多 warp 一行，warp reduce + shared | 适配长序列/融合 |

详见 [norm/README.md](norm/README.md)。

---

## 如何跑 NCU

- **reduce**：`cd reduce && ./reduce.sh all`
- **norm**：`cd norm && ./rmsnorm.sh all`
- **sgemm**：各 `.cu` 独立编译后执行 `ncu --set basic -o <out> ./<exe>`

用 Nsight Compute 打开生成的 `.ncu-rep` 查看各 kernel 的 Memory、Compute、Warp Efficiency、Sync 等指标，并与各子目录 README 中的表格对照。
