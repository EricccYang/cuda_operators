# SGEMM 优化系列

单精度矩阵乘 C = A×B 的多种实现，A[M,K]、B[K,N]、C[M,N]。逐步优化访存、共享内存布局与计算/加载重叠。

---

## 1. 各版本优化与效果概览

| 版本 | 文件 | 主要优化 | 效果简述 |
|------|------|----------|----------|
| **Naive** | `gemm_naive.cu` | 每线程算一个 C 元素，K 维在 global 上循环 | 基线；global 访存次数多、带宽利用率低，无复用 |
| **V1 Shared Memory** | `gemm_sm.cu` | Block 分块：BM=128, BN=128, BK=8；TM=TN=8 每线程 8×8 结果；FLOAT4 向量化 load/store | 大幅减少 global 访问，共享内存复用 A/B 块，算术强度提升 |
| **V2 Bank Conflict** | `gemm_bank_cf.cu` | s_a 存为 `[BK][BM]`（按 K 行存），计算时按 k 索引取；寄存器 r_comp_a/r_comp_b 缓存 | 消除 shared 按行读时的 bank conflict，提高 SM 访存吞吐 |
| **V3 Double Buffer** | `gemm_double_buffer.cu` | s_a/s_b 双缓冲 `[2][BK][BM]`、`[2][BK][BN]`；当前步算上一块、同时加载下一块 | 隐藏 global→shared 延迟，计算与加载重叠，提高 SM 利用率 |
| **Example** | `gemm_v1_example.cu` | 与 V3 同思路的双缓冲实现，可作为参考 | 与 `gemm_double_buffer.cu` 对照 |

---

## 2. NCU 指标预期变化与原因

| 优化阶段 | 预期变化的 NCU 指标 | 变化方向 | 原因简述 |
|----------|----------------------|----------|----------|
| **Naive → V1 (sm)** | DRAM Throughput、Global Load/Store 次数 | ↓ 明显下降 | 分块后每数据从 global 只读一次、在 shared 中复用，总 global 流量大幅减少 |
| **Naive → V1** | Compute Throughput、SM 利用率 | ↑ 提升 | 算术强度提高，更多时间在算而非等内存 |
| **V1 → V2 (bank_cf)** | Shared Memory Throughput、Bank Conflicts | 冲突 ↓，有效带宽 ↑ | s_a 按 K 维连续存，同一 warp 读不同 k 时不再命中同一 bank |
| **V2 → V3 (double_buffer)** | Pipeline 利用率、Memory Latency 隐藏 | ↑ 提升 | 加载下一块与计算当前块重叠，减少 stall，SM 更饱和 |

说明：实际数值需用 `ncu` 生成报告后，在 Nsight Compute 中查看对应 kernel 的 Memory Workload、Compute、Scheduler 等指标。

---

## 3. 各版本逻辑简述

- **Naive**：`(ix, iy)` 对应 C 的一格，内层 k 循环在 global 上累加 A[iy,k]*B[k,ix]，结果写 C[iy,ix]。
- **V1 (gemm_sm)**：每个 block 负责 C 的 128×128 块；按 K 方向每次取 8，把 A 的 128×8、B 的 8×128 装入 s_a、s_b；块内每线程算 8×8 的 C 子块，用寄存器累加；FLOAT4 读写。
- **V2 (gemm_bank_cf)**：在 V1 基础上，A 装入 shared 时按“k 为行”存成 s_a[BK][BM]，计算时沿 k 取不会 bank 冲突；并用寄存器缓存当前 k 的 A/B 行/列以减轻 shared 读压力。
- **V3 (gemm_double_buffer)**：s_a/s_b 为双缓冲；循环中“算”用上一轮加载的缓冲，“加载”写当前轮缓冲，下一轮交换，实现算与加载重叠。

---

## 4. 编译与运行

各 `.cu` 为独立可执行程序，编译示例：

```bash
cd sgemm
nvcc -o gemm_naive gemm_naive.cu
nvcc -o gemm_sm gemm_sm.cu
# ...
```

生成 NCU 报告（若已安装 ncu）：

```bash
ncu --set basic -o gemm_sm ./gemm_sm
```

在 Nsight Compute 中打开生成的 `.ncu-rep` 查看各 kernel 的详细指标。
