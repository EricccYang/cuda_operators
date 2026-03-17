# Norm 优化系列（RMSNorm）

对每行做 RMS 归一化：`x_i / sqrt(mean(x^2))`。多种实现，从“每线程单元素 + 共享内存归约”到“每线程多元素 + warp shuffle”，以及固定 head 维度的专用版本。

---

## 1. 各版本优化与效果概览

| 版本 | 文件 | 主要优化 | 效果简述 |
|------|------|----------|----------|
| **v1** | `rmsnorm_v1.cu` | 每线程 1 元素，平方与数据放 shared，树形归约后写回 | 基线；每行 512 线程，shared 读写与 sync 较多 |
| **v2** | `rmsnorm_v2.cu` | 每线程 4 元素（DATA_PER_THREAD=4），寄存器存 r_data/squ_sum，仅部分和进 shared 归约 | 每 block 管 512×4 个数，grid 缩小，寄存器复用、减少 shared 占用 |
| **v3** | `rmsnorm_v3.cu` | 每线程 2 元素；树归约到 stride=32 后，用 volatile 展开的 warp 内归约 | 最后 32 不用 __syncthreads，减少同步与 shared 访问 |
| **v4** | `rmsnorm_v4.cu` | 同 v3 思路，最后 32 用 __shfl_down_sync 做 warp reduce | 与 v3 等价语义，warp shuffle 更省 shared、无 bank 冲突 |
| **v5** | `rmsnorm_v5.cu` | 树归约部分按 stride 展开（256/128/64/32），再 warp shuffle | 减少循环分支与 __syncthreads 次数，归约路径更固定 |
| **head128 / head256** | `rmsnorm_head128.cu` | 固定 head 维 128/256：1 warp 一行，4/8 元素每线程，仅 warp shuffle，无 shared | 适配 attention head；无 shared、无 block sync，延迟最低 |
| **flashinfer 风格** | `rmsnorm_flashinfer.cu` | 2D block，每 warp 管一行的一段；warp 内 shuffle 得部分和，再 shared 跨 warp 归约 | 多 warp 协作一行，适合较长序列或与其它 kernel 融合 |

---

## 2. NCU 指标预期变化与原因

| 优化阶段 | 预期变化的 NCU 指标 | 变化方向 | 原因简述 |
|----------|----------------------|----------|----------|
| **v1 → v2** | Grid size、Shared Memory 使用量 | grid ↓，smem 压力 ↓ | 每 block 管 4 倍数据，block 数减少；平方和在寄存器合并后再写 shared，shared 只存 512 个部分和 |
| **v2 → v3/v4** | __syncthreads 次数、Warp Execution Efficiency | sync ↓，warp 内更高效 | 最后 32 用 shuffle/volatile 归约，不再用 shared + 多次 sync |
| **v4 → v5** | 分支、Sync 开销 | 略降 | 树归约 stride 展开为固定步，循环更短、行为更可预测 |
| **通用 v → head128/256** | Shared Memory、Sync、Block 配置 | smem=0，sync=0 | 一行 1 warp，全程寄存器 + shuffle，无 block 级同步，适合小固定维 |
| **head128/256** | Memory Throughput、Occupancy | 视 shape 而定 | 小 head 时 block 数多、每 block 工作量小，可能更易受 launch/调度影响；大 batch 时收益明显 |

说明：实际数值需在 `norm` 目录下用 `rmsnorm.sh` 生成 `.ncu-rep` 后，在 Nsight Compute 中查看对应 kernel 指标。

---

## 3. 各版本逻辑简述

- **v1**：每行 512 线程，每线程 1 元素 → shared 存元素与平方 → 树归约得 sum_sq → 开方得 rms → 写回 normalized。
- **v2**：每线程 4 元素，寄存器算平方和并合并为 1 个部分和 → 写入 shared[512] → 树归约 → rms → 用寄存器中的 r_data 写回 4 个结果。
- **v3/v4**：每线程 2 元素，部分和进 shared；树归约到 32 后，v3 用 volatile 展开加和，v4 用 __shfl_down_sync 归约到 lane0，再开方、写回。
- **v5**：与 v4 相同数据与 warp 归约，树归约部分用 if(BLOCK_SIZE>=512) 等展开为 256→128→64→32。
- **head128/256**：一行 128 或 256 维，1 block 1 行、1 warp 32 线程；每线程 4 或 8 元素，warp_reduce_sum(squ_sum) 后 lane0 算 rms，__shfl_sync 广播，再写回；无 shared、无 __syncthreads。
- **flashinfer**：2D block，每 warp 负责一段数据，warp 内先 shuffle 得到部分和，再写入 shared，由部分线程做跨 warp 归约，得到 rms 后写回。

---

## 4. 如何复现 NCU 报告

```bash
cd norm
./rmsnorm.sh all    # 编译 v1～v5 并生成 ncu-rep
# 或
./rmsnorm.sh build && ./rmsnorm.sh analyze
```

`rmsnorm_head128.cu`、`rmsnorm_flashinfer.cu` 需单独编译。生成的 `rmsnorm_v*.ncu-rep` 用 Nsight Compute 打开，在对应 kernel 的 Summary 中查看 Memory、Compute、Warp Execution Efficiency、Sync 等指标。
