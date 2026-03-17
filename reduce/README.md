# Reduce 优化系列

对 `int` 数组做 block 内求和归约的多种实现，逐步优化访存与并行效率。数据规模默认 `n = 2<<24`，block 大小 512。

---

## 1. 各版本优化与效果概览

| 版本 | 主要优化 | 效果简述 |
|------|----------|----------|
| **v1** | 朴素归约：`tid % (stride*2)==0` 参与加和，in-place 写回 global memory | 基线；warp 内大量分支分化，且每轮都读写的都是 global memory，延迟高 |
| **v2** | 连续索引：`idx = tid*2*stride`，只让“前一半”线程参与，且索引连续 | 消除 warp 分化，同一 warp 内线程同进同出，提高 Warp Execution Efficiency |
| **v3** | 二分树 + 递减 stride：`stride = blockDim.x/2` 开始每次减半，`tid < stride` 的线程参与 | 归约树更规整，访存模式更连续，减少 global 读写的总轮次与冲突 |
| **v4** | 每 block 管 2 段数据：先把相邻第二个 half-block 加到第一个，再在首段上做树归约；`grid = grid_size/2` | 启动的 block 数减半，kernel 启动与调度开销减半；每 block 工作量翻倍，更好地隐藏延迟 |
| **v5-unroll8** | 每 block 管 8 段数据：`#pragma unroll` 循环 7 次做 8 段累加；最后 32 元素用 volatile 展开的 warp 内归约；`grid = grid_size/8` | block 数变为 1/8，启动与全局调度开销大幅下降；每 block 做 8× 数据，算术强度提高；最后 32 不用 `__syncthreads`，减少同步开销 |
| **v*_sm** | 在对应版本逻辑基础上，用 `__shared__` 做归约（先 load 到 smem，再在 smem 上树归约，最后写回 1 个结果到 global） | 大幅减少对 global memory 的读写次数与带宽占用，提高 DRAM/L2 利用率与带宽占比的“有效性” |

---

## 2. NCU 指标预期变化与原因

使用 `reduce.sh` 可编译并生成 NCU 报告（`ncu --set basic -o <name> ./<name>`）。下面给出各优化对**典型 NCU 指标**的预期影响及原因。

| 优化阶段 | 预期变化的 NCU 指标 | 变化方向 | 原因简述 |
|----------|----------------------|----------|----------|
| **v1 → v2** | Warp Execution Efficiency | ↑ 明显提升 | v1 中 `tid % (stride*2)==0` 导致同一 warp 内部分线程干活、部分不干活，严重分化；v2 连续索引使 warp 内线程行为一致 |
| **v1 → v2** | SM occupancy / Active warps | 可能略好 | 分化减少，更多 warp 能有效执行，调度更均衡 |
| **v2 → v3** | Memory Throughput (DRAM/L2)、Global Load/Store | 优化（带宽更“用在刀刃上”） | 递减 stride 的树归约访存更规整，重复读写减少，相同数据量下有效带宽利用更好 |
| **v3 → v4** | Grid size、Kernel launch / scheduler | grid 减半 | block 数减半，启动与调度次数减半；单 block 内计算/访存比提高 |
| **v4 → v5-unroll8** | Grid size、Blocks per SM | grid 再降为约 1/4 | block 数变为 1/8，启动与全局调度开销进一步下降；每 block 做 8 段累加，算术强度更高 |
| **v5-unroll8**（最后 32 用 warp 内归约） | `__syncthreads()` 次数、Sync overhead | 减少 | 最后 32 元素不再用 shared memory + `__syncthreads()`，改用 volatile 或 warp shuffle，同步开销下降 |
| **任意 v → v_sm** | DRAM Throughput、Global Load/Store 次数 | ↓ 明显下降 | 归约过程在 shared memory 内完成，每个 block 只对 global 做 1 次 block 宽度 load + 1 次 1 个元素的 store |
| **任意 v → v_sm** | Shared Memory 使用、L1/Shared 带宽 | ↑ 增加 | 显式使用 `__shared__` 做中间归约，L1/Shared 带宽占比提升，利于把带宽留给有用负载 |
| **任意 v → v_sm** | Memory [%] 或 Compute [%] | Compute 占比相对提升 | 同一算力下 global 访存减少，kernel 从“访存 bound”向“计算 bound”或更均衡过渡 |

说明：实际数值需在本地用 `ncu` 跑出 `.ncu-rep` 后，在 Nsight Compute UI 中查看对应 kernel 的上述指标并填入上表。

---

## 3. 各版本逻辑简述

- **v1**：在 global `idata` 上，按 `stride=1,2,4,...` 做归约，仅 `tid % (stride*2)==0` 的线程加 `idata[tid+stride]`，写回 `idata[tid]`；warp 分化严重。
- **v2**：用 `idx = tid*2*stride` 做连续索引，`idx < blockDim.x` 时 `idata[idx] += idata[idx+stride]`，无分支分化；仍全在 global 上操作。
- **v3**：`stride` 从 `blockDim.x/2` 递减到 1，`tid < stride` 时 `idata[tid] += idata[tid+stride]`；仍是 global in-place，但树形规整。
- **v4**：每个 block 负责 2 个连续 half-block：先把后一半加到前一半，再在前一半上做 v3 的树归约；grid 为原来的 1/2。
- **v5-unroll8**：每个 block 负责 8 个连续 block 的数据，用 unroll 循环做 7 次加和到第一段，再在第一段上做树归约；最后 32 个数用展开的 warp 内归约（volatile 或可改为 `__shfl_down_sync`）；grid 为原来的 1/8。
- **v*_sm**：对应版本的“归约阶段”改为：先把本 block 要读的数据 load 到 `s_d[]`，在 `s_d[]` 上做树归约，最后 `g_odata[blockIdx.x] = s_d[0]`；减少对 global 的反复读写。

---

## 4. 如何复现 NCU 报告与填表

```bash
cd reduce
./reduce.sh all    # 编译并生成各版本的 ncu-rep
# 或
./reduce.sh build && ./reduce.sh analyze
```

生成的 `reduce_v1.ncu-rep`、`reduce_v2.ncu-rep`、…、`reduce_v5-unroll8_sm.ncu-rep` 用 Nsight Compute 打开后，在 kernel `reduce` 的 Summary 里可查看：

- **Warp Execution Efficiency**
- **Memory Workload Analysis**：DRAM Throughput、L2 Throughput、Global Load/Store
- **Scheduler**：Active warps、Blocks per SM、Grid/Block 配置
- **Sync**：若有 `__syncthreads()` 会显示同步相关开销

把上述指标按版本填入第 2 节表格的“实际数值”列，即可对比每次优化对 NCU 指标的具体影响。
