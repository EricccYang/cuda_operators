# Hopper GEMM 要完全理解细节，必须先搞懂的 3 个概念

上下文：为了真正看懂 [TensorRT-LLM PR #13253](https://github.com/NVIDIA/TensorRT-LLM/pull/13253) 这种"选 Pingpong 还是 Cooperative"的决策，以及所有 sm_90 生产级 GEMM kernel，下面 3 件事是必经之路。现在只是知道名词，要学到**能手写 / 能调参**的水平。

先列最重要的三个。后续补充进来 PtrArray、FP8 FastAccum、epilogue fusion、stream-K、cluster launch 等。

---

## 概念 1：Warp Specialization（producer/consumer 分工）

### 是什么

**把一个 block 里的 warp 按角色拆成两组**：
- **Producer warps**（通常 1 个 warp = 32 线程）：**只发 TMA**，把 gmem 数据搬到 smem，arrive 到 mbarrier
- **Consumer warps**（1~2 个 warp group = 128~256 线程）：**只做 WGMMA**，等 mbarrier 放行后消费 smem 数据

Producer 和 consumer **完全独立运行**，通过 mbarrier 做生产者-消费者信号传递。

### 为什么是基础

这是 sm_90 所有主流内核模板（`KernelTmaWarpSpecialized*`）的骨架。没搞懂这个，后面的 Pingpong/Cooperative、TMA multicast、persistent kernel 全是空中楼阁。

它解决的问题：在我们的 `gemm_wgmma_cute.cu` 里，同一批 threads 既发 cp.async 又跑 wgmma，**issue slot 互相挤**；WS 之后 producer 的 TMA 指令和 consumer 的 wgmma 指令在不同 warp，完全不抢占。

### 需要搞懂到什么程度

- [ ] 能说清为啥要专门拿 1 个 warp 做 producer，而不是让所有 warp 都发 TMA
- [ ] 能在 PTX 层描述出 `setmaxnreg` 的作用（producer warp 把寄存器预算让给 consumer）
- [ ] 能画出 producer/consumer 用 mbarrier 做 ping-pong 的时序图（"arrive/wait" 两侧）
- [ ] 能写出一个 minimal 的 "1 producer + 1 consumer warp group" kernel，用 inline PTX 或 CuTe

### 要读的东西

- **CUTLASS**: `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp`
  尤其看它的 `operator()` 里怎么分 warp 角色，`if (warp_group_idx == 0)` 分支那段
- **CUTLASS**: `include/cutlass/pipeline/sm90_pipeline.hpp`
  `PipelineTmaAsync` 是 producer/consumer mbarrier 同步的标准抽象
- **PTX docs**: `mbarrier.arrive.expect_tx`、`mbarrier.try_wait`、`setmaxnreg`

### 最小可跑验证

改造我们的 `gemm_wgmma_cute.cu`：
1. block 扩到 1 wg consumer (128) + 1 warp producer (32) = 160 threads
2. producer 发 cp.async（先不上 TMA），arrive 到 `cuda::barrier`
3. consumer 等 barrier，做 wgmma
4. 对比性能和 ncu 的 `No Eligible %`

---

## 概念 2：TMA + mbarrier 的完整配对

### 是什么

**TMA**（Tensor Memory Accelerator）是 sm_90 新增的硬件单元，一条指令搬整个 2D/3D tile：

```
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_dst], [tensor_map, {coord_x, coord_y}], [mbarrier]
```

工作流：
1. **Host** 端用 `cuTensorMapEncodeTiled` 构造 `CUtensorMap`（描述 gmem 张量的形状、步长、swizzle 模式）
2. **Device** 端 1 个线程发射 `cp.async.bulk.tensor`，传入 tensor_map、smem 目标、gmem 坐标、mbarrier
3. 硬件 Copy Engine 异步搬运，完成后在 mbarrier 上累加 tx_count
4. Consumer 用 `mbarrier.try_wait` 等 tx_count 达标

### 为什么是基础

- TMA 是 Hopper 达到 gmem 饱和带宽的**唯一**靠谱方式
- 它的 swizzle 模式（`CU_TENSOR_MAP_SWIZZLE_128B` 等）和 wgmma descriptor 的 swizzle 字段**必须匹配**，否则数据错位
- mbarrier 的 `arrive_and_expect_tx` + `complete_tx` 是 producer/consumer 的通信基石

### 需要搞懂到什么程度

- [ ] 能手写 host 端 `cuTensorMapEncodeTiled` 调用，每个参数（`globalDim`、`globalStride`、`boxDim`、`elementStrides`、`swizzle`）都知道含义
- [ ] 理解 `tx_count`（transaction count）和 arrival count 的区别——TMA 只贡献前者，普通 thread `arrive` 贡献后者
- [ ] 能描述 TMA multicast（1 条 TMA 把同一块 gmem 搬到多个 CTA 的 smem）
- [ ] 知道为什么 TMA 要求 smem 128-byte 对齐，以及 swizzle 模式怎么选（128B/64B/32B/None 和 box inner dim 的关系）

### 要读的东西

- **CUTLASS CuTe**: `include/cute/atom/copy_traits_sm90_tma.hpp`
  看 `Copy_Traits<SM90_TMA_LOAD>` 怎么包 descriptor
- **CUTLASS CuTe**: `include/cute/atom/copy_traits_sm90_tma_swizzle.hpp:45-112`
  swizzle 从 smem layout 自动推导到 `CUtensorMap` 参数
- **NVIDIA Hopper Architecture Whitepaper**: TMA 章节（包括 multicast）
- **PTX docs**: `cp.async.bulk.tensor`, `mbarrier.arrive.expect_tx`

### 最小可跑验证

我们的 `gemm_tma_wmma.cu` 已经走到 "单线程发 TMA + mbarrier" 这一步。需要继续的：
1. 加上 `CU_TENSOR_MAP_SWIZZLE_128B`，和 wgmma 匹配（需要换 wgmma）
2. 试 TMA multicast：2 个 block 共享一次 A tile load
3. ncu 里对比 "1 thread TMA" vs "256 threads cp.async"，看 `DRAM Throughput` 和指令数

---

## 概念 3：CUTLASS 的 `KernelSchedule` + `CollectiveMainloop` 模板组合系统

### 是什么

CUTLASS 的 Hopper kernel 不是一个单体，是**一堆可插拔的 policy 模板**组合出来的：

```cpp
// Kernel level (选调度策略)
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
//                                      ↑  ↑                 ↑
//                              TMA?   WS?         Pingpong / Cooperative / Auto

// Mainloop level (A、B 怎么从 gmem 进 smem 再喂 wgmma)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,           // <-- 比如 Shape<_64, _64, _128>
    ClusterShape,        // <-- 比如 Shape<_1, _1, _1>
    cutlass::gemm::collective::StageCountAuto,
    KernelSchedule       // <-- 把上面那个传进来
>::CollectiveOp;

// Epilogue level (C 怎么从 acc reg 写回 gmem，中间可以接 activation/bias/reduce)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ...
>::CollectiveOp;

// 最终组装
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler
>;
```

### 为什么是基础

- PR #13253 改的就是这个层面：`std::conditional_t<(CTA_M<128), Pingpong, Cooperative>` 在选 `KernelSchedule`
- 所有 production GEMM（cuBLAS、TensorRT-LLM、vLLM、SGLang）的 sm_90 路径都走这套
- 理解它之后，"改 tile 大小"、"换 epilogue fusion"、"加 stream-K"都变成改几行模板参数

### 需要搞懂到什么程度

- [ ] 能说清 `KernelSchedule`、`CollectiveMainloop`、`CollectiveEpilogue`、`TileScheduler` 这 4 个层面各自的职责
- [ ] 能读懂 PR 里的 `std::conditional_t<...>` 选型逻辑
- [ ] 能写出一个完整的 CUTLASS SGEMM host 端代码（不用 CuTe 手写 kernel，用 CollectiveBuilder）
- [ ] 能指出 `KernelSchedule` 里每个 token 的意思（`Tma`、`WarpSpecialized`、`Pingpong`、`Cooperative`、`PtrArray`、`FP8FastAccum`）
- [ ] 知道 `TileScheduler` 可选哪些（`Default`、`StreamK`、`PersistentTileScheduler`）

### 要读的东西

- **CUTLASS**: `include/cutlass/gemm/collective/collective_builder.hpp`
  CollectiveBuilder 的 dispatch 入口
- **CUTLASS**: `include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`
  sm_90 的具体分发
- **CUTLASS examples**: `examples/48_hopper_warp_specialized_gemm/`
  完整可跑的 Hopper GEMM，不是 tutorial 而是 production 写法
- **CUTLASS docs**: `media/docs/cpp/gemm_api_3x.md`
  介绍 3.x API（CuTe + Collective）的设计哲学

### 最小可跑验证

1. 把我们的 `gemm_wgmma_cute.cu`（手写 CuTe 内核）改成用 `CollectiveBuilder`
2. 对比 `KernelSchedule = KernelTmaWarpSpecialized{Pingpong,Cooperative}` 两种的 ncu 指标
3. 切 `TileScheduler` 到 StreamK，看对 small batch 的影响

---

## 三者之间的关系

```
     KernelSchedule (选什么策略)
       │
       ├─── CollectiveMainloop (拿对应实现)
       │      ├── TMA load  ←── 概念 2
       │      ├── WGMMA compute
       │      └── Warp Specialization  ←── 概念 1
       │
       └─── Pingpong / Cooperative (consumer warp group 调度)
              ↑
         这是 PR #13253 直接改的那个开关
```

**搞定顺序**：先 1（理解 WS），再 2（理解 TMA），最后 3（知道 CUTLASS 怎么把前两个包起来）。每个概念都有最小可跑验证，在我们已有的 kernel 基础上逐步加。

---

## 下次补充清单（暂记）

- [ ] **PtrArray grouped GEMM**（MoE 特化）
- [ ] **FP8 FastAccum**（精度换吞吐的累加路径）
- [ ] **Epilogue fusion**（store 时顺带做 activation / reduce）
- [ ] **Stream-K / Split-K**（小 batch 高 occupancy 调度）
- [ ] **Cluster launch + TMA multicast**（多 CTA 协作的硬件机制）
- [ ] **Persistent kernel + TileScheduler**（一个 block 跑多个 tile 的循环）
