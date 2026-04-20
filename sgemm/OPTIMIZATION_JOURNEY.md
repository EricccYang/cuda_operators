# SGEMM 优化迭代记录（H100 篇）

从 V3 double_buffer 作为 SIMT baseline，逐步往 Hopper 原生路线推进的迭代日志。每一版记录：**改了什么、期望什么、实测怎样、ncu 看到什么、学到什么**。

测试平台：**NVIDIA H100 PCIe**（driver 570, CUDA 12.8, ncu 2025.1.1）。所有 ncu 采样均为 M=N=2048, K=1024 的第 86 次 launch (`--launch-skip 85 --launch-count 1`)。GFlops 均为 M=N=4096, K=1024 下的 10 次平均。

---

## 1. 版本一览（按时间）

| # | 版本 | 文件 | 4096 GFlops | Duration (us) | vs dbuf | 一句话 |
|---|---|---|---:|---:|---:|---|
| 0 | dbuf | `gemm_double_buffer.cu` | **34440** | 502 | — | SIMT + 2-stage 寄存器中转 baseline |
| 1 | multistage | `gemm_multistage.cu` | 22215 | 715 | **-35%** ❌ | cp.async + STAGES=3 反而慢 |
| 2 | multistage_padded | `gemm_multistage_padded.cu` | 23673 | 671 | -31% ❌ | 给 s_a 的 M 维加 PAD=4 补救 bank |
| 3 | warpspec | `gemm_warpspec.cu` | 19318 | 870 | -44% ❌ | 角色拆分实验，producer warp 拖累 |
| 4 | tc_wmma | `gemm_tc_wmma.cu` | **50177** | 331 | **+46%** ✅ | 换 TensorCore (wmma TF32 m16n16k8) 第一次真正跳档 |
| 5 | tma_wmma | `gemm_tma_wmma.cu` | 43721 | 339 | +27% | 换 TMA，BK 扩到 16，bank 反而更差 |
| 6 | tc_wmma_padded | `gemm_tc_wmma_padded.cu` | 49186 | 337 | +43% | K 维 PAD=4 破 bank，确认诊断用 |
| 7 | **wgmma_cute** | `gemm_wgmma_cute.cu` | **117552** | **124** | **+241%** 🔥 | CUTLASS CuTe + SW128 + wgmma，swizzle 闭环 |

---

## 2. 逐版本解析

### V0. dbuf (baseline)

SIMT 每 thread 算 8×8 外积，s_a 转置存 `[BK][BM]`，gmem→reg→smem 做隐式转置。

**关键点**：
- 1 条 `FLOAT4` LDG + 4 条 STS 写转置 smem，寄存器中转
- double buffer 2 stage，隐式 async via warp scheduler

**NCU**：SM throughput 59%, IPC 3.23, No Eligible 19%（相当健康）。

**结论**：SIMT 外积的理论天花板附近，H100 FP32 CUDA core 峰值 ~67 TFLOPS，我们占了 52%。

---

### V1. multistage (cp.async + STAGES=3)

**改动**：把 gmem→smem 换成 `__pipeline_memcpy_async`，stage 数从 2 扩到 3。期望更好地隐藏 gmem 延迟。

**实测**：慢了 35%。NCU 真相：
- `cp.async` 一条最多 16B，但我们要把 A 转置到 s_a[k][m]，gmem 连续 16B 要写到 smem 4 个不同位置 → **每 thread 拆成 4× 4B cp.async**
- 4B cp.async 读 gmem 粒度极小 → **59% uncoalesced global sectors**（ncu 里明写）
- 指令数从 151M 涨到 158M

**教训**：**"先进的 pipeline 不一定更快"**。H100 DRAM 本来就不是瓶颈（DRAM throughput 只有 1.9%），cp.async 的"异步隐藏延迟"无处发力，反而被 4B 拆分的副作用吃光。

---

### V2. multistage_padded (给 s_a 加 M 维 padding)

**改动**：`s_a[STAGES][BK][BM + 4]`。PAD_M=4 让 4 行 smem 的 bank 位置正好错开半圈（`4×132 mod 32 = 16`），避免写 smem 时偶奇 tid 撞 bank。

**实测**：速度从 22215 → 23673（+6%），但**bank conflict 比例基本没变**（38% → 37%）。

原因：**主要瓶颈不是 smem 写的 bank conflict**（那只是症状），是 **4B cp.async 导致的 gmem uncoalesced 59%**。padding 补的是错的地方。

**教训**：ncu 会报多个 OPT 诊断，要定位**真正的瓶颈**，不是随便挑一个。

---

### V3. warpspec（实验失败存档）

**改动**：blockDim 从 256 扩到 320，加 2 个 producer warp 专门发 cp.async，其他 warp 只做 compute。

**实测**：19318 GFlops，更糟。原因：compute warp 数没变，**额外的 producer 只是稀释 SM**，occupancy 从 23% 掉到 15%。

**教训**：真正的 WS 要配合硬件 async path（TMA + mbarrier），手动拆 warp 做 producer 在 SIMT cp.async 上不对味。

---

### V4. tc_wmma (换 TensorCore, 跳档)

**改动**：
- A 不再转置，`s_a[STAGES][BM][BK]` 自然布局 → cp.async 回到 16B
- compute 换成 wmma `mma.sync.m16n16k8` TF32，一条顶 2048 条 FFMA
- Tile 规划：BM×BN=128×128, warp tile 64×32（8 warps 2×4 排布），MMA 16×16×8

**实测**：**50177 GFlops (+46%)**，Duration 502→332us。NCU：
- Executed Instructions 从 150.9M **砍到 46.4M**（指令密度跳级）
- DRAM Throughput 从 1.9% 升到 2.95%（cp.async 本来想做的事情现在能做了）
- 但 **No Eligible 从 19% 飙到 60%**，L1/TEX throughput 65% → warp 大量时间在等 LDS

**结论**：第一次真正跨越 SIMT 天花板。但瓶颈转移到 **smem→register 的 LDS bank conflict**（62% excessive wavefronts, 2.7-way bank conflict）。

---

### V5. tma_wmma (换 TMA，BK 扩到 16)

**改动**：
- gmem→smem 用 `cp.async.bulk.tensor`（TMA），host 侧构造 `CUtensorMap`
- 同步用 `cuda::barrier` (mbarrier)
- BK 从 8 扩到 16 摊 TMA 启动开销
- STAGES=2（smem 够用）

**实测**：速度**反而退到 43721 GFlops**（比 tc_wmma 慢 13%）。NCU：
- 指令数降一点点 (46.4M → 46.0M)，TMA issue 极省
- 但 **excessive wavefronts 从 62% 飙到 75%**，bank conflict 从 2.7-way 恶化到 **4.0-way**

**原因**：**BK=16 把 wmma 的 LDS bank 打得更糟**。行步长从 8 floats 变 16 floats = 16 banks。wmma 读 16 行 fragment，周期变成 2 行同 bank → 最坏情况。

**结论**：
1. TMA 本身没帮上（gmem 本来也没瓶颈）
2. 为了摊 TMA 开销扩 BK，却把 bank conflict 搞得更严重
3. **TMA 的真正价值要配合 swizzle 才能兑现**，不是单独换 TMA 就行

---

### V7. wgmma_cute (CUTLASS CuTe + SW128 swizzle + wgmma) 🔥

**改动**：
- 用 CUTLASS CuTe (`<cute/tensor.hpp>`) 构建 kernel
- smem layout = `GMMA::Layout_K_SW128_Atom<tfloat32_t>` → 物理地址按 128B XOR 打散
- MMA = `SM90_64x64x8_F32TF32TF32_SS_TN` (wgmma SS，TF32 精度)
- 一个 warp group = 128 线程，替代之前的 256 线程两倍 warp
- BM=BN=128, BK=32, STAGES=2, 64KB 动态 smem (cudaFuncSetAttribute)
- 元素类型从 `float` 换成 `cutlass::tfloat32_t`（bit 兼容 float）
- "TN" 布局：cpuSgemm 也改成 C[m,n] = Σ A[m,k] * B[n,k]，B 按 (N,K) 读

**实测**：**4096 跑到 117552 GFlops，2.34× 于 tc_wmma**。H100 TF32 峰值占比从 5.5% 跳到 **12%**。

**NCU 对比**（M=N=2048, K=1024）：
- Duration **332 → 124 us** (-63%)
- Executed Instructions **46.4M → 6.4M** (-86%，每条 wgmma 顶 N 条 mma.sync)
- DRAM Throughput **2.95% → 7.56%** (+156%，真正开始吃 gmem)
- **Bank conflict 消失**（ncu 不再报 excessive wavefronts / N-way bank）
- **No Eligible 60% → 87%** ← 瓶颈转移到 DRAM

**这是整个迭代链的最大跳档**。swizzle 让 wgmma 直读 swizzled smem，LDS pipe 这一段整个消失。

**新瓶颈**：`No Eligible 87%` + `DRAM 7.56%` 指向 gmem 带宽跟不上 compute，而不是算力或 smem 问题。

**下一步可选**：
- 换真正的 TMA（`cp.async.bulk.tensor`），比 cp.async 更省指令，搬得更快
- warp-specialization（producer warp 只做 TMA，consumer warp group 只做 wgmma）
- 更大的 stage 数 / cluster launch 做 CTA 间协同

---

### V6. tc_wmma_padded (K 维 PAD=4, 验证诊断)

**改动**：给 `gemm_tc_wmma.cu` 的 `s_a[STAGES][BM][BK + 4]` 加 4 个 float padding，行步长从 8 → 12。数学上 `gcd(12, 32)=4, 周期=8` → 16 行 fragment 里 2-way 冲突（理论从 4-way 降到 2-way）。

**实测**：
- bank conflict **从 2.7-way 降到 2.0-way** ✅
- excessive wavefronts **62% → 58%** ✅
- L1/TEX throughput **65% → 50%** ✅
- **但 Duration 微涨 (331 → 337us，指令数 +3M)**

**结论**：padding 确实降了 bank conflict，但：
1. smem 变大 50%，编译器生成的 LDS/STS 指令略不同
2. No Eligible 只从 60% → 58%，bank conflict 只是众多 stall 源之一
3. MMA 依赖链、cp.async wait 等其他 stall 没动

**教训**：bank conflict 存在，也能缓解，但**光缓解 bank 不够治根**。要速度真涨，**LDS 这一步得整个跳过**——只有 wgmma 能做到（它直接 smem→TensorCore，没中间 LDS 步骤）。

---

## 3. NCU 关键指标对比总表

M=N=2048, K=1024 (launch-skip 85)：

| Metric | dbuf | ms | ms_pad | ws | tc_wmma | tma_wmma | tc_pad | **wgmma** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Duration (us) | 502 | 716 | 671 | 870 | 332 | 340 | 337 | **124** |
| Executed Instructions | 150.9M | 157.7M | 157.7M | 153.7M | 46.4M | 46.0M | 49.1M | **6.4M** |
| Executed IPC | 3.23 | 2.40 | 2.56 | 1.87 | 1.62 | 1.55 | 1.68 | 0.51 |
| Compute SM % | 59 | 44 | 36 | 35 | 33 | 28 | 35 | 28 |
| DRAM Throughput % | 1.9 | 1.4 | 1.5 | 1.2 | 2.95 | 2.83 | 2.89 | **7.56** |
| L1/TEX Throughput % | 72 | 73 | 72 | 73 | 65 | 88 | 50 | 61 |
| L1/TEX Hit Rate % | 0 | 59 | 59 | 69 | 4 | 66 | 3 | 5 |
| No Eligible % | 19 | 40 | 36 | 53 | 60 | 61 | 58 | **87** |
| Achieved Occupancy % | 22 | 23 | 24 | 16 | 23 | 23 | 23 | 13 |
| Excessive Wavefronts | 4% | 38% | 37% | 57% | 62% | 75% | 58% | **消失** |
| Bank Conflict | 2.4-way | — | — | — | 2.7-way | 4.0-way | 2.0-way | **消失** |
| Uncoalesced Global | — | 59% | 59% | — | — | — | — | — |

---

## 4. 瓶颈转移链（最重要的洞察）

```
dbuf          ─→  CUDA core FFMA 吞吐                (SIMT 天花板)
multistage    ─→  A 转置 + 4B cp.async + gmem 59% uncoalesced  (自己作)
tc_wmma       ─→  smem → reg 的 LDS bank conflict     (TC 带来的新瓶颈)
tma_wmma      ─→  LDS bank conflict 恶化（BK=16 周期踩雷）
wgmma_cute    ─→  gmem 带宽 (DRAM 7.56%, No Eligible 87%)  ✓ 验证
```

每一步**瓶颈都在移动**。优化的精神不是"全面提升"，是**识别当前瓶颈 + 针对性干掉**。

---

## 5. 关键教训（按重要性）

1. **"更先进的 pipeline" 要配合对的计算单元才能兑现**。cp.async 在 SIMT 上不值，在 TC 上才值。TMA 在 wmma 上不值，在 wgmma 上才值。
2. **瓶颈在哪决定优化在哪**。H100 DRAM 太快，我们前半段所有"gmem 优化"都没兑现收益；直到 compute 被 TC 拉起来，gmem 才第一次成为值得优化的侧。
3. **smem 的"方向"决定一侧顺一侧不顺**。SIMT 要转置 A（外积需求）导致 cp.async 难受；TC 不要转置但 wmma 的 LDS 又撞 bank。两边是镜像问题。
4. **swizzle 不是单独能加的**。swizzle 必须 "gmem 写入端 + compute 读取端" 同时理解 XOR 公式。wmma 的 fragment 布局编译期静态，不读 descriptor，用不了 swizzle。只有 wgmma 的 descriptor 机制能闭环。
5. **ncu 报多个 OPT 不代表每个都要改**。要看主瓶颈，改对了 Duration 才会降。padding 降了 bank 没降 Duration，就是因为 bank 不是主瓶颈。
6. **BK 越大不一定越好**。BK 对 32 banks 循环周期的命中决定 LDS 的命运。BK=8 周期 4，BK=16 周期 2（更糟），BK=32 周期 1（最糟）。要选周期 >= 16 的 BK 或加 swizzle。
7. **指令数是好的早期信号**。tc_wmma 一眼看到指令数 150M→46M 就知道要赢了。指令数是 compute 密度的代理指标。

---

## 6. 下一步：wgmma + TMA + swizzle

要同时做三件事，不能拆开：

1. **smem 换 swizzled layout**（CuTe 的 `Layout_K_SW128_Atom`）
2. **gmem→smem 用 TMA** 且带 `CU_TENSOR_MAP_SWIZZLE_128B`
3. **compute 换 wgmma**（读 descriptor 的版本）

三件事的 swizzle 公式必须一致，硬件才能闭环。推荐从 CUTLASS `examples/cute/tutorial/wgmma_sm90.cu` 改起（300 行 HGEMM 例子，改成 TF32 SGEMM）。

预期效果：
- bank conflict 彻底消失（硬件按 swizzle 反解）
- No Eligible 降到 <30%
- Compute throughput 从 33% 拉到 70%+
- 期望打到 100+ TFLOPS（当前 50 TFLOPS，还有 2× 空间）

---

## 附录：所有实验产物

代码：`sgemm/gemm_*.cu`
NCU 报告：`~/Downloads/ncu/sgemm/ncu_gemm_*_details.txt` 和 `.ncu-rep`
原始数据：`results/gemm_*.log`（远端 H100）
