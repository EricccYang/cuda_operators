# 128/256 维 Attention Head Reduce 优化设计

## 目标
对每行 128 或 256 维做 reduce（如 RMSNorm 的平方和、softmax 的 sum），让 GPU 最优。

## 设计原则

| 维度 | 策略 | 原因 |
|------|------|------|
| 128  | **1 warp (32 threads) × 4 元素/线程** | 整好 128，只用 warp shuffle，**不用 shared memory** |
| 256  | **1 warp (32 threads) × 8 元素/线程** | 整好 256，同样只用 warp shuffle |

要点：
- **Warp 内 reduce 用 shuffle**：延迟低、不占 shared memory、无 bank conflict。
- **每线程多元素**：合并访存（stride=32），寄存器累加，减少 warp 内步数。
- **Block 内尽量 1 个 warp 搞定**：128/256 维小，不需要 block 级 tree reduce。

## 128 维 kernel 设计

```
Block: 32 线程 (1 warp)
每线程: 4 个 float，连续 stride=32 的全局索引 → 合并读
Reduce: 每线程 4 个平方和 → 1 个寄存器，再 warp shuffle 归约到 lane0
RMS: lane0 算 sqrt(sum/128)，shuffle 广播（或写 shared 1 个 float）
写回: 每线程除自己的 4 个，合并写
```

- **Shared memory**: 只需 1 个 float 存 RMS（或用 shuffle 广播），甚至可省略。
- **Grid**: `(batch * num_heads * seq_len, 1)`，一个 block 管一行。

## 256 维 kernel 设计

- 同上，每线程 8 个元素，warp shuffle 归约。
- 若想少一点寄存器：64 线程 × 4 元素，2 个 warp 各 shuffle 得 2 个部分和，再 shared 或 lane0 加一次。

## 与 v4 的对比

| 项目        | v4 (1024 维/block) | 128/256 维 head |
|-------------|--------------------|------------------|
| Block 大小  | 512                | 32 (或 64)       |
| 每线程元素  | 2                  | 4 或 8           |
| Block 内 reduce | shared tree + warp | **仅 warp shuffle** |
| Shared 用量 | 512 float          | 0~1 float        |

小维度时 block 小、shared 少，**占用率**靠 grid 里大量 block 撑满 SM。

## 索引与访存（128 维示例）

- `base = blockIdx.x * 128`
- Thread `tid` (0..31) 负责下标: `base + tid`, `base + tid + 32`, `base + tid + 64`, `base + tid + 96`
- 全局内存连续 128 个 float，按 32 段分给 32 线程 → 每线程读 4 次，每次 32 个 thread 连续 32 个 float → **完全合并**。

## 可选：多行一个 Block（提高占用）

若 SM 上 block 数偏少，可让一个 block 管多行（如 4 行 128 维）：
- 32×4=128 线程，每线程仍 4 元素/行，行内 warp shuffle 按 lane 分组（如 lane_id % 32 同一行）。实现稍复杂，通常 128 维下一行 32 线程已经足够，先单行/block 再测 occupancy。
