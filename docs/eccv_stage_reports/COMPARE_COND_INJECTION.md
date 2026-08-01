# ECCV 条件注入消融 · 统一对比

> 官方指标（`min_eval/eval.py`）：`Summary = valid_ratio × (Surface_F1 + Edge_F1 + Vertex_F1 + Topo_F1) / 4`，`Topo_F1 = (F-E_F1 + E-V_F1)/2`。`gen_success` = 能 rebuild 出合法 STEP 的样本占比（≈ valid_ratio 上限）。
> 评测集：full val（694）；R1 为 GT test。所有训练 run 冻 AR+FSQ，仅训条件侧。

## 主表（full official）

| 版本 | 注入方式 | gen_success | valid_ratio | Surface_F1 | Edge_F1 | Vertex_F1 | Topo_F1 | **Summary** | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| **R1 260728-201614** (GT test) | —（best+analytic 参照） | 0.500 | 1.00 | 0.064 | 0.017 | 0.015 | 0.016 | **0.042** | 参照 |
| 旧 GT 260725-002218 | —（参照） | 0.353 | — | — | — | — | — | **0.046** | 参照 |
| **P1-A** `65e84657` | per-prim → compressor(64) + fuse **xattn** | 0.233 | 1.00 | 0.037 | 0.008 | 0.008 | 0.008 | **0.025** | ❌ 失败 |
| **prim-direct** `8de616d4` | per-prim **直进** prepend（pad/trunc 64），无 compressor/xattn | 0.248 | 0.994 | 0.021 | 0.006 | 0.006 | 0.006 | **0.016** | ❌ 失败 |
| **MMA** `ac395002` | prim + decoder **xattn** + surf-type head | （崩溃） | — | — | — | — | — | 未评 | ❌ ep42 SegFault |
| **topo-sketch** `5e0b8a3d` | prim-direct + loop topo 草图 + 计数 aux | （崩溃） | — | — | — | — | — | 未评 | ❌ ep5 SegFault |

> P1-A / prim-direct 的 fast_val ppl 均随训练**恶化**（冻 AR 下条件注入损害 CE），与 summary 低一致。

## 读数

1. **所有条件注入版本都显著低于 R1 / 旧 GT**（summary 0.016–0.025 vs 0.042–0.046；gen 0.25 vs 0.35–0.50）。冻结 AR 下，无论 prefix 还是 cross-attn，**条件注入目前是净伤害**。
2. **prim-direct（0.016）比 P1-A（0.025）更差**：去掉 compressor 的「每图元直进」并未改善，反而更糟。→ 问题不在 compressor/xattn 的形式，而在**条件信号本身没被 AR 用上**，反而干扰。
3. gen_success 普遍 ~0.25，远低于 R1 的 0.50：复杂件（高 n_prims）rebuild/decode 失败是主战场（MMA 中 n_prims 52/81/128 全崩）。
4. valid_ratio 都 ≈1.0（能生成的 STEP 都合法）→ 瓶颈是 **gen（能否生成）+ 几何/拓扑 F1**，不是 STEP 合法性。

## 根因假设（待验证）

- **A. 对齐失败**：AR 没有学会把 prepend 的条件 token 与输出 BRep token 对齐；条件成了噪声前缀。证据：加条件后 summary 反降。
- **B. 数据/视图划分污染**：三视图切分不可靠（见 TECHDRAW_VIEW_SPLIT.md），条件几何本身有错，误导模型。
- **C. 无条件先验更强**：R1/旧 GT 不靠三视图条件反而更好，说明当前条件管线引入的噪声 > 收益。

## 下一步（优先级）

1. **修 DataLoader SegFault**（MMA/topo 同款崩溃，num_workers 相关）→ 先恢复训练稳定性，否则无法迭代。
2. **A 验证**：做一个「条件置零」对照——同结构但 drop 全部条件，看 summary 是否回到 R1 水平，定位是条件有害还是实现有害。
3. **B 验证**：修三视图划分（数据侧），再重训最简条件。
4. gen 主战场：rebuild/decode 鲁棒性（与条件正交），先把 gen_success 从 0.25 → 0.5。
