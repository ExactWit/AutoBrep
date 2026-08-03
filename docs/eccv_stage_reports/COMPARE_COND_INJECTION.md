# ECCV 条件注入消融 · 统一对比

> 官方指标（`min_eval/eval.py`）：`Summary = valid_ratio × (Surface_F1 + Edge_F1 + Vertex_F1 + Topo_F1) / 4`，`Topo_F1 = (F-E_F1 + E-V_F1)/2`。
> `gen_success` = rebuild 出合法 STEP 的样本占比。**等效分 ≈ gen_success × summary**（交作业口径近似；缺失 pred 在官方 Space 计 invalid）。
> **方法对比口径（2026-08-03 起）**：各方法 **best.ckpt（按 val_loss 选）→ official test split（n=348）**。

## 主表（best → test）

| 版本 | 注入 / 推理 | gen | Surface | Edge | Vertex | Topo | Summary | 等效分 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| **R1** `260728-201614` | 无条件 AR 参照 + analytic | 0.500 | 0.064 | 0.017 | 0.015 | 0.016 | **0.042** | **0.021** | hard 仅 4%；多数 medium |
| **topo-v2** `#39` `60f1b9f4` | prim-direct + topo sketch；`from_condition` | 0.480 | 0.023 | 0.007 | 0.006 | 0.007 | **0.016** | **0.0077** | hard 误分 40% |
| **prim-direct** `#40` `d7c58f70` | per-prim 直进 prefix | 0.471 | 0.020 | 0.004 | 0.004 | 0.004 | **0.013** | **0.0060** | 与 topo 同结构偏低 |
| **P1-A** `#41` `3faf0c45` | compressor(64)+fuse xattn | 0.394 | 0.034 | 0.010 | 0.007 | 0.008 | **0.021** | **0.0083** | best→test 完成 |

配对 bootstrap（topo vs direct，n=348，未生成=0）：Δsummary=+0.0017，95%CI 跨 0 → **方向偏好 topo，但不显著**。

## 推理侧消融（进行中，均基于 topo-v2 best）

| 实验 | 配置 | 状态 |
|---|---|---|
| Phase1a medium | `complexity=medium` 固定 | 排队 |
| Phase1a2 easy | `complexity=easy` 固定 | 排队 |
| Phase1d retry4 | medium + `gen_retries=4` | 排队 |
| Phase1e rerank | medium + retries=4 + `gen_rerank=1` | 排队 |

复杂度分析（topo-v2 `from_condition`）：`pred=medium` 在所有 GT 桶上成功率都高于 `pred=hard`；**hard token 有害**。固定 medium 预期 gen ≈0.55。

## 训练侧（排队）

| 实验 | 配置 | 状态 |
|---|---|---|
| `topo-conddrop04` | cond_dropout=0.4 | 排队 |
| `topo-conddrop04-unfreeze2` | cond_dropout=0.4 + unfreeze 顶 2 层 | 排队 |

代码：`feat/topo-sketch-prefix` @ `ea92cf4`（cond_dropout / unfreeze / gen_retries / gen_rerank）。

## 归档：旧 val 口径数字（勿与主表混比）

| 版本 | split | gen | Summary | 状态 |
|---|---|---|---|---|
| P1-A `65e84657` | val? | 0.233 | 0.025 | 旧 |
| prim-direct `8de616d4` | val? | 0.248 | 0.016 | 旧 |
| MMA `ac395002` | — | — | — | ep42 SegFault |
| topo-sketch `5e0b8a3d` | — | — | — | ep5 SegFault |

## 读数（更新）

1. **瓶颈分层**：复杂度错分（hard）→ gen 失败（rebuild/decode）→ 条件净伤害（成功样本质量仍远低于 R1）。
2. **topo-v2 medium 子集成功率 55% > R1 medium 50%**：生成器本身不差，被复杂度 token 拖累。
3. 绝对量级仍低：R1 summary 0.042 距 smoke 上界 ~0.99 差 ~20×；交作业等效分 R1≈0.021。
4. 下一步优先：固定 medium / retry 推理开关（不等待重训），再等 cond_dropout 训练验证「条件净伤害」是否可消。
