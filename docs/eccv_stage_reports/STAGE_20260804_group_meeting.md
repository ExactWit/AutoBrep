# ECCV 挑战赛 · 阶段性实验情报（组会报告）

> 日期：2026-08-04  
> 基线仓库：AutoBrep（条件注入消融 + 推理侧优化）  
> 评测口径：**best.ckpt（按 val_loss）→ official test（n=348）**；交作业近似分 **等效分 ≈ gen_success × summary**  
> 官方 Summary：`valid_ratio × (Surface_F1 + Edge_F1 + Vertex_F1 + Topo_F1) / 4`（各 F1∈[0,1]，匹配阈值 Chamfer&lt;0.1）

---

## 1. 一句话结论

1. **条件注入尚未打败无条件 R1**：各条件模型在 test 上的等效分仍低于 R1（0.021）。  
2. **推理侧 retry 是当前最大实证收益**：`medium + gen_retries=4` 把 gen 从 ~0.48 提到 **0.88**，等效分 **0.0077 → 0.0124**（约 R1 的 59%）。  
3. **固定复杂度 / rerank 无效或有害**；复杂度 `from_condition` 把大量样本推到 hard，hard token 反而压低成功率。  
4. **public_test 已出包**：805/927 STEP（gen≈0.868），`submission.zip` 已生成，可交 Space。

---

## 2. 指标含义（报告时统一口径）

| 符号 | 含义 | 值域 | 备注 |
|---|---|---|---|
| gen_success | 能 decode+rebuild 出合法 STEP 的比例 | [0,1] | 交作业时缺失 pred ≈ invalid |
| Surface/Edge/Vertex F1 | 点云匈牙利匹配后 F1（阈值 0.1） | [0,1] | 仅在成功样本上宏平均 |
| Topo F1 | (F–E F1 + E–V F1)/2 | [0,1] | 依赖几何匹配 |
| Summary | valid_ratio × 四项 F1 均值 | [0,1] | 完美复制 smoke ≈0.99 |
| **等效分** | gen × summary（本地近似） | [0,1] | 便于方法对比；官方 Space 对全量算分 |

参照：R1 等效分 ≈ 0.50 × 0.042 = **0.021**。上界接近 1.0，当前全员仍在低分区。

---

## 3. 方法对比：条件注入（best → test）

冻结 AR+FSQ，只训条件侧；同一 test 集。

| 版本 | jid | 注入方式 | gen | Surface | Summary | **等效分** |
|---|---|---|---|---|---|---|
| **R1**（参照） | 260728-201614 | 无条件预训练 AR + analytic | 0.500 | 0.064 | **0.042** | **0.021** |
| **P1-A** | `3faf0c45` #41 | per-prim → compressor(64) + fuse | 0.394 | 0.034 | **0.021** | 0.0083 |
| **topo-v2** | `60f1b9f4` #39 | prim-direct + loop topo sketch | 0.480 | 0.023 | 0.016 | 0.0077 |
| **prim-direct** | `d7c58f70` #40 | 每图元 1 token 直进 prefix | 0.471 | 0.020 | 0.013 | 0.0060 |

### 可佐证的情报

| # | 情报 | 证据 |
|---|---|---|
| A1 | **条件注入相对 R1 仍是净弱势** | 等效分 0.006–0.008 vs R1 0.021；成功样本 Surface 亦低于 R1（0.02–0.03 vs 0.064） |
| A2 | **P1-A 质量最好、生成最差** | summary 0.021 居条件模型之首，但 gen 仅 0.39 → 等效分仍低 |
| A3 | **topo 相对 prim-direct 有弱正向但不显著** | 配对 bootstrap（n=348，未生成=0）Δsummary=+0.0017，95% CI 跨 0；方向一致偏好 topo |
| A4 | **问题不在「有没有 compressor」 alone** | 去掉 compressor 的 prim-direct 更差 → 根因更像「条件信号未被 AR 用上 / 干扰」 |

---

## 4. 推理侧消融（均基于 topo-v2 best.ckpt）

| 实验 | jid | 配置 | gen | Summary | **等效分** |
|---|---|---|---|---|---|
| 基线 | #39 | `from_condition` | 0.480 | 0.0160 | 0.0077 |
| Phase1a | #42 `0733aff4` | complexity=**medium** | 0.497 | 0.0138 | 0.0069 |
| Phase1a2 | #43 `f52d3605` | complexity=**easy** | **0.529** | 0.0123 | 0.0065 |
| **Phase1d** | #44 `5b84c99f` | medium + **retries=4** | **0.879** | 0.0141 | **0.0124** |
| Phase1e | #45 `eddb30cd` | medium + retries=4 + **rerank** | 0.853 | 0.0145 | 0.0123 |

### Retry 细节（#44，可写进 slide）

- 成功样本按 attempt：1 次 189 · 2 次 61 · 3 次 33 · 4 次 23（合计 306/348）  
- 仍失败：rebuild 27 + decode 15  
- 墙钟：约 **2.6 h / 348**（含最多 4 轮重采样）

### 可佐证的情报

| # | 情报 | 证据 |
|---|---|---|
| B1 | **「采更多候选直到 rebuild 成功」远大于「改复杂度策略」** | retry 等效分 +61%（相对 #39）；固定 medium/easy 等效分不升反降 |
| B2 | **hard 复杂度 token 有害** | topo-v2 在 hard 上成功率远低于 medium；GT=hard 且 pred=medium 时成功率 0.40，pred=hard 仅 0.18 |
| B3 | **R1 几乎从不选 hard（~4%）反而是对的** | R1 退化成「几乎全 medium」；条件模型把 40% 样本推向 hard → 长序列 → rebuild/decode 崩 |
| B4 | **固定 medium 单独不够** | gen 仅 +1.7pt，summary 下降 → 等效分变差；必须与采样多样性结合 |
| B5 | **轻量 rerank（桶内 face 数启发式）无收益** | #45 vs #44：gen 略降，等效分持平 → 可关 |
| B6 | **topo-v2 在 medium 子集上生成器不弱** | 同桶成功率可高于 R1 medium；瓶颈在「何时用 hard」与「单次采样失败」 |

---

## 5. Public test 交作业进度

| 项 | 数值 |
|---|---|
| 配置 | topo-v2 best + `complexity=medium` + `gen_retries=4` |
| jid | `5e1f021a` #46（finished） |
| 生成 | **805 / 927**（gen ≈ **0.868**） |
| 成功 attempt 分布 | 1:426 · 2:209 · 3:117 · 4:53 |
| 仍失败 | rebuild 81 + decode 41 |
| 产物 | `/data/hdd/outputs/.../260804-092611/topo-sketch-w0-v2__public_infer/submission.zip`（~34MB） |
| 说明 | public 无 GT，本地无法算 Summary；需上传 HF Space 拿 leaderboard |

**情报 C1**：retry 策略在 public 上同样成立（gen≈0.87，与 test 0.88 一致），不是 test 集过拟合。

---

## 6. 工程与流程上已坐实的事

| # | 情报 | 说明 |
|---|---|---|
| D1 | 评测必须 **best→test**，不能用 epoch 末权重比方法 | ModelCheckpoint 按 val_loss；launcher 优先 `best.ckpt` |
| D2 | `__len__` 高估会导致验证静默跳过 | topo-w0 曾 50 epoch 无 val；已修精确行计数 |
| D3 | 推理必须传 `prim_group_ids` | 否则 topo sketch 退化成单点 token（train/infer 不一致） |
| D4 | DataLoader `num_workers>0` 易 SegFault | 正式训用 `num_workers=0` |
| D5 | 临时 git worktree 会挡住 launcher checkout | 曾导致整队 Phase1 `failed`；已清理并 repend |

代码落点（`feat/topo-sketch-prefix`）：`gen_retries` / `gen_rerank`、`cond_dropout`、`unfreeze_decoder_layers`、`best.ckpt` 别名。

---

## 7. 尚未被实验证伪/证实（待办，勿在组会当结论）

| 假设 | 状态 |
|---|---|
| 条件 dropout（p≈0.4）能否消除「条件净伤害」 | 训练 job 曾排队后 defer，**尚未跑完** |
| 解冻 decoder 顶 2 层联合微调 | 同上 |
| MTP / 自投机解码能否在不损质量下加速 AR | **仅方案分析**，未实验；动机：单样本 AR≈15s，与 retry 相乘 |
| 条件版绝对质量追上 R1 Surface≈0.064 | 未达成 |

---

## 8. 组会可强调的叙事线

```text
瓶颈分层（已由实验切开）:

  ① 复杂度错分（hard）     → 固定策略收益有限
  ② 单次采样 gen 失败       → retry×4 实证大幅缓解（主战果）
  ③ 成功样本几何/拓扑质量   → 条件模型仍远低于 R1（下一主战场）
```

**对标**：先把交作业等效分从 0.008 拉到 0.012+（已完成），下一步要同时：**保 gen≥0.85** + **抬成功样本 Summary（向 R1 0.042）**。

---

## 9. 建议下一步（按 ROI）

1. **立刻**：上传 #46 `submission.zip` 拿 Space 分数（建立外部锚点）。  
2. **短线**：以 `#44 配置` 为推理默认；关 rerank；若卡允许可试 `retries=8` smoke。  
3. **训练**：重挂 `cond_dropout=0.4` ± `unfreeze=2`，验证条件是否从「净伤害」变为可用。  
4. **加速（并行）**：Medusa/MTP 浅头自投机，降低 AR 墙钟，使更大 K 可负担。  
5. **质量**：在 gen 已高的前提下，专攻成功样本 Surface/Topo（对齐条件、解冻顶层、结构化 MTP）。

---

## 10. 关键 run 索引

| 标签 | jid | 作用 |
|---|---|---|
| R1 参照 | 260728-201614 / R1-ckpt-host | 无条件天花板 |
| topo-v2 train | 260802-195832 | best val_loss=0.2518 @ step 10352 |
| #39 topo best→test | `60f1b9f4` | from_condition 基线 |
| #40 prim-direct | `d7c58f70` | 直进 prefix |
| #41 P1-A | `3faf0c45` | compressor+fuse |
| #42/#43 | `0733aff4` / `f52d3605` | 固定 medium / easy |
| #44/#45 | `5b84c99f` / `eddb30cd` | retry4 / retry4+rerank |
| #46 public | `5e1f021a` | 交作业包 |

更细的对比表见同目录 [`COMPARE_COND_INJECTION.md`](COMPARE_COND_INJECTION.md)（将同步 Phase1 完成态）。
