# ADR-3：Multi-Agent 拓扑选择

> **项目**: 5G 基站告警智能诊断系统 (BaseStation-MAS)  
> **日期**: 2026-06-13  
> **状态**: 已采纳  
> **版本**: v2.0 — 基于已验证的 TelecomTS 数据集 + Framework 蓝图重写

---

## 决策

**选择 Supervisor 模式 + Evaluator 质检闭环**，部署 3 个异构 Worker：

```
Detection (纯ML) ∥ [Diagnosis (ML+GraphRAG+LLM)] → Reporter (LLM) → Evaluator (LLM)
```

GraphRAG 知识图谱基于 Framework 图中**真实网络实体**（BaseStation / Jammer / Zone A/B/C / Mobile），不再需要虚拟拓扑。

---

## 背景

### 原代码与 Framework 的 gap（已验证）

```
Framework 蓝图                      原代码实现
─────────────────────────────────────────────────────────
5 大下游任务                          4 个（缺 Network-Level Q&A 和 Time Series Q&A）
多模态（数值 + 文本）                   纯数值（只用 KPIs 字段）
Network description and logs         只在 README 中提及，代码未读取
Q&A 推理链（220 万条）                 完全未使用
Jamming（唯一真实异常）                被代码在所有非 detection 任务中主动排除
Zone / Mobility / Application 标签    存在但未被读取
知识图谱                              不存在
```

### 这意味着什么

原代码只是整个 Framework 蓝图的前半程（纯 ML 分类器）。**你的 Multi-Agent 系统要补上后半程**：把被丢弃的 Jamming 数据用起来、把 description 和 labels 建成 GraphRAG、把 Q&A 推理链作为 Reporter 的参考范式。

### 项目池四项要求

| 要求 | 当前代码状态 | Multi-Agent 落点 |
|------|:--:|------|
| 加载 TelecomTS 数据集 | ✅ 已实现 | 复用 `load_dataset()` |
| 引入 GraphRAG 理解基站拓扑 | ❌ 完全缺失 | Diagnosis Worker Step 2 — 基于 Framework 实体构建 KG |
| AI 监控告警流 | ⚠️ 只有 batch 评估 | Detection Worker 持续扫描 |
| 输出根因分析报告 | ❌ 只打印 Accuracy | Diagnosis (ML+KG+LLM) → Reporter (LLM) |

---

## 考虑的方案

### 方案 A：Pipeline（流水线）

```
Detection → Diagnosis → Reporter
依次串行
```

| 优点 | 缺点 |
|------|------|
| 实现简单 | Detection 完成后才能开始 Diagnosis，即使 ML 结果已出也必须等 |
| | 无法表达 "无异常时跳过 Diagnosis" 的条件逻辑 |
| | 三个步骤串行，响应时间 = 三者之和 |

### 方案 B：Swarm（群组并行）

```
Detection ─┐
Diagnosis ─┼─→ Reporter
           ─┘
全部并行
```

| 优点 | 缺点 |
|------|------|
| 并发效率最高 | Diagnosis 的 GraphRAG 查询需要知道 "哪个 KPI 异常"，依赖 Detection 输出 |
| | Reporter 需要 Diagnosis 的根因结果，不能盲生成 |
| | 无法实现条件触发（无异常时 GraphRAG 白查） |

### 方案 C：Supervisor + Evaluator（选择方案）

```
Orchestrator (LLM):
  ├─ Detection  (纯ML)     ──→ { has_anomaly, score, affected_kpis }
  ├─ Diagnosis  (ML+KG+LLM)──→ { root_cause, CoT, duration }  ← 条件触发
  └─ Reporter   (LLM)      ──→ Markdown 诊断报告
Evaluator (LLM) → validate → passed / retry
```

| 优点 | 缺点 |
|------|------|
| 条件路由：has_anomaly=false → 跳过 Diagnosis + GraphRAG | Orchestrator 是单点 |
| GraphRAG 查询上下文来自 Detection 的 affected_kpis，精准不浪费 | 实现复杂度高 |
| Detection 结果直传 Diagnosis，中间不丢失信息 | 需严格定义 Worker 契约 |
| Evaluator 闭环保证输出质量 | |

---

## 选择理由

### 1. 任务依赖图天然是 Supervisor 的 DAG

```
Detection(纯ML) ──→ has_anomaly? ──(yes)──→ Diagnosis(ML+KG+LLM)
                                       └──(no)──→ 跳过
                                                      ↓
                                               Reporter(LLM)
```

- Pipeline 无法表达条件分支
- Swarm 无法表达前后依赖
- **只有 Supervisor 能同时处理 "条件触发 + 前后依赖"**

### 2. GraphRAG 需要上下文才能精准查询

```
错误做法（Swarm 盲查）:
  GraphRAG.query("网络有什么问题？") → 返回整个拓扑，噪声大

正确做法（Supervisor 精准查询）:
  Detection → {affected_kpis: ["RSRP", "BLER"]}
  GraphRAG.query("哪个实体影响了RSRP和BLER？Zone B的干扰源？")
  → 精准返回: Jammer →[interference]→ BaseStation →[covers]→ Zone B
```

### 3. Jamming 是最佳核心场景

- Jamming 是数据集中唯一**真实采集**的异常（其余 10 种是合成）
- 原代码在所有非 detection 任务中**过滤掉了 Jamming**（`item["anomalies"]["type"] == "Jamming": continue`）
- GraphRAG 中 Jammer 是独立实体 → 可以展示完整的 "干扰源 → 基站 → 覆盖区" 推理路径
- **这也是项目池要求 "输出根因分析报告" 最具说服力的场景**

### 4. 决策卡验证

| 问题 | 答案 | 指向 |
|------|------|------|
| 任务之间有前后依赖？ | 是，Diagnosis 依赖 Detection 的 anomaly_result | Supervisor |
| 步骤是固定线性的？ | 否，无异常时跳过 Diagnosis | 排除 Pipeline |
| 可以完全并行独立？ | 否，GraphRAG 需要 Detection 输出做精准查询 | 排除 Swarm |

---

## Worker 职责边界（最终版）

### 📡 Detection Worker — 纯 ML

```
原代码对应: preprocess("anomaly detection")
模型: TimesNet
输入: 18通道 KPI × 128时间步
输出: { has_anomaly, anomaly_score, affected_kpis }
LLM: ❌
GraphRAG: ❌
```

### 🛠️ Diagnosis Worker — ML + GraphRAG + LLM

```
原代码对应: preprocess("root-cause analysis") + "anomaly duration"
             + description/labels/QnA（原代码未用）

三级推理流水线:

Step 1 — ML 分类（纯ML）
  模型: Autoformer
  输入: KPI数据 + anomaly_result
  输出: Top-3 根因候选（11类）
  LLM: ❌

Step 2 — GraphRAG 拓扑查询（KG）

  知识图谱来源: Framework 图 + labels + description + troubleshooting_tickets
  
  节点:  BaseStation, Jammer, Zone_A/B/C, Mobile
  边:   interference, covers, located_in, moves_to, adjacent
  属性: description 文本嵌入 + labels 结构化标签

  查询示例: "Jammer是否活跃？Zone B的干扰路径？Mobile是否正在从Zone A移入Zone B？"
  输出: { 干扰源, 影响路径, 相关实体状态, 历史类似案例 }
  LLM: ❌（Embedding 除外）

Step 3 — LLM 综合推理
  输入: ML Top-3 + GraphRAG 拓扑上下文 + affected_kpis
  Prompt 模板:
    "ML模型认为最可能的3个根因是：[...]
     网络拓扑查询显示：[...]
     KPIs异常：[RSRP下降8dB, BLER突发]
     请综合推理，给出最终根因、置信度、推理链。"
  输出: { root_cause, confidence, duration, reasoning(CoT), topology_evidence }
  LLM: ✅

输出: { root_cause, confidence, duration, reasoning, topology_evidence }
LLM: ✅ (仅 Step 3)
```

### 💬 Reporter Worker — LLM

```
原代码对应: 无（新增）
            参考 TelecomTS QnA 的 reasoning 风格

输入: user_query + detection_result + diagnosis_result
参考: TelecomTS QnA 220万条推理链的写作范式

Prompt 模板:
  "你是一个5G基站运维诊断助手。
   用户问题: {user_query}
   诊断结果: {全部Worker输出}
   请参照以下风格生成诊断报告: {few-shot QnA reasoning 示例}
   要求: ①证据链完整 ②有修复建议 ③运维新人也能看懂"

输出: { report (Markdown), reasoning, suggestions }
LLM: ✅
```

---

## 参照 OpenClaw 自检

| OpenClaw 模式 | 本项目对应 | 职责清晰？ |
|---------------|-----------|:--:|
| Gateway = Orchestrator | `orchestrator.py` | ✅ 只分解+路由+汇总 |
| Pi Engine = Worker | 3 个 Worker | ✅ 各自有独立推理能力 |
| worker ≠ orchestrator | 无双重角色 | ✅ 消除了职责模糊 |

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| Orchestrator 单点故障 | Docker `restart: unless-stopped` + `healthcheck` |
| GraphRAG 初始构建成本 | MVP 用 NetworkX + 静态 JSON，不依赖外部图数据库 |
| Reporter 生成幻觉 | Evaluator 比对 report 与 Worker 事实输出，不一致打回 |
| Context Rot | Worker 间只传结构化 JSON 摘要，不传原始序列 |
| Jamming 样本稀少 | 保留所有 Jamming 样本用于诊断场景（修复原代码的过滤问题） |

---

## 后续演进

- **V2.1**: GraphRAG 从静态 JSON → Neo4j，支持动态拓扑更新
- **V2.2**: 引入 TelecomTS QnA 推理链做 Reporter Worker 的 Few-shot 示例
- **V3.0**: 接入实时 KPI 流（从 batch → streaming）
