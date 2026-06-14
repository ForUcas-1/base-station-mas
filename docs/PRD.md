# PRD：5G 基站告警智能诊断系统（BaseStation-MAS）

> **版本**: v2.0  
> **变更**: 基于已下载验证的 TelecomTS 数据集 + Framework 蓝图重写  
> **项目来源**: 三阶段实战项目池 · 第三轮 #4  
> **数据集**: TelecomTS (ICLR 2026) — 已就绪 (`/root/projects/TelecomTS/data_cache/`)  
> **技术栈**: Python 3.11 + PyTorch + HuggingFace + GraphRAG + Multi-Agent + Docker

---

## 1. 产品概述

### 1.1 一句话描述

面向 5G 基站运维工程师的 AI 智能诊断助手。基于 TelecomTS 多模态数据集（32K 样本，18 通道 KPI + 220 万条带推理链的 Q&A），利用 Multi-Agent 架构自动完成**异常检测 → 根因分析（含 GraphRAG 拓扑推理）→ 自然语言诊断报告**的完整流程。

### 1.2 解决的痛点

| 当前痛点 | 系统如何解决 |
|----------|------------|
| 5G 基站告警后排查根因依赖专家经验，耗时长 | Diagnosis Worker 三级流水线（ML 分类 + GraphRAG 拓扑 + LLM 推理）秒级输出根因 + CoT |
| 告警是孤立的数值，缺少上下文 | GraphRAG 引入 Jammer/Zone/Mobility 等网络实体关系，定位异常传播路径 |
| 运维新人看不懂 18 通道 KPI 的专业含义 | Reporter Worker 利用 TelecomTS 自带的 220 万条 Q&A 推理链风格，生成自然语言诊断报告 |
| 纯 ML 分类器只能输出标签，不能解释 | LLM 综合 ML 候选 + 拓扑上下文 + 领域知识，输出可解释的 CoT 推理 |

### 1.3 目标用户

- **主要**: 5G 基站现场运维工程师
- **次要**: NOC 值班人员、网优工程师

---

## 2. 功能需求

### 2.1 核心功能

| 编号 | 功能 | 描述 | 优先级 |
|------|------|------|--------|
| F1 | **异常检测** | 输入 18 通道 KPI 时序（128 时间步），输出异常有无 + 异常分数 + 受影响 KPI 列表。纯 ML（TimesNet），不做语言推理 | P0 |
| F2 | **根因分析（含 GraphRAG）** | ML（Autoformer）→ 11 类根因候选 → GraphRAG 查询网络拓扑图（Jammer/Zone/Mobility）→ LLM 综合推理输出最终根因 + CoT + 异常时长 | P0 |
| F3 | **自然语言诊断报告** | 汇总 F1–F2 结果，参照 TelecomTS QnA 的 reasoning 风格，生成 Markdown 诊断报告 + 修复建议 | P0 |
| F4 | **交互式问答** | 运维工程师可追问具体 KPI 细节或拓扑关系 | P2 |

### 2.2 非功能需求

| 编号 | 需求 | 指标 |
|------|------|------|
| NF1 | 响应时间 | 单次完整诊断 ≤ 30 秒 |
| NF2 | 异常检测准确率 | ≥ 90%（原代码已验证） |
| NF3 | 根因分析准确率 | ≥ 80% |
| NF4 | 部署便捷性 | `docker compose up -d` 一键启动，数据集首次自动拉取 |
| NF5 | 可扩展性 | 每个 Worker 独立 config，编码器可热切换（8 选 1） |

---

## 3. 用户故事

### US-1：干扰告警诊断（核心场景）

> **作为** 现场运维工程师  
> **当我** 收到 Zone B 的 RSRP 骤降告警时  
> **我希望** 系统结合网络拓扑告诉我：是外部干扰器（Jammer）还是邻区同频干扰，干扰影响了哪些 Zone，终端移动是否加剧了问题  
> **以便** 我能直接去正确位置排查

**验收标准**:
- Diagnosis Worker 输出含 GraphRAG 拓扑证据（如 `Jammer →[interference]→ BaseStation →[covers]→ Zone B`）
- LLM 推理链引用 ML 候选 + 拓扑查询 + Mobility 时间线
- 报告中包含 Jamming 特有的修复建议

### US-2：例行巡检

> **作为** NOC 值班工程师  
> **我希望** 输入 "检查当前网络状态"，系统自动扫描各 Zone 的 KPI，报告异常并解释根因  
> **以便** 30 秒内完成一次全网诊断

### US-3：历史案例学习

> **作为** 新入职运维  
> **我希望** 询问 "RSRP 下降通常是什么原因"，系统能结合 TelecomTS 的 Q&A 推理链给出解释  
> **以便** 快速积累诊断经验

---

## 4. 数据与模型

### 4.1 数据集（已验证）

| 属性 | 值 | 状态 |
|------|------|:--:|
| 数据集 | `AliMaatouk/TelecomTS` on HuggingFace | ✅ 已下载 |
| 样本量 | 32,000 时序样本 | ✅ |
| 通道数 | 18（10 float + 6 integer + 2 categorical） | ✅ |
| 文本描述 | `description` 字段 — NL 网络状态摘要 | ✅ |
| Q&A | 2,210,185 条，含 `network`/`timeseries`/`anomalies` 三类 + `reasoning` 推理链 | ✅ |
| 标签 | `zone` (A/B/C) · `application` · `mobility` · `congestion` | ✅ |
| 异常 | 11 类型（含 Jamming，唯一真实采集的） · `troubleshooting_tickets` 文本 | ✅ |
| 统计 | `statistics` — 每 KPI 的 mean/variance/trend/periodicity | ✅ |

### 4.2 知识图谱（GraphRAG）

基于 Framework 图中的真实网络实体构建：

```yaml
节点:
  BaseStation:    # 主基站 gNB
    attributes:   [description 文本, 频段, 功率]
  Jammer:         # 干扰源（数据集唯一真实异常！）
    attributes:   [active, broadband noise]
  Zone_A:         # 正常覆盖区 (0-3m)
    attributes:   [zone=A, 信号强, RSRP正常]
  Zone_B:         # 受干扰区 (3-6m)
    attributes:   [zone=B, RSRP下降, BLER突发]
  Zone_C:         # 移动性区 (>6m)
    attributes:   [zone=C, 信号弱, 切换频繁]
  Mobile:         # 移动终端
    attributes:   [application, mobility状态]

边:
  Jammer    →[interference]→ BaseStation
  BaseStation →[covers]→    Zone_A/B/C
  Mobile    →[located_in]→  Zone_A/B/C
  Mobile    →[moves_to]→    Zone_A/B/C
  Zone_A    →[adjacent]→    Zone_B
  Zone_B    →[adjacent]→    Zone_C

节点属性来源:
  - description 字段（NL 文本 → 嵌入）
  - labels 字段（结构化标签）
  - troubleshooting_tickets（诊断文本）
```

### 4.3 模型与 LLM 分布

| 组件 | 类型 | 默认模型 | LLM? | 理由 |
|------|------|---------|:--:|------|
| **Detection Worker** | 纯 ML | TimesNet (CNN+FFT) | ❌ | 二分类是数学问题 |
| **Diagnosis Step 1** | 纯 ML | Autoformer | ❌ | 11 类分类是数学问题 |
| **Diagnosis Step 2** | 知识图谱 | GraphRAG | ❌ | 图遍历 + 向量检索 |
| **Diagnosis Step 3** | LLM 推理 | Claude/GPT | ✅ | 多源信息综合推理 |
| **Reporter Worker** | LLM | Claude/GPT | ✅ | 自然语言生成 |
| **Orchestrator** | LLM | Claude/GPT | ✅ | 意图解析 + 任务分解 |
| **Evaluator** | LLM | Claude/GPT | ✅ | 语义一致性检查 |

---

## 5. 系统边界

### 5.1 MVP 包含

- 3 Worker（Detection / Diagnosis / Reporter）+ Orchestrator + Evaluator
- GraphRAG 知识图谱（基于 Framework 实体，NetworkX + 向量索引）
- 原代码 5 个 encoder 模型复用（TimesNet/Autoformer/Chronos 等）
- TelecomTS 全部数据字段的完整利用（包括此前被代码忽略的 description/QnA/labels/troubleshooting）
- Docker Compose 一键部署
- Jamming 作为核心诊断场景（修复原代码把它过滤掉的问题）

### 5.2 MVP 不包含

- 实时流数据接入（使用 batch 模式）
- 可视化仪表盘
- 告警自动推送

---

## 6. 里程碑

| 阶段 | 时间 | 交付物 |
|------|------|--------|
| M1 | W10–11 | Multi-Agent 骨架、Docker 部署、GraphRAG 知识图谱构建、文档 |
| M2 | W12–13 | 3 Worker 完整实现（含 Diagnosis ML+KG+LLM）、Evaluator 闭环 |
| M3 | W14 | Reporter 完整诊断报告、Jamming 场景端到端验收 |
