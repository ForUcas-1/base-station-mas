# BaseStation-MAS 系统架构文档

> **项目**: 5G 基站告警智能诊断系统  
> **架构模式**: Orchestrator-Worker-Evaluator（Supervisor + 3 异构 Worker）  
> **知识图谱**: 基于 TelecomTS Framework 图真实实体  
> **数据集**: TelecomTS 32K 样本 · 已下载到本地  
> **日期**: 2026-06-13 · v2.0

---

## 1. 系统架构全景图

```mermaid
graph TB
    subgraph 用户层
        USER[👤 运维工程师<br/>自然语言查询]
    end

    subgraph 调度层
        ORCH[🧠 Orchestrator<br/>────────────<br/>LLM 意图解析<br/>任务分解 · 条件路由<br/>结果汇总]
    end

    subgraph Worker层
        DET["📡 Detection Worker<br/>────────────<br/>纯 ML · TimesNet<br/>任务: 异常检测 二分类<br/>LLM: ❌  /  KG: ❌"]

        subgraph DIAG_INTERNAL["🛠️ Diagnosis Worker（核心智能体）"]
            direction TB
            D1["Step 1: ML 分类<br/>Autoformer → 11类<br/>Top-3 候选根因<br/>LLM: ❌"]
            D2["Step 2: GraphRAG<br/>查询网络拓扑图<br/>Jammer/Zone/Mobility<br/>LLM: ❌"]
            D3["Step 3: LLM 推理<br/>综合 ML+KG<br/>→ 根因 + CoT<br/>LLM: ✅"]
            D1 --> D3
            D2 --> D3
        end

        REP["💬 Reporter Worker<br/>────────────<br/>纯 LLM<br/>参考 TelecomTS QnA 推理链<br/>输出: Markdown 诊断报告<br/>LLM: ✅  /  KG: ❌"]
    end

    subgraph 质检层
        EVAL[✅ Evaluator<br/>────────────<br/>LLM 质检<br/>格式校验 + 事实一致性<br/>不通过 → 打回 Reporter]
    end

    subgraph 数据与知识层
        HF["🤗 TelecomTS 数据集<br/>32K 样本 · 18通道 · 128步<br/>已验证字段:<br/>✅ KPIs · description<br/>✅ labels · anomalies<br/>✅ QnA 220万 · statistics<br/>✅ troubleshooting_tickets"]
        
        KG["🏗️ GraphRAG 知识图谱<br/>────────────<br/>节点: BaseStation · Jammer<br/>      Zone A/B/C · Mobile<br/>边: interference · covers<br/>     located_in · moves_to<br/>属性: description 文本嵌入<br/>      labels 结构化标签"]
    end

    USER -->|"Zone B RSRP为什么骤降？"| ORCH
    ORCH -->|并行| DET
    ORCH -->|条件: has_anomaly=true| DIAG_INTERNAL
    ORCH -->|汇总后| REP
    REP -->|report| EVAL
    EVAL -->|passed| ORCH
    EVAL -->|retry| REP
    ORCH -->|📊 诊断报告| USER

    DET -.->|读取 KPIs| HF
    D1 -.->|读取 KPIs + anomalies| HF
    D2 -.->|查询拓扑| KG
    REP -.->|参考 QnA 推理链| HF

    style ORCH fill:#4A90D9,color:#fff
    style DET fill:#2ECC71,color:#fff
    style DIAG_INTERNAL fill:#E67E22,color:#fff,stroke:#E74C3C,stroke-width:3px
    style D1 fill:#F39C12,color:#fff
    style D2 fill:#16A085,color:#fff
    style D3 fill:#C0392B,color:#fff
    style REP fill:#9B59B6,color:#fff
    style EVAL fill:#F1C40F,color:#333
    style HF fill:#95A5A6,color:#fff
    style KG fill:#E74C3C,color:#fff
```

---

## 2. 核心时序图：Jamming 干扰诊断

```mermaid
sequenceDiagram
    participant User as 👤 运维工程师
    participant Orch as 🧠 Orchestrator (LLM)
    participant DetW as 📡 Detection<br/>(纯ML)
    participant DiagML as 🛠️ Diag Step1<br/>(Autoformer)
    participant GraphRAG as 🏗️ Diag Step2<br/>(GraphRAG)
    participant DiagLLM as 🧠 Diag Step3<br/>(LLM推理)
    participant RepW as 💬 Reporter<br/>(LLM)
    participant Eval as ✅ Evaluator<br/>(LLM)
    participant HF as 🤗 TelecomTS

    User->>Orch: "Zone B 的 RSRP 为什么突然恶化？"

    rect rgb(74, 144, 217, 0.1)
        Note over Orch: ⚡ Phase 1: LLM 意图解析
        Orch->>Orch: LLM解析 → subtasks:["detect","diagnose","report"]<br/>condition: if has_anomaly → "diagnose"
    end

    rect rgb(46, 204, 113, 0.1)
        Note over DetW,HF: 📡 Phase 2: 异常检测（纯ML）
        Orch->>DetW: detect(Zone B KPI数据)
        DetW->>HF: load_dataset("AliMaatouk/TelecomTS")
        DetW->>DetW: preprocess("anomaly detection")<br/>TimesNet.forward()<br/>head: Linear(d,2)→Softmax
        DetW-->>Orch: {has_anomaly: true,<br/>  score: 0.95,<br/>  affected_kpis: ["RSRP","BLER","UL_MCS"]}
    end

    rect rgb(230, 126, 34, 0.12)
        Note over DiagML,DiagLLM: 🛠️ Phase 3: 根因分析（条件触发 · ML+KG+LLM）
        
        alt has_anomaly == true
            Orch->>DiagML: Step1: classify(data, anomaly_result)
            DiagML->>DiagML: Autoformer → 11类<br/>（含 Jamming！原代码过滤了这里修复）
            DiagML-->>Orch: Top-3: ["Jamming(0.52)","同频干扰(0.31)","覆盖不足(0.10)"]

            Orch->>GraphRAG: Step2: query("Zone Bの干扰源？<br/>RSRP下降与Mobility是否相关？")
            GraphRAG->>GraphRAG: 图遍历:<br/>  Zone_B →[covered_by]→ BaseStation<br/>  BaseStation →[affected_by]→ Jammer<br/>向量检索: "RSRP jamming Zone B"<br/>→ 匹配 troubleshooting_tickets
            GraphRAG-->>Orch: {干扰源: Jammer(active),<br/>  影响路径: Jammer→BS→Zone_B,<br/>  Mobility: Mobile从ZoneA移入ZoneB,<br/>  症状: RSRP -8dB + BLER突发,<br/>  历史: Jamming场景RSRP降幅8-15dB}

            Orch->>DiagLLM: Step3: reason(ML+KG)
            DiagLLM->>DiagLLM: Prompt:<br/>"ML Top-1=Jamming(52%)<br/>GraphRAG确认Jammer活跃<br/>Mobility显示终端刚进入干扰区<br/>RSRP降幅(-8dB)符合Jamming特征<br/>→ 排除同频干扰(邻区未见异常)<br/>→ 确认为人为宽带干扰"
            DiagLLM-->>Orch: {root_cause: "Jamming",<br/>  confidence: 0.94,<br/>  duration: {start:42,end:89},<br/>  reasoning: "[CoT] 1)ML指向Jamming<br/>  2)KG确认Jammer→BS→ZoneB链路<br/>  3)Mobility时间线与异常onset吻合<br/>  →唯一合理解释",<br/>  topology_evidence: "Jammer→BS→ZoneB"}
        else has_anomaly == false
            Orch->>Orch: 跳过 Diagnosis，不调 GraphRAG
        end
    end

    rect rgb(155, 89, 182, 0.1)
        Note over RepW: 💬 Phase 4: LLM 报告生成
        Orch->>Orch: 汇总 → summary
        Orch->>RepW: generate_report(summary, query)
        RepW->>RepW: LLM 参照 TelecomTS QnA reasoning 风格<br/>生成 Markdown 报告
        RepW-->>Orch: "📊 # Zone B 诊断报告<br/>## ⚠️ 异常: RSRP骤降 (95%)<br/>## 🛠️ 根因: 人为宽带干扰(94%)<br/>  **证据链**:<br/>  1. ML: Jamming(52%)<br/>  2. GraphRAG: Jammer→BS→ZoneB<br/>  3. Mobility: 终端移入时间线吻合<br/>## ⏱️ 持续: 42-89步 (4.7s)<br/>## 💡 建议: 定位Jammer物理位置"
    end

    rect rgb(241, 196, 15, 0.12)
        Note over Eval: ✅ Phase 5: LLM 质检
        Orch->>Eval: validate(report, worker_outputs)
        Eval->>Eval: ① Markdown格式完整? ✓<br/>② report.根因 == diagnosis.root_cause? ✓<br/>③ 修复建议引用了topology_evidence? ✓<br/>④ 覆盖所有affected_kpis? ✓
        Eval-->>Orch: {passed: true}
    end

    Orch-->>User: 📊 诊断报告 + 🔗 完整证据链
```

---

## 3. Diagnosis Worker 内部三级流水线

```mermaid
flowchart TD
    INPUT["输入: KPI数据 + anomaly_result<br/>来自 Detection Worker"]

    INPUT --> S1
    INPUT --> S2

    subgraph PARALLEL["Step 1 & 2 可并行"]
        S1["📊 Step 1: ML 分类<br/>────────────<br/>Autoformer.forward()<br/>→ Top-3 根因候选 11类<br/>⚠️ 修复: Jamming 不再被过滤<br/>LLM: ❌ / 耗时: ~0.1s"]
        
        S2["🏗️ Step 2: GraphRAG<br/>────────────<br/>知识图谱查询<br/>节点: BS/Jammer/Zone/Mobile<br/>边: interference/covers/moves_to<br/>LLM: ❌ / 耗时: ~0.5s"]
    end

    S1 --> S3
    S2 --> S3

    S3["🧠 Step 3: LLM 推理<br/>────────────<br/>Prompt = ML Top-3<br/>+ GraphRAG 拓扑上下文<br/>+ affected_kpis<br/>+ telecom 领域知识<br/>→ 最终根因 + CoT + 置信度<br/>LLM: ✅ / 耗时: ~3-5s"]
    
    S3 --> OUTPUT["输出: { root_cause,<br/>  confidence,<br/>  duration,<br/>  reasoning CoT,<br/>  topology_evidence }"]

    style S1 fill:#F39C12,color:#fff
    style S2 fill:#16A085,color:#fff
    style S3 fill:#C0392B,color:#fff
    style INPUT fill:#95A5A6,color:#fff
    style OUTPUT fill:#4A90D9,color:#fff
```

---

## 4. 知识图谱结构图

```mermaid
graph LR
    subgraph 干扰源
        JAMMER[🔴 Jammer<br/>干扰器<br/>active: true<br/>type: broadband]
    end

    subgraph 核心网络
        BS[📡 BaseStation<br/>gNB<br/>18 KPI channels]
    end

    subgraph 覆盖区域
        ZA[🟢 Zone A<br/>0-3m · 信号强<br/>正常覆盖]
        ZB[🟡 Zone B<br/>3-6m · 受干扰<br/>RSRP下降·BLER突发]
        ZC[🟠 Zone C<br/>>6m · 信号弱<br/>移动性相关]
    end

    subgraph 终端
        M1[📱 Mobile<br/>app: YouTube<br/>mobility: Yes<br/>located: Zone A]
        M2[📱 Mobile<br/>app: File<br/>mobility: No<br/>located: Zone B]
    end

    JAMMER -->|"interference<br/>宽带噪声压制"| BS
    BS -->|covers| ZA
    BS -->|covers| ZB
    BS -->|covers| ZC
    M1 -->|located_in| ZA
    M2 -->|located_in| ZB
    M1 -.->|"moves_to<br/>（切换延迟增加）"| ZB
    ZA ---|adjacent| ZB
    ZB ---|adjacent| ZC

    style JAMMER fill:#E74C3C,color:#fff
    style BS fill:#4A90D9,color:#fff
    style ZA fill:#2ECC71,color:#fff
    style ZB fill:#F1C40F,color:#333
    style ZC fill:#E67E22,color:#fff
    style M1 fill:#9B59B6,color:#fff
    style M2 fill:#9B59B6,color:#fff
```

---

## 5. 条件分支：无异常场景

```mermaid
sequenceDiagram
    participant User as 👤 运维
    participant Orch as 🧠 Orchestrator
    participant DetW as 📡 Detection (ML)
    participant RepW as 💬 Reporter (LLM)

    User->>Orch: "网络状态巡检"

    Orch->>DetW: detect(all zones)
    DetW-->>Orch: {has_anomaly: false, score: 0.08}

    Note over Orch: has_anomaly==false<br/>→ 跳过 Diagnosis<br/>→ 不调 GraphRAG<br/>→ 不调 Diag LLM<br/>省 3 次调用

    Orch->>RepW: generate_report({status: normal})
    RepW-->>Orch: "✅ 全网正常 · 无异常"

    Orch-->>User: 巡检报告
```

---

## 6. 数据流：原代码 gap → Multi-Agent 补齐

```mermaid
flowchart LR
    subgraph 原代码["原代码（已有）"]
        A1["load_dataset()<br/>✅ 32K样本"] 
        A2["preprocess()<br/>⚠️ 4 tasks · 只读 KPIs"]
        A3["encoder.forward()<br/>✅ 8模型"]
        A4["evaluate()<br/>⚠️ 只打印 Accuracy/RMSE"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph 被忽略["被忽略的数据（现在要用）"]
        B1["description<br/>NL 网络状态摘要"]
        B2["labels<br/>zone/mobility/app"]
        B3["QnA<br/>220万条·reasoning"]
        B4["troubleshooting_tickets<br/>诊断文本"]
        B5["Jamming 样本<br/>唯一真实异常"]
    end

    subgraph MultiAgent["Multi-Agent 补齐"]
        C1["Detection Worker<br/>复用 A2 + A3"]
        C2["Diagnosis Worker<br/>ML(11类·含Jamming)<br/>+ GraphRAG(B1+B2+B4)<br/>+ LLM 推理"]
        C3["Reporter Worker<br/>LLM · 参照 B3 风格"]
    end

    A2 -.->|"复用"| C1
    A3 -.->|"复用"| C2
    B1 & B2 & B4 -.->|"构建"| C2
    B3 -.->|"参考范式"| C3
    B5 -.->|"放回诊断流程"| C2

    style 原代码 fill:#95A5A6,color:#fff
    style 被忽略 fill:#E74C3C,color:#fff
    style MultiAgent fill:#2ECC71,color:#fff
```

---

## 7. Docker 部署架构

```mermaid
graph TB
    subgraph "WSL2 Ubuntu 22.04 · Docker Compose"
        ORCH_CONT[🐳 orchestrator<br/>port: 8000<br/>restart: unless-stopped]
        DET_CONT[🐳 detection-worker<br/>纯ML · TimesNet]
        DIAG_CONT[🐳 diagnosis-worker<br/>ML + GraphRAG + LLM]
        REP_CONT[🐳 reporter-worker<br/>LLM]
        EVAL_CONT[🐳 evaluator<br/>LLM]
    end

    subgraph 数据卷
        HF_CACHE[📁 TelecomTS data_cache/<br/>32K样本 · 只读]
        KG_DATA[📁 knowledge_graph/<br/>topology.json + 向量索引]
        CONFIG_VOL[📁 configs/:ro]
    end

    User[👤 curl localhost:8000] --> ORCH_CONT
    ORCH_CONT --> DET_CONT & DIAG_CONT & REP_CONT
    REP_CONT --> EVAL_CONT

    DET_CONT -.-> HF_CACHE
    DIAG_CONT -.-> HF_CACHE
    DIAG_CONT -.-> KG_DATA
```

---

## 8. Token / 时间预算

```
场景                        LLM调用              耗时       Token
─────────────────────────────────────────────────────────────
有异常 (Jamming诊断):
  Orchestrator 意图解析       1次                   ~1s       ~500
  Detection (ML)              0次                   ~0.01s    0
  Diagnosis ML (Autoformer)   0次                   ~0.1s     0
  Diagnosis GraphRAG          0次                   ~0.5s     0
  Diagnosis LLM 推理          1次                   ~4s       ~2000
  Reporter 报告生成           1次                   ~3s       ~1500
  Evaluator 质检              1次                   ~1s       ~800
  ─────────────────────────────────────────────────────────
  总计                       4次 LLM               ~10s      ~4800

无异常 (巡检):
  Orchestrator 意图解析       1次                   ~1s       ~500
  Detection (ML)              0次                   ~0.01s    0
  Reporter 简单报告           1次                   ~1s       ~500
  ─────────────────────────────────────────────────────────
  总计                       2次 LLM               ~2s       ~1000
```
