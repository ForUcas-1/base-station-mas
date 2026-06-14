# BaseStation-MAS — 5G 基站告警智能诊断系统

基于 Multi-Agent 架构的 5G 基站 KPI 异常检测与根因分析系统。

## 架构

```
User Query → Orchestrator(LLM) → Detection(TimesNet) → {anomaly?}
  → YES: Diagnosis(Autoformer+GraphRAG+LLM) → Reporter(LLM) → Evaluator(LLM)
  → NO:  Reporter(LLM)
```

| 组件 | 类型 | 模型 | 职责 |
|------|------|------|------|
| Orchestrator | LLM | Claude/GPT | 意图解析、任务分解、条件路由 |
| Detection Worker | 纯 ML | TimesNet | 异常二分类 |
| Diagnosis Worker | ML+KG+LLM | Autoformer + GraphRAG + LLM | 11 类根因分析 + CoT |
| Reporter Worker | LLM | Claude/GPT | Markdown 诊断报告 |
| Evaluator | LLM | Claude/GPT | 质检（格式/事实/幻觉） |

## 快速开始

### 环境要求

- Python 3.10+
- WSL2 / Linux
- 6GB+ 磁盘空间（含数据集缓存）

### 安装

```bash
cd /root/projects/base-station-mas
bash setup_env.sh
source venv/bin/activate
cp .env.example .env
# 编辑 .env 填入 ANTHROPIC_API_KEY 或 OPENAI_API_KEY
```

### 构建知识图谱

```bash
python knowledge_graph/build_graph.py
```

### 启动服务

```bash
python src/main.py
```

### 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 提交诊断
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"query": "Zone B 的 RSRP 为什么突然恶化？"}'
```

### API 文档

启动后访问 http://localhost:8000/docs

## 项目结构

```
base-station-mas/
├── configs/              # YAML 配置文件
├── data/
│   ├── cache/            # TelecomTS 数据集 (1.1 GB, gitignored)
│   └── checkpoints/      # 预训练模型权重
├── knowledge_graph/
│   ├── topology.json     # 静态拓扑（7 节点 + 9 边）
│   └── embeddings/       # FAISS 向量索引（生成）
├── src/
│   ├── main.py           # FastAPI 入口
│   ├── encoders/         # 8 个预训练编码器（TimesNet, Autoformer, etc.）
│   ├── orchestrator/     # 意图路由 + Supervisor
│   ├── workers/          # 5 个 Worker
│   ├── graphrag/         # NetworkX + FAISS 知识图谱
│   ├── models/           # 编码器注册表 + 任务头
│   ├── llm/              # LLM API 客户端
│   └── utils/            # 配置加载 + 日志
├── contracts/            # Pydantic Schema（Worker 契约）
├── tests/                # 单元测试
├── docs/                 # 设计文档（PRD/ADR/Architecture）
└── requirements.txt
```

## 数据集

TelecomTS (ICLR 2026)：32,000 时序样本，18 通道 KPI × 128 时间步，11 种异常类型，220 万条 Q&A。

## 运行测试

```bash
pytest tests/ -v
```


