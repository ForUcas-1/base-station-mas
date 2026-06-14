# BaseStation-MAS — 5G 基站告警智能诊断系统

基于 Multi-Agent 架构的 5G 基站 KPI 异常检测与根因分析系统。

## 架构

```
User Query → Orchestrator(LLM) → Detection(TimesNet) → {anomaly?}
  → YES: Diagnosis(Autoformer+GraphRAG+LLM) → Reporter(LLM) → Evaluator(LLM)
  → NO:  跳过 Diagnosis，直接输出正常报告
```

| 组件 | 类型 | 模型 | 职责 |
|------|------|------|------|
| Orchestrator | LLM | DeepSeek/GPT/Claude | 意图解析、任务分解、条件路由 |
| Detection Worker | 纯 ML | TimesNet | 异常二分类（ground truth / ML 推理） |
| Diagnosis Worker | ML+KG+LLM | Autoformer + GraphRAG + LLM | 11 类根因分析 + CoT 推理 |
| Reporter Worker | LLM | DeepSeek/GPT/Claude | Markdown 诊断报告 |
| Evaluator | LLM | DeepSeek/GPT/Claude | 质检（格式/事实/幻觉） |

## 快速开始

### 环境要求

- Python 3.10+
- WSL2 / Linux
- 6GB+ 磁盘空间（含数据集缓存）

### 1. 克隆项目

```bash
git clone https://github.com/ForUcas-1/base-station-mas.git
cd base-station-mas
```

### 2. 下载数据集

从 [TelecomTS](https://github.com/Ali-maatouk/TelecomTS) 下载数据集，放入 `data/cache/` 目录：

```bash
# 方式一：用 HuggingFace 下载（推荐）
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速
python -c "
from datasets import load_dataset
ds = load_dataset('AliMaatouk/TelecomTS', data_files={'full': '**/chunked.jsonl'})['full']
print(f'Downloaded {len(ds)} samples')
"

# 方式二：从 TelecomTS 项目复制已有缓存
cp -r /path/to/TelecomTS/data_cache/* data/cache/
```

数据集包含 32,000 时序样本，18 通道 KPI × 128 时间步，11 种异常类型，220 万条 Q&A。

### 3. 安装依赖

```bash
bash setup_env.sh
source venv/bin/activate
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY（支持 deepseek / openai / anthropic）
```

### 4. 配置 LLM

编辑 `.env` 即可切换模型，所有 Worker 统一生效：

```bash
LLM_PROVIDER=deepseek           # deepseek | openai | anthropic
LLM_MODEL=deepseek-v4-pro       # 模型名
DEEPSEEK_API_KEY=sk-xxx         # API Key
```

### 5. 构建知识图谱及模型

```bash
# 预下载嵌入模型（一次性）
python scripts/download_models.py

# 构建 GraphRAG 知识图谱 + FAISS 索引
python knowledge_graph/build_graph.py
```

### 6. 启动服务

```bash
python src/main.py
```

浏览器打开 `http://localhost:8000` 进入 Web 仪表盘。

### 7. Docker 部署（可选）

```bash
# 构建并启动
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

**注意事项：**

| 项目 | 说明 |
|------|------|
| 数据集 | 不打包进镜像，通过 volume 挂载 `./data/cache:/app/data/cache:ro` |
| 嵌入模型 | 已打包进镜像（`data/models/`），无需额外下载 |
| SQLite | `data/basestation.db` 持久化到宿主机，重启不丢失 |
| KG 索引 | `knowledge_graph/embeddings/` 挂载到宿主机 |
| API Key | 通过 `.env` 文件传入，不打包进镜像 |
| 端口 | 默认 `8000`，可通过 `docker-compose.yml` 修改 |

**跨机器移植：**

```bash
# 源机器导出
docker save basestation-mas | gzip > basestation-mas.tar.gz

# 目标机器导入
docker load < basestation-mas.tar.gz

# 单独复制数据集和配置
scp -r data/cache/ user@target:~/base-station-mas/data/cache/
scp .env user@target:~/base-station-mas/.env

# 启动
docker compose up -d
```

## Web 仪表盘

启动后可访问交互式诊断面板，提供以下功能：

### 页面布局

```
┌──────────────────────────────────────────────────────┐
│  📡 5G 基站告警智能诊断系统                             │
├───────────────────────────┬───────┬──────────────────┤
│  网络拓扑图 (SVG)         │ Agent │  诊断事件 + 历史  │
│  实时高亮异常路径          │ 工作流 │                  │
├───────────────────────────┴───────┤                  │
│  异常概率 | 根因 | 置信度 | KPI | 耗时 | 样本       │
├───────────────────────────────────┴──────────────────┤
│  [正常] [异常] [▶一次性] [⏱定时] [⏹停止]  间隔: 15s  │
└──────────────────────────────────────────────────────┘
```

### 操作说明

| 功能 | 说明 |
|------|------|
| **样本选择** | 点击 [正常] 或 [异常] 随机选取数据集样本 |
| **一次性判断** | 对当前样本跑完整检测 + 诊断 + 报告 |
| **定时判断** | 每 N 秒自动跑一轮，检测到异常生成报告后自动停止 |
| **停止** | 手动中断定时监测，立即关闭所有 LLM 连接 |
| **拓扑图** | 根因诊断后自动高亮受影响节点和路径 |
| **Agent 工作流** | 实时显示各 Agent 状态（空闲/运行中/完成） |
| **历史记录** | 所有诊断结果持久化到 SQLite，支持查看和删除 |

### 通信协议

前端与后端通过 **WebSocket** 实时通信，命令和事件统一走 `/ws` 端点：

```
前端 → 后端: {"cmd":"select_sample"/"run_oneshot"/"start_monitor"/"stop_monitor"}
后端 → 前端: {"type":"detection_done"/"round_complete"/"agent_status"/"topology_highlight"/...}
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 仪表盘 |
| GET | `/health` | 健康检查 |
| POST | `/diagnose` | 自然语言诊断（curl 兼容） |
| WS | `/ws` | WebSocket 实时通信 |
| GET | `/api/topology` | 知识图谱数据 |
| GET | `/api/history` | 历史诊断记录 |
| DELETE | `/api/history/{id}` | 删除诊断记录 |

## 项目结构

```
base-station-mas/
├── configs/              # YAML 配置文件
├── data/
│   ├── cache/            # TelecomTS 数据集（需自行下载）
│   ├── checkpoints/      # 预训练模型权重
│   └── models/           # 预下载的嵌入模型
├── knowledge_graph/
│   ├── topology.json     # 静态拓扑（7 节点 + 9 边）
│   └── embeddings/       # FAISS 向量索引（生成）
├── scripts/              # 工具脚本
├── src/
│   ├── main.py           # FastAPI 入口
│   ├── encoders/         # 8 个预训练编码器（TimesNet, Autoformer, ...）
│   ├── orchestrator/     # Supervisor DAG 引擎
│   ├── workers/          # 5 个 Worker（Detection/Diagnosis/Reporter/Evaluator）
│   ├── graphrag/         # NetworkX + FAISS 知识图谱
│   ├── data/             # 数据集加载 + 预处理
│   ├── models/           # 编码器注册表 + 任务头工厂
│   ├── web/              # Web 仪表盘
│   ├── db/               # SQLite 持久化
│   ├── llm/              # LLM API 客户端（DeepSeek/OpenAI/Anthropic）
│   └── utils/            # 工具函数
├── contracts/            # Pydantic Schema
├── tests/                # 单元测试
└── requirements.txt
```

## 运行测试

```bash
pytest tests/ -v
```
