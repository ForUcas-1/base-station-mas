#!/bin/bash
# BaseStation-MAS 环境初始化脚本
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== BaseStation-MAS 环境初始化 ==="
echo "项目路径: $PROJECT_ROOT"

# 1. 检查 Python 版本
echo ""
echo "[1/6] 检查 Python 版本..."
PYTHON=$(which python3.11 2>/dev/null || which python3 2>/dev/null || which python 2>/dev/null)
PY_VER=$($PYTHON --version 2>&1)
echo "  Python: $PY_VER ($PYTHON)"

if ! $PYTHON -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "  ❌ 需要 Python >= 3.10"
    exit 1
fi
echo "  ✓"

# 2. 安装 python3-venv（Ubuntu/Debian 需要）
echo ""
echo "[2/6] 检查 venv 支持..."
PY_MAJOR_MINOR=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if command -v apt &>/dev/null; then
    if ! dpkg -s "python${PY_MAJOR_MINOR}-venv" >/dev/null 2>&1; then
        echo "  ⚠ python${PY_MAJOR_MINOR}-venv 未安装，正在安装..."
        sudo apt install -y "python${PY_MAJOR_MINOR}-venv"
    fi
    echo "  ✓ python${PY_MAJOR_MINOR}-venv 已就绪"
else
    echo "  ✓ 非 Debian 系统，跳过"
fi

# 3. 创建虚拟环境
echo ""
echo "[3/6] 创建虚拟环境..."
if [ -f "venv/bin/activate" ]; then
    echo "  venv/ 已存在，跳过"
else
    # 清理之前失败残留的空目录
    [ -d "venv" ] && rm -rf venv
    $PYTHON -m venv venv
    echo "  ✓ venv/ 已创建"
fi

# 4. 安装依赖
echo ""
echo "[4/6] 安装 Python 依赖..."
source venv/bin/activate
pip install --upgrade pip
pip install -v -r requirements.txt
echo "  ✓ 依赖安装完成"

# 5. 复制 .env（如果不存在）
echo ""
echo "[5/6] 检查 .env 配置..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ⚠ 已创建 .env（从 .env.example），请编辑填入 API Key"
else
    echo "  .env 已存在，跳过"
fi

# 6. 预下载模型（若本地已有则跳过）
echo ""
echo "[6/6] 预下载模型..."
if [ -d "data/models/sentence_transformers" ]; then
    echo "  模型已缓存，跳过"
else
    echo "  下载 all-MiniLM-L6-v2 ..."
    python scripts/download_models.py
    echo "  ✓ 模型下载完成"
fi

echo ""
echo "=== 初始化完成 ==="
echo ""
echo "激活环境:"
echo "  source venv/bin/activate"
echo ""
echo "构建知识图谱索引:"
echo "  python knowledge_graph/build_graph.py"
echo ""
echo "启动服务:"
echo "  python src/main.py"
