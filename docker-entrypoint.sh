#!/bin/bash
set -e

echo "=== BaseStation-MAS Docker Entrypoint ==="

# Build knowledge graph index (requires dataset at /app/data/cache)
if [ -d "/app/data/cache" ] && [ -n "$(ls -A /app/data/cache 2>/dev/null)" ]; then
    echo "[1/2] Building GraphRAG knowledge graph..."
    python knowledge_graph/build_graph.py
else
    echo "[1/2] ⚠ Dataset not found at /app/data/cache — skipping KG build"
    echo "    Mount your TelecomTS dataset: -v /path/to/cache:/app/data/cache"
fi

echo "[2/2] Starting BaseStation-MAS..."
exec python src/main.py
