#!/usr/bin/env python3
"""Build the GraphRAG knowledge graph from TelecomTS dataset.

This script:
  1. Reads the static topology from knowledge_graph/topology.json
  2. Scans the TelecomTS dataset samples (description, labels, anomalies,
     troubleshooting_tickets, statistics)
  3. Enriches the topology with dataset-derived attributes
  4. Generates text embeddings for all nodes via SentenceTransformer
  5. Builds a FAISS vector index for semantic retrieval

Usage:
    python knowledge_graph/build_graph.py
"""

import os
import sys
from pathlib import Path

# Ensure src/ and project root are on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    from src.data.loader import DatasetLoader
    from src.graphrag.builder import GraphBuilder
    from src.graphrag.embedder import GraphEmbedder
    from src.utils.config import load_yaml_config
    from src.utils.logging import get_logger

    logger = get_logger("build_graph")

    # Load config
    config_path = _PROJECT_ROOT / "configs" / "graphrag.yaml"
    config = load_yaml_config(str(config_path))
    graphrag_cfg = config.get("graphrag", config)

    # Build topology
    topology_path = _PROJECT_ROOT / graphrag_cfg.get("topology", {}).get(
        "path", "knowledge_graph/topology.json"
    )
    logger.info(f"Loading dataset and building topology → {topology_path}")

    loader = DatasetLoader()
    logger.info(f"Dataset loaded: {len(loader)} samples")

    builder = GraphBuilder(loader, graphrag_cfg)
    topology = builder.build_topology(topology_path)
    logger.info(
        f"Topology built: {len(topology.get('nodes', []))} nodes, "
        f"{len(topology.get('edges', []))} edges"
    )

    # Build embeddings
    embedding_cfg = graphrag_cfg.get("embedding", {})
    output_dir = _PROJECT_ROOT / graphrag_cfg.get("vector_store", {}).get(
        "faiss", {}
    ).get("output_dir", "knowledge_graph/embeddings")

    logger.info(f"Building FAISS index → {output_dir}")
    logger.info(f"Embedding model: {embedding_cfg.get('model', 'all-MiniLM-L6-v2')}")

    embedder = GraphEmbedder(
        model_name=embedding_cfg.get("model", "all-MiniLM-L6-v2"),
        vector_store=graphrag_cfg.get("vector_store", {}).get("type", "faiss"),
        device=embedding_cfg.get("device", "cpu"),
        cache_dir=str(_PROJECT_ROOT / "data" / "models"),
    )

    index_path = embedder.build_index(
        topology_path=str(topology_path),
        output_dir=str(output_dir),
    )

    logger.info(f"✓ FAISS index saved to: {index_path}")
    logger.info("Knowledge graph build complete!")


if __name__ == "__main__":
    main()
