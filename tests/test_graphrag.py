"""Tests for GraphRAG knowledge graph module."""

import json
import tempfile
from pathlib import Path

import pytest


TOPOLOGY = {
    "nodes": [
        {"id": "BS_001", "type": "BaseStation", "label": "gNB"},
        {"id": "JAM_001", "type": "Jammer", "label": "Jammer", "attributes": {"active": True}},
        {"id": "ZB_001", "type": "Zone_B", "label": "Zone B", "attributes": {"zone": "B"}},
        {"id": "MOB_001", "type": "Mobile", "label": "UE", "attributes": {"mobility": True, "current_zone": "Zone_A"}},
    ],
    "edges": [
        {"id": "E_001", "source": "JAM_001", "target": "BS_001", "type": "interference"},
        {"id": "E_002", "source": "BS_001", "target": "ZB_001", "type": "covers"},
        {"id": "E_003", "source": "MOB_001", "target": "ZB_001", "type": "moves_to"},
    ],
}


class TestKnowledgeGraph:
    """Test the NetworkX graph wrapper."""

    def test_load_and_summary(self):
        from src.graphrag.graph import KnowledgeGraph

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(TOPOLOGY, f)
            tmp_path = f.name

        try:
            kg = KnowledgeGraph(tmp_path)
            summary = kg.summary()
            assert summary["num_nodes"] == 4
            assert summary["num_edges"] == 3
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_find_paths(self):
        from src.graphrag.graph import KnowledgeGraph

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(TOPOLOGY, f)
            tmp_path = f.name

        try:
            kg = KnowledgeGraph(tmp_path)
            paths = kg.find_paths("JAM_001", "ZB_001")
            assert len(paths) > 0
            # Should find JAM_001 -> BS_001 -> ZB_001
            assert any("JAM_001" in p and "ZB_001" in p for p in paths)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_get_entity_context(self):
        from src.graphrag.graph import KnowledgeGraph

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(TOPOLOGY, f)
            tmp_path = f.name

        try:
            kg = KnowledgeGraph(tmp_path)
            ctx = kg.get_entity_context("ZB_001")
            assert ctx["type"] == "Zone_B"
            assert len(ctx["neighbors"]) > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_query_interference_path(self):
        from src.graphrag.graph import KnowledgeGraph

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(TOPOLOGY, f)
            tmp_path = f.name

        try:
            kg = KnowledgeGraph(tmp_path)
            result = kg.query_interference_path("Zone_B")
            assert "interference_source" in result
            assert len(result["impact_paths"]) > 0
            # Should detect mobility
            assert len(result["mobility_events"]) > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
