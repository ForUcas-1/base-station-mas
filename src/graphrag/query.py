"""High-level GraphRAG query API used by Diagnosis Worker Step 2.

Combines:
  1. Structured graph traversal (KnowledgeGraph)
  2. Semantic vector search (GraphEmbedder)

To produce a rich GraphRAGContext for the LLM reasoning step.
"""

from pathlib import Path
from typing import Any

from contracts.diagnosis import (
    EntityState,
    GraphRAGContext,
    HistoricalCase,
    TopologyPath,
)
from contracts.task import OrchestratorTask
from src.graphrag.embedder import GraphEmbedder
from src.graphrag.graph import KnowledgeGraph


class GraphRAGQuery:
    """Simplified query interface for the Diagnosis Worker Step 2.

    Abstracts away the hybrid retrieval internals.

    Usage:
        rag = GraphRAGQuery(
            kg_path="knowledge_graph/topology.json",
            embeddings_dir="knowledge_graph/embeddings",
        )
        context = await rag.query(task)
    """

    def __init__(
        self,
        kg_path: str | Path,
        embeddings_dir: str | Path = "knowledge_graph/embeddings",
        vector_store: str = "faiss",
        embedding_model: str = "all-MiniLM-L6-v2",
        model_cache_dir: str | None = None,
    ):
        self.kg = KnowledgeGraph(kg_path)
        self.embedder = GraphEmbedder(
            model_name=embedding_model,
            vector_store=vector_store,
            cache_dir=model_cache_dir,
        )
        self.embeddings_dir = Path(embeddings_dir)
        self.index_path = str(self.embeddings_dir / "index.faiss")
        # Warm up — load model now, not lazily during first query
        _ = self.embedder.model

    async def query(self, task: OrchestratorTask) -> GraphRAGContext:
        """Execute a hybrid GraphRAG query.

        1. Structured graph traversal for interference paths
        2. Vector search for similar historical cases
        3. Mobility context if relevant
        4. Return unified GraphRAGContext

        Args:
            task: The orchestrator task with DetectionResult populated.

        Returns:
            Structured GraphRAGContext for LLM reasoning.
        """
        detection = task.context.detection_result
        zone = task.zone

        # Infer zone from affected_kpis context if not explicitly given
        if not zone and detection:
            # Default to Zone_B (most likely interference zone)
            zone = "B"

        zone = f"Zone_{zone}" if zone and not zone.startswith("Zone_") else zone or "Zone_B"

        # --- 1. Graph traversal ---
        # Use ML top-1 candidate to determine root cause
        ml_root_cause = ""
        if detection:
            ml_root_cause = detection.root_cause if hasattr(detection, 'root_cause') else ""

        rc_info = self.kg.query_by_root_cause(ml_root_cause or "Unknown", zone)
        topology_info = self.kg.query_interference_path(zone)

        impact_paths = [
            TopologyPath(
                path=p["path"],
                edges=p.get("edges", []),
            )
            for p in topology_info.get("paths_detail", [])
        ]
        # Add root-cause-specific evidence
        if rc_info.get("highlight_edges"):
            impact_paths.append(TopologyPath(
                path=rc_info.get("highlight_nodes", []),
                edges=rc_info.get("highlight_edges", []),
            ))

        # Entity states
        entity_states = []
        for key in ("interference_source", "affected_zone"):
            entity = topology_info.get(key)
            if entity and isinstance(entity, dict):
                entity_states.append(EntityState(
                    entity_type=entity.get("type", "?"),
                    entity_id=entity.get("node_id", ""),
                    attributes=entity.get("attributes", {}),
                ))

        # --- 2. Vector search ---
        similar_cases = []
        if detection and detection.affected_kpis:
            query_text = self._build_query_text(task, zone)
            try:
                results = self.embedder.search(
                    query_text,
                    top_k=5,
                    index_path=self.index_path,
                )
                for r in results:
                    similar_cases.append(HistoricalCase(
                        anomaly_type="",  # populated if tickets indexed
                        description=f"{r.get('type', '?')}: {r.get('label', '')}",
                        similarity=r.get("similarity", 0.0),
                    ))
            except Exception:
                # FAISS index may not exist yet; gracefully degrade
                pass

        # --- 3. Mobility context ---
        mobility_events = topology_info.get("mobility_events", [])

        # --- 4. Build serialized natural language context ---
        raw_context = self._serialize_context(
            topology_info, similar_cases, mobility_events, zone
        )

        return GraphRAGContext(
            interference_source=(
                topology_info.get("interference_source", {}).get("label", "")
                or None
            ),
            impact_paths=impact_paths,
            entity_states=entity_states,
            similar_cases=similar_cases,
            mobility_events=mobility_events,
            raw_context=raw_context,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_query_text(self, task: OrchestratorTask, zone: str) -> str:
        """Build a natural language query string for vector search."""
        detection = task.context.detection_result
        parts = [task.user_query]

        if detection:
            kpis = ", ".join(detection.affected_kpis)
            parts.append(f"Affected KPIs: {kpis}")
            parts.append(f"Anomaly score: {detection.anomaly_score:.2f}")

        parts.append(f"Zone: {zone}")
        return " | ".join(parts)

    @staticmethod
    def _serialize_context(
        topology_info: dict[str, Any],
        similar_cases: list[HistoricalCase],
        mobility_events: list[str],
        zone: str,
    ) -> str:
        """Serialize the GraphRAG context into a natural language paragraph
        suitable for LLM prompt injection."""
        lines = []

        # Interference source
        src = topology_info.get("interference_source", {})
        if src and src.get("type"):
            lines.append(
                f"Interference source: {src.get('type')} '{src.get('label', '')}' "
                f"(active={src.get('attributes', {}).get('active', '?')})"
            )

        # Impact paths
        paths = topology_info.get("impact_paths", [])
        if paths:
            lines.append("Impact paths:")
            for p in paths[:5]:
                lines.append(f"  - {p}")

        # Affected zone
        zone_info = topology_info.get("affected_zone", {})
        if zone_info:
            symptoms = zone_info.get("attributes", {}).get("typical_symptoms", [])
            if symptoms:
                lines.append(
                    f"{zone} typical symptoms: {', '.join(symptoms)}"
                )

        # Mobility
        if mobility_events:
            lines.append("Mobility events:")
            for event in mobility_events:
                lines.append(f"  - {event}")

        # Historical cases
        if similar_cases:
            lines.append("Similar historical cases:")
            for case in similar_cases[:3]:
                lines.append(f"  - {case.description} (sim={case.similarity:.2f})")

        return "\n".join(lines)
