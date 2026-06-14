"""NetworkX-based knowledge graph for GraphRAG.

Wraps the static topology from topology.json and provides
graph traversal, entity query, and pattern matching.
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx


NODE_TYPES = ["BaseStation", "Jammer", "Zone_A", "Zone_B", "Zone_C", "Mobile"]
EDGE_TYPES = ["interference", "covers", "located_in", "moves_to", "adjacent"]


class KnowledgeGraph:
    """NetworkX MultiDiGraph wrapping the base station network topology.

    Usage:
        kg = KnowledgeGraph("knowledge_graph/topology.json")
        paths = kg.find_paths("JAM_001", "ZB_001")
        context = kg.get_entity_context("ZB_001")
    """

    def __init__(self, topology_path: str | Path):
        self.topology_path = Path(topology_path)
        self.graph = nx.MultiDiGraph()
        self._node_index: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load topology from JSON file."""
        with open(self.topology_path, encoding="utf-8") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            attrs = dict(node.get("attributes", {}))
            # Merge type/label into attrs; use node_type to avoid conflict
            attrs["node_type"] = node["type"]
            attrs["label"] = node.get("label", "")
            self.graph.add_node(node["id"], **attrs)
            self._node_index[node["id"]] = node

        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                key=edge["id"],
                type=edge["type"],
                **edge.get("attributes", {}),
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_node(self, node_id: str) -> dict | None:
        """Get node attributes by ID."""
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    def get_node_by_type(self, node_type: str) -> list[str]:
        """Get all node IDs of a given type."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == node_type
        ]

    def get_entity_context(self, node_id: str) -> dict[str, Any]:
        """Get a node's attributes and its directly connected entities.

        Returns a dict ready for LLM prompt injection.
        """
        node = self.get_node(node_id)
        if node is None:
            return {"error": f"Node '{node_id}' not found"}

        neighbors = []
        for _, neighbor, edge_data in self.graph.edges(node_id, data=True):
            neighbor_node = self.get_node(neighbor) or {}
            neighbors.append({
                "node_id": neighbor,
                "type": neighbor_node.get("node_type", "?"),
                "label": neighbor_node.get("label", ""),
                "relation": edge_data.get("type", "?"),
            })

        # Also check incoming edges
        for src, _, edge_data in self.graph.in_edges(node_id, data=True):
            if src not in {n["node_id"] for n in neighbors}:
                src_node = self.get_node(src) or {}
                neighbors.append({
                    "node_id": src,
                    "type": src_node.get("node_type", "?"),
                    "label": src_node.get("label", ""),
                    "relation": f"← {edge_data.get('type', '?')}",
                })

        return {
            "node_id": node_id,
            "type": node.get("node_type", "?"),
            "label": node.get("label", ""),
            "attributes": {k: v for k, v in node.items()
                          if k not in ("node_type", "label")},
            "neighbors": neighbors,
        }

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------
    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """Find all simple paths between source and target (up to max_depth).

        Returns list of node-id paths.
        """
        try:
            return list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_depth,
            ))
        except nx.NodeNotFound:
            return []

    def find_paths_by_type(
        self,
        source_type: str,
        target_type: str,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Find all paths between any source-type and target-type nodes.

        Returns list of {path: [...], edges: [...]} dicts.
        """
        sources = self.get_node_by_type(source_type)
        targets = self.get_node_by_type(target_type)
        results = []

        for src in sources:
            for tgt in targets:
                if src == tgt:
                    continue
                for path in self.find_paths(src, tgt, max_depth):
                    edge_list = []
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                        if edge_data:
                            # MultiDiGraph: edge_data is {key: attrs}
                            edge_type = next(iter(edge_data.values())).get("type", "?")
                            edge_list.append(edge_type)
                    results.append({
                        "path": path,
                        "edges": edge_list,
                    })
        return results

    def query_by_root_cause(
        self, root_cause: str, zone: str = "Zone_B",
    ) -> dict[str, Any]:
        """Return topology evidence (nodes + edges) for a specific root cause.

        Maps each of the 11 anomaly types to the relevant KG paths.
        """
        result = {"highlight_nodes": [], "highlight_edges": [], "context": ""}
        z = zone[-1]  # "A", "B", "C"

        jammer = self.get_node_by_type("Jammer")
        bs = self.get_node_by_type("BaseStation")
        zone_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("zone") == z]
        mobiles = self.get_node_by_type("Mobile")

        rc = root_cause.lower() if root_cause else ""

        if "jamming" in rc:
            result["highlight_nodes"] = jammer + bs + zone_nodes
            result["highlight_edges"] = [
                [jammer[0], bs[0]], [bs[0], zone_nodes[0]]
            ] if jammer and bs and zone_nodes else []
            result["context"] = "干扰器 → 基站 → 受扰区"

        elif "antenna" in rc:
            result["highlight_nodes"] = bs
            result["highlight_edges"] = []
            result["context"] = "基站天线故障"

        elif "co-channel" in rc or "同频" in rc:
            result["highlight_nodes"] = zone_nodes
            if zone_nodes:
                result["highlight_edges"] = [[zone_nodes[0], "ZB_001"]]
            result["context"] = "同频干扰"

        elif "doppler" in rc:
            # Mobile moving causes Doppler
            result["highlight_nodes"] = mobiles[:1] + zone_nodes[:1]
            result["highlight_edges"] = [
                [mobiles[0], zone_nodes[0]]
            ] if mobiles and zone_nodes else []
            result["context"] = "终端移动 → Doppler频移"

        elif "handover" in rc:
            result["highlight_nodes"] = mobiles[:1] + zone_nodes[:1]
            result["highlight_edges"] = [
                [mobiles[0], zone_nodes[0]]
            ] if mobiles and zone_nodes else []
            result["context"] = "切换算法故障"

        elif "buffer" in rc or "overflow" in rc:
            result["highlight_nodes"] = bs
            result["highlight_edges"] = []
            result["context"] = "缓冲区溢出"

        elif "resource" in rc:
            result["highlight_nodes"] = bs
            result["highlight_edges"] = []
            result["context"] = "资源分配异常"

        elif "congestion" in rc:
            result["highlight_nodes"] = bs + zone_nodes
            result["highlight_edges"] = [
                [bs[0], zone_nodes[0]]
            ] if bs and zone_nodes else []
            result["context"] = "网络拥塞"

        elif "filter" in rc:
            result["highlight_nodes"] = bs
            result["highlight_edges"] = []
            result["context"] = "射频滤波器故障"

        else:
            # Default: show BS + zone
            result["highlight_nodes"] = bs + zone_nodes
            result["highlight_edges"] = [
                [bs[0], zone_nodes[0]]
            ] if bs and zone_nodes else []
            result["context"] = f"Zone {z} 异常"

        return result

    def query_interference_path(
        self, zone: str = "Zone_B",
    ) -> dict[str, Any]:
        """Specialized query: find interference path from Jammer to a Zone.

        This is the primary GraphRAG query for the Jamming scenario.
        """
        jammer_nodes = self.get_node_by_type("Jammer")
        zone_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("zone") == zone[-1]  # "A", "B", "C"
        ]

        if not jammer_nodes or not zone_nodes:
            return {"error": "Jammer or Zone node not found"}

        paths = self.find_paths_by_type("Jammer", f"Zone_{zone[-1]}")

        # Build structured context
        interference_source = self.get_entity_context(jammer_nodes[0])
        zone_context = self.get_entity_context(zone_nodes[0])

        # Check mobility
        mobility_events = []
        for node_id in self.get_node_by_type("Mobile"):
            node = self.get_node(node_id) or {}
            if node.get("mobility"):
                mobile_ctx = self.get_entity_context(node_id)
                for nb in mobile_ctx.get("neighbors", []):
                    if nb.get("relation") == "moves_to" and zone in str(nb.get("node_id", "")):
                        mobility_events.append(
                            f"UE '{node.get('label', node_id)}' moving "
                            f"from {node.get('current_zone', '?')} to {zone}"
                        )

        return {
            "interference_source": interference_source,
            "affected_zone": zone_context,
            "impact_paths": [
                " → ".join(p["path"]) for p in paths
            ],
            "mobility_events": mobility_events,
            "paths_detail": paths,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Return a summary of the graph for logging/debugging."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": {
                nt: len(self.get_node_by_type(nt))
                for nt in NODE_TYPES
            },
            "edge_types": EDGE_TYPES,
        }
