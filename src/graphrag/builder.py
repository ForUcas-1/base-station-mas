"""Knowledge graph builder from TelecomTS dataset."""

import json
from pathlib import Path
from typing import Any

from src.data.loader import DatasetLoader


class GraphBuilder:
    """Constructs the knowledge graph from TelecomTS dataset fields.

    Enriches the static topology.json with:
      - description field → text embeddings for semantic search
      - labels (zone, mobility, application) → node attributes
      - anomalies → Jammer.active, Zone signal strength
      - troubleshooting_tickets → historical cases for vector search
      - statistics → per-KPI stats on BaseStation
    """

    def __init__(self, loader: DatasetLoader, config: dict[str, Any] | None = None):
        self.loader = loader
        self.config = config or {}

    def build_topology(self, output_path: str | Path) -> dict[str, Any]:
        """Generate a complete topology.json from dataset samples.

        Iterates the dataset to collect statistics, then writes an
        enriched topology file.

        Args:
            output_path: Where to write the enriched topology JSON.

        Returns:
            The topology dict.
        """
        output_path = Path(output_path)

        # Start from static template
        template_path = output_path.parent / "topology.json"
        if template_path.exists():
            with open(template_path, encoding="utf-8") as f:
                topology = json.load(f)
        else:
            topology = {"schema_version": "1.0", "nodes": [], "edges": []}

        # Collect per-zone statistics from samples
        zone_stats = self._collect_zone_stats()

        # Collect troubleshooting tickets
        tickets = self._collect_tickets()

        # Enrich nodes with dataset-derived attributes
        self._enrich_nodes(topology, zone_stats, tickets)

        # Write enriched topology
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(topology, f, indent=2, ensure_ascii=False)

        return topology

    def _collect_zone_stats(self) -> dict[str, list[dict]]:
        """Collect KPI statistics per zone from dataset samples (sample up to 500)."""
        zone_data: dict[str, list[dict]] = {"A": [], "B": [], "C": []}

        max_samples = min(len(self.loader), 500)
        for i in range(max_samples):
            sample = self.loader[i]
            labels = sample.get("labels", {})
            zone = labels.get("zone", "")
            if zone in zone_data and sample.get("statistics"):
                zone_data[zone].append(sample["statistics"])

        return zone_data

    def _collect_tickets(self, max_tickets: int = 200) -> list[dict]:
        """Collect troubleshooting tickets from anomalous samples."""
        tickets = []
        max_samples = min(len(self.loader), 2000)
        for i in range(max_samples):
            sample = self.loader[i]
            anomalies = sample.get("anomalies", {})
            ticket = sample.get("troubleshooting_tickets", "")
            if anomalies and anomalies.get("type") and ticket:
                tickets.append({
                    "sample_index": i,
                    "anomaly_type": anomalies["type"],
                    "text": str(ticket),
                })
            if len(tickets) >= max_tickets:
                break
        return tickets

    def _enrich_nodes(
        self,
        topology: dict,
        zone_stats: dict[str, list[dict]],
        tickets: list[dict],
    ) -> None:
        """Add dataset-derived attributes to topology nodes."""
        for node in topology.get("nodes", []):
            node_type = node.get("type", "")

            if node_type.startswith("Zone_"):
                zone_letter = node_type[-1]
                stats_list = zone_stats.get(zone_letter, [])
                if stats_list:
                    node.setdefault("attributes", {})
                    node["attributes"]["sample_count"] = len(stats_list)

            elif node_type == "Jammer":
                # Check if any samples have Jamming anomalies
                jamming_count = sum(
                    1 for t in tickets
                    if t.get("anomaly_type") == "Jamming"
                )
                node.setdefault("attributes", {})
                node["attributes"]["jamming_samples"] = jamming_count
                node["attributes"]["active"] = jamming_count > 0
