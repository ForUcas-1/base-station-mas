"""Server-side topology graph renderer using matplotlib + networkx.

Renders the knowledge graph as a PNG image. Supports highlighting
anomaly paths (e.g., Jammer→BS→Zone_B) when evidence is provided.
"""

import io
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

TOPOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge_graph" / "topology.json"

NODE_COLORS = {
    "BaseStation": "#58a6ff",
    "Jammer": "#f85149",
    "Zone_A": "#3fb950",
    "Zone_B": "#d2991d",
    "Zone_C": "#db6d28",
    "Mobile": "#a371f7",
}

NODE_SHAPES = {
    "BaseStation": "s",   # square
    "Jammer": "D",         # diamond
    "Zone_A": "o",
    "Zone_B": "o",
    "Zone_C": "o",
    "Mobile": "^",         # triangle
}

NODE_SIZES = {
    "BaseStation": 800,
    "Jammer": 600,
    "Zone_A": 500,
    "Zone_B": 500,
    "Zone_C": 500,
    "Mobile": 400,
}


def render_topology(
    highlight_path: list[str] | None = None,
    highlight_color: str = "#f85149",
    highlight_width: int = 3,
) -> bytes:
    """Render the knowledge graph as a PNG and return bytes.

    Args:
        highlight_path: Optional list of node IDs to highlight (e.g., ['JAM_001', 'BS_001', 'ZB_001']).
        highlight_color: Color for highlighted nodes/edges.

    Returns:
        PNG image bytes.
    """
    with open(TOPOLOGY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.MultiDiGraph()

    # Add nodes
    for node in data.get("nodes", []):
        G.add_node(node["id"], **node.get("attributes", {}),
                   node_type=node["type"], label=node.get("label", node["type"]))

    # Add edges
    for edge in data.get("edges", []):
        G.add_edge(edge["source"], edge["target"],
                   type=edge["type"], **edge.get("attributes", {}))

    # Build layout
    pos = _build_layout(G, data)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # Highlight set
    hl_nodes = set(highlight_path or [])

    # Draw edges
    for u, v, key, edata in G.edges(keys=True, data=True):
        in_hl = u in hl_nodes and v in hl_nodes
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=highlight_color if in_hl else "#484f58",
            width=highlight_width if in_hl else 1,
            alpha=0.8, ax=ax,
            connectionstyle="arc3,rad=0.1",
            arrows=True, arrowsize=12,
        )
        # Edge label
        if not in_hl:
            mid = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
            ax.text(mid[0], mid[1] + 0.03, edata.get("type", ""),
                    fontsize=7, color="#8b949e", ha="center", va="bottom",
                    bbox=dict(facecolor="#0d1117", edgecolor="none", pad=0))

    # Draw nodes
    for ntype in NODE_COLORS:
        nodelist = [n for n, d in G.nodes(data=True) if d.get("node_type") == ntype]
        if not nodelist:
            continue
        colors = []
        sizes = []
        for n in nodelist:
            colors.append(highlight_color if n in hl_nodes else NODE_COLORS.get(ntype, "#484f58"))
            sizes.append(NODE_SIZES.get(ntype, 400) * 1.3 if n in hl_nodes else NODE_SIZES.get(ntype, 400))
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist,
            node_color=colors, node_size=sizes,
            node_shape=NODE_SHAPES.get(ntype, "o"),
            ax=ax, edgecolors="#30363d", linewidths=1,
        )

    # Labels
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(
        G, pos, labels, font_size=9, font_color="#c9d1d9",
        font_family="sans-serif", ax=ax,
    )

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor="#0d1117",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_layout(G: nx.MultiDiGraph, data: dict) -> dict:
    """Build a manual layout that matches the architecture diagram."""
    pos = {
        "JAM_001": (0.0, 0.75),
        "BS_001": (0.5, 0.75),
        "ZA_001": (0.3, 0.35),
        "ZB_001": (0.7, 0.35),
        "ZC_001": (0.5, 0.0),
        "MOB_001": (0.15, 0.35),
        "MOB_002": (0.85, 0.35),
    }
    # Fill any missing nodes
    for n in G.nodes():
        if n not in pos:
            pos[n] = (0.5, 0.5)
    return pos
