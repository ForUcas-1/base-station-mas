"""Text embedding and vector index for GraphRAG nodes."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class GraphEmbedder:
    """Embeds text fields from knowledge graph nodes into a vector index.

    Uses sentence-transformers for embedding and FAISS for indexing.

    Embeddable fields per node:
      - BaseStation: description text
      - Jammer: constructed description
      - Zone_A/B/C: zone description + typical symptoms
      - Mobile: application + mobility description
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_store: str = "faiss",
        device: str = "cpu",
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.vector_store = vector_store
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._index = None
        self._id_to_meta: dict[int, dict] = {}

    @property
    def model(self):
        """Lazy-load the sentence-transformer model from local path.

        Resolves model_name to a local snapshot path under cache_dir.
        No network requests — model must be pre-downloaded.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            local_path = self._resolve_local_path()
            self._model = SentenceTransformer(local_path, device=self.device)
        return self._model

    def _resolve_local_path(self) -> str:
        """Resolve 'all-MiniLM-L6-v2' to the local snapshot path under cache_dir.

        HuggingFace cache layout:
          <cache_dir>/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/
        """
        if self.cache_dir:
            from pathlib import Path
            import glob

            cache = Path(self.cache_dir)
            model_dir = cache / f"models--sentence-transformers--{self.model_name.replace('/', '--')}"
            snapshots = list((model_dir / "snapshots").glob("*"))
            if snapshots:
                # Return the first (and usually only) snapshot
                return str(snapshots[0])

        # Fallback: let SentenceTransformer download from HF (network required)
        return self.model_name

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build_index(
        self,
        topology_path: str | Path,
        output_dir: str | Path,
        descriptions: dict[str, str] | None = None,
    ) -> str:
        """Build a FAISS index from knowledge graph node descriptions.

        Args:
            topology_path: Path to topology.json.
            output_dir: Directory to write index.faiss and meta.json.
            descriptions: Optional dict mapping embedding_id → text.
                          If None, auto-generates descriptions from node attrs.

        Returns:
            Path to the created FAISS index file.
        """
        import faiss

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(topology_path, encoding="utf-8") as f:
            topology = json.load(f)

        # Collect texts and IDs
        texts = []
        ids = []
        meta_map: dict[int, dict] = {}

        for i, node in enumerate(topology.get("nodes", [])):
            embed_id = node.get("attributes", {}).get("description_embedding_id", "")
            if descriptions and embed_id in descriptions:
                text = descriptions[embed_id]
            else:
                text = self._node_to_text(node)

            if not text.strip():
                text = f"{node.get('type', 'Unknown')}: {node.get('label', '')}"

            texts.append(text)
            ids.append(i)
            meta_map[i] = {
                "node_id": node["id"],
                "type": node["type"],
                "label": node.get("label", ""),
                "embedding_id": embed_id,
            }

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Build FAISS index (cosine similarity via inner product on normalized vectors)
        dim = embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

        # Save
        index_path = str(output_dir / "index.faiss")
        faiss.write_index(index, index_path)

        meta_path = output_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_map, f, indent=2, ensure_ascii=False)

        self._index = index
        self._id_to_meta = meta_map

        return index_path

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        index_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for top-k similar nodes by text embedding.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            index_path: Path to FAISS index. If None, uses cached index.

        Returns:
            List of {node_id, type, label, similarity, ...} dicts.
        """
        import faiss

        if self._index is None:
            if index_path is None:
                raise ValueError("No index loaded and no index_path provided")
            self._index = faiss.read_index(index_path)
            self._load_meta(index_path)

        # Encode query
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._id_to_meta.get(int(idx), {})
            results.append({
                **meta,
                "similarity": float(score),
            })

        return results

    def _load_meta(self, index_path: str) -> None:
        meta_path = str(Path(index_path).parent / "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self._id_to_meta = {
                    int(k): v for k, v in json.load(f).items()
                }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _node_to_text(node: dict) -> str:
        """Convert a topology node to a searchable text string."""
        parts = [node.get("type", ""), node.get("label", "")]
        attrs = node.get("attributes", {})

        for key in ("typical_symptoms", "impact", "description"):
            val = attrs.get(key, "")
            if isinstance(val, list):
                parts.append(", ".join(str(v) for v in val))
            elif val:
                parts.append(str(val))

        return " | ".join(p for p in parts if p)
