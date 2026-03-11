"""Hybrid graph memory: in-memory property graph + optional vector index.

Supports:
  - MemoryNode with provenance, confidence, and type metadata
  - MemoryEdge with relationship types and weights
  - Bag-of-hash embedding for toy similarity search
  - Optional FAISS integration for fast top-k retrieval
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryNode:
    node_id: str
    text: str
    node_type: str = "FACT"  # CONCEPT, FACT, PROCEDURE, EXAMPLE
    source_type: str = "user"  # teacher, execution, user, web
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MemoryEdge:
    edge_id: str
    source_id: str
    target_id: str
    rel_type: str = "RELATED_TO"  # RELATED_TO, IMPLIES, USED_IN, CONTRADICTS
    weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def embed_text(text: str, dim: int = 64) -> np.ndarray:
    """Toy embedding: bag-of-hash projection, L2-normalized."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        v[hash(tok) % dim] += 1.0
    n = np.linalg.norm(v) + 1e-8
    return v / n


class GraphMemory:
    """In-memory hybrid graph + vector store."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        self.embs: List[np.ndarray] = []
        self.ids: List[str] = []
        self.faiss_index = None

        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(dim)
        except ImportError:
            pass

    def add_node(self, node: MemoryNode):
        self.nodes[node.node_id] = node
        e = embed_text(node.text, self.dim)
        self.embs.append(e)
        self.ids.append(node.node_id)
        if self.faiss_index is not None:
            self.faiss_index.add(e.reshape(1, -1))

    def add_edge(self, source_id: str, target_id: str, rel_type: str = "RELATED_TO", weight: float = 1.0):
        edge_id = str(uuid.uuid4())
        edge = MemoryEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            weight=weight,
        )
        self.edges[edge_id] = edge
        return edge

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (node_id, score) pairs by cosine similarity."""
        if not self.ids:
            return []
        q = embed_text(query, self.dim).reshape(1, -1)
        if self.faiss_index is not None:
            scores, idx = self.faiss_index.search(q, min(k, len(self.ids)))
            return [(self.ids[i], float(scores[0][j])) for j, i in enumerate(idx[0]) if i >= 0]
        # brute force fallback
        sims = []
        for nid, emb in zip(self.ids, self.embs):
            score = float(np.dot(q.flatten(), emb))
            sims.append((nid, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[Tuple[str, MemoryEdge]]:
        """Return (neighbor_id, edge) pairs for a given node."""
        result = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                result.append((edge.target_id, edge))
            elif edge.target_id == node_id:
                result.append((edge.source_id, edge))
        return result

    def prune_low_confidence(self, threshold: float = 0.1):
        """Remove nodes below confidence threshold."""
        to_remove = [nid for nid, node in self.nodes.items() if node.confidence < threshold]
        for nid in to_remove:
            del self.nodes[nid]
            # Remove associated edges
            edge_ids_to_remove = [
                eid for eid, e in self.edges.items()
                if e.source_id == nid or e.target_id == nid
            ]
            for eid in edge_ids_to_remove:
                del self.edges[eid]
        # Note: FAISS index would need rebuilding after pruning in production
        return to_remove
