"""Optional FAISS vector index wrapper for memory retrieval."""

from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaissIndex:
    """Thin wrapper around FAISS IndexFlatIP for inner-product similarity search."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.index = None
        self.ids: List[str] = []
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)

    @property
    def available(self) -> bool:
        return self.index is not None

    def add(self, node_id: str, embedding: np.ndarray):
        if not self.available:
            return
        self.ids.append(node_id)
        self.index.add(embedding.reshape(1, -1).astype(np.float32))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if not self.available or not self.ids:
            return []
        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, idx = self.index.search(q, min(k, len(self.ids)))
        return [
            (self.ids[i], float(scores[0][j]))
            for j, i in enumerate(idx[0])
            if i >= 0
        ]

    def reset(self):
        if self.available:
            self.index.reset()
            self.ids.clear()
