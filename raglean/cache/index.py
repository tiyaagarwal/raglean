"""Vector index abstraction. `NumpyFlatIndex` is a brute-force cosine search — perfectly
fine up to tens of thousands of cached queries and has zero extra dependencies. Swap in
a FAISS-backed index (IndexFlatIP for exact, IndexIVFFlat/HNSW for scale) by
implementing the same `VectorIndex` interface once cache size demands it — the
docstring below shows exactly what that swap looks like.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..embeddings import cosine_sim_matrix


class VectorIndex(ABC):
    @abstractmethod
    def add(self, vecs: np.ndarray, ids: list[int]) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int = 1) -> list[tuple[int, float]]:
        """Returns up to k (id, cosine_similarity) pairs, best first."""
        ...

    @abstractmethod
    def remove(self, ids: list[int]) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...


class NumpyFlatIndex(VectorIndex):
    """Brute-force exact cosine search.

    FAISS swap-in (production, >~50k entries), same interface:

        import faiss
        class FaissFlatIndex(VectorIndex):
            def __init__(self, dim):
                self._index = faiss.IndexFlatIP(dim)
                self._ids = []
            def add(self, vecs, ids):
                self._index.add(vecs); self._ids.extend(ids)
            def search(self, query_vec, k=1):
                scores, idxs = self._index.search(query_vec[None, :], k)
                return [(self._ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[int] = []
        self._id_to_row: dict[int, int] = {}

    def add(self, vecs: np.ndarray, ids: list[int]) -> None:
        assert vecs.shape[0] == len(ids)
        for i, vec_id in enumerate(ids):
            self._id_to_row[vec_id] = self._vecs.shape[0] + i
        self._vecs = np.vstack([self._vecs, vecs]) if self._vecs.shape[0] else vecs.copy()
        self._ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int = 1) -> list[tuple[int, float]]:
        if self._vecs.shape[0] == 0:
            return []
        sims = cosine_sim_matrix(query_vec[None, :], self._vecs)[0]
        k = min(k, sims.shape[0])
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self._ids[i], float(sims[i])) for i in top_idx]

    def remove(self, ids: list[int]) -> None:
        remove_set = set(ids)
        keep_rows = [i for i, vid in enumerate(self._ids) if vid not in remove_set]
        self._vecs = self._vecs[keep_rows] if keep_rows else np.zeros((0, self.dim), dtype=np.float32)
        self._ids = [self._ids[i] for i in keep_rows]
        self._id_to_row = {vid: i for i, vid in enumerate(self._ids)}

    def __len__(self) -> int:
        return len(self._ids)
