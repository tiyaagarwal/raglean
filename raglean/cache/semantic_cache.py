"""Retrieval-layer semantic cache.

The key design choice vs. a generic LLM-gateway cache (e.g. GPTCache-style full-prompt
caching): this cache keys off the *user query* (and optionally a coarse context
fingerprint), not the fully-assembled RAG prompt. Two requests with the same question
but slightly different retrieved-chunk ordering, or minor prompt-template changes, still
hit the same cache entry — which is exactly the case a full-prompt cache misses.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from ..embeddings import BaseEmbedder, LocalHashEmbedder
from ..providers.base import LLMResponse
from .index import NumpyFlatIndex, VectorIndex

_id_counter = itertools.count()


@dataclass
class CacheEntry:
    entry_id: int
    query: str
    response: LLMResponse
    created_at: float
    ttl_s: float | None
    hits: int = 0

    def is_expired(self, now: float) -> bool:
        return self.ttl_s is not None and (now - self.created_at) > self.ttl_s


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class SemanticCache:
    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        index: VectorIndex | None = None,
        similarity_threshold: float = 0.86,
        ttl_s: float | None = 3600.0,
        max_entries: int = 20_000,
    ):
        self.embedder = embedder or LocalHashEmbedder()
        self.index = index or NumpyFlatIndex(dim=self.embedder.dim)
        self.similarity_threshold = similarity_threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._entries: dict[int, CacheEntry] = {}
        self.stats = CacheStats()

    def lookup(self, query: str) -> CacheEntry | None:
        self._evict_expired()
        if len(self.index) == 0:
            self.stats.misses += 1
            return None

        query_vec = self.embedder.embed_one(query)
        candidates = self.index.search(query_vec, k=1)
        if not candidates:
            self.stats.misses += 1
            return None

        best_id, score = candidates[0]
        entry = self._entries.get(best_id)
        if entry is None or entry.is_expired(time.time()):
            self.stats.misses += 1
            return None

        if score >= self.similarity_threshold:
            entry.hits += 1
            self.stats.hits += 1
            return entry

        self.stats.misses += 1
        return None

    def store(self, query: str, response: LLMResponse) -> CacheEntry:
        if len(self._entries) >= self.max_entries:
            self._evict_oldest(n=max(1, self.max_entries // 20))  # evict oldest 5%

        entry_id = next(_id_counter)
        vec = self.embedder.embed_one(query)
        self.index.add(vec[None, :], [entry_id])
        entry = CacheEntry(
            entry_id=entry_id,
            query=query,
            response=response,
            created_at=time.time(),
            ttl_s=self.ttl_s,
        )
        self._entries[entry_id] = entry
        return entry

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [eid for eid, e in self._entries.items() if e.is_expired(now)]
        if expired:
            self.index.remove(expired)
            for eid in expired:
                del self._entries[eid]

    def _evict_oldest(self, n: int) -> None:
        oldest = sorted(self._entries.values(), key=lambda e: e.created_at)[:n]
        ids = [e.entry_id for e in oldest]
        self.index.remove(ids)
        for eid in ids:
            del self._entries[eid]

    def __len__(self) -> int:
        return len(self._entries)
