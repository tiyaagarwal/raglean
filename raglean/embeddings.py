"""Shared embedding backend used by the semantic cache, the router, and the compressor.

Default backend is a stateless local embedder (HashingVectorizer + L2 normalize) so the
whole project runs offline, deterministically, with no model download and no API key.
This is a deliberate, documented trade-off (see docs/ARCHITECTURE.md, section 6):
for production use, swap in `SentenceTransformerEmbedder` (needs the optional
`sentence-transformers` dependency) or an API-based embedder without touching any other
module — everything downstream depends only on the `BaseEmbedder` interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) L2-normalized float32 array."""
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class LocalHashEmbedder(BaseEmbedder):
    """Stateless, offline embedder. Not as semantically rich as a trained sentence
    encoder, but deterministic, fast, and requires no download — good enough to
    demonstrate and benchmark the *system* (caching, routing, compression logic)
    without coupling the whole project to a model download / API key.
    """

    def __init__(self, n_features: int = 512, ngram_range: tuple[int, int] = (1, 2)):
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            ngram_range=ngram_range,
            norm=None,
            lowercase=True,
        )
        self.dim = n_features

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        mat = normalize(mat, norm="l2", axis=1)
        return mat


class SentenceTransformerEmbedder(BaseEmbedder):
    """Production swap-in. Requires `pip install sentence-transformers`.
    Not used by default so the project has zero heavy/networked dependencies out of
    the box — pass `embedder=SentenceTransformerEmbedder()` to `SemanticCache` or
    `ContextCompressor` once that's installed and you want real semantic quality
    instead of the offline hash embedder.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` to use this embedder."
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def cosine_sim_matrix(query_vecs: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """query_vecs: (n_q, d) L2-normalized, corpus_vecs: (n_c, d) L2-normalized.
    Returns (n_q, n_c) cosine similarity matrix. Plain numpy — swappable for a FAISS
    IndexFlatIP (or IVF/HNSW for scale) via cache/index.py's `VectorIndex` interface.
    """
    if corpus_vecs.shape[0] == 0:
        return np.zeros((query_vecs.shape[0], 0), dtype=np.float32)
    return query_vecs @ corpus_vecs.T
