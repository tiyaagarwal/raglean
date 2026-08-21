"""Context compressor: trims retrieved chunks down to the sentences most relevant to
the query before they're sent to the LLM, instead of forwarding raw chunks verbatim.

Approach: sentence-level extractive scoring (cosine similarity between each sentence's
embedding and the query embedding) + greedy selection under a token budget, restoring
original order so the final context still reads coherently. This is a simpler cousin of
approaches like LLMLingua / context-compressor libraries; it's deliberately
lightweight (no extra model) so it runs with the same embedder already used by the
cache and router — swapping in a learned compressor is a documented v2, not a
prerequisite for the pipeline to work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..embeddings import BaseEmbedder, LocalHashEmbedder, cosine_sim_matrix

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class CompressionResult:
    compressed_text: str
    tokens_before: int
    tokens_after: int
    sentences_kept: int
    sentences_total: int

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio(self) -> float:
        return 1 - (self.tokens_after / self.tokens_before) if self.tokens_before else 0.0


class ContextCompressor:
    def __init__(self, embedder: BaseEmbedder | None = None, token_budget: int = 300):
        self.embedder = embedder or LocalHashEmbedder()
        self.token_budget = token_budget

    def compress(self, query: str, chunks: list[str], token_budget: int | None = None) -> CompressionResult:
        budget = token_budget or self.token_budget
        full_text = "\n".join(chunks)
        tokens_before = _approx_tokens(full_text)

        sentences: list[str] = []
        for chunk in chunks:
            sentences.extend(_split_sentences(chunk))

        if not sentences:
            return CompressionResult(full_text, tokens_before, tokens_before, 0, 0)

        if tokens_before <= budget:
            # Already under budget — nothing to trim.
            return CompressionResult(full_text, tokens_before, tokens_before, len(sentences), len(sentences))

        query_vec = self.embedder.embed_one(query)
        sentence_vecs = self.embedder.embed(sentences)
        scores = cosine_sim_matrix(query_vec[None, :], sentence_vecs)[0]

        order = np.argsort(-scores)  # most relevant first
        kept_idx: list[int] = []
        running_tokens = 0
        for idx in order:
            cost = _approx_tokens(sentences[idx])
            if running_tokens + cost > budget and kept_idx:
                continue
            kept_idx.append(int(idx))
            running_tokens += cost
            if running_tokens >= budget:
                break

        kept_idx.sort()  # restore original reading order
        compressed = " ".join(sentences[i] for i in kept_idx)
        tokens_after = _approx_tokens(compressed)

        return CompressionResult(
            compressed_text=compressed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            sentences_kept=len(kept_idx),
            sentences_total=len(sentences),
        )
