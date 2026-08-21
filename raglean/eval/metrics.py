"""Answer-quality metric for the eval harness.

Uses token-level F1 against a gold answer (the same style of metric used by SQuAD),
rather than an LLM-judge, so the eval harness runs offline with zero API cost/keys.
This is a proxy for quality: it captures whether the answer's *content* survived
caching/routing/compression, not the fluency a real judge model would score. Swapping
in a RAGAS-style LLM-judge is a documented, config-flagged upgrade for when API budget
is available — see docs/ARCHITECTURE.md, section 6.
"""
from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
