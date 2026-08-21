"""Simulated production traffic for the benchmark, sampled from the eval dataset with a
Zipfian popularity skew — real support/FAQ traffic is dominated by a small number of
frequently-repeated questions, which is exactly the pattern semantic caching is meant to
exploit. A uniform-random sample would understate real-world cache effectiveness; a
fixed skew parameter keeps the simulation honest about *why* the hit rate is what it is.
"""
from __future__ import annotations

import random

from ..eval.dataset import load_eval_dataset


def generate_workload(n_requests: int = 500, seed: int = 7, zipf_skew: float = 1.2) -> list[dict]:
    rng = random.Random(seed)
    dataset = load_eval_dataset()

    order = list(range(len(dataset)))
    rng.shuffle(order)  # randomize which questions are "popular" this run
    weights = [1.0 / ((rank + 1) ** zipf_skew) for rank in range(len(dataset))]

    sampled_indices = rng.choices(order, weights=weights, k=n_requests)
    return [dataset[i] for i in sampled_indices]
