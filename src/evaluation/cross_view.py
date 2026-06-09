"""Helpers for cross-view residual-linkability evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def max_similarity_to_gallery(query_embedding: np.ndarray, gallery_embeddings: np.ndarray) -> float:
    """Return the maximum cosine similarity between one query and a gallery."""
    if gallery_embeddings.size == 0:
        return 0.0
    similarities = np.dot(gallery_embeddings, query_embedding)
    return float(np.max(similarities))


def build_control_groups(rows: Iterable[dict[str, object]]) -> dict[tuple[str, str], list[int]]:
    """Index row positions by day and exocentric stream for mismatched-timestamp controls."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(str(row["day_id"]), str(row["exocentric_stream_id"]))].append(idx)
    return grouped
