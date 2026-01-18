import numpy as np
from typing import List, Tuple
from .similarity import batch_cosine_similarity


def brute_force_search(query: np.ndarray, vectors: np.ndarray, k: int) -> List[Tuple[int, float]]:
    similarities = batch_cosine_similarity(query, vectors)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [(int(idx), float(similarities[idx])) for idx in top_k_indices]
