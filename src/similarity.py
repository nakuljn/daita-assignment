import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity for pre-normalized vectors. For unnormalized vectors, divide by norms."""
    return float(np.dot(vec_a, vec_b))


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return 1.0 - np.dot(vec_a, vec_b)


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.dot(vectors, query)
