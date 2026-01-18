"""Unit tests for brute force search."""
import numpy as np
from src.brute_force import brute_force_search


def test_returns_k_results():
    vectors = np.random.randn(100, 10)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0]
    results = brute_force_search(query, vectors, k=5)
    assert len(results) == 5


def test_most_similar_is_self():
    vectors = np.random.randn(100, 10)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0]
    results = brute_force_search(query, vectors, k=1)
    assert results[0][0] == 0


def test_results_sorted_descending():
    vectors = np.random.randn(100, 10)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0]
    results = brute_force_search(query, vectors, k=10)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)
