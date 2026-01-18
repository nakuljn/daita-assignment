"""Unit tests for HNSW index."""
import numpy as np
from src.hnsw_index import HNSWIndex
from src.brute_force import brute_force_search


def test_build_creates_layers():
    np.random.seed(42)
    vectors = np.random.randn(100, 16)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    index.build(vectors, verbose=False)

    stats = index.get_stats()
    assert stats["vectors"] == 100
    assert stats["layers"] >= 1


def test_query_returns_k_results():
    np.random.seed(42)
    vectors = np.random.randn(100, 16)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    index = HNSWIndex(dim=16, M=8, ef_construction=50)
    index.build(vectors, verbose=False)

    results = index.query(vectors[0], k=5)
    assert len(results) == 5


def test_self_is_top_result():
    np.random.seed(42)
    vectors = np.random.randn(100, 16)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    index = HNSWIndex(dim=16, M=16, ef_construction=100)
    index.build(vectors, verbose=False)

    results = index.query(vectors[0], k=5)
    assert results[0][0] == 0


def test_recall_above_threshold():
    np.random.seed(42)
    vectors = np.random.randn(500, 32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    index = HNSWIndex(dim=32, M=16, ef_construction=100)
    index.build(vectors, verbose=False)

    total_recall = 0
    for i in range(10):
        query = vectors[i]
        hnsw_results = set(r[0] for r in index.query(query, k=10, ef_search=100))
        bf_results = set(r[0] for r in brute_force_search(query, vectors, k=10))
        total_recall += len(hnsw_results & bf_results) / 10

    avg_recall = total_recall / 10
    assert avg_recall >= 0.7
