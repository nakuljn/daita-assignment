"""Unit tests for cosine similarity functions."""
import numpy as np
import pytest
from src.similarity import cosine_similarity, batch_cosine_similarity


def test_identical_vectors():
    vec = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_orthogonal_vectors():
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)


def test_opposite_vectors():
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([-1.0, 0.0, 0.0])
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)


def test_batch_similarity():
    query = np.array([1.0, 0.0])
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    result = batch_cosine_similarity(query, vectors)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
