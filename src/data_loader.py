import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple


def load_vectors(data_dir: str = "data/syndata-vectors") -> Tuple[np.ndarray, np.ndarray]:
    path = Path(data_dir)
    vectors = np.load(path / "vectors.npy")
    vector_ids = np.load(path / "vector_ids.npy")
    return vectors, vector_ids


def load_queries(data_dir: str = "data/syndata-queries") -> Tuple[np.ndarray, np.ndarray]:
    path = Path(data_dir)
    queries = np.load(path / "queries.npy")
    query_ids = np.load(path / "query_ids.npy")
    return queries, query_ids


def load_ground_truth(data_dir: str = "data/syndata-queries") -> Dict:
    path = Path(data_dir) / "ground_truth.json"
    with open(path, "r") as f:
        return json.load(f)


def load_all(base_dir: str = "data") -> Dict:
    vectors, vector_ids = load_vectors(f"{base_dir}/syndata-vectors")
    queries, query_ids = load_queries(f"{base_dir}/syndata-queries")
    ground_truth = load_ground_truth(f"{base_dir}/syndata-queries")
    
    return {
        "vectors": vectors,
        "vector_ids": vector_ids,
        "queries": queries,
        "query_ids": query_ids,
        "ground_truth": ground_truth
    }
