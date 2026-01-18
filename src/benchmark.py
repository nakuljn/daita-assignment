import numpy as np
import time
from typing import List, Dict, Callable, Any, Tuple


def compute_recall(predicted: List[int], ground_truth: List[int]) -> float:
    if not ground_truth:
        return 1.0
    found = len(set(predicted) & set(ground_truth))
    return found / len(ground_truth)


def time_query(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def run_benchmark(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: Dict,
    brute_force_fn: Callable,
    hnsw_query_fn: Callable,
    k: int = 10,
    num_queries: int = 10
) -> Dict:
    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    
    bf_times, bf_recalls = [], []
    hnsw_times, hnsw_recalls = [], []
    
    for i in range(num_queries):
        query = queries[i]
        query_id = f"query_{i:03d}"
        true_idx = ground_truth[query_id]["indices"][:k]
        
        bf_res, bf_t = time_query(brute_force_fn, query, vectors, k)
        bf_times.append(bf_t)
        bf_recalls.append(compute_recall([r[0] for r in bf_res], true_idx))
        
        hnsw_res, hnsw_t = time_query(hnsw_query_fn, query, k, 100)
        hnsw_times.append(hnsw_t)
        hnsw_recalls.append(compute_recall([r[0] for r in hnsw_res], true_idx))
    
    bf_avg_t, bf_avg_r = np.mean(bf_times), np.mean(bf_recalls)
    hnsw_avg_t, hnsw_avg_r = np.mean(hnsw_times), np.mean(hnsw_recalls)
    speedup = bf_avg_t / hnsw_avg_t if hnsw_avg_t > 0 else 0
    
    print(f"\nQueries: {num_queries}, k: {k}")
    print("-" * 60)
    print(f"{'Method':<15} {'Latency':<15} {'Recall@k':<15}")
    print("-" * 60)
    print(f"{'Brute Force':<15} {bf_avg_t:.3f} ms{'':<6} {bf_avg_r:.2%}")
    print(f"{'HNSW':<15} {hnsw_avg_t:.3f} ms{'':<6} {hnsw_avg_r:.2%}")
    print("-" * 60)
    print(f"Speedup: {speedup:.2f}x")
    
    return {
        "brute_force": {"latency_ms": bf_avg_t, "recall": bf_avg_r},
        "hnsw": {"latency_ms": hnsw_avg_t, "recall": hnsw_avg_r},
        "speedup": speedup
    }
