#!/usr/bin/env python3
"""Query HNSW index."""

import argparse
import pickle
import time
import json
import numpy as np
from pathlib import Path

from .brute_force import brute_force_search
from .benchmark import compute_recall


def load_index(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Query HNSW index")
    parser.add_argument("--index", type=str, default="output/index.pkl")
    parser.add_argument("--queries", type=str, default="data/syndata-queries")
    parser.add_argument("--query", type=int, nargs="*", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOADING")
    print("=" * 60)
    
    index = load_index(args.index)
    print(f"Index: {index.get_stats()}")
    
    queries_path = Path(args.queries)
    queries = np.load(queries_path / "queries.npy")
    print(f"Queries: {len(queries)}")
    
    gt_path = queries_path / "ground_truth.json"
    ground_truth = None
    if gt_path.exists():
        with open(gt_path) as f:
            ground_truth = json.load(f)
    
    if args.query is not None:
        print("\n" + "=" * 60)
        print("QUERY MODE")
        print("=" * 60)
        
        for qi in args.query:
            query = queries[qi]
            
            start = time.perf_counter()
            results = index.query(query, args.k, ef_search=args.ef)
            latency = (time.perf_counter() - start) * 1000
            
            print(f"\n[Query {qi}] ({latency:.2f}ms)")
            for idx, sim in results[:5]:
                print(f"  {idx}: {sim:.4f}")
            
            if ground_truth:
                true_idx = ground_truth[f"query_{qi:03d}"]["indices"][:args.k]
                recall = compute_recall([r[0] for r in results], true_idx)
                print(f"  Recall: {recall:.2%}")
    else:
        print("\n" + "=" * 60)
        print("BENCHMARK MODE")
        print("=" * 60)
        
        num_test = min(10, len(queries))
        hnsw_times, hnsw_recalls = [], []
        bf_times = []
        
        for i in range(num_test):
            query = queries[i]
            
            start = time.perf_counter()
            hnsw_res = index.query(query, args.k, ef_search=args.ef)
            hnsw_times.append((time.perf_counter() - start) * 1000)
            
            start = time.perf_counter()
            bf_res = brute_force_search(query, index.vectors, args.k)
            bf_times.append((time.perf_counter() - start) * 1000)
            
            if ground_truth:
                true_idx = ground_truth[f"query_{i:03d}"]["indices"][:args.k]
                hnsw_recalls.append(compute_recall([r[0] for r in hnsw_res], true_idx))
        
        print(f"\nQueries: {num_test}, k: {args.k}")
        print("-" * 60)
        print(f"Brute Force: {np.mean(bf_times):.3f} ms")
        print(f"HNSW: {np.mean(hnsw_times):.3f} ms, Recall: {np.mean(hnsw_recalls):.2%}")
        print(f"Speedup: {np.mean(bf_times)/np.mean(hnsw_times):.2f}x")


if __name__ == "__main__":
    main()
