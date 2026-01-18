#!/usr/bin/env python3
"""Demo: build + query + benchmark."""

import time
from . import load_all, brute_force_search, HNSWIndex, run_benchmark


def main():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    data = load_all()
    vectors = data["vectors"]
    queries = data["queries"]
    ground_truth = data["ground_truth"]
    
    print(f"Vectors: {vectors.shape}")
    print(f"Queries: {queries.shape}")
    
    print("\n" + "=" * 60)
    print("BUILDING INDEX")
    print("=" * 60)
    
    start = time.time()
    index = HNSWIndex(dim=128, M=16, ef_construction=100)
    index.build(vectors)
    build_time = time.time() - start
    
    print(f"\nBuild time: {build_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("SINGLE QUERY TEST")
    print("=" * 60)
    
    query = queries[0]
    
    print("\nBrute Force:")
    for idx, sim in brute_force_search(query, vectors, 5):
        print(f"  {idx}: {sim:.4f}")
    
    print("\nHNSW:")
    for idx, sim in index.query(query, 5):
        print(f"  {idx}: {sim:.4f}")
    
    results = run_benchmark(
        vectors=vectors,
        queries=queries,
        ground_truth=ground_truth,
        brute_force_fn=brute_force_search,
        hnsw_query_fn=index.query,
        k=10,
        num_queries=10
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Build: {build_time:.2f}s")
    print(f"Recall: {results['hnsw']['recall']:.2%}")
    print(f"Speedup: {results['speedup']:.2f}x")


if __name__ == "__main__":
    main()
