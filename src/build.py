#!/usr/bin/env python3
"""Build HNSW index from vectors."""

import argparse
import pickle
import time
import numpy as np
from pathlib import Path

from .hnsw_index import HNSWIndex


def main():
    parser = argparse.ArgumentParser(description="Build HNSW index")
    parser.add_argument("--data", type=str, default="data/syndata-vectors")
    parser.add_argument("--output", type=str, default="output/index.pkl")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef", type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOADING VECTORS")  
    print("=" * 60)
    
    data_path = Path(args.data)
    vectors = np.load(data_path / "vectors.npy")
    print(f"Loaded: {vectors.shape[0]} vectors, {vectors.shape[1]} dimensions")
    
    print("\n" + "=" * 60)
    print("BUILDING INDEX")
    print("=" * 60)
    
    start = time.time()
    index = HNSWIndex(dim=vectors.shape[1], M=args.M, ef_construction=args.ef)
    index.build(vectors)
    build_time = time.time() - start
    
    print(f"\nBuild time: {build_time:.2f}s")
    print(f"Stats: {index.get_stats()}")
    
    print("\n" + "=" * 60)
    print("SAVING INDEX")
    print("=" * 60)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
    
    print(f"Saved: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
